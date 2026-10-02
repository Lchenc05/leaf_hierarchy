"""Train a configured classification experiment with validation-only selection."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path

from .checkpoints import save_checkpoint
from .config import add_common_arguments, apply_overrides, load_config
from .consistency import HierarchicalConsistencyLoss, loss_metadata
from .data import build_taxonomy, load_manifest, make_loader, split_fingerprint
from .engine import evaluate_loader, loss_components
from .models import create_model
from .reporting import save_evaluation_results
from .runtime import ROOT, SPLITS, configure_runtime, select_device, write_json

import pandas as pd
import torch
import torchvision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Set learning rate and weight decay in the TOML [training] table (lr and weight_decay).",
    )
    add_common_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Parent for a timestamped run; default: configured run_root/experiment.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weights", choices=("imagenet", "none"), default=None,
                        help="none permits an offline smoke test and changes the experiment.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--consistency-weight", type=float, default=None,
                        help="Nonnegative hierarchical consistency weight; positive values require multitask heads.")
    return parser.parse_args(argv)


def source_provenance() -> dict:
    """Record the exact Python sources, plus Git state when available."""
    package = Path(__file__).resolve().parent
    files = {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(package.rglob("*.py"))}
    result = {"source_files_sha256": files}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True, timeout=5).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                                capture_output=True, text=True, timeout=5).stdout
        result.update(git_commit=commit, git_dirty=bool(status.strip()))
    except (OSError, subprocess.SubprocessError):
        result.update(git_commit=None, git_dirty=None)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    settings = config["training"]
    device = select_device(config["runtime"]["device"])
    configure_runtime(settings["seed"], config["runtime"]["num_threads"])
    data_dir = Path(config["data"]["data_dir"])
    split_file = Path(config["data"]["split_file"])
    data, species_mapping = load_manifest(split_file, data_dir)
    taxonomy = build_taxonomy(data)
    mappings = taxonomy["class_mappings"]
    if mappings["species"] != species_mapping:
        raise ValueError("Dataset taxonomy and manifest class ordering disagree.")
    spec = {key: config["model"][key] for key in ("architecture", "tasks")}
    tasks = spec["tasks"]
    consistency_weight = settings["consistency_weight"]
    consistency = HierarchicalConsistencyLoss(taxonomy).to(device) if len(tasks) == 3 else None
    loaders = {
        split: make_loader(data, split, data_dir, settings["batch_size"], settings["seed"],
                           class_mappings=mappings, tasks=tasks, preprocessing=config["preprocessing"],
                           augmentation=config["augmentation"])
        for split in ("train", "validation")
    }

    # Preserve baseline RNG timing: reset after loader construction, before the model.
    configure_runtime(settings["seed"], config["runtime"]["num_threads"])
    model = create_model(spec, mappings, pretrained=config["model"]["weights"] == "imagenet").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    run_parent = (args.output_dir.expanduser().resolve() if args.output_dir is not None else
                  Path(config["output"]["run_root"]) / config["experiment"])
    run_dir = run_parent / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    config.update({
        "model_spec": spec, "seed": settings["seed"], "batch_size": settings["batch_size"],
        "optimizer": "AdamW", "device": str(device), "split_unit": "group_id",
        "split_manifest_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "split_records_sha256": split_fingerprint(data), "split_fingerprint_version": 2,
        "split_sizes": {split: int(data["split"].eq(split).sum()) for split in SPLITS},
        "class_mappings": mappings, "run_dir": str(run_dir), "source": source_provenance(),
        "selection_tie_breaker": "lower_validation_loss",
        "loss": loss_metadata(tasks, consistency_weight),
        "versions": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            **{name: metadata.version(name) for name in ("numpy", "pandas", "Pillow", "scikit-learn")},
        },
    })
    if args.config is not None:
        source_file = args.config.expanduser().resolve()
        config["config_source"] = {"path": str(source_file), "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest()}
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    print(f"Device: {device}\nRun directory: {run_dir}", flush=True)
    print(f"{len(species_mapping)} species | train: {len(loaders['train'].dataset)} images | "
          f"validation: {len(loaders['validation'].dataset)} images", flush=True)
    if consistency is not None:
        print(f"Hierarchical consistency weight: {consistency_weight:g}", flush=True)

    best_score = (-1.0, float("-inf"))
    history = []
    selection = config["selection"]
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        train_loss = 0.0
        component_sums = {key: 0.0 for key in ("classification_loss", "consistency_loss", "weighted_consistency_loss")}
        for batch, (images, targets) in enumerate(loaders["train"], 1):
            images = images.to(device)
            targets = {task: labels.to(device) for task, labels in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            components = loss_components(outputs, targets, tasks, consistency=consistency,
                                         consistency_weight=consistency_weight)
            loss = components["loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
            for key in component_sums:
                component_sums[key] += components[key].item() * len(images)
            if batch % 50 == 0:
                print(f"Epoch {epoch}/{settings['epochs']}, batch {batch}/{len(loaders['train'])}", flush=True)

        # Test images never participate in parameter updates or checkpoint selection.
        validation = evaluate_loader(model, loaders["validation"], device, tasks=tasks, class_mappings=mappings,
                                     taxonomy=taxonomy, consistency_weight=consistency_weight)
        val_loss = validation["loss"]
        selected_metric = validation["tasks"][selection["task"]][selection["metric"]]
        if (selected_metric, -val_loss) > best_score:
            best_score = (selected_metric, -val_loss)
            selected_metrics = {"validation_loss": val_loss, "validation": validation["tasks"],
                                "validation_classification_loss": validation["classification_loss"]}
            if consistency is not None:
                selected_metrics.update({f"validation_{key}": validation[key]
                                         for key in ("consistency_loss", "weighted_consistency_loss", "hierarchy")})
            save_checkpoint(run_dir / "best.pt", model, config=config, taxonomy=taxonomy, epoch=epoch,
                            metrics=selected_metrics)
            save_evaluation_results(
                run_dir / "validation", split="validation", result=validation,
                frame=loaders["validation"].dataset.frame, class_mappings=mappings, selected_epoch=epoch,
            )
        row = {"epoch": epoch, "train_loss": train_loss / len(loaders["train"].dataset), "val_loss": val_loss}
        row.update(train_classification_loss=component_sums["classification_loss"] / len(loaders["train"].dataset),
                   val_classification_loss=validation["classification_loss"])
        if consistency is not None:
            for key in ("consistency_loss", "weighted_consistency_loss"):
                row[f"train_{key}"] = component_sums[key] / len(loaders["train"].dataset)
                row[f"val_{key}"] = validation[key]
            row.update(val_coherence_rate=validation["hierarchy"]["coherence_rate"],
                       val_invalid_paths=validation["hierarchy"]["invalid_paths"])
        for task, metrics in validation["tasks"].items():
            row.update({f"val_{task}_{key}": value for key, value in metrics.items() if key != "num_classes"})
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8")
        print(f"Epoch {epoch}/{settings['epochs']} | train loss: {row['train_loss']:.4f} | "
              f"validation loss: {val_loss:.4f} | {selection['task']} {selection['metric']}: {selected_metric:.4f}", flush=True)
        if consistency is not None:
            print(f"  Validation CE: {validation['classification_loss']:.4f} | "
                  f"consistency: {validation['consistency_loss']:.4f} | "
                  f"independent coherence: {validation['hierarchy']['coherence_rate']:.4f}", flush=True)
    print(f"Checkpoint: {run_dir / 'best.pt'}", flush=True)
    print("Training completed. Use evaluate --checkpoint PATH --split validation or --split test.", flush=True)


if __name__ == "__main__":
    main()
