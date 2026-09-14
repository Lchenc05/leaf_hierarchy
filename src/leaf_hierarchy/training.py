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
from .data import build_taxonomy, load_manifest, make_loader, split_fingerprint
from .engine import evaluate_loader, task_loss
from .models import create_model
from .runtime import ROOT, SPLITS, configure_runtime, select_device, write_json

import pandas as pd
import torch
import torchvision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Parent for a timestamped run; default: configured run_root/experiment.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weights", choices=("imagenet", "none"), default=None,
                        help="none permits an offline smoke test and changes the experiment.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
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

    best_score = (-1.0, float("-inf"))
    history = []
    selection = config["selection"]
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch, (images, targets) in enumerate(loaders["train"], 1):
            images = images.to(device)
            targets = {task: labels.to(device) for task, labels in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = task_loss(outputs, targets, tasks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
            if batch % 50 == 0:
                print(f"Epoch {epoch}/{settings['epochs']}, batch {batch}/{len(loaders['train'])}", flush=True)

        # Test images never participate in parameter updates or checkpoint selection.
        validation = evaluate_loader(model, loaders["validation"], device, tasks=tasks, class_mappings=mappings)
        val_loss = validation["loss"]
        selected_metric = validation["tasks"][selection["task"]][selection["metric"]]
        if (selected_metric, -val_loss) > best_score:
            best_score = (selected_metric, -val_loss)
            save_checkpoint(run_dir / "best.pt", model, config=config, taxonomy=taxonomy, epoch=epoch,
                            metrics={"validation_loss": val_loss, "validation": validation["tasks"]})
        row = {"epoch": epoch, "train_loss": train_loss / len(loaders["train"].dataset), "val_loss": val_loss}
        for task, metrics in validation["tasks"].items():
            row.update({f"val_{task}_{key}": value for key, value in metrics.items() if key != "num_classes"})
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8")
        print(f"Epoch {epoch}/{settings['epochs']} | train loss: {row['train_loss']:.4f} | "
              f"validation loss: {val_loss:.4f} | {selection['task']} {selection['metric']}: {selected_metric:.4f}", flush=True)
    print(f"Checkpoint: {run_dir / 'best.pt'}", flush=True)
    print("Training completed. Evaluate this checkpoint on the held-out test set.", flush=True)


if __name__ == "__main__":
    main()
