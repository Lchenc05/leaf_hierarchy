"""Evaluate a selected checkpoint on the held-out test split, for every output task."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .checkpoints import check_split_identity, load_checkpoint
from .config import add_common_arguments, apply_overrides, load_config
from .data import build_taxonomy, load_manifest, make_loader
from .engine import evaluate_loader
from .runtime import MODEL_SEED, configure_runtime, select_device, write_json

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <checkpoint directory>/test.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    device = select_device(config["runtime"]["device"])
    configure_runtime(num_threads=config["runtime"]["num_threads"])
    checkpoint_path = args.checkpoint.expanduser().resolve()
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    saved_config = checkpoint["config"]
    # An explicit config controls data location; otherwise use the recorded paths.
    if args.config is None:
        for key in ("data_dir", "split_file"):
            if getattr(args, key) is None and key in saved_config.get("data", {}):
                config["data"][key] = saved_config["data"][key]
    data_dir, split_file = (Path(config["data"][key]) for key in ("data_dir", "split_file"))
    tasks = checkpoint["model_spec"]["tasks"]
    mappings = checkpoint["taxonomy"]["class_mappings"]
    data, _ = load_manifest(split_file, data_dir, mappings["species"])
    current_taxonomy = build_taxonomy(data)
    for task in tasks:
        if current_taxonomy["class_mappings"].get(task) != mappings.get(task):
            raise ValueError(f"Checkpoint class order differs from the manifest for task {task!r}.")
    check_split_identity(checkpoint, data)
    if args.batch_size is None and args.config is None:
        config["training"]["batch_size"] = saved_config.get("training", {}).get("batch_size", saved_config.get("batch_size", 16))
    seed = saved_config.get("training", {}).get("seed", saved_config.get("seed", MODEL_SEED))
    loader = make_loader(data, "test", data_dir, config["training"]["batch_size"], seed,
                         class_mappings=mappings, tasks=tasks, preprocessing=checkpoint["preprocessing"])
    result = evaluate_loader(model, loader, device, tasks=tasks, class_mappings=mappings)
    metrics = {"selected_epoch": int(checkpoint.get("epoch", -1)), "test_loss": result["loss"],
               "test_images": len(loader.dataset), "tasks": result["tasks"]}
    if tasks == ["species"]:
        species_metrics = result["tasks"]["species"]
        metrics.update(test_accuracy=species_metrics["accuracy"], test_macro_f1=species_metrics["macro_f1"],
                       num_classes=species_metrics["num_classes"])
    output_dir = (args.output_dir or checkpoint_path.parent / "test").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "test_metrics.json", metrics)
    write_json(output_dir / "evaluation_config.json", {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "data": {"data_dir": str(data_dir), "split_file": str(split_file)}, "device": str(device),
        "batch_size": config["training"]["batch_size"], "seed": seed,
        "num_threads": config["runtime"]["num_threads"], "preprocessing": checkpoint["preprocessing"],
        "tasks": tasks,
    })
    identity_columns = [column for column in ("image_path", "species", "genus", "family", "group_id")
                        if column in loader.dataset.frame.columns]
    predictions = loader.dataset.frame[identity_columns].copy()
    for task in tasks:
        names = [name for name, _ in sorted(mappings[task].items(), key=lambda pair: pair[1])]
        labels = list(range(len(names)))
        true, predicted = result["true_labels"][task], result["predicted_labels"][task]
        report = classification_report(true, predicted, labels=labels, target_names=names,
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(output_dir / f"test_{task}_classification_report.csv",
                                              encoding="utf-8", index_label="label")
        pd.DataFrame(confusion_matrix(true, predicted, labels=labels), index=names, columns=names).to_csv(
            output_dir / f"test_{task}_confusion_matrix.csv", encoding="utf-8", index_label=f"true_{task}")
        predictions[f"predicted_{task}"] = [names[index] for index in predicted]
        task_metrics = result["tasks"][task]
        print(f"Test {task} | accuracy: {task_metrics['accuracy']:.4f} | macro F1: {task_metrics['macro_f1']:.4f}")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")
    print(f"Selected model: epoch {metrics['selected_epoch']}")
    print(f"Saved metrics and predictions: {output_dir}")


if __name__ == "__main__":
    main()
