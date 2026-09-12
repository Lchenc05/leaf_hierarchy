"""Evaluate a selected ResNet18 checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from model_utils import (
    DEFAULT_DATA_DIR, DEFAULT_SPLIT_FILE, MODEL_SEED, check_split_identity,
    configure_runtime, evaluate_loader, load_checkpoint, load_manifest,
    make_loader, select_device, write_json,
)

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory; default: <checkpoint directory>/test.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Default: checkpoint batch size, or 16 for a historical checkpoint.")
    parser.add_argument("--num-threads", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    configure_runtime(num_threads=args.num_threads)
    model, checkpoint = load_checkpoint(args.checkpoint.expanduser().resolve(), device)
    data, class_to_idx = load_manifest(args.split_file, args.data_dir, checkpoint["class_to_idx"])
    check_split_identity(checkpoint, data)
    config = checkpoint.get("config", {})
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 16)
    loader = make_loader(data, "test", args.data_dir, batch_size, config.get("seed", MODEL_SEED))
    test_loss, true_labels, predicted_labels = evaluate_loader(model, loader, device)
    species_names = [name for name, index in sorted(class_to_idx.items(), key=lambda pair: pair[1])]
    labels = list(range(len(species_names)))
    metrics = {
        "selected_epoch": int(checkpoint.get("epoch", -1)), "test_loss": float(test_loss),
        "test_accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "test_macro_f1": float(f1_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)),
        "test_images": len(true_labels), "num_classes": len(species_names),
    }
    output_dir = (args.output_dir or args.checkpoint.parent / "test").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "test_metrics.json", metrics)
    report = classification_report(true_labels, predicted_labels, labels=labels,
                                   target_names=species_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir / "test_classification_report.csv", encoding="utf-8", index_label="label")
    pd.DataFrame(confusion_matrix(true_labels, predicted_labels, labels=labels), index=species_names,
                 columns=species_names).to_csv(output_dir / "test_confusion_matrix.csv", encoding="utf-8", index_label="true_species")
    predictions = loader.dataset.frame[["image_path", "Species", "group_id"]].copy()
    predictions["predicted_species"] = [species_names[index] for index in predicted_labels]
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")
    print(f"Selected model: epoch {metrics['selected_epoch']}")
    print(f"Test | accuracy: {metrics['test_accuracy']:.4f} | macro F1: {metrics['test_macro_f1']:.4f}")
    print(f"Saved metrics and predictions: {output_dir}")


if __name__ == "__main__":
    main()
