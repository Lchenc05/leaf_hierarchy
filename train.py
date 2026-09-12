"""Train the reproducible species-only ResNet18 baseline."""

from __future__ import annotations

import argparse
import hashlib
import platform
from datetime import datetime
from importlib import metadata
from pathlib import Path

from model_utils import (
    DEFAULT_DATA_DIR, DEFAULT_SPLIT_FILE, MODEL_SEED, ROOT, SPLITS,
    configure_runtime, create_model, evaluate_loader, evaluation_transform,
    load_manifest, make_loader, select_device, split_fingerprint, write_json,
)

import pandas as pd
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Dataset directory containing train/ (default: %(default)s).")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "resnet18",
                        help="Parent directory for a new timestamped run.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet",
                        help="Use none for an offline smoke test; it changes the experiment.")
    parser.add_argument("--seed", type=int, default=MODEL_SEED)
    parser.add_argument("--num-threads", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.split_file = args.split_file.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be between 0 and 4294967295.")
    device = select_device(args.device)
    configure_runtime(args.seed, args.num_threads)
    data, class_to_idx = load_manifest(args.split_file, args.data_dir)
    loaders = {split: make_loader(data, split, args.data_dir, args.batch_size, args.seed)
               for split in ("train", "validation")}

    # Reset immediately before model construction, matching the original baseline.
    configure_runtime(args.seed, args.num_threads)
    model = create_model(len(class_to_idx), pretrained=(args.weights == "imagenet")).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    run_dir = args.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "architecture": "resnet18_species_baseline",
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": 1e-4, "weight_decay": 1e-4, "optimizer": "AdamW",
        "weights": "IMAGENET1K_V1" if args.weights == "imagenet" else "none",
        "device": str(device), "num_threads": args.num_threads,
        "num_classes": len(class_to_idx), "class_to_idx": class_to_idx,
        "preprocessing": str(evaluation_transform()),
        "augmentation": "HorizontalFlip(0.5), VerticalFlip(0.5), training only",
        "selection": "Maximum validation macro F1; ties resolved by lower validation loss",
        "split_unit": "group_id",
        "split_manifest_sha256": hashlib.sha256(args.split_file.read_bytes()).hexdigest(),
        "split_records_sha256": split_fingerprint(data),
        "split_sizes": {split: int(data["split"].eq(split).sum()) for split in SPLITS},
        "versions": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            **{name: metadata.version(name) for name in ("numpy", "pandas", "Pillow", "scikit-learn")},
        },
    }
    write_json(run_dir / "config.json", config)
    print(f"Device: {device}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"{len(class_to_idx)} species | train: {len(loaders['train'].dataset)} images | "
          f"validation: {len(loaders['validation'].dataset)} images", flush=True)
    best_score = (-1.0, float("-inf"))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch, (images, labels) in enumerate(loaders["train"], 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
            if batch % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs}, batch {batch}/{len(loaders['train'])}", flush=True)

        # Test images never participate in checkpoint selection or parameter updates.
        val_loss, true_labels, predicted_labels = evaluate_loader(model, loaders["validation"], device)
        val_accuracy = float(accuracy_score(true_labels, predicted_labels))
        val_f1 = float(f1_score(true_labels, predicted_labels, labels=range(len(class_to_idx)),
                                average="macro", zero_division=0))
        if (val_f1, -val_loss) > best_score:
            best_score = (val_f1, -val_loss)
            torch.save({"model_state_dict": model.state_dict(), "class_to_idx": class_to_idx,
                        "epoch": epoch, "val_f1": val_f1, "val_loss": float(val_loss),
                        "config": config}, run_dir / "resnet18.pt")
        history.append({"epoch": epoch, "train_loss": train_loss / len(loaders["train"].dataset),
                        "val_loss": float(val_loss), "val_accuracy": val_accuracy, "val_f1": val_f1})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8")
        print(f"Epoch {epoch}/{args.epochs} | train loss: {history[-1]['train_loss']:.4f} | "
              f"validation loss: {val_loss:.4f} | validation accuracy: {val_accuracy:.4f} | "
              f"validation macro F1: {val_f1:.4f}", flush=True)
    print(f"Checkpoint: {run_dir / 'resnet18.pt'}", flush=True)
    print("Training completed. Run evaluate.py with this checkpoint to evaluate the held-out test set.", flush=True)


if __name__ == "__main__":
    main()
