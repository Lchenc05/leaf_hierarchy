"""Evaluate a selected checkpoint on the held-out test split, for every output task."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .checkpoints import check_split_identity, load_checkpoint
from .config import add_common_arguments, apply_overrides, data_dir_from_env, load_config
from .data import build_taxonomy, load_manifest, make_loader
from .engine import evaluate_loader
from .reporting import save_evaluation_results
from .runtime import MODEL_SEED, configure_runtime, select_device, write_json


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
    # Reuse recorded paths unless a config, CLI flag or dataset environment override is supplied.
    if args.config is None:
        for key in ("data_dir", "split_file"):
            if key == "data_dir" and data_dir_from_env() is not None:
                continue
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
    output_dir = (args.output_dir or checkpoint_path.parent / "test").expanduser().resolve()
    metrics = save_evaluation_results(
        output_dir, split="test", result=result, frame=loader.dataset.frame,
        class_mappings=mappings, selected_epoch=checkpoint.get("epoch", -1),
    )
    write_json(output_dir / "evaluation_config.json", {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "data": {"data_dir": str(data_dir), "split_file": str(split_file)}, "device": str(device),
        "batch_size": config["training"]["batch_size"], "seed": seed,
        "num_threads": config["runtime"]["num_threads"], "preprocessing": checkpoint["preprocessing"],
        "tasks": tasks,
    })
    for task in tasks:
        task_metrics = result["tasks"][task]
        print(f"Test {task} | accuracy: {task_metrics['accuracy']:.4f} | macro F1: {task_metrics['macro_f1']:.4f}")
    print(f"Selected model: epoch {metrics['selected_epoch']}")
    print(f"Saved metrics and predictions: {output_dir}")


if __name__ == "__main__":
    main()
