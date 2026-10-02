"""Evaluate validation or test; automatically compare all decoding methods for multitask models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .checkpoints import check_split_identity, load_checkpoint
from .config import add_common_arguments, apply_overrides, evaluation_settings, load_config
from .consistency import loss_metadata, saved_consistency_weight
from .data import build_taxonomy, load_manifest, make_loader
from .engine import evaluate_loader
from .hierarchy import METHODS, PROTOCOL, TASKS, compare_predictions, decode_logits
from .reporting import comparison_rows, print_comparison, save_comparison, save_evaluation_results
from .runtime import configure_runtime, select_device, write_json

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test",
                        help="Images to evaluate; default: test. Multitask models compare all three methods automatically.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <checkpoint directory>/<split>. Repeated evaluations replace generated results.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    device = select_device(config["runtime"]["device"])
    configure_runtime(num_threads=config["runtime"]["num_threads"])
    checkpoint_path = args.checkpoint.expanduser().resolve()
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    saved_config = checkpoint["config"]
    data_paths, batch_size, seed = evaluation_settings(config, args, saved_config)
    data_dir, split_file = (Path(data_paths[key]) for key in ("data_dir", "split_file"))
    tasks = checkpoint["model_spec"]["tasks"]
    output_dir = (args.output_dir or checkpoint_path.parent / args.split).expanduser().resolve()
    previous_metadata = output_dir / "evaluation_config.json"
    if previous_metadata.is_file():
        previous_tasks = json.loads(previous_metadata.read_text(encoding="utf-8")).get("tasks")
        if previous_tasks is not None and set(previous_tasks) != set(tasks):
            raise ValueError("Evaluation output contains results for different tasks. "
                             "Choose another --output-dir to preserve the previous results.")
    mappings = checkpoint["taxonomy"]["class_mappings"]
    data, _ = load_manifest(split_file, data_dir, mappings["species"])
    current_taxonomy = build_taxonomy(data)
    for task in tasks:
        if current_taxonomy["class_mappings"].get(task) != mappings.get(task):
            raise ValueError(f"Checkpoint class order differs from the manifest for task {task!r}.")
    check_split_identity(checkpoint, data)
    consistency_weight = saved_consistency_weight(saved_config, tasks)
    loader = make_loader(data, args.split, data_dir, batch_size, seed,
                         class_mappings=mappings, tasks=tasks, preprocessing=checkpoint["preprocessing"])
    multitask = set(tasks) == set(TASKS)
    methods = list(METHODS) if multitask else ["independent"]
    print(f"Checkpoint: {checkpoint_path.parent.name} | {args.split} | methods: {', '.join(methods)}", flush=True)
    result = evaluate_loader(model, loader, device, tasks=tasks, class_mappings=mappings,
                             taxonomy=checkpoint["taxonomy"], consistency_weight=consistency_weight,
                             collect_scores=multitask, progress=True)
    if multitask:
        predictions, log_probs = decode_logits(result["logits"], checkpoint["taxonomy"])
        result["methods"] = compare_predictions(result["true_labels"], predictions, checkpoint["taxonomy"])
    metadata = {
        "status": "running", "split": args.split, "methods": methods,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "data": {"data_dir": str(data_dir), "split_file": str(split_file)}, "device": str(device),
        "batch_size": batch_size, "seed": seed,
        "num_threads": config["runtime"]["num_threads"], "preprocessing": checkpoint["preprocessing"],
        "tasks": tasks, "loss": loss_metadata(tasks, consistency_weight),
        "selected_epoch": checkpoint.get("epoch", -1),
        "split_records_sha256": saved_config.get("split_records_sha256"),
        "taxonomy_sha256": checkpoint.get("taxonomy_sha256"),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("evaluation.py", "engine.py", "hierarchy.py", "reporting.py", "consistency.py")},
    }
    if multitask:
        metadata["protocol"] = PROTOCOL
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation_config.json", metadata)
    metrics = save_evaluation_results(
        output_dir, split=args.split, result=result, frame=loader.dataset.frame,
        class_mappings=mappings, selected_epoch=checkpoint.get("epoch", -1),
    )
    if multitask:
        save_comparison(output_dir, loader.dataset.frame, checkpoint["taxonomy"], result["logits"],
                        result["true_labels"], predictions, log_probs, result["methods"], prefix="comparison_")
        pd.DataFrame(comparison_rows(result["methods"], checkpoint=checkpoint_path, seed=seed,
                                     split=args.split)).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8")
        print_comparison(result["methods"])
    else:
        for task in tasks:
            task_metrics = result["tasks"][task]
            print(f"{args.split.capitalize()} {task} | accuracy: {task_metrics['accuracy']:.4f} | "
                  f"macro F1: {task_metrics['macro_f1']:.4f}")
    if "hierarchy" in result:
        print(f"{args.split.capitalize()} independent coherence: {result['hierarchy']['coherence_rate']:.4f} | "
              f"consistency loss: {result['consistency_loss']:.4f} | weight: {consistency_weight:g}")
    metadata["status"] = "complete"
    write_json(output_dir / "evaluation_config.json", metadata)
    print(f"Selected model: epoch {metrics['selected_epoch']}")
    print(f"Saved metrics and predictions: {output_dir}")


if __name__ == "__main__":
    main()
