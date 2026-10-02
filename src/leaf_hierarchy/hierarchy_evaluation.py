"""Compatibility command for one or more multitask checkpoints; use evaluate for a single model."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path

from . import runtime  # Set deterministic runtime variables before importing torch.
from .checkpoints import check_split_identity, load_checkpoint
from .config import add_common_arguments, apply_overrides, evaluation_settings, load_config
from .consistency import loss_metadata, saved_consistency_weight
from .data import build_taxonomy, load_manifest, make_loader
from .engine import evaluate_loader
from .hierarchy import PROTOCOL, TASKS, compare_predictions, decode_logits
from .reporting import comparison_rows, print_comparison, save_comparison
from .runtime import ROOT, configure_runtime, select_device, write_json

import numpy as np
import pandas as pd
import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True,
                        help="One or more multitask best.pt files evaluated on the same split.")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path,
                        help="New comparison directory; refuses existing paths. Default: timestamp under results/analysis/hierarchy.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paths = [p.expanduser().resolve() for p in args.checkpoint]
    if len(set(paths)) != len(paths):
        raise ValueError("Each checkpoint may appear only once in a comparison.")
    output_dir = (args.output_dir or ROOT / "results/analysis/hierarchy" /
                  datetime.now().strftime("%Y%m%d_%H%M%S_%f")).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Comparison output already exists: {output_dir}")
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("At least one checkpoint path does not exist.")
    config = apply_overrides(load_config(args.config), args)
    device = select_device(config["runtime"]["device"])
    configure_runtime(num_threads=config["runtime"]["num_threads"])
    output_dir.mkdir(parents=True)
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("hierarchy_evaluation.py", "hierarchy.py", "engine.py", "reporting.py", "consistency.py")}
    comparison = {"status": "running", "split": args.split, "protocol": PROTOCOL,
                  "checkpoints": [str(p) for p in paths], "source_sha256": source_hashes,
                  "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__}}
    write_json(output_dir / "comparison_config.json", comparison)
    reference = None
    summary = []
    for index, path in enumerate(paths, 1):
        print(f"Checkpoint {index}/{len(paths)}: {path.parent.name} | {args.split}", flush=True)
        model, checkpoint = load_checkpoint(path, device)
        if set(checkpoint["model_spec"]["tasks"]) != set(TASKS):
            raise ValueError("Use a multitask checkpoint with family, genus and species outputs.")
        saved = checkpoint["config"]
        data_paths, batch_size, seed = evaluation_settings(config, args, saved)
        configure_runtime(seed, config["runtime"]["num_threads"])
        taxonomy = checkpoint["taxonomy"]
        frame, _ = load_manifest(Path(data_paths["split_file"]), Path(data_paths["data_dir"]),
                                 taxonomy["class_mappings"]["species"])
        current_taxonomy = build_taxonomy(frame)
        for task in TASKS:
            if current_taxonomy["class_mappings"][task] != taxonomy["class_mappings"][task]:
                raise ValueError(f"Checkpoint class order differs from the manifest for task {task!r}.")
        check_split_identity(checkpoint, frame)
        identity = (saved.get("split_records_sha256"), checkpoint["taxonomy_sha256"], checkpoint["preprocessing"])
        if reference is not None and identity != reference:
            raise ValueError("Compared checkpoints must share the manifest, taxonomy and preprocessing.")
        reference = identity
        loader = make_loader(frame, args.split, Path(data_paths["data_dir"]), batch_size, seed,
                             tasks=TASKS, class_mappings=taxonomy["class_mappings"],
                             preprocessing=checkpoint["preprocessing"])
        weight = saved_consistency_weight(saved, TASKS)
        result = evaluate_loader(model, loader, device, tasks=TASKS, class_mappings=taxonomy["class_mappings"],
                                 taxonomy=taxonomy, consistency_weight=weight, collect_scores=True, progress=True)
        logits, truth = result["logits"], result["true_labels"]
        predictions, log_probs = decode_logits(logits, taxonomy)
        metrics = compare_predictions(truth, predictions, taxonomy)
        destination = output_dir / f"{index:02d}_seed_{seed}"
        destination.mkdir()
        metadata = {"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "seed": seed, "selected_epoch": checkpoint.get("epoch"), "split": args.split,
                    "split_records_sha256": saved.get("split_records_sha256"), "taxonomy_sha256": checkpoint["taxonomy_sha256"],
                    "data": data_paths, "preprocessing": checkpoint["preprocessing"], "device": str(device),
                    "batch_size": batch_size, "num_threads": config["runtime"]["num_threads"], "protocol": PROTOCOL}
        metadata["training_loss"] = loss_metadata(TASKS, weight)
        write_json(destination / "config.json", metadata)
        save_comparison(destination, loader.dataset.frame, taxonomy, logits, truth, predictions, log_probs, metrics)
        print_comparison(metrics)
        summary.extend(comparison_rows(metrics, checkpoint=path, seed=seed, split=args.split))
        del model
    rows = pd.DataFrame(summary)
    rows.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8")
    aggregates = rows.groupby(["method", "task"], sort=False).agg(
        num_runs=("seed", "size"), accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        coherence_mean=("coherence_rate", "mean"), coherence_sd=("coherence_rate", "std"),
        complete_path_accuracy_mean=("complete_path_accuracy", "mean"))
    aggregates.to_csv(output_dir / "aggregate.csv", encoding="utf-8")
    comparison["status"] = "complete"
    write_json(output_dir / "comparison_config.json", comparison)
    print(f"Saved hierarchy comparison: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
