"""Compare independent, species-derived and joint taxonomic inference without training."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path

from . import runtime  # Set deterministic runtime variables before importing torch.
from .checkpoints import check_split_identity, load_checkpoint
from .config import add_common_arguments, apply_overrides, data_dir_from_env, load_config
from .data import load_manifest, make_loader
from .hierarchy import METHODS, PROTOCOL, TASKS, coherence_flags, compare_predictions, decode_logits
from .runtime import MODEL_SEED, ROOT, configure_runtime, select_device, write_json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
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


def collect_logits(model, loader, device):
    """One deterministic image pass supplies all three inference methods."""
    model.eval()
    logits, truth = {t: [] for t in TASKS}, {t: [] for t in TASKS}
    with torch.inference_mode():
        for batch, (images, targets) in enumerate(loader, 1):
            outputs = model(images.to(device))
            if set(outputs) != set(TASKS):
                raise ValueError("Hierarchy comparison requires all three output heads.")
            for task in TASKS:
                logits[task].append(outputs[task].cpu().numpy())
                truth[task].append(targets[task].numpy())
            if batch % 10 == 0 or batch == len(loader):
                print(f"  Batches: {batch}/{len(loader)}", flush=True)
    return ({t: np.concatenate(logits[t]) for t in TASKS},
            {t: np.concatenate(truth[t]) for t in TASKS})


def save_comparison(destination, frame, taxonomy, logits, truth, predictions, log_probs, metrics):
    """Preserve scores, source-image order and per-class evidence for each method."""
    write_json(destination / "taxonomy.json", taxonomy)
    write_json(destination / "metrics.json", metrics)
    columns = [c for c in ("image_path", "species", "genus", "family", "group_id") if c in frame]
    output = frame[columns].reset_index(drop=True).copy()
    arrays = {f"logits_{t}": logits[t] for t in TASKS}
    arrays.update({f"log_probs_{t}": log_probs[t] for t in TASKS})
    arrays["image_path"] = output["image_path"].to_numpy(dtype=str)
    for task in TASKS:
        mapping = taxonomy["class_mappings"][task]
        arrays[f"classes_{task}"] = np.asarray(sorted(mapping, key=mapping.get), dtype=str)
        arrays[f"true_{task}"] = truth[task]
    np.savez_compressed(destination / "scores.npz", **arrays)
    for method in METHODS:
        method_dir = destination / method
        method_dir.mkdir()
        for task in TASKS:
            names = arrays[f"classes_{task}"].tolist()
            labels = list(range(len(names)))
            values = predictions[method][task]
            output[f"{method}_{task}"] = [names[i] for i in values]
            report = classification_report(truth[task], values, labels=labels, target_names=names,
                                           output_dict=True, zero_division=0)
            pd.DataFrame(report).transpose().to_csv(method_dir / f"{task}_classification_report.csv",
                                                   encoding="utf-8", index_label="label")
            pd.DataFrame(confusion_matrix(truth[task], values, labels=labels), index=names, columns=names).to_csv(
                method_dir / f"{task}_confusion_matrix.csv", encoding="utf-8", index_label=f"true_{task}")
        for key, values in coherence_flags(predictions[method], taxonomy).items():
            output[f"{method}_{key}"] = values
    output.to_csv(destination / "predictions.csv", encoding="utf-8", index=False)


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
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__), Path(__file__).with_name("hierarchy.py"))}
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
        data_paths = dict(config["data"])
        if args.config is None:
            for key in ("data_dir", "split_file"):
                if key == "data_dir" and data_dir_from_env() is not None:
                    continue
                if getattr(args, key) is None and key in saved.get("data", {}):
                    data_paths[key] = saved["data"][key]
        batch_size = config["training"]["batch_size"]
        if args.batch_size is None and args.config is None:
            batch_size = saved.get("training", {}).get("batch_size", 16)
        seed = saved.get("training", {}).get("seed", saved.get("seed", MODEL_SEED))
        configure_runtime(seed, config["runtime"]["num_threads"])
        taxonomy = checkpoint["taxonomy"]
        frame, _ = load_manifest(Path(data_paths["split_file"]), Path(data_paths["data_dir"]),
                                 taxonomy["class_mappings"]["species"])
        check_split_identity(checkpoint, frame)
        identity = (saved.get("split_records_sha256"), checkpoint["taxonomy_sha256"], checkpoint["preprocessing"])
        if reference is not None and identity != reference:
            raise ValueError("Compared checkpoints must share the manifest, taxonomy and preprocessing.")
        reference = identity
        loader = make_loader(frame, args.split, Path(data_paths["data_dir"]), batch_size, seed,
                             tasks=TASKS, class_mappings=taxonomy["class_mappings"],
                             preprocessing=checkpoint["preprocessing"])
        logits, truth = collect_logits(model, loader, device)
        predictions, log_probs = decode_logits(logits, taxonomy)
        metrics = compare_predictions(truth, predictions, taxonomy)
        destination = output_dir / f"{index:02d}_seed_{seed}"
        destination.mkdir()
        metadata = {"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "seed": seed, "selected_epoch": checkpoint.get("epoch"), "split": args.split,
                    "split_records_sha256": saved.get("split_records_sha256"), "taxonomy_sha256": checkpoint["taxonomy_sha256"],
                    "data": data_paths, "preprocessing": checkpoint["preprocessing"], "device": str(device),
                    "batch_size": batch_size, "num_threads": config["runtime"]["num_threads"], "protocol": PROTOCOL}
        write_json(destination / "config.json", metadata)
        save_comparison(destination, loader.dataset.frame, taxonomy, logits, truth, predictions, log_probs, metrics)
        for method in METHODS:
            m = metrics[method]
            print(f"  {method}: coherent {m['valid_paths']}/{m['images']} | "
                  f"species accuracy {m['tasks']['species']['accuracy']:.4f} | "
                  f"macro F1 {m['tasks']['species']['macro_f1']:.4f}", flush=True)
            for task in TASKS:
                summary.append({"checkpoint": str(path), "seed": seed, "split": args.split,
                                "method": method, "task": task, "images": m["images"],
                                "coherence_rate": m["coherence_rate"], "invalid_paths": m["invalid_paths"],
                                "complete_path_accuracy": m["complete_path_accuracy"], **m["tasks"][task]})
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
