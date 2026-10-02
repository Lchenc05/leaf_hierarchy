"""Persist split results with independent metrics and reports for each task."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .runtime import write_json
from .hierarchy import METHODS, TASKS, coherence_flags


def save_evaluation_results(output_dir: Path, *, split: str, result: dict,
                            frame, class_mappings: dict, selected_epoch: int) -> dict:
    """Write results from one selected model; rows retain evaluation loader order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(result["tasks"])
    metrics = {"selected_epoch": int(selected_epoch), f"{split}_loss": result["loss"],
               f"{split}_images": len(frame), "tasks": result["tasks"]}
    for key in ("classification_loss", "consistency_loss", "weighted_consistency_loss", "consistency_weight"):
        if key in result:
            metrics[f"{split}_{key}"] = result[key]
    if "hierarchy" in result:
        metrics["hierarchy"] = result["hierarchy"]
    if "methods" in result:
        metrics["methods"] = result["methods"]
    if tasks == ["species"]:
        species = result["tasks"]["species"]
        metrics.update({f"{split}_accuracy": species["accuracy"],
                        f"{split}_macro_f1": species["macro_f1"], "num_classes": species["num_classes"]})
    write_json(output_dir / f"{split}_metrics.json", metrics)
    identity_columns = [column for column in ("image_path", "species", "genus", "family", "group_id")
                        if column in frame.columns]
    predictions = frame[identity_columns].copy()
    for task in tasks:
        write_json(output_dir / f"{split}_{task}_metrics.json", {
            "task": task, "split": split, "selected_epoch": int(selected_epoch),
            "num_images": len(frame), **result["tasks"][task],
        })
        names = [name for name, _ in sorted(class_mappings[task].items(), key=lambda pair: pair[1])]
        labels = list(range(len(names)))
        true, predicted = result["true_labels"][task], result["predicted_labels"][task]
        report = classification_report(true, predicted, labels=labels, target_names=names,
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(output_dir / f"{split}_{task}_classification_report.csv",
                                              encoding="utf-8", index_label="label")
        pd.DataFrame(confusion_matrix(true, predicted, labels=labels), index=names, columns=names).to_csv(
            output_dir / f"{split}_{task}_confusion_matrix.csv", encoding="utf-8", index_label=f"true_{task}")
        predictions[f"predicted_{task}"] = [names[index] for index in predicted]
    for key, values in result.get("coherence_flags", {}).items():
        predictions[key] = values
    predictions.to_csv(output_dir / f"{split}_predictions.csv", index=False, encoding="utf-8")
    return metrics


def save_comparison(destination, frame, taxonomy, logits, truth, predictions, log_probs, metrics,
                    *, prefix=""):
    """Write the same comparison artifacts for unified and legacy evaluation commands."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "taxonomy.json", taxonomy)
    write_json(destination / f"{prefix}metrics.json", metrics)
    columns = [c for c in ("image_path", "species", "genus", "family", "group_id") if c in frame]
    output = frame[columns].reset_index(drop=True).copy()
    arrays = {f"logits_{t}": logits[t] for t in TASKS}
    arrays.update({f"log_probs_{t}": log_probs[t] for t in TASKS})
    arrays["image_path"] = output["image_path"].to_numpy(dtype=str)
    for task in TASKS:
        mapping = taxonomy["class_mappings"][task]
        arrays[f"classes_{task}"] = np.asarray(sorted(mapping, key=mapping.get), dtype=str)
        arrays[f"true_{task}"] = np.asarray(truth[task])
    np.savez_compressed(destination / "scores.npz", **arrays)
    for method in METHODS:
        method_dir = destination / method
        method_dir.mkdir(exist_ok=True)
        write_json(method_dir / "metrics.json", metrics[method])
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
    output.to_csv(destination / f"{prefix}predictions.csv", encoding="utf-8", index=False)


def comparison_rows(metrics, *, checkpoint, seed, split):
    """A common tabular schema for one checkpoint or a multi-checkpoint summary."""
    return [
        {"checkpoint": str(checkpoint), "seed": seed, "split": split,
         "method": method, "task": task, "images": m["images"],
         "coherence_rate": m["coherence_rate"], "invalid_paths": m["invalid_paths"],
         "complete_path_accuracy": m["complete_path_accuracy"], **m["tasks"][task]}
        for method, m in metrics.items() for task in TASKS
    ]


def print_comparison(metrics):
    """Print all levels so species-path changes to genus/family are visible too."""
    print("  Method          Level      Accuracy   Macro F1   Coherent paths", flush=True)
    for method in METHODS:
        m = metrics[method]
        for task in TASKS:
            values = m["tasks"][task]
            print(f"  {method:<15} {task:<10} {values['accuracy']:.4f}     "
                  f"{values['macro_f1']:.4f}     {m['valid_paths']}/{m['images']}", flush=True)
