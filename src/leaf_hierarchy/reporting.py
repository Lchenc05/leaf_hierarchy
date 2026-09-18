"""Persist split results with independent metrics and reports for each task."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .runtime import write_json


def save_evaluation_results(output_dir: Path, *, split: str, result: dict,
                            frame, class_mappings: dict, selected_epoch: int) -> dict:
    """Write results from one selected model; rows retain evaluation loader order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(result["tasks"])
    metrics = {"selected_epoch": int(selected_epoch), f"{split}_loss": result["loss"],
               f"{split}_images": len(frame), "tasks": result["tasks"]}
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
    predictions.to_csv(output_dir / f"{split}_predictions.csv", index=False, encoding="utf-8")
    return metrics
