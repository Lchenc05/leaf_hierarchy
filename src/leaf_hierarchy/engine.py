"""Shared task-aware loss and evaluation, independent of the model backbone."""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .consistency import HierarchicalConsistencyLoss, validate_consistency_weight
from .hierarchy import TASKS, coherence_flags


def task_loss(outputs: dict, targets: dict, tasks: tuple[str, ...] | list[str],
              loss_weights: dict[str, float] | None = None) -> torch.Tensor:
    """Compute cross-entropy for every declared output, rejecting task mismatches."""
    expected = set(tasks)
    if not expected or len(tasks) != len(expected):
        raise ValueError("Tasks must be nonempty and unique.")
    if not isinstance(outputs, dict) or set(outputs) != expected:
        raise ValueError(f"Model outputs must contain exactly these tasks: {sorted(expected)}")
    if not isinstance(targets, dict) or set(targets) != expected:
        raise ValueError(f"Targets must contain exactly these tasks: {sorted(expected)}")
    weights = loss_weights if loss_weights is not None else {task: 1.0 for task in tasks}
    if set(weights) != expected or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                                       for value in weights.values()):
        raise ValueError("Loss weights must be positive and finite for every declared task.")
    criterion = nn.CrossEntropyLoss()
    # Preserve the exact original single-task loss operation when its weight is one.
    terms = [criterion(outputs[task], targets[task]) for task in tasks]
    if len(terms) == 1 and weights[tasks[0]] == 1.0:
        return terms[0]
    return sum(term * weights[task] for task, term in zip(tasks, terms))


def evaluate_loader(model: nn.Module, loader, device: torch.device, *,
                    tasks: tuple[str, ...] | list[str], class_mappings: dict,
                    loss_weights: dict[str, float] | None = None,
                    taxonomy: dict | None = None, consistency_weight: float = 0.0,
                    collect_scores: bool = False, progress: bool = False) -> dict:
    """Evaluate tasks, optionally retaining logits for decoding without another image pass."""
    model.eval()
    weight = validate_consistency_weight(consistency_weight, tasks)
    consistency = (HierarchicalConsistencyLoss(taxonomy).to(device)
                   if taxonomy is not None and set(tasks) == set(TASKS) else None)
    if weight and consistency is None:
        raise ValueError("A taxonomy is required for a positive consistency_weight.")
    if consistency is not None and any(taxonomy["class_mappings"][task] != class_mappings[task] for task in tasks):
        raise ValueError("Consistency taxonomy and evaluation class mappings disagree.")
    if not len(loader.dataset):
        raise ValueError("Cannot evaluate an empty dataset.")
    for task in tasks:
        if task not in class_mappings or not class_mappings[task]:
            raise ValueError(f"Missing class mapping for task {task!r}.")
    total_loss = 0.0
    component_sums = {key: 0.0 for key in ("classification_loss", "consistency_loss", "weighted_consistency_loss")}
    task_losses = {task: 0.0 for task in tasks}
    truths = {task: [] for task in tasks}
    predictions = {task: [] for task in tasks}
    logits = {task: [] for task in tasks} if collect_scores else None
    with torch.no_grad():
        for batch, (images, targets) in enumerate(loader, 1):
            images = images.to(device)
            targets = {task: values.to(device) for task, values in targets.items()}
            outputs = model(images)
            components = loss_components(outputs, targets, tasks, loss_weights,
                                         consistency=consistency, consistency_weight=weight)
            total_loss += components["loss"].item() * len(images)
            for key in component_sums:
                component_sums[key] += components[key].item() * len(images)
            for task in tasks:
                if outputs[task].ndim != 2 or outputs[task].shape[1] != len(class_mappings[task]):
                    raise ValueError(f"Output class count does not match task {task!r}.")
                task_losses[task] += nn.functional.cross_entropy(outputs[task], targets[task]).item() * len(images)
                truths[task].extend(targets[task].cpu().tolist())
                predictions[task].extend(outputs[task].argmax(1).cpu().tolist())
                if collect_scores:
                    logits[task].append(outputs[task].cpu().numpy())
            if progress and (batch % 10 == 0 or batch == len(loader)):
                print(f"  Batches: {batch}/{len(loader)}", flush=True)
    metrics = {
        task: {
            "loss": task_losses[task] / len(loader.dataset),
            "accuracy": float(accuracy_score(truths[task], predictions[task])),
            "macro_f1": float(f1_score(truths[task], predictions[task],
                                      labels=range(len(class_mappings[task])), average="macro", zero_division=0)),
            "num_classes": len(class_mappings[task]),
        } for task in tasks
    }
    result = {"loss": total_loss / len(loader.dataset), "tasks": metrics,
              "classification_loss": component_sums["classification_loss"] / len(loader.dataset),
              "true_labels": truths, "predicted_labels": predictions}
    if collect_scores:
        result["logits"] = {task: np.concatenate(values) for task, values in logits.items()}
    if consistency is not None:
        result.update({key: component_sums[key] / len(loader.dataset)
                       for key in ("consistency_loss", "weighted_consistency_loss")})
        result["consistency_weight"] = weight
        flags = coherence_flags(predictions, taxonomy)
        result["coherence_flags"] = flags
        result["hierarchy"] = {
            "images": len(loader.dataset), "valid_paths": int(flags["valid_path"].sum()),
            "invalid_paths": int((~flags["valid_path"]).sum()),
            "coherence_rate": float(flags["valid_path"].mean()),
            "pair_consistency": {key: float(values.mean()) for key, values in flags.items() if key != "valid_path"},
        }
    return result


def loss_components(outputs: dict, targets: dict, tasks, loss_weights=None, *,
                    consistency: HierarchicalConsistencyLoss | None = None,
                    consistency_weight: float = 0.0) -> dict[str, torch.Tensor]:
    """Keep the original CE path exact at zero weight, with separate diagnostics."""
    weight = validate_consistency_weight(consistency_weight, tasks)
    if weight and consistency is None:
        raise ValueError("A consistency loss is required for a positive consistency_weight.")
    classification = task_loss(outputs, targets, tasks, loss_weights)
    penalty = consistency(outputs) if consistency is not None else classification.new_zeros(())
    weighted = weight * penalty if weight else classification.new_zeros(())
    return {"loss": classification + weighted if weight else classification,
            "classification_loss": classification, "consistency_loss": penalty,
            "weighted_consistency_loss": weighted}
