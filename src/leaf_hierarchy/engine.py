"""Shared task-aware loss and evaluation, independent of the model backbone."""

from __future__ import annotations

import math

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn


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
                    loss_weights: dict[str, float] | None = None) -> dict:
    """Evaluate every configured task without changing weights or sample order."""
    model.eval()
    if not len(loader.dataset):
        raise ValueError("Cannot evaluate an empty dataset.")
    for task in tasks:
        if task not in class_mappings or not class_mappings[task]:
            raise ValueError(f"Missing class mapping for task {task!r}.")
    total_loss = 0.0
    truths = {task: [] for task in tasks}
    predictions = {task: [] for task in tasks}
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = {task: values.to(device) for task, values in targets.items()}
            outputs = model(images)
            loss = task_loss(outputs, targets, tasks, loss_weights)
            total_loss += loss.item() * len(images)
            for task in tasks:
                if outputs[task].ndim != 2 or outputs[task].shape[1] != len(class_mappings[task]):
                    raise ValueError(f"Output class count does not match task {task!r}.")
                truths[task].extend(targets[task].cpu().tolist())
                predictions[task].extend(outputs[task].argmax(1).cpu().tolist())
    metrics = {
        task: {
            "accuracy": float(accuracy_score(truths[task], predictions[task])),
            "macro_f1": float(f1_score(truths[task], predictions[task],
                                      labels=range(len(class_mappings[task])), average="macro", zero_division=0)),
            "num_classes": len(class_mappings[task]),
        } for task in tasks
    }
    return {"loss": total_loss / len(loader.dataset), "tasks": metrics,
            "true_labels": truths, "predicted_labels": predictions}
