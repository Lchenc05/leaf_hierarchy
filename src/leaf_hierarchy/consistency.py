"""Differentiable consistency between adjacent taxonomic probability distributions."""

from __future__ import annotations

import math

import torch
from torch import nn

from .hierarchy import TASKS, taxonomy_paths


def validate_consistency_weight(value, tasks) -> float:
    """Allow zero for every model; positive weights require all three heads."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("consistency_weight must be a finite, nonnegative number.")
    if value > 0 and set(tasks) != set(TASKS):
        raise ValueError("Positive consistency_weight requires family, genus and species tasks.")
    return float(value)


def loss_metadata(tasks, consistency_weight: float) -> dict:
    """Record the objective independently of architecture and inference decoding."""
    weight = validate_consistency_weight(consistency_weight, tasks)
    metadata = {
        "name": "cross_entropy_with_hierarchical_consistency" if weight else "cross_entropy",
        "reduction": "sum_of_task_means", "weights": {task: 1.0 for task in tasks},
    }
    if set(tasks) == set(TASKS):
        metadata["consistency"] = {
            "name": "jensen_shannon", "version": 1, "weight": weight,
            "pairs": ["species_to_genus", "genus_to_family"],
            "reduction": "mean_of_pair_batch_means", "log_base": "e",
        }
    return metadata


def saved_consistency_weight(config: dict, tasks) -> float:
    """Older checkpoints have no consistency setting and retain their CE objective."""
    settings = config.get("training", {})
    if not isinstance(settings, dict):
        raise ValueError("Checkpoint training settings must be a dictionary.")
    weight = validate_consistency_weight(settings.get("consistency_weight", 0.0), tasks)
    saved_loss = config.get("loss", {})
    if not isinstance(saved_loss, dict):
        raise ValueError("Checkpoint loss metadata must be a dictionary.")
    if weight > 0 and "consistency" not in saved_loss:
        raise ValueError("Checkpoint with positive consistency_weight requires consistency loss metadata.")
    if "consistency" in saved_loss:
        expected = loss_metadata(tasks, weight).get("consistency")
        if expected is None or saved_loss["consistency"] != expected:
            raise ValueError("Checkpoint consistency loss metadata disagrees with the supported objective.")
    return weight


class HierarchicalConsistencyLoss(nn.Module):
    """Average JS(S→G, G) and JS(G→F, F), with gradients into both distributions.

    Uses natural logs, sums over parent classes and averages over images and edges.
    This regularizes marginals; it does not constrain independent argmax labels.
    """

    def __init__(self, taxonomy: dict):
        super().__init__()
        paths = taxonomy_paths(taxonomy)
        self.class_counts = {task: len(taxonomy["class_mappings"][task]) for task in TASKS}
        species_parents = torch.as_tensor(paths["species_to_genus"], dtype=torch.long)
        genus_parents = torch.as_tensor(paths["genus_to_family"], dtype=torch.long)
        self.register_buffer("species_groups", nn.functional.one_hot(
            species_parents, num_classes=self.class_counts["genus"]).T.bool())
        self.register_buffer("genus_groups", nn.functional.one_hot(
            genus_parents, num_classes=self.class_counts["family"]).T.bool())

    @staticmethod
    def _marginal(log_probabilities: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        # Groups are indexed by saved parent IDs. Empty groups have exactly zero mass.
        present = groups.any(dim=1)
        values = log_probabilities[:, None, :].masked_fill(~groups[None, :, :], -torch.inf)
        # Avoid logsumexp(all -inf), whose backward is undefined even when masked later.
        values = torch.where(present[None, :, None], values, torch.zeros_like(values))
        return torch.logsumexp(values, dim=2).masked_fill(~present[None, :], -torch.inf)

    @staticmethod
    def _js(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
        log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
        # A zero-mass parent contributes zero; avoid evaluating 0 * (-inf).
        safe_p = log_p.masked_fill(~torch.isfinite(log_p), 0.0)
        safe_q = log_q.masked_fill(~torch.isfinite(log_q), 0.0)
        divergence = 0.5 * (log_p.exp() * (safe_p - log_m)
                            + log_q.exp() * (safe_q - log_m)).sum(dim=1)
        return divergence.mean().clamp_min(0.0)

    def forward(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(outputs, dict) or set(outputs) != set(TASKS):
            raise ValueError("Consistency loss requires family, genus and species logits.")
        reference = outputs["species"]
        if not isinstance(reference, torch.Tensor) or reference.ndim != 2 or not len(reference):
            raise ValueError("Consistency logits must be nonempty two-dimensional tensors.")
        for task in TASKS:
            values = outputs[task]
            if (not isinstance(values, torch.Tensor) or values.shape != (len(reference), self.class_counts[task])
                    or not values.is_floating_point() or not torch.isfinite(values).all()
                    or values.device != reference.device or values.dtype != reference.dtype):
                raise ValueError(f"Invalid consistency logits for {task!r}.")
        if reference.device != self.species_groups.device:
            raise ValueError("Move the consistency loss to the logits' device before evaluation.")
        dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
        probabilities = {task: nn.functional.log_softmax(outputs[task].to(dtype), dim=1) for task in TASKS}
        genus_from_species = self._marginal(probabilities["species"], self.species_groups)
        family_from_genus = self._marginal(probabilities["genus"], self.genus_groups)
        return 0.5 * (self._js(probabilities["genus"], genus_from_species)
                      + self._js(probabilities["family"], family_from_genus))
