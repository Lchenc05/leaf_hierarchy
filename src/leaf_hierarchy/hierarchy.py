"""Fixed, label-free decoding of family/genus/species scores along valid paths."""

from __future__ import annotations

import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import accuracy_score, f1_score

TASKS = ("family", "genus", "species")
METHODS = ("independent", "species_path", "joint_path")
PROTOCOL = {
    "version": 1,
    "joint_score": "log p(family|image) + log p(genus|image) + log p(species|image)",
    "weights": {task: 1.0 for task in TASKS},
    "temperature": 1.0,
    "tie_breaker": "lowest saved species class index",
    "selection": "fixed in advance; no parameter search or fitting on validation/test",
    "interpretation": "product-of-heads score over valid paths; not a calibrated joint probability",
}


def taxonomy_paths(taxonomy: dict) -> dict[str, np.ndarray]:
    """Index parent labels in checkpoint order, never inferred from name spelling."""
    mappings = taxonomy.get("class_mappings", {})
    if set(mappings) != set(TASKS):
        raise ValueError("Hierarchy decoding requires family, genus and species vocabularies.")
    for task, mapping in mappings.items():
        if (not mapping or any(type(i) is not int for i in mapping.values())
                or set(mapping.values()) != set(range(len(mapping)))):
            raise ValueError(f"Invalid saved class indices for {task}.")
    for relation, child, parent in (("species_to_genus", "species", "genus"),
                                     ("genus_to_family", "genus", "family")):
        edges = taxonomy.get(relation, {})
        if set(edges) != set(mappings[child]) or any(v not in mappings[parent] for v in edges.values()):
            raise ValueError(f"Incomplete or invalid taxonomy relationship: {relation}.")
    species = sorted(mappings["species"], key=mappings["species"].get)
    genera = sorted(mappings["genus"], key=mappings["genus"].get)
    s_to_g = np.asarray([mappings["genus"][taxonomy["species_to_genus"][s]] for s in species])
    g_to_f = np.asarray([mappings["family"][taxonomy["genus_to_family"][g]] for g in genera])
    return {"species_to_genus": s_to_g, "genus_to_family": g_to_f,
            "species_to_family": g_to_f[s_to_g]}


def decode_logits(logits: dict, taxonomy: dict) -> tuple[dict, dict]:
    """Decode the same logits three ways; neither labels nor split enter this rule."""
    paths = taxonomy_paths(taxonomy)
    if set(logits) != set(TASKS):
        raise ValueError("Logits must contain exactly family, genus and species.")
    log_probs = {}
    n = None
    for task in TASKS:
        values = np.asarray(logits[task], dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != len(taxonomy["class_mappings"][task])
                or not len(values) or not np.isfinite(values).all()):
            raise ValueError(f"Invalid or nonfinite logits for {task}.")
        if n is not None and len(values) != n:
            raise ValueError("All heads must score the same images in the same order.")
        n = len(values)
        log_probs[task] = log_softmax(values, axis=1)
    independent = {task: log_probs[task].argmax(axis=1) for task in TASKS}

    def from_species(species):
        return {"species": species.copy(), "genus": paths["species_to_genus"][species],
                "family": paths["species_to_family"][species]}

    scores = (log_probs["species"] + log_probs["genus"][:, paths["species_to_genus"]]
              + log_probs["family"][:, paths["species_to_family"]])
    return {"independent": independent, "species_path": from_species(independent["species"]),
            "joint_path": from_species(scores.argmax(axis=1))}, log_probs


def coherence_flags(predicted: dict, taxonomy: dict) -> dict[str, np.ndarray]:
    """A valid triplet respects both adjacent edges of the saved taxonomy."""
    paths = taxonomy_paths(taxonomy)
    sg = paths["species_to_genus"][predicted["species"]] == predicted["genus"]
    gf = paths["genus_to_family"][predicted["genus"]] == predicted["family"]
    sf = paths["species_to_family"][predicted["species"]] == predicted["family"]
    return {"valid_path": sg & gf, "species_genus_consistent": sg,
            "genus_family_consistent": gf, "species_family_consistent": sf}


def compare_predictions(truth: dict, predictions: dict, taxonomy: dict) -> dict:
    """Separate coherence from correctness, including fixes and regressions."""
    baseline = predictions["independent"]
    results = {}
    for method in METHODS:
        predicted = predictions[method]
        flags = coherence_flags(predicted, taxonomy)
        n = len(predicted["species"])
        tasks = {}
        for task in TASKS:
            actual = np.asarray(truth[task])
            if actual.shape != (n,):
                raise ValueError("Truth and predictions must have the same nonempty sample count.")
            correct = actual == predicted[task]
            baseline_correct = actual == baseline[task]
            tasks[task] = {
                "accuracy": float(accuracy_score(actual, predicted[task])),
                "macro_f1": float(f1_score(actual, predicted[task],
                    labels=range(len(taxonomy["class_mappings"][task])), average="macro", zero_division=0)),
                "correct": int(correct.sum()),
                "changed_from_independent": int(np.count_nonzero(predicted[task] != baseline[task])),
                "fixed_from_independent": int((correct & ~baseline_correct).sum()),
                "harmed_from_independent": int((~correct & baseline_correct).sum()),
            }
        all_correct = np.logical_and.reduce([np.asarray(truth[t]) == predicted[t] for t in TASKS])
        results[method] = {
            "images": n, "valid_paths": int(flags["valid_path"].sum()),
            "invalid_paths": int((~flags["valid_path"]).sum()),
            "coherence_rate": float(flags["valid_path"].mean()),
            "complete_path_accuracy": float(all_correct.mean()),
            "pair_consistency": {k: float(v.mean()) for k, v in flags.items() if k != "valid_path"},
            "tasks": tasks,
        }
    return results
