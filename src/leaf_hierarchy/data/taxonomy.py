"""Validated label hierarchy shared by models, metrics and representation analyses."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .manifest import normalize_manifest_columns

TAXONOMIC_TASKS = ("species", "genus", "family")


def build_taxonomy(frame: pd.DataFrame) -> dict:
    """Build a deterministic hierarchy from dataset labels, without renaming taxa.

    This hierarchy describes supervision. The frozen botanical snapshot used for
    tree selection is a separate source and does not replace the dataset labels.
    """
    data = normalize_manifest_columns(frame)
    for task in TAXONOMIC_TASKS:
        if task not in data or data[task].isna().any() or data[task].astype(str).str.strip().eq("").any():
            raise ValueError(f"The taxonomy contains missing {task} labels.")
    if data.empty:
        raise ValueError("The taxonomy requires at least one labelled image.")
    relations = {}
    for child, parent in (("species", "genus"), ("genus", "family")):
        ambiguity = data.groupby(child)[parent].nunique().gt(1)
        if ambiguity.any():
            label = ambiguity[ambiguity].index[0]
            raise ValueError(f"Inconsistent taxonomy: {child} {label!r} has multiple {parent} labels.")
        relations[f"{child}_to_{parent}"] = dict(
            data[[child, parent]].drop_duplicates().sort_values(child).itertuples(index=False, name=None))
    return {"schema_version": 1, "source": "dataset_labels",
            "class_mappings": {task: {name: i for i, name in enumerate(sorted(data[task].unique()))}
                               for task in TAXONOMIC_TASKS}, **relations}


def taxonomy_fingerprint(taxonomy: dict) -> str:
    """Identify labels, class order, relations and stated provenance together."""
    serialized = json.dumps(taxonomy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
