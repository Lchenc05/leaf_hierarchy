"""Common image manifests and an explicit reader for historical field names."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import pandas as pd

from leaf_hierarchy.runtime import SPLITS

HISTORICAL_DATASET_PREFIX = "PlantCLEF2015TrainingData"
MANIFEST_COLUMNS = ("image_path", "species", "genus", "family", "class_id",
                    "observation_id", "group_id", "split", "content")
COLUMN_ALIASES = {
    "registro_origen": "source_record",
    "Species": "species", "Genus": "genus", "Family": "family",
    "ClassId": "class_id", "ObservationId": "observation_id", "Content": "content",
    "ruta_imagen": "image_path", "especie": "species", "genero": "genus",
    "familia": "family", "id_clase": "class_id", "id_observacion": "observation_id",
    "id_grupo": "group_id", "particion": "split", "contenido": "content",
}


def canonical_image_path(value: str) -> str:
    """Return a safe dataset-relative path, accepting the historical prefix.

    Dataset adapters may emit the original prefixed paths to preserve stable
    image-group identifiers. Consumers resolve both spellings identically.
    """
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or PureWindowsPath(value).drive
            or ".." in path.parts or any(":" in part for part in path.parts)):
        raise ValueError(f"Image paths must be relative dataset paths: {value!r}")
    parts = path.parts
    if parts and parts[0] == HISTORICAL_DATASET_PREFIX:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Image path must identify a file: {value!r}")
    return PurePosixPath(*parts).as_posix()


def resolve_image_path(value: str, data_dir: Path) -> Path:
    """Resolve a manifest entry below the dataset, including symlink checks."""
    relative = PurePosixPath(canonical_image_path(value))
    root = Path(data_dir).expanduser().resolve()
    candidate = root.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Image path resolves outside --data-dir: {value!r}")
    return candidate


def normalize_manifest_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt historical headers without silently choosing conflicting aliases."""
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names occur in the manifest.")
    result = frame.copy()
    for alias, canonical in COLUMN_ALIASES.items():
        if alias not in result:
            continue
        if canonical in result:
            left = result[alias].fillna("").astype(str)
            right = result[canonical].fillna("").astype(str)
            if canonical == "image_path":
                left, right = left.map(canonical_image_path), right.map(canonical_image_path)
            if not left.equals(right):
                raise ValueError(f"Contradictory manifest columns {alias!r} and {canonical!r}.")
            result = result.drop(columns=alias)
        else:
            result = result.rename(columns={alias: canonical})
    return result


def validate_class_mapping(mapping: dict, *, minimum_classes: int = 2) -> dict[str, int]:
    """Require unique contiguous integer labels for nonempty class names."""
    if not isinstance(mapping, dict) or len(mapping) < minimum_classes:
        raise ValueError(f"The class mapping must contain at least {minimum_classes} classes.")
    if any(not isinstance(name, str) or not name.strip() for name in mapping):
        raise ValueError("Class names must be nonempty strings.")
    if any(type(index) is not int for index in mapping.values()):
        raise ValueError("Class indices must be integers.")
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("Class indices must be unique and contiguous from zero.")
    return mapping


def load_manifest(split_file: Path, data_dir: Path, class_to_idx: dict | None = None
                  ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate a common manifest, complete taxonomy, groups and active images."""
    split_file = Path(split_file).expanduser().resolve()
    if not split_file.is_file():
        raise FileNotFoundError(f"Prepare the dataset first. Missing split manifest: {split_file}")
    # Pandas otherwise silently mangles repeated CSV headings before validation.
    import csv
    with split_file.open(encoding="utf-8-sig", newline="") as handle:
        headings = next(csv.reader(handle), [])
    if len(set(headings)) != len(headings):
        raise ValueError("Duplicate column names occur in the manifest.")
    data = normalize_manifest_columns(pd.read_csv(split_file, dtype=str, keep_default_na=False))
    required = set(MANIFEST_COLUMNS) - {"content"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Split manifest is missing columns: {', '.join(missing)}")
    unexpected = sorted(set(data["split"]) - set(SPLITS) - {"excluded"})
    if unexpected:
        raise ValueError(f"Unknown split labels: {unexpected}")
    data = data.loc[data["split"].isin(SPLITS)].copy()
    if data.empty:
        raise ValueError("The manifest does not contain active train, validation or test images.")
    for field in required:
        if data[field].str.strip().eq("").any():
            raise ValueError(f"The active manifest contains missing {field} values.")
    if "content" not in data:
        data["content"] = ""
    data["image_path"] = data["image_path"].map(canonical_image_path)
    data = data.sort_values("image_path").reset_index(drop=True)
    if not data["image_path"].is_unique:
        raise ValueError("Duplicate image paths occur in the active split manifest.")
    for source, target in (("species", "class_id"), ("class_id", "species")):
        if (data.groupby(source)[target].nunique() != 1).any():
            raise ValueError(f"A {source} has multiple {target} values in the split manifest.")
    from .taxonomy import build_taxonomy
    build_taxonomy(data)
    for field in ("observation_id", "group_id"):
        if (data.groupby(field)["split"].nunique() != 1).any():
            raise ValueError(f"Data leakage: {field} values overlap across splits.")
    # Regenerated manifests contain these hashes; historical manifests may not.
    for field in ("sha256", "pixel_sha256"):
        if field in data:
            known = data.loc[data[field].str.strip().ne("")]
            if known.groupby(field)["split"].nunique().gt(1).any():
                raise ValueError(f"Data leakage: {field} values overlap across splits.")
    species_names = sorted(data["species"].unique())
    if len(species_names) < 2:
        raise ValueError("The split manifest must contain at least two active species.")
    expected_mapping = {name: index for index, name in enumerate(species_names)}
    if class_to_idx is None:
        class_to_idx = expected_mapping
    else:
        validate_class_mapping(class_to_idx)
        if class_to_idx != expected_mapping:
            raise ValueError("Checkpoint class order does not match the manifest's sorted species labels.")
    for split in SPLITS:
        observed = set(data.loc[data["split"] == split, "species"])
        if observed != set(species_names):
            raise ValueError(f"Split {split!r} must contain every active species.")
    resolved_paths = data["image_path"].map(lambda value: resolve_image_path(value, data_dir))
    missing_paths = resolved_paths.loc[~resolved_paths.map(Path.is_file)]
    if len(missing_paths):
        raise FileNotFoundError(f"{len(missing_paths)} images are missing. First missing image: {missing_paths.iloc[0]}")
    data["target"] = data["species"].map(class_to_idx)
    return data, class_to_idx


def _fingerprint_records(frame: pd.DataFrame, *, legacy: bool) -> str:
    data = normalize_manifest_columns(frame)
    records = data.reindex(columns=MANIFEST_COLUMNS, fill_value="").fillna("").astype(str)
    records["image_path"] = records["image_path"].map(canonical_image_path)
    if legacy:
        records["image_path"] = records["image_path"].map(
            lambda value: f"{HISTORICAL_DATASET_PREFIX}/{value}")
        records = records.rename(columns={"species": "Species", "genus": "Genus",
            "family": "Family", "class_id": "ClassId", "observation_id": "ObservationId",
            "content": "Content"})
    records = records.sort_values("image_path")
    serialized = json.dumps(records.to_dict(orient="records"), ensure_ascii=False,
                            sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def split_fingerprint(frame: pd.DataFrame) -> str:
    """Hash schema-v1 canonical records independently of CSV layout and aliases."""
    return _fingerprint_records(frame, legacy=False)


def legacy_split_fingerprint(frame: pd.DataFrame) -> str:
    """Reproduce the original uppercase-key, PlantCLEF-prefixed checkpoint hash."""
    return _fingerprint_records(frame, legacy=True)
