"""Versioned checkpoints and an explicit adapter for the reference experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import warnings

from . import runtime  # Configure CUDA before torch initialization.
from .models import create_model, validate_model_spec
from .preprocessing import default_preprocessing, validate_preprocessing

FORMAT_VERSION = 2


def _validate_mapping(mapping, *, minimum=1):
    if (not isinstance(mapping, dict) or len(mapping) < minimum
            or any(not isinstance(name, str) or not name.strip() for name in mapping)
            or any(type(index) is not int for index in mapping.values())
            or set(mapping.values()) != set(range(len(mapping)))):
        raise ValueError("Checkpoint class indices must be contiguous integers for nonempty labels.")
    return mapping


def _validate_taxonomy(taxonomy):
    if not isinstance(taxonomy, dict) or taxonomy.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint taxonomy schema.")
    mappings = taxonomy.get("class_mappings")
    if not isinstance(mappings, dict) or set(mappings) != {"species", "genus", "family"}:
        raise ValueError("Checkpoint taxonomy must describe species, genus and family vocabularies.")
    for task, mapping in mappings.items():
        _validate_mapping(mapping, minimum=2 if task == "species" else 1)
    for relation, child, parent in (("species_to_genus", "species", "genus"),
                                    ("genus_to_family", "genus", "family")):
        edges = taxonomy.get(relation)
        if (not isinstance(edges, dict) or set(edges) != set(mappings[child])
                or any(value not in mappings[parent] for value in edges.values())):
            raise ValueError(f"Incomplete or invalid checkpoint taxonomy relation: {relation}.")
    return mappings


def save_checkpoint(path: Path, model, *, config: dict, taxonomy: dict, epoch: int, metrics: dict):
    import torch
    from .data.taxonomy import taxonomy_fingerprint

    _validate_taxonomy(taxonomy)
    spec = validate_model_spec(config["model_spec"])
    preprocessing = validate_preprocessing(config["preprocessing"])
    if config.get("split_fingerprint_version") != 2 or not config.get("split_records_sha256"):
        raise ValueError("A new checkpoint requires the canonical training manifest fingerprint (version 2).")
    payload = {
        "format_version": FORMAT_VERSION,
        "model_spec": spec,
        "model_state_dict": model.state_dict(),
        "taxonomy": deepcopy(taxonomy),
        "taxonomy_sha256": taxonomy_fingerprint(taxonomy),
        "preprocessing": preprocessing,
        "epoch": int(epoch),
        "metrics": deepcopy(metrics),
        "config": deepcopy(config),
        "class_to_idx": deepcopy(taxonomy["class_mappings"]["species"]),
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device):
    """Load weights safely; only the historical format assumes a ResNet18 baseline."""
    import torch
    from .data.taxonomy import taxonomy_fingerprint

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict and model metadata.")
    if not isinstance(checkpoint.get("config", {}), dict):
        raise ValueError("Checkpoint config must be a dictionary.")
    version = checkpoint.get("format_version", 1)
    if type(version) is not int or version not in (1, FORMAT_VERSION):
        raise ValueError(f"Unsupported checkpoint format version: {version!r}.")
    checkpoint["config"] = checkpoint.get("config", {})
    if version == 1:
        architecture = checkpoint["config"].get("architecture", "resnet18_species_baseline")
        if architecture not in ("resnet18_species_baseline", "resnet18"):
            raise ValueError(f"Unsupported historical architecture: {architecture!r}.")
        mapping = _validate_mapping(checkpoint.get("class_to_idx"), minimum=2)
        spec = {"architecture": "resnet18", "tasks": ["species"]}
        mappings = {"species": mapping}
        # Earlier checkpoints contain no family/genus vocabulary; do not invent it.
        checkpoint["taxonomy"] = {"schema_version": 0, "source": "historical_checkpoint",
                                  "class_mappings": mappings}
        checkpoint["model_spec"] = spec
        checkpoint["preprocessing"] = default_preprocessing()
        state = {"backbone." + name: tensor for name, tensor in checkpoint["model_state_dict"].items()}
    else:
        spec = validate_model_spec(checkpoint.get("model_spec"))
        mappings = _validate_taxonomy(checkpoint.get("taxonomy"))
        if checkpoint.get("taxonomy_sha256") != taxonomy_fingerprint(checkpoint["taxonomy"]):
            raise ValueError("Checkpoint taxonomy fingerprint does not match its labels and relationships.")
        if checkpoint.get("class_to_idx") != mappings["species"]:
            raise ValueError("Checkpoint species vocabularies disagree.")
        if "preprocessing" not in checkpoint:
            raise ValueError("Checkpoint preprocessing specification is missing.")
        checkpoint["preprocessing"] = validate_preprocessing(checkpoint["preprocessing"])
        if (checkpoint["config"].get("split_fingerprint_version") != 2
                or not checkpoint["config"].get("split_records_sha256")):
            raise ValueError("Checkpoint training manifest fingerprint is missing or unsupported.")
        if checkpoint["config"].get("model_spec") != spec:
            raise ValueError("Checkpoint and experiment model specifications disagree.")
        if validate_preprocessing(checkpoint["config"].get("preprocessing")) != checkpoint["preprocessing"]:
            raise ValueError("Checkpoint and experiment preprocessing specifications disagree.")
        state = checkpoint["model_state_dict"]
    checkpoint["format_version"] = version
    model = create_model(spec, mappings, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def check_split_identity(checkpoint: dict, data) -> None:
    """Check records using the fingerprint schema that was saved during training."""
    from .data.manifest import legacy_split_fingerprint, split_fingerprint
    from .data.taxonomy import build_taxonomy, taxonomy_fingerprint

    config = checkpoint.get("config", {})
    expected = config.get("split_records_sha256")
    if expected is None:
        if checkpoint.get("format_version", 1) != 1:
            raise ValueError("Checkpoint training manifest fingerprint is missing.")
        warnings.warn("Historical checkpoint has no canonical split fingerprint. Class mapping and group separation "
                      "were validated, but its original partition cannot be verified automatically.", UserWarning,
                      stacklevel=2)
        return
    version = config.get("split_fingerprint_version", 1)
    if version not in (1, 2):
        raise ValueError(f"Unsupported split fingerprint version: {version}.")
    actual = legacy_split_fingerprint(data) if version == 1 else split_fingerprint(data)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError("The active split records differ from the training checkpoint. Use the original manifest.")
    if checkpoint.get("format_version") == FORMAT_VERSION:
        if taxonomy_fingerprint(build_taxonomy(data)) != checkpoint["taxonomy_sha256"]:
            raise ValueError("The manifest taxonomy differs from the training checkpoint.")
