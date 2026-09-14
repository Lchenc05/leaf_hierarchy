"""Dataset-independent manifest, taxonomy and image-loader interfaces."""

from .manifest import (canonical_image_path, legacy_split_fingerprint, load_manifest,
                       normalize_manifest_columns, resolve_image_path,
                       split_fingerprint, validate_class_mapping)
from .taxonomy import build_taxonomy, taxonomy_fingerprint


def make_loader(*args, **kwargs):
    # Dataset preparation and manifest validation do not require PyTorch.
    from .dataset import make_loader as implementation
    return implementation(*args, **kwargs)


__all__ = ["build_taxonomy", "canonical_image_path", "legacy_split_fingerprint",
           "load_manifest", "make_loader", "normalize_manifest_columns",
           "resolve_image_path", "split_fingerprint", "taxonomy_fingerprint",
           "validate_class_mapping"]
