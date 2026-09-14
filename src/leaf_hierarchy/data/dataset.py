"""Image loading with labels identified by classification task."""

from __future__ import annotations

from pathlib import Path

from leaf_hierarchy.runtime import MODEL_SEED, SPLITS
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset

from leaf_hierarchy.preprocessing import build_transform
from .manifest import normalize_manifest_columns, resolve_image_path, validate_class_mapping
from .taxonomy import build_taxonomy


class LeafDataset(Dataset):
    """Load RGB images and task labels, retaining the ordered source manifest."""

    def __init__(self, frame, data_dir: Path, transform, *, class_mappings, tasks=("species",)):
        self.frame = normalize_manifest_columns(frame).reset_index(drop=True)
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.transform = transform
        self.tasks = tuple(tasks)
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("At least one distinct classification task is required.")
        self.class_mappings = {}
        for task in self.tasks:
            if task not in self.frame or task not in class_mappings:
                raise ValueError(f"Missing labels or class mapping for task {task!r}.")
            mapping = validate_class_mapping(class_mappings[task], minimum_classes=1)
            if not set(self.frame[task]).issubset(mapping):
                raise ValueError(f"Unknown labels for task {task!r}.")
            self.class_mappings[task] = mapping

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(resolve_image_path(row["image_path"], self.data_dir)) as original:
            image = self.transform(ImageOps.exif_transpose(original).convert("RGB"))
        return image, {task: self.class_mappings[task][row[task]] for task in self.tasks}


def make_loader(frame, split: str, data_dir: Path, batch_size: int, seed: int = MODEL_SEED,
                *, class_mappings=None, tasks=("species",), preprocessing=None, augmentation=None):
    """Shuffle and augment training only; use portable deterministic workers."""
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if split not in SPLITS:
        raise ValueError(f"Unknown data split {split!r}.")
    data = normalize_manifest_columns(frame)
    if class_mappings is None:
        class_mappings = build_taxonomy(data)["class_mappings"]
    selected = data.loc[data["split"] == split]
    if selected.empty:
        raise ValueError(f"The manifest does not contain images in split {split!r}.")
    transform = build_transform(preprocessing, training=(split == "train"), augmentation=augmentation)
    dataset = LeafDataset(selected, data_dir, transform, class_mappings=class_mappings, tasks=tasks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=0,
                      generator=torch.Generator().manual_seed(seed))
