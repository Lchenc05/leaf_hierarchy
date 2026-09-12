"""Shared image loading, split checks and ResNet18 baseline utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
import warnings
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "PlantCLEF2015TrainingData"
DEFAULT_SPLIT_FILE = ROOT / "results" / "analysis" / "particion_propuesta.csv"
DATASET_NAME = "PlantCLEF2015TrainingData"
SPLITS = ("train", "validation", "test")
MODEL_SEED = 20260910

# Configure deterministic CUDA operations before importing PyTorch.
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def select_device(name: str) -> torch.device:
    """Resolve an explicit device without silently falling back from CUDA."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    return torch.device(name)


def configure_runtime(seed: int = MODEL_SEED, num_threads: int = 6) -> None:
    """Reset the random state and use the original deterministic settings."""
    if num_threads < 1:
        raise ValueError("--num-threads must be at least 1.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def canonical_image_path(value: str) -> str:
    """Normalize a manifest path while preventing access outside the dataset."""
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError(f"Image paths must be relative dataset paths: {value!r}")
    parts = path.parts
    if parts and parts[0] == DATASET_NAME:
        parts = parts[1:]
    if len(parts) < 2 or parts[0] != "train":
        raise ValueError(f"Expected {DATASET_NAME}/train/<image>, received {value!r}.")
    return str(PurePosixPath(DATASET_NAME, *parts))


def resolve_image_path(value: str, data_dir: Path) -> Path:
    """Map canonical manifest paths onto an independently located dataset."""
    relative = PurePosixPath(canonical_image_path(value)).relative_to(DATASET_NAME)
    root = data_dir.expanduser().resolve()
    candidate = (root / Path(*relative.parts)).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Image path resolves outside --data-dir: {value!r}")
    return candidate


def validate_class_mapping(mapping: dict) -> dict[str, int]:
    """Require one contiguous integer target for each nonempty species name."""
    if not isinstance(mapping, dict) or len(mapping) < 2:
        raise ValueError("The checkpoint must contain at least two species in class_to_idx.")
    if any(not isinstance(name, str) or not name.strip() for name in mapping):
        raise ValueError("Checkpoint species names must be nonempty strings.")
    if any(type(index) is not int for index in mapping.values()):
        raise ValueError("Checkpoint class indices must be integers.")
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("Checkpoint class indices must be unique and contiguous from zero.")
    return mapping


def load_manifest(split_file: Path, data_dir: Path, class_to_idx: dict | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate taxonomy, complete splits, source groups and all active images."""
    split_file = split_file.expanduser().resolve()
    if not split_file.is_file():
        raise FileNotFoundError(f"Run analyze_data.py first. Missing split manifest: {split_file}")
    data = pd.read_csv(split_file, dtype=str, keep_default_na=False)
    required = {"image_path", "Species", "ClassId", "ObservationId", "group_id", "split"}
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
    data["image_path"] = data["image_path"].map(canonical_image_path)
    data = data.sort_values("image_path").reset_index(drop=True)
    if not data["image_path"].is_unique:
        raise ValueError("Duplicate image paths occur in the active split manifest.")
    if (data.groupby("Species")["ClassId"].nunique() != 1).any():
        raise ValueError("A species has multiple ClassId values in the split manifest.")
    if (data.groupby("ClassId")["Species"].nunique() != 1).any():
        raise ValueError("A ClassId has multiple species names in the split manifest.")
    for field in ("ObservationId", "group_id"):
        if (data.groupby(field)["split"].nunique() != 1).any():
            raise ValueError(f"Data leakage: {field} values overlap across splits.")
    species_names = sorted(data["Species"].unique())
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
        observed_species = set(data.loc[data["split"] == split, "Species"])
        if observed_species != set(species_names):
            raise ValueError(f"Split {split!r} must contain every active species.")
    resolved_paths = data["image_path"].map(lambda value: resolve_image_path(value, data_dir))
    missing_paths = resolved_paths.loc[~resolved_paths.map(Path.is_file)]
    if len(missing_paths):
        raise FileNotFoundError(f"{len(missing_paths)} images are missing. First missing image: {missing_paths.iloc[0]}")
    data["target"] = data["Species"].map(class_to_idx)
    return data, class_to_idx


def split_fingerprint(data: pd.DataFrame) -> str:
    """Hash canonical active records independently of CSV formatting and row order."""
    columns = ["image_path", "Species", "ClassId", "ObservationId", "group_id", "split", "Genus", "Family", "Content"]
    records = data.reindex(columns=columns, fill_value="").copy()
    records["image_path"] = records["image_path"].map(canonical_image_path)
    records = records.sort_values("image_path")
    serialized = json.dumps(records.to_dict(orient="records"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def check_split_identity(checkpoint: dict, data: pd.DataFrame) -> None:
    """Reject changed active records for checkpoints created by these scripts."""
    expected = checkpoint.get("config", {}).get("split_records_sha256")
    if expected is None:
        warnings.warn(
            "Historical checkpoint has no canonical split fingerprint. Class mapping and group separation "
            "were validated, but the original training partition cannot be verified automatically.",
            UserWarning,
            stacklevel=2,
        )
    elif not isinstance(expected, str) or expected != split_fingerprint(data):
        raise ValueError("The active split records differ from the training checkpoint. Use the original split manifest.")


def evaluation_transform():
    """Use the baseline ImageNet resize, center crop and normalization."""
    return models.ResNet18_Weights.IMAGENET1K_V1.transforms()


class LeafDataset(Dataset):
    """Load a leaf image with the original EXIF orientation and RGB conversion."""

    def __init__(self, frame: pd.DataFrame, data_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.data_dir = data_dir.expanduser().resolve()
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(resolve_image_path(row["image_path"], self.data_dir)) as image:
            image = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        return image, int(row["target"])


def make_loader(data: pd.DataFrame, split: str, data_dir: Path, batch_size: int, seed: int = MODEL_SEED) -> DataLoader:
    """Shuffle and augment training data only; keep Windows loading portable."""
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    transform = evaluation_transform()
    if split == "train":
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transform])
    dataset = LeafDataset(data.loc[data["split"] == split], data_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def create_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Load tensor-only checkpoints, including those from the historical baseline."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("The checkpoint must contain model_state_dict and class_to_idx.")
    class_to_idx = validate_class_mapping(checkpoint.get("class_to_idx"))
    if "config" in checkpoint and not isinstance(checkpoint["config"], dict):
        raise ValueError("Checkpoint config must be a dictionary.")
    model = create_model(len(class_to_idx))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    """Return mean cross-entropy and predictions without updating model weights."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, true_labels, predicted_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(outputs.argmax(1).cpu().tolist())
    return total_loss / len(loader.dataset), true_labels, predicted_labels


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
