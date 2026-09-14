"""Shared paths, reproducible runtime settings and JSON output."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "PlantCLEF2015TrainingData"
DEFAULT_SPLIT_FILE = ROOT / "results" / "data" / "plantclef2015" / "v1" / "split_manifest.csv"
MODEL_SEED = 20260910
SPLITS = ("train", "validation", "test")

# These settings must precede torch initialization.
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def select_device(name: str):
    import torch
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {name}")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but PyTorch cannot access a CUDA device.")
    return torch.device(name)


def configure_runtime(seed: int = MODEL_SEED, num_threads: int = 6) -> None:
    import numpy as np
    import torch
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 4294967295.")
    if not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError("num_threads must be a positive integer.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
