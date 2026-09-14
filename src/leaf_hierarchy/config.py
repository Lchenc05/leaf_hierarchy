"""Validated experiment configuration and explicit command-line overrides."""

from __future__ import annotations

import argparse
import copy
import math
import re
import tomllib
from pathlib import Path

from .models import validate_model_spec
from .preprocessing import (default_augmentation, default_preprocessing,
                            validate_augmentation, validate_preprocessing)
from .runtime import DEFAULT_DATA_DIR, DEFAULT_SPLIT_FILE, MODEL_SEED, ROOT


def default_config() -> dict:
    """Return the current experiment defaults without sharing mutable values."""
    return {
        "experiment": "resnet18_species",
        "data": {"data_dir": str(DEFAULT_DATA_DIR), "split_file": str(DEFAULT_SPLIT_FILE)},
        "model": {"architecture": "resnet18", "tasks": ["species"], "weights": "imagenet"},
        "training": {"epochs": 5, "batch_size": 16, "seed": MODEL_SEED,
                     "lr": 1e-4, "weight_decay": 1e-4},
        "runtime": {"device": "auto", "num_threads": 6},
        "output": {"run_root": str(ROOT / "results" / "runs")},
        "selection": {"task": "species", "metric": "macro_f1"},
        "preprocessing": default_preprocessing(),
        "augmentation": default_augmentation(),
    }


def _merge(base: dict, values: dict, prefix: str = "") -> None:
    for key, value in values.items():
        qualified = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown configuration key: {qualified}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Configuration {qualified} must be a table.")
            _merge(base[key], value, qualified)
        else:
            base[key] = value


def validate_config(config: dict) -> None:
    """Reject unsupported experiments and invalid values before opening a run."""
    name = config["experiment"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("experiment must be a nonempty directory-safe name.")
    model = config["model"]
    validate_model_spec({key: model[key] for key in ("architecture", "tasks")})
    if model["weights"] not in ("imagenet", "none"):
        raise ValueError("model.weights must be 'imagenet' or 'none'.")
    for section, key in (("training", "epochs"), ("training", "batch_size"), ("runtime", "num_threads")):
        if type(config[section][key]) is not int or config[section][key] < 1:
            raise ValueError(f"{section}.{key} must be a positive integer.")
    seed = config["training"]["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("training.seed must be an integer between 0 and 4294967295.")
    for key in ("lr", "weight_decay"):
        value = config["training"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == "lr" and value == 0):
            raise ValueError(f"training.{key} must be finite and {'positive' if key == 'lr' else 'nonnegative'}.")
    if config["runtime"]["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("runtime.device must be 'auto', 'cpu', or 'cuda'.")
    if config["selection"]["task"] not in model["tasks"]:
        raise ValueError("selection.task must be an output task of the model.")
    if config["selection"]["metric"] not in ("macro_f1", "accuracy"):
        raise ValueError("selection.metric must be 'macro_f1' or 'accuracy'.")
    for section, key in (("data", "data_dir"), ("data", "split_file"), ("output", "run_root")):
        if not isinstance(config[section][key], str) or not config[section][key].strip():
            raise ValueError(f"{section}.{key} must be a nonempty path string.")
    validate_preprocessing(config["preprocessing"])
    validate_augmentation(config["augmentation"])


def load_config(path: Path | None = None) -> dict:
    """Read TOML; file paths in configuration always refer to the repository root."""
    config = default_config()
    if path is not None:
        path = path.expanduser().resolve()
        with path.open("rb") as source:
            values = tomllib.load(source)
        _merge(config, values)
    validate_config(config)
    for section, key in (("data", "data_dir"), ("data", "split_file"), ("output", "run_root")):
        location = Path(config[section][key]).expanduser()
        config[section][key] = str((ROOT / location).resolve() if not location.is_absolute() else location.resolve())
    return config


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Only explicit CLI arguments override TOML; CLI paths use the current directory."""
    config = copy.deepcopy(config)
    options = {
        "data_dir": ("data", "data_dir"), "split_file": ("data", "split_file"),
        "epochs": ("training", "epochs"), "batch_size": ("training", "batch_size"),
        "seed": ("training", "seed"), "lr": ("training", "lr"),
        "weight_decay": ("training", "weight_decay"), "weights": ("model", "weights"),
        "device": ("runtime", "device"), "num_threads": ("runtime", "num_threads"),
    }
    for option, (section, key) in options.items():
        value = getattr(args, option, None)
        if value is not None:
            config[section][key] = str(value.expanduser().resolve()) if isinstance(value, Path) else value
    validate_config(config)
    return config


def add_common_arguments(parser: argparse.ArgumentParser, *, data: bool = True) -> None:
    parser.add_argument("--config", type=Path, help="Experiment TOML; explicit flags override its values.")
    if data:
        parser.add_argument("--data-dir", type=Path, default=None)
        parser.add_argument("--split-file", type=Path, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--num-threads", type=int, default=None)
