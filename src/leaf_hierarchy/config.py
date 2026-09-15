"""Validated experiment configuration and explicit command-line overrides."""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import tomllib
from pathlib import Path

from .models import validate_model_spec
from .preprocessing import (default_augmentation, default_preprocessing,
                            validate_augmentation, validate_preprocessing)
from .runtime import DEFAULT_DATA_DIR, DEFAULT_SPLIT_FILE, MODEL_SEED, ROOT


def data_dir_from_env() -> Path | None:
    """Read the optional dataset location relative to the current directory."""
    value = os.environ.get("LEAF_HIERARCHY_DATA_DIR")
    if value is None or not value.strip():
        return None
    return Path(value).expanduser().resolve()


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


def _read_toml(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.expanduser().resolve().open("rb") as source:
        return tomllib.load(source)


def _resolve_config_path(value: str, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty path string.")
    return str((ROOT / Path(value).expanduser()).resolve())


def _resolve_data_paths(data: dict) -> dict:
    data = {key: _resolve_config_path(value, f"data.{key}") for key, value in data.items()}
    environment_data_dir = data_dir_from_env()
    if environment_data_dir is not None:
        data["data_dir"] = str(environment_data_dir)
    return data


def load_data_config(path: Path | None = None) -> dict:
    """Read only [data] for preparation, independent of model and training settings."""
    config = {"data": default_config()["data"]}
    _merge(config, {"data": _read_toml(path).get("data", {})})
    return _resolve_data_paths(config["data"])


def load_config(path: Path | None = None) -> dict:
    """Read repository-relative TOML paths, then apply the dataset environment override."""
    config = default_config()
    _merge(config, _read_toml(path))
    validate_config(config)
    config["data"] = _resolve_data_paths(config["data"])
    config["output"]["run_root"] = _resolve_config_path(config["output"]["run_root"], "output.run_root")
    return config


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Explicit CLI arguments override TOML and environment; CLI paths use the current directory."""
    config = copy.deepcopy(config)
    options = {
        "data_dir": ("data", "data_dir"), "split_file": ("data", "split_file"),
        "epochs": ("training", "epochs"), "batch_size": ("training", "batch_size"),
        "seed": ("training", "seed"), "weights": ("model", "weights"),
        "device": ("runtime", "device"), "num_threads": ("runtime", "num_threads"),
    }
    for option, (section, key) in options.items():
        value = getattr(args, option, None)
        if value is not None:
            config[section][key] = str(value.expanduser().resolve()) if isinstance(value, Path) else value
    validate_config(config)
    return config


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="Experiment TOML; explicit flags override its values.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset directory; overrides [data].data_dir and any environment setting.")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--num-threads", type=int, default=None)
