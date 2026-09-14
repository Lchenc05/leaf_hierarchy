"""Serializable image preprocessing shared by training and saved-model inference."""

from __future__ import annotations

import math


def default_preprocessing() -> dict:
    return {"resize_size": 256, "crop_size": 224,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "interpolation": "bilinear", "antialias": True}


def default_augmentation() -> dict:
    return {"horizontal_flip": 0.5, "vertical_flip": 0.5}


def validate_preprocessing(config=None) -> dict:
    defaults = default_preprocessing()
    if config is not None:
        if not isinstance(config, dict) or set(config) - set(defaults):
            raise ValueError("Preprocessing must contain only the supported image transform fields.")
        defaults.update(config)
    for key in ("resize_size", "crop_size"):
        if type(defaults[key]) is not int or defaults[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("mean", "std"):
        values = defaults[key]
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"{key} must contain three RGB values.")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError(f"{key} values must be finite numbers.")
        if key == "std" and any(value <= 0 for value in values):
            raise ValueError("Normalization standard deviations must be positive.")
        defaults[key] = list(values)
    if defaults["interpolation"] not in {"bilinear", "bicubic", "nearest"}:
        raise ValueError("Unsupported interpolation mode.")
    if type(defaults["antialias"]) is not bool:
        raise ValueError("antialias must be boolean.")
    return defaults


def validate_augmentation(config=None) -> dict:
    result = default_augmentation()
    if config is not None:
        if not isinstance(config, dict) or set(config) - set(result):
            raise ValueError("Unsupported augmentation fields.")
        result.update(config)
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in result.values()):
        raise ValueError("Flip probabilities must be finite numbers between zero and one.")
    return result


def build_transform(config=None, training=False, augmentation=None):
    # Import the runtime before torch/torchvision to configure deterministic CUDA.
    from . import runtime  # noqa: F401
    from torchvision import transforms
    spec = validate_preprocessing(config)
    interpolation = transforms.InterpolationMode(spec["interpolation"])
    transform = transforms.Compose([
        transforms.Resize(spec["resize_size"], interpolation=interpolation, antialias=spec["antialias"]),
        transforms.CenterCrop(spec["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize(spec["mean"], spec["std"]),
    ])
    if training:
        flips = validate_augmentation(augmentation)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(flips["horizontal_flip"]),
            transforms.RandomVerticalFlip(flips["vertical_flip"]), transform,
        ])
    return transform
