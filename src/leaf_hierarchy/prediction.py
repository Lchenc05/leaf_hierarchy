"""Predict each configured classification task using checkpoint preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoints import load_checkpoint
from .config import add_common_arguments, apply_overrides, load_config
from .preprocessing import build_transform
from .runtime import configure_runtime, select_device

import torch
from PIL import Image, ImageOps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, data=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Print a JSON object instead of plain text.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    configure_runtime(num_threads=config["runtime"]["num_threads"])
    device = select_device(config["runtime"]["device"])
    model, checkpoint = load_checkpoint(args.checkpoint.expanduser().resolve(), device)
    tasks = checkpoint["model_spec"]["tasks"]
    with Image.open(image_path) as image:
        inputs = build_transform(checkpoint["preprocessing"])(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(device)
    predictions = {}
    with torch.no_grad():
        outputs = model(inputs)
        if not isinstance(outputs, dict) or set(outputs) != set(tasks):
            raise ValueError("Model output tasks do not match the checkpoint specification.")
        for task in tasks:
            mapping = checkpoint["taxonomy"]["class_mappings"][task]
            if outputs[task].ndim != 2 or outputs[task].shape != (1, len(mapping)):
                raise ValueError(f"Model output size does not match task {task!r}.")
            probabilities = outputs[task].softmax(dim=1)[0]
            predicted_index = int(probabilities.argmax().item())
            idx_to_name = {index: name for name, index in mapping.items()}
            predictions[task] = {"label": idx_to_name[predicted_index],
                                 "softmax_score": float(probabilities[predicted_index].item())}
    result = {"image": str(image_path), "predictions": predictions,
              "selected_epoch": int(checkpoint.get("epoch", -1))}
    if tasks == ["species"]:
        result.update(predicted_species=predictions["species"]["label"], softmax_score=predictions["species"]["softmax_score"])
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        for task, prediction in predictions.items():
            print(f"Predicted {task}: {prediction['label']}")
            print(f"Softmax score: {prediction['softmax_score']:.4f}")


if __name__ == "__main__":
    main()
