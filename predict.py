"""Predict one image's species with a saved ResNet18 baseline checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_utils import configure_runtime, evaluation_transform, load_checkpoint, select_device

import torch
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--json", action="store_true", help="Print a JSON object instead of plain text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    configure_runtime(num_threads=args.num_threads)
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint.expanduser().resolve(), device)
    with Image.open(image_path) as image:
        inputs = evaluation_transform()(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(inputs).softmax(dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
    idx_to_class = {index: name for name, index in checkpoint["class_to_idx"].items()}
    result = {"image": str(image_path), "predicted_species": idx_to_class[predicted_index],
              "softmax_score": float(probabilities[predicted_index].item()),
              "selected_epoch": int(checkpoint.get("epoch", -1))}
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"Predicted species: {result['predicted_species']}")
        print(f"Softmax score: {result['softmax_score']:.4f}")


if __name__ == "__main__":
    main()
