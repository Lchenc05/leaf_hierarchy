"""Model construction from an explicit experiment specification."""

from __future__ import annotations


def validate_model_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != {"architecture", "tasks"}:
        raise ValueError("A model specification must contain architecture and tasks.")
    if spec["architecture"] != "resnet18":
        raise ValueError(f"Unsupported architecture: {spec['architecture']!r}.")
    if spec["tasks"] != ["species"]:
        raise ValueError("The implemented ResNet18 experiment supports tasks=['species']; multitask models are planned.")
    return {"architecture": spec["architecture"], "tasks": list(spec["tasks"])}


def create_model(spec: dict, class_mappings: dict, pretrained: bool = False):
    spec = validate_model_spec(spec)
    from .resnet import ResNetClassifier
    mapping = class_mappings.get("species", {})
    if (len(mapping) < 2 or any(not isinstance(name, str) or not name.strip() for name in mapping)
            or any(type(index) is not int for index in mapping.values())
            or set(mapping.values()) != set(range(len(mapping)))):
        raise ValueError("The species vocabulary must have at least two contiguous class indices.")
    return ResNetClassifier(len(mapping), pretrained=pretrained)
