"""Model construction from an explicit experiment specification."""

from __future__ import annotations


def validate_model_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != {"architecture", "tasks"}:
        raise ValueError("A model specification must contain architecture and tasks.")
    if spec["architecture"] != "resnet18":
        raise ValueError(f"Unsupported architecture: {spec['architecture']!r}.")
    if spec["tasks"] not in (["species"], ["family", "genus", "species"]):
        raise ValueError("ResNet18 supports tasks=['species'] or ['family', 'genus', 'species'].")
    return {"architecture": spec["architecture"], "tasks": list(spec["tasks"])}


def create_model(spec: dict, class_mappings: dict, pretrained: bool = False):
    spec = validate_model_spec(spec)
    from .resnet import ResNetClassifier, ResNetMultitaskClassifier
    counts = {}
    for task in spec["tasks"]:
        mapping = class_mappings.get(task, {})
        minimum = 2 if task == "species" else 1
        if (not isinstance(mapping, dict) or len(mapping) < minimum
                or any(not isinstance(name, str) or not name.strip() for name in mapping)
                or any(type(index) is not int for index in mapping.values())
                or set(mapping.values()) != set(range(len(mapping)))):
            raise ValueError(f"The {task} vocabulary requires at least {minimum} contiguous class indices.")
        counts[task] = len(mapping)
    if spec["tasks"] == ["species"]:
        return ResNetClassifier(counts["species"], pretrained=pretrained)
    return ResNetMultitaskClassifier(counts, pretrained=pretrained)
