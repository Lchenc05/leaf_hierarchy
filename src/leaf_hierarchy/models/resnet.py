"""ResNet species classifier with explicit access to its learned representations."""

from .. import runtime  # Configure CUDA before importing torch.
import torch
from torch import nn
from torchvision import models


class ResNetClassifier(nn.Module):
    tasks = ("species",)

    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    @property
    def feature_layer(self):
        """Spatial features available to a future gradient-based explanation method."""
        return self.backbone.layer4

    def forward_features(self, images):
        """Return one pooled embedding per input image, preserving batch order."""
        model = self.backbone
        features = model.maxpool(model.relu(model.bn1(model.conv1(images))))
        features = model.layer4(model.layer3(model.layer2(model.layer1(features))))
        return torch.flatten(model.avgpool(features), 1)

    def forward(self, images):
        return {"species": self.backbone.fc(self.forward_features(images))}
