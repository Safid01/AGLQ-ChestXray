import torch.nn as nn
from torchvision import models


class ResNet50Baseline(nn.Module):
    """ResNet50 baseline for multi-label chest X-ray classification."""

    def __init__(self, num_classes: int = 14, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, images):
        # Return logits. Sigmoid is applied later for metrics/inference.
        return self.model(images)


def build_resnet50(
    num_classes: int = 14,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> ResNet50Baseline:
    return ResNet50Baseline(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
