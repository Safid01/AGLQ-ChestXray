import torch.nn as nn
from torchvision import models


class DenseNet121Baseline(nn.Module):
    """DenseNet121 baseline for multi-label chest X-ray classification."""

    def __init__(self, num_classes: int = 14, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, images):
        # Return logits. Sigmoid is applied later for metrics/inference.
        return self.model(images)


def build_densenet121(
    num_classes: int = 14,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> DenseNet121Baseline:
    return DenseNet121Baseline(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
