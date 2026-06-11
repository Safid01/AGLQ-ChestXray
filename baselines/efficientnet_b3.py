import torch.nn as nn
from torchvision import models


class EfficientNetB3Baseline(nn.Module):
    """EfficientNet-B3 baseline for multi-label chest X-ray classification."""

    def __init__(self, num_classes: int = 14, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_b3(weights=weights)

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, images):
        # Return logits. Sigmoid is applied later for metrics/inference.
        return self.model(images)


def build_efficientnet_b3(
    num_classes: int = 14,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> EfficientNetB3Baseline:
    return EfficientNetB3Baseline(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
