import torch
import torch.nn as nn
from torchvision import models


class Query2Label(nn.Module):
    """Query2Label baseline with a DenseNet121 image backbone.

    The model keeps the baseline simple:
    1. DenseNet121 extracts a spatial feature map.
    2. The feature map is flattened into visual tokens.
    3. Each disease label has one learnable query.
    4. Transformer decoder cross-attention lets label queries attend to image tokens.
    5. Each label query predicts one disease logit.
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_decoder_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)

        self.backbone = densenet.features
        backbone_channels = densenet.classifier.in_features
        self.num_classes = num_classes

        self.image_projection = nn.Linear(backbone_channels, hidden_dim)
        self.label_queries = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.dropout = nn.Dropout(dropout)

        self.classifier_weight = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)
        self.classifier_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(images)
        feature_map = torch.relu(feature_map)
        batch_size = feature_map.size(0)

        # [B, C, H, W] -> [B, H*W, C] -> [B, H*W, hidden_dim]
        image_tokens = feature_map.flatten(2).transpose(1, 2)
        image_tokens = self.image_projection(image_tokens)

        # [num_classes, hidden_dim] -> [B, num_classes, hidden_dim]
        label_queries = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded_queries = self.decoder(tgt=label_queries, memory=image_tokens)
        decoded_queries = self.dropout(decoded_queries)

        logits = (decoded_queries * self.classifier_weight.unsqueeze(0)).sum(dim=-1)
        logits = logits + self.classifier_bias.unsqueeze(0)
        return logits


def build_query2label(
    num_classes: int = 14,
    pretrained: bool = True,
    hidden_dim: int = 256,
    num_heads: int = 4,
    num_decoder_layers: int = 2,
    dropout: float = 0.2,
) -> Query2Label:
    return Query2Label(
        num_classes=num_classes,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
    )
