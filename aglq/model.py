import torch
import torch.nn as nn
from torchvision import models


class CrossAttentionBlock(nn.Module):
    """One lightweight block where disease queries attend to image tokens."""

    def __init__(self, query_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 2, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        attended_queries, _ = self.cross_attention(
            query=queries,
            key=image_tokens,
            value=image_tokens,
            need_weights=False,
        )
        queries = self.norm1(queries + attended_queries)
        queries = self.norm2(queries + self.feed_forward(queries))
        return queries


class DenseNet121_AGLQ(nn.Module):
    """DenseNet121 with Adaptive Label Queries for multi-label diagnosis.

    This is the first AG-LQ experiment only:
    - DenseNet121 extracts image feature maps.
    - Each disease has one learnable label query.
    - Global image features adapt the label queries for the current image.
    - Cross-attention lets each disease query attend to image tokens.

    Dynamic label dependency / label self-attention is intentionally not used
    in this version.
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        query_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)

        # Keep only the convolutional feature extractor. DenseNet121 outputs
        # 1024 feature channels before global pooling/classification.
        self.backbone = densenet.features
        self.backbone_channels = densenet.classifier.in_features
        self.num_classes = num_classes
        self.query_dim = query_dim

        # Convert DenseNet image tokens from 1024 channels to query_dim.
        self.image_projection = nn.Linear(self.backbone_channels, query_dim)

        # One learnable query per NIH disease label.
        self.label_queries = nn.Parameter(torch.randn(num_classes, query_dim) * 0.02)

        # Query guidance: global image context creates a per-label adjustment.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.query_guidance = nn.Sequential(
            nn.Linear(self.backbone_channels, query_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim, num_classes * query_dim),
        )

        self.attention_blocks = nn.ModuleList(
            [CrossAttentionBlock(query_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # Per-label classifiers: each query produces its own disease logit.
        self.classifier_weight = nn.Parameter(torch.randn(num_classes, query_dim) * 0.02)
        self.classifier_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(images)
        feature_map = torch.relu(feature_map)

        batch_size, channels, height, width = feature_map.shape

        # Image tokens: [B, C, H, W] -> [B, H*W, C] -> [B, H*W, query_dim]
        image_tokens = feature_map.flatten(2).transpose(1, 2)
        image_tokens = self.image_projection(image_tokens)

        # Global context adapts the fixed disease queries for each image.
        global_feature = self.global_pool(feature_map).view(batch_size, channels)
        query_adjustment = self.query_guidance(global_feature)
        query_adjustment = query_adjustment.view(batch_size, self.num_classes, self.query_dim)

        base_queries = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = base_queries + query_adjustment

        for block in self.attention_blocks:
            queries = block(queries, image_tokens)

        queries = self.dropout(queries)
        logits = (queries * self.classifier_weight.unsqueeze(0)).sum(dim=-1)
        logits = logits + self.classifier_bias.unsqueeze(0)
        return logits


def build_densenet121_aglq(
    num_classes: int = 14,
    pretrained: bool = True,
    query_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 1,
    dropout: float = 0.1,
) -> DenseNet121_AGLQ:
    return DenseNet121_AGLQ(
        num_classes=num_classes,
        pretrained=pretrained,
        query_dim=query_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
