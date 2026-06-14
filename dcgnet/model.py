from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torchvision import models


def build_cooccurrence_adjacency(
    csv_path: str | Path,
    class_names: list[str],
    num_classes: int,
) -> torch.Tensor:
    """Build a row-normalized disease co-occurrence graph from train.csv."""
    data = pd.read_csv(csv_path)
    labels = torch.tensor(data[class_names].values, dtype=torch.float32)

    adjacency = labels.t().matmul(labels)
    adjacency = adjacency + torch.eye(num_classes, dtype=torch.float32)

    row_sum = adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adjacency / row_sum


class GraphConvolutionBlock(nn.Module):
    """Residual graph convolution over the disease nodes."""

    def __init__(self, graph_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(graph_dim, graph_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(graph_dim)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.matmul(adjacency, features)
        updated = self.linear(aggregated)
        updated = self.activation(updated)
        updated = self.dropout(updated)
        return self.norm(features + updated)


class DenseNet121_DCGNet(nn.Module):
    """Disease Co-occurrence Guided Network with a DenseNet121 backbone."""

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        graph_dim: int = 256,
        num_graph_layers: int = 2,
        dropout: float = 0.2,
        beta: float = 0.1,
        adjacency_source: str | Path = "datasets/ChestXray14/train.csv",
        class_names: list[str] | None = None,
    ):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)

        self.backbone = densenet.features
        self.backbone_channels = densenet.classifier.in_features
        self.num_classes = num_classes
        self.beta = beta

        if class_names is None:
            class_names = [str(index) for index in range(num_classes)]

        adjacency = build_cooccurrence_adjacency(adjacency_source, class_names, num_classes)
        self.register_buffer("adjacency", adjacency)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_dropout = nn.Dropout(dropout)
        self.global_classifier = nn.Linear(self.backbone_channels, num_classes)

        self.disease_embeddings = nn.Parameter(torch.randn(num_classes, graph_dim) * 0.02)
        self.global_to_graph = nn.Sequential(
            nn.Linear(self.backbone_channels, graph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim, graph_dim),
        )

        self.graph_layers = nn.ModuleList(
            [GraphConvolutionBlock(graph_dim, dropout) for _ in range(num_graph_layers)]
        )
        self.graph_dropout = nn.Dropout(dropout)
        self.graph_classifier_weight = nn.Parameter(torch.randn(num_classes, graph_dim) * 0.02)
        self.graph_classifier_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.training:
            feature_map = grad_checkpoint(self.backbone, images, use_reentrant=False)
        else:
            feature_map = self.backbone(images)

        feature_map = torch.relu(feature_map)
        batch_size, channels, _, _ = feature_map.shape

        global_feature = self.global_pool(feature_map).view(batch_size, channels)
        global_logits = self.global_classifier(self.global_dropout(global_feature))

        image_context = self.global_to_graph(global_feature).unsqueeze(1)
        graph_features = self.disease_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        graph_features = graph_features + image_context

        adjacency = self.adjacency.to(device=graph_features.device, dtype=graph_features.dtype)
        for graph_layer in self.graph_layers:
            graph_features = graph_layer(graph_features, adjacency)

        graph_features = self.graph_dropout(graph_features)
        graph_logits = (graph_features * self.graph_classifier_weight.unsqueeze(0)).sum(dim=-1)
        graph_logits = graph_logits + self.graph_classifier_bias.unsqueeze(0)

        logits = global_logits + self.beta * graph_logits
        return {
            "logits": logits,
            "global_logits": global_logits,
            "graph_logits": graph_logits,
        }


def build_densenet121_dcgnet(
    num_classes: int = 14,
    pretrained: bool = True,
    graph_dim: int = 256,
    num_graph_layers: int = 2,
    dropout: float = 0.2,
    beta: float = 0.1,
    adjacency_source: str | Path = "datasets/ChestXray14/train.csv",
    class_names: list[str] | None = None,
) -> DenseNet121_DCGNet:
    return DenseNet121_DCGNet(
        num_classes=num_classes,
        pretrained=pretrained,
        graph_dim=graph_dim,
        num_graph_layers=num_graph_layers,
        dropout=dropout,
        beta=beta,
        adjacency_source=adjacency_source,
        class_names=class_names,
    )
