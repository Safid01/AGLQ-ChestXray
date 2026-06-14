import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for imbalanced multi-label classification.

    This downweights easy negative labels, which are very common in
    ChestX-ray14. It is optional and selected from the YAML config.
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        pos_probs = probabilities
        neg_probs = 1.0 - probabilities

        if self.clip > 0:
            neg_probs = torch.clamp(neg_probs + self.clip, max=1.0)

        pos_loss = targets * torch.log(torch.clamp(pos_probs, min=self.eps))
        neg_loss = (1.0 - targets) * torch.log(torch.clamp(neg_probs, min=self.eps))

        pos_weight = torch.pow(1.0 - pos_probs, self.gamma_pos)
        neg_weight = torch.pow(1.0 - neg_probs, self.gamma_neg)
        loss = pos_weight * pos_loss + neg_weight * neg_loss
        return -loss.mean()


def get_bce_loss(pos_weight: torch.Tensor | None = None) -> nn.BCEWithLogitsLoss:
    """Binary cross entropy loss for multi-label classification."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_loss(config: dict, pos_weight: torch.Tensor | None = None) -> nn.Module:
    """Build the configured loss while keeping BCE as the safe default."""
    loss_name = config.get("training", {}).get("loss", "bce").lower()

    if loss_name in {"balanced_bce", "weighted_bce", "pos_weight_bce"}:
        return get_bce_loss(pos_weight=pos_weight)

    if loss_name in {"asl", "asymmetric", "asymmetric_loss"}:
        loss_cfg = config.get("loss", {})
        return AsymmetricLoss(
            gamma_pos=loss_cfg.get("gamma_pos", 1.0),
            gamma_neg=loss_cfg.get("gamma_neg", 4.0),
            clip=loss_cfg.get("clip", 0.05),
        )

    return get_bce_loss()
