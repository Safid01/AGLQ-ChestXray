import torch.nn as nn


def get_bce_loss() -> nn.BCEWithLogitsLoss:
    """Binary cross entropy loss for multi-label classification."""
    return nn.BCEWithLogitsLoss()
