import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _safe_mean(values: list[float]) -> float:
    valid_values = [value for value in values if not np.isnan(value)]
    if not valid_values:
        return float("nan")
    return float(np.mean(valid_values))


def _build_thresholds(search_min: float, search_max: float, search_step: float) -> list[float]:
    thresholds = []
    current = search_min
    while current <= search_max + 1e-8:
        thresholds.append(round(current, 4))
        current += search_step
    return thresholds


def tune_threshold_for_macro_f1(
    probabilities: np.ndarray,
    targets: np.ndarray,
    search_min: float,
    search_max: float,
    search_step: float,
) -> tuple[float, float]:
    """Find one validation threshold that gives the best macro F1."""
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in _build_thresholds(search_min, search_max, search_step):
        predictions = (probabilities >= threshold).astype(int)
        score = float(f1_score(targets, predictions, average="macro", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
    threshold_config: dict | None = None,
) -> dict:
    """Compute multi-label classification metrics from model logits."""
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    threshold_config = threshold_config or {}
    default_threshold = threshold_config.get("default", 0.5)
    tune_threshold = threshold_config.get("tune_on_validation", False)

    if tune_threshold:
        threshold, macro_f1 = tune_threshold_for_macro_f1(
            probabilities=probabilities,
            targets=targets,
            search_min=threshold_config.get("search_min", 0.1),
            search_max=threshold_config.get("search_max", 0.9),
            search_step=threshold_config.get("search_step", 0.05),
        )
    else:
        threshold = default_threshold
        predictions = (probabilities >= threshold).astype(int)
        macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))

    average_precisions = []
    per_class_auroc = {}
    aurocs = []

    for class_index, class_name in enumerate(class_names):
        class_targets = targets[:, class_index]
        class_probs = probabilities[:, class_index]

        if len(np.unique(class_targets)) < 2:
            average_precisions.append(float("nan"))
            per_class_auroc[class_name] = float("nan")
            continue

        average_precisions.append(float(average_precision_score(class_targets, class_probs)))
        class_auroc = float(roc_auc_score(class_targets, class_probs))
        per_class_auroc[class_name] = class_auroc
        aurocs.append(class_auroc)

    metrics = {
        "mAP": _safe_mean(average_precisions),
        "macro_auroc": _safe_mean(aurocs),
        "macro_f1": macro_f1,
        "threshold": threshold,
        "per_class_auroc": per_class_auroc,
    }
    return metrics
