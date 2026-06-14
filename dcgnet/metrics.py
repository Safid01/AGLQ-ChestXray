import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


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


def tune_per_class_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    default_threshold: float,
    search_min: float,
    search_max: float,
    search_step: float,
) -> np.ndarray:
    """Find one F1 threshold per disease label on validation predictions."""
    thresholds = np.full(probabilities.shape[1], default_threshold, dtype=np.float32)

    for class_index in range(probabilities.shape[1]):
        class_targets = targets[:, class_index]
        class_probs = probabilities[:, class_index]

        if len(np.unique(class_targets)) < 2:
            continue

        best_threshold = default_threshold
        best_f1 = -1.0

        for threshold in _build_thresholds(search_min, search_max, search_step):
            class_predictions = (class_probs >= threshold).astype(int)
            score = float(f1_score(class_targets, class_predictions, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        thresholds[class_index] = best_threshold

    return thresholds


def _threshold_dict(thresholds: np.ndarray, class_names: list[str]) -> dict:
    return {
        class_name: float(thresholds[class_index])
        for class_index, class_name in enumerate(class_names)
    }


def _load_per_class_thresholds(
    threshold_config: dict,
    class_names: list[str],
    default_threshold: float,
) -> np.ndarray | None:
    saved_thresholds = threshold_config.get("per_class_threshold")
    if saved_thresholds is None:
        saved_thresholds = threshold_config.get("per_class_thresholds")
    if saved_thresholds is None:
        return None

    if isinstance(saved_thresholds, dict):
        return np.array(
            [float(saved_thresholds.get(class_name, default_threshold)) for class_name in class_names],
            dtype=np.float32,
        )

    return np.array(saved_thresholds, dtype=np.float32)


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
    use_per_class_threshold = threshold_config.get("per_class", False)
    per_class_threshold = None

    if tune_threshold and use_per_class_threshold:
        threshold_values = tune_per_class_thresholds(
            probabilities=probabilities,
            targets=targets,
            default_threshold=default_threshold,
            search_min=threshold_config.get("search_min", 0.1),
            search_max=threshold_config.get("search_max", 0.9),
            search_step=threshold_config.get("search_step", 0.05),
        )
        predictions = (probabilities >= threshold_values.reshape(1, -1)).astype(int)
        threshold = float(np.mean(threshold_values))
        per_class_threshold = _threshold_dict(threshold_values, class_names)
    elif use_per_class_threshold:
        threshold_values = _load_per_class_thresholds(threshold_config, class_names, default_threshold)
        if threshold_values is None:
            threshold_values = np.full(len(class_names), default_threshold, dtype=np.float32)
        predictions = (probabilities >= threshold_values.reshape(1, -1)).astype(int)
        threshold = float(np.mean(threshold_values))
        per_class_threshold = _threshold_dict(threshold_values, class_names)
    elif tune_threshold:
        threshold, macro_f1 = tune_threshold_for_macro_f1(
            probabilities=probabilities,
            targets=targets,
            search_min=threshold_config.get("search_min", 0.1),
            search_max=threshold_config.get("search_max", 0.9),
            search_step=threshold_config.get("search_step", 0.05),
        )
        predictions = (probabilities >= threshold).astype(int)
    else:
        threshold = default_threshold
        predictions = (probabilities >= threshold).astype(int)

    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))

    average_precisions = []
    per_class_auroc = {}
    per_class_ap = {}
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
    aurocs = []

    for class_index, class_name in enumerate(class_names):
        class_targets = targets[:, class_index]
        class_probs = probabilities[:, class_index]
        class_predictions = predictions[:, class_index]

        per_class_f1[class_name] = float(f1_score(class_targets, class_predictions, zero_division=0))
        per_class_precision[class_name] = float(precision_score(class_targets, class_predictions, zero_division=0))
        per_class_recall[class_name] = float(recall_score(class_targets, class_predictions, zero_division=0))

        if len(np.unique(class_targets)) < 2:
            average_precisions.append(float("nan"))
            per_class_auroc[class_name] = float("nan")
            per_class_ap[class_name] = float("nan")
            continue

        class_ap = float(average_precision_score(class_targets, class_probs))
        average_precisions.append(class_ap)
        per_class_ap[class_name] = class_ap

        class_auroc = float(roc_auc_score(class_targets, class_probs))
        per_class_auroc[class_name] = class_auroc
        aurocs.append(class_auroc)

    if len(np.unique(targets.ravel())) < 2:
        micro_auroc = float("nan")
    else:
        micro_auroc = float(roc_auc_score(targets.ravel(), probabilities.ravel()))

    metrics = {
        "mAP": _safe_mean(average_precisions),
        "macro_auroc": _safe_mean(aurocs),
        "micro_auroc": micro_auroc,
        "macro_f1": macro_f1,
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(targets, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(targets, predictions, average="micro", zero_division=0)),
        "threshold": threshold,
        "per_class_auroc": per_class_auroc,
        "per_class_ap": per_class_ap,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }
    if per_class_threshold is not None:
        metrics["per_class_threshold"] = per_class_threshold
    return metrics
