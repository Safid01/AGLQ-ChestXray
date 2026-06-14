import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from baselines.densenet121 import build_densenet121
from baselines.efficientnet_b3 import build_efficientnet_b3
from baselines.query2label import build_query2label
from baselines.resnet50 import build_resnet50
from dcgnet.config import load_config
from dcgnet.data import get_dataloaders
from dcgnet.losses import get_bce_loss, get_loss
from dcgnet.metrics import compute_metrics
from dcgnet.model import build_densenet121_dcgnet
from dcgnet.trainer import Trainer


MODEL_BUILDERS = {
    "densenet121": build_densenet121,
    "densenet121_dcgnet": build_densenet121_dcgnet,
    "densenet121_dcg_net": build_densenet121_dcgnet,
    "resnet50": build_resnet50,
    "efficientnet_b3": build_efficientnet_b3,
    "efficientnet-b3": build_efficientnet_b3,
    "query2label": build_query2label,
    "query_2_label": build_query2label,
}


MODEL_FILE_NAMES = {
    "densenet121": "densenet121",
    "densenet121_dcgnet": "densenet121_dcgnet",
    "densenet121_dcg_net": "densenet121_dcgnet",
    "resnet50": "resnet50",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet-b3": "efficientnet_b3",
    "query2label": "query2label",
    "query_2_label": "query2label",
}


def get_model_key(config: dict) -> str:
    model_cfg = config["model"]
    model_name = model_cfg.get("name") or model_cfg.get("backbone", "densenet121")
    model_key = str(model_name).lower().replace(" ", "_").replace("-", "_")
    if model_key not in MODEL_BUILDERS:
        supported = ", ".join(sorted(MODEL_FILE_NAMES.values()))
        raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {supported}")
    return model_key


def get_model_file_stem(config: dict) -> str:
    return MODEL_FILE_NAMES[get_model_key(config)]


def get_checkpoint_name(config: dict) -> str:
    return f"best_{get_model_file_stem(config)}.pth"


def get_results_name(config: dict) -> str:
    return f"{get_model_file_stem(config)}_test_results.csv"


def is_dcgnet_model(config: dict) -> bool:
    return get_model_key(config) in {"densenet121_dcgnet", "densenet121_dcg_net"}


def build_model(config: dict):
    model_key = get_model_key(config)
    model_cfg = config["model"]

    if is_dcgnet_model(config):
        dcg_cfg = config.get("dcg", {})
        adjacency_source = PROJECT_ROOT / dcg_cfg.get("adjacency_source", "datasets/ChestXray14/train.csv")
        return MODEL_BUILDERS[model_key](
            num_classes=config["dataset"]["num_classes"],
            pretrained=model_cfg.get("pretrained", True),
            graph_dim=dcg_cfg.get("graph_dim", 256),
            num_graph_layers=dcg_cfg.get("num_graph_layers", 2),
            dropout=dcg_cfg.get("dropout", model_cfg.get("dropout", 0.2)),
            beta=dcg_cfg.get("beta", 0.1),
            adjacency_source=adjacency_source,
            class_names=config["classes"],
        )

    if model_key in {"query2label", "query_2_label"}:
        return MODEL_BUILDERS[model_key](
            num_classes=config["dataset"]["num_classes"],
            pretrained=model_cfg.get("pretrained", True),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_heads=model_cfg.get("num_heads", 4),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
        )

    return MODEL_BUILDERS[model_key](
        num_classes=config["dataset"]["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
    )


def initialize_dcgnet_from_densenet(model, checkpoint_path: str | Path) -> None:
    checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        print(f"DenseNet121 checkpoint not found, skipping initialization: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    densenet_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    copied_keys = 0

    for key, value in densenet_state.items():
        target_key = None

        if key.startswith("model.features."):
            target_key = key.replace("model.features.", "backbone.")
        elif key == "model.classifier.1.weight":
            target_key = "global_classifier.weight"
        elif key == "model.classifier.1.bias":
            target_key = "global_classifier.bias"

        if target_key in model_state and model_state[target_key].shape == value.shape:
            model_state[target_key] = value
            copied_keys += 1

    model.load_state_dict(model_state)
    print(f"Initialized DCG-Net from DenseNet121 checkpoint: {checkpoint_path}")
    print(f"Copied {copied_keys} DenseNet121 parameter tensors")


def get_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(
    train_dataset,
    class_names: list[str],
    device: torch.device,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    labels = torch.tensor(train_dataset.data[class_names].values, dtype=torch.float32)
    positives = labels.sum(dim=0)
    negatives = labels.size(0) - positives
    pos_weight = negatives / torch.clamp(positives, min=1.0)
    pos_weight = torch.clamp(pos_weight, max=max_pos_weight)
    return pos_weight.to(device)


def build_optimizer(config: dict, model):
    training_cfg = config["training"]
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate DCG-Net and baselines on ChestXray14.")
    parser.add_argument("--config", default="configs/chestxray14_dcgnet.yaml", help="Path to YAML config.")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Run training or test evaluation.")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override, e.g. DenseNet121_DCGNet, DenseNet121, Query2Label.",
    )
    parser.add_argument("--loss", default=None, help="Optional loss override: bce, balanced_bce, or asl.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--no-densenet-init",
        action="store_true",
        help="Disable DenseNet121 checkpoint initialization for DCG-Net.",
    )
    return parser.parse_args()


def train_model(config: dict, device: torch.device, skip_densenet_init: bool = False) -> None:
    train_loader, val_loader, _ = get_dataloaders(config)
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Train and validation CSV files must contain at least one row.")

    model_stem = get_model_file_stem(config)
    print(f"Training {model_stem}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = build_model(config)
    init_cfg = config.get("initialization", {})
    should_initialize = (
        is_dcgnet_model(config)
        and init_cfg.get("from_densenet", False)
        and init_cfg.get("densenet_checkpoint")
        and not skip_densenet_init
    )
    if should_initialize:
        initialize_dcgnet_from_densenet(model, init_cfg["densenet_checkpoint"])

    pos_weight = None
    if config["training"].get("loss", "bce").lower() in {"balanced_bce", "weighted_bce", "pos_weight_bce"}:
        pos_weight = compute_pos_weight(
            train_loader.dataset,
            config["classes"],
            device,
            max_pos_weight=config.get("loss", {}).get("max_pos_weight", 20.0),
        )
        print("Using class-balanced BCE with pos_weight from train.csv")

    criterion = get_loss(config, pos_weight=pos_weight)
    optimizer = build_optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        class_names=config["classes"],
        checkpoint_dir=PROJECT_ROOT / config["checkpoint"]["save_dir"],
        checkpoint_name=get_checkpoint_name(config),
        print_frequency=config.get("logging", {}).get("print_frequency", 20),
        early_stopping_patience=(
            config["early_stopping"].get("patience", 8)
            if config.get("early_stopping", {}).get("enabled", False)
            else None
        ),
        threshold_config=config.get("threshold", {}),
        graph_auxiliary_weight=config.get("dcg", {}).get("graph_aux_weight", 0.0),
    )
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=config["training"]["epochs"])


@torch.no_grad()
def evaluate_test(config: dict, device: torch.device) -> None:
    _, _, test_loader = get_dataloaders(config)
    if len(test_loader.dataset) == 0:
        raise ValueError("test.csv must contain at least one row.")

    checkpoint_path = PROJECT_ROOT / config["checkpoint"]["save_dir"] / get_checkpoint_name(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Evaluating {get_model_file_stem(config)}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    criterion = get_bce_loss()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        model_output = model(images)
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    checkpoint_metrics = checkpoint.get("metrics", {})
    checkpoint_threshold = checkpoint_metrics.get("threshold", config.get("threshold", {}).get("default", 0.5))
    checkpoint_per_class_threshold = checkpoint_metrics.get("per_class_threshold")
    threshold_config = {
        "tune_on_validation": False,
        "default": checkpoint_threshold,
        "per_class": checkpoint_per_class_threshold is not None,
        "per_class_threshold": checkpoint_per_class_threshold,
    }

    metrics = compute_metrics(all_logits, all_labels, config["classes"], threshold_config)
    metrics["loss"] = total_loss / len(test_loader.dataset)

    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test mAP: {metrics['mAP']:.4f}")
    print(f"Test macro AUROC: {metrics['macro_auroc']:.4f}")
    print(f"Test micro AUROC: {metrics['micro_auroc']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(f"Test micro F1: {metrics['micro_f1']:.4f}")

    rows = [
        {"metric": "test_loss", "class": "all", "value": metrics["loss"]},
        {"metric": "test_mAP", "class": "all", "value": metrics["mAP"]},
        {"metric": "test_macro_auroc", "class": "all", "value": metrics["macro_auroc"]},
        {"metric": "test_micro_auroc", "class": "all", "value": metrics["micro_auroc"]},
        {"metric": "test_macro_f1", "class": "all", "value": metrics["macro_f1"]},
        {"metric": "test_micro_f1", "class": "all", "value": metrics["micro_f1"]},
        {"metric": "test_macro_precision", "class": "all", "value": metrics["macro_precision"]},
        {"metric": "test_macro_recall", "class": "all", "value": metrics["macro_recall"]},
        {"metric": "test_micro_precision", "class": "all", "value": metrics["micro_precision"]},
        {"metric": "test_micro_recall", "class": "all", "value": metrics["micro_recall"]},
        {"metric": "f1_threshold", "class": "all", "value": metrics["threshold"]},
    ]
    for metric_name in [
        "per_class_auroc",
        "per_class_ap",
        "per_class_f1",
        "per_class_precision",
        "per_class_recall",
        "per_class_threshold",
    ]:
        if metric_name not in metrics:
            continue
        for class_name, value in metrics[metric_name].items():
            rows.append({"metric": metric_name, "class": class_name, "value": value})

    results_dir = PROJECT_ROOT / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / get_results_name(config)
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"Saved test results to: {results_path}")


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)

    if args.model is not None:
        config["model"]["name"] = args.model
        config["model"]["backbone"] = args.model
    if args.loss is not None:
        config["training"]["loss"] = args.loss
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    set_seed(config["training"].get("seed", 42))
    device = get_device()

    if args.mode == "train":
        train_model(config, device, skip_densenet_init=args.no_densenet_init)
    else:
        evaluate_test(config, device)


if __name__ == "__main__":
    main()
