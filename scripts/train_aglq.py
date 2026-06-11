import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from aglq.config import load_config
from aglq.data import get_dataloaders
from aglq.losses import get_bce_loss
from aglq.model import build_densenet121_aglq
from aglq.trainer import Trainer
from baselines.densenet121 import build_densenet121
from baselines.efficientnet_b3 import build_efficientnet_b3
from baselines.query2label import build_query2label
from baselines.resnet50 import build_resnet50


BASELINE_BUILDERS = {
    "densenet121": build_densenet121,
    "densenet121_aglq": build_densenet121_aglq,
    "densenet121_ag_lq": build_densenet121_aglq,
    "resnet50": build_resnet50,
    "efficientnet_b3": build_efficientnet_b3,
    "efficientnet-b3": build_efficientnet_b3,
    "query2label": build_query2label,
    "query_2_label": build_query2label,
}


MODEL_FILE_NAMES = {
    "densenet121": "densenet121",
    "densenet121_aglq": "densenet121_aglq",
    "densenet121_ag_lq": "densenet121_aglq",
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
    if model_key not in BASELINE_BUILDERS:
        supported = ", ".join(sorted(MODEL_FILE_NAMES.values()))
        raise ValueError(f"Unsupported baseline model '{model_name}'. Choose one of: {supported}")
    return model_key


def get_model_file_stem(config: dict) -> str:
    return MODEL_FILE_NAMES[get_model_key(config)]


def build_model(config: dict):
    model_key = get_model_key(config)
    model_cfg = config["model"]

    if model_key in {"densenet121_aglq", "densenet121_ag_lq"}:
        aglq_cfg = config.get("aglq", {})
        return BASELINE_BUILDERS[model_key](
            num_classes=config["dataset"]["num_classes"],
            pretrained=model_cfg.get("pretrained", True),
            query_dim=aglq_cfg.get("query_dim", model_cfg.get("hidden_dim", 256)),
            num_heads=aglq_cfg.get("num_attention_heads", model_cfg.get("num_heads", 4)),
            num_layers=aglq_cfg.get("num_layers", 1),
            dropout=aglq_cfg.get("dropout", 0.1),
        )

    if model_key in {"query2label", "query_2_label"}:
        return BASELINE_BUILDERS[model_key](
            num_classes=config["dataset"]["num_classes"],
            pretrained=model_cfg.get("pretrained", True),
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_heads=model_cfg.get("num_heads", 4),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
        )

    return BASELINE_BUILDERS[model_key](
        num_classes=config["dataset"]["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
    )


def get_checkpoint_name(config: dict) -> str:
    return f"best_{get_model_file_stem(config)}.pth"


def get_results_name(config: dict) -> str:
    return f"{get_model_file_stem(config)}_test_results.csv"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate baseline models on ChestX-ray14.")
    parser.add_argument("--config", default="configs/chestxray14_aglq.yaml", help="Path to YAML config from project root.")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Run training or test evaluation.")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override, e.g. densenet121, resnet50, efficientnet_b3, query2label.",
    )
    return parser.parse_args()


def train_baseline(config: dict, device: torch.device) -> None:
    train_loader, val_loader, _ = get_dataloaders(config)
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Train and validation CSV files must contain at least one row.")

    model_stem = get_model_file_stem(config)
    print(f"Training {model_stem} experiment")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = build_model(config)
    criterion = get_bce_loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        class_names=config["classes"],
        checkpoint_dir=PROJECT_ROOT / config["checkpoint"]["save_dir"],
        checkpoint_name=get_checkpoint_name(config),
        print_frequency=config["logging"].get("print_frequency", 20),
        early_stopping_patience=(
            config["early_stopping"].get("patience", 3)
            if config.get("early_stopping", {}).get("enabled", False)
            else None
        ),
        threshold_config=config.get("threshold", {}),
    )
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
    )


@torch.no_grad()
def evaluate_test(config: dict, device: torch.device) -> None:
    _, _, test_loader = get_dataloaders(config)
    if len(test_loader.dataset) == 0:
        raise ValueError("test.csv must contain at least one row.")

    checkpoint_path = PROJECT_ROOT / config["checkpoint"]["save_dir"] / get_checkpoint_name(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Evaluating {get_model_file_stem(config)} checkpoint")
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

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    checkpoint_threshold = checkpoint.get("metrics", {}).get("threshold", config.get("threshold", {}).get("default", 0.5))
    threshold_config = {
        "tune_on_validation": False,
        "default": checkpoint_threshold,
    }
    from aglq.metrics import compute_metrics

    metrics = compute_metrics(all_logits, all_labels, config["classes"], threshold_config)
    metrics["loss"] = total_loss / len(test_loader.dataset)

    print(f"Test loss: {metrics['loss']:.4f}")
    print(f"Test mAP: {metrics['mAP']:.4f}")
    print(f"Test macro AUROC: {metrics['macro_auroc']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(f"F1 threshold: {metrics['threshold']:.2f}")
    print("Per-class AUROC:")
    for class_name, auroc in metrics["per_class_auroc"].items():
        print(f"  {class_name}: {auroc:.4f}")

    results_dir = PROJECT_ROOT / config["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / get_results_name(config)

    rows = [
        {"metric": "test_loss", "class": "all", "value": metrics["loss"]},
        {"metric": "test_mAP", "class": "all", "value": metrics["mAP"]},
        {"metric": "test_macro_auroc", "class": "all", "value": metrics["macro_auroc"]},
        {"metric": "test_macro_f1", "class": "all", "value": metrics["macro_f1"]},
        {"metric": "f1_threshold", "class": "all", "value": metrics["threshold"]},
    ]
    for class_name, auroc in metrics["per_class_auroc"].items():
        rows.append({"metric": "per_class_auroc", "class": class_name, "value": auroc})

    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"Saved test results to: {results_path}")


def main() -> None:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    if args.model is not None:
        config["model"]["name"] = args.model
        config["model"]["backbone"] = args.model

    set_seed(config["training"].get("seed", 42))
    device = get_device()

    if args.mode == "train":
        train_baseline(config, device)
    else:
        evaluate_test(config, device)


if __name__ == "__main__":
    main()
