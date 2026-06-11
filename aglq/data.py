from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ChestXray14Dataset(Dataset):
    """Dataset for ChestX-ray14 multi-label classification."""

    def __init__(self, csv_path: str | Path, image_dir: str | Path, labels: list[str], image_size: int):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.labels = labels

        self.data = pd.read_csv(self.csv_path)
        self._check_columns()

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _check_columns(self) -> None:
        required_columns = ["image_path", *self.labels]
        missing_columns = [column for column in required_columns if column not in self.data.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"{self.csv_path} is missing required column(s): {missing}")

    def __len__(self) -> int:
        return len(self.data)

    def _get_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.image_dir / path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[index]
        image_path = self._get_image_path(str(row["image_path"]))

        # Chest X-ray images may be grayscale. Convert to RGB so ImageNet
        # normalization and pretrained backbones receive 3-channel input.
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label_values = row[self.labels].astype("float32").values
        labels = torch.tensor(label_values, dtype=torch.float32)

        return image, labels


def _raw_config(config: Any) -> dict[str, Any]:
    """Support either a plain dict or the project's Config wrapper."""
    return config.raw if hasattr(config, "raw") else config


def get_dataloaders(config: Any) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test dataloaders from the YAML config."""
    cfg = _raw_config(config)
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    labels = cfg["classes"]

    train_dataset = ChestXray14Dataset(
        csv_path=dataset_cfg["train_csv"],
        image_dir=dataset_cfg["image_dir"],
        labels=labels,
        image_size=dataset_cfg["image_size"],
    )
    val_dataset = ChestXray14Dataset(
        csv_path=dataset_cfg["val_csv"],
        image_dir=dataset_cfg["image_dir"],
        labels=labels,
        image_size=dataset_cfg["image_size"],
    )
    test_dataset = ChestXray14Dataset(
        csv_path=dataset_cfg["test_csv"],
        image_dir=dataset_cfg["image_dir"],
        labels=labels,
        image_size=dataset_cfg["image_size"],
    )

    batch_size = training_cfg["batch_size"]
    num_workers = training_cfg.get("num_workers", 0)
    # Keep pinned memory disabled here so the dataloader stays device-neutral.
    pin_memory = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=len(train_dataset) > 0,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
