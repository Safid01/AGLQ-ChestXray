import argparse
import random
from pathlib import Path

import pandas as pd


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ChestX-ray14 train/val/test CSV splits.")
    parser.add_argument(
        "--metadata",
        default="datasets/ChestXray14/Data_Entry_2017_v2020.csv",
        help="Path to NIH ChestX-ray14 metadata CSV from project root.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/ChestXray14",
        help="Directory where train.csv, val.csv, and test.csv will be saved.",
    )
    parser.add_argument(
        "--image-dir",
        default="datasets/ChestXray14/images",
        help="Directory containing ChestX-ray14 PNG images from project root.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_patient_ids(patient_ids: list[int], seed: int) -> tuple[set[int], set[int], set[int]]:
    """Create 70/10/20 patient-level splits to avoid patient leakage."""
    patient_ids = patient_ids.copy()
    random.Random(seed).shuffle(patient_ids)

    total = len(patient_ids)
    train_end = int(total * 0.70)
    val_end = int(total * 0.80)

    train_ids = set(patient_ids[:train_end])
    val_ids = set(patient_ids[train_end:val_end])
    test_ids = set(patient_ids[val_end:])

    return train_ids, val_ids, test_ids


def add_label_columns(data: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame()
    output["image_path"] = data["Image Index"]

    findings = data["Finding Labels"].fillna("").str.split("|")
    for label in LABELS:
        raw_label = label.replace("_", " ")
        output[label] = findings.apply(lambda row_labels: int(raw_label in row_labels))

    return output


def save_split(name: str, data: pd.DataFrame, patient_ids: set[int], output_dir: Path) -> None:
    split_data = data[data["Patient ID"].isin(patient_ids)]
    split_data = add_label_columns(split_data)
    output_path = output_dir / f"{name}.csv"
    split_data.to_csv(output_path, index=False)
    print(f"{name}: {len(split_data)} images, {len(patient_ids)} patients -> {output_path}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    metadata_path = project_root / args.metadata
    output_dir = project_root / args.output_dir
    image_dir = project_root / args.image_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(metadata_path)
    image_names = {path.name for path in image_dir.glob("*.png")}
    before_count = len(data)
    data = data[data["Image Index"].isin(image_names)].copy()
    dropped_count = before_count - len(data)
    print(f"Found {len(image_names)} PNG images in {image_dir}")
    print(f"Using {len(data)} metadata rows; skipped {dropped_count} rows without local images.")

    if len(data) == 0:
        raise ValueError("No metadata rows matched local images. Check --image-dir.")

    patient_ids = data["Patient ID"].drop_duplicates().tolist()
    train_ids, val_ids, test_ids = split_patient_ids(patient_ids, args.seed)

    save_split("train", data, train_ids, output_dir)
    save_split("val", data, val_ids, output_dir)
    save_split("test", data, test_ids, output_dir)


if __name__ == "__main__":
    main()
