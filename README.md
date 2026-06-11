# AGLQ Chest X-ray

Project skeleton for training an AGLQ-style multi-label classifier on ChestX-ray14.

## Structure

- `configs/`: experiment configuration files
- `baselines/`: baseline model definitions
- `aglq/`: package code for data, model, losses, metrics, and training
- `scripts/`: runnable entry points

## Quick Start

```bash
pip install -r requirements.txt
bash run.sh
```

Expected data layout:

```text
data/ChestXray14/
  images/
  train.csv
  val.csv
  test.csv
```

CSV files should include an `image` column and one binary column for each label listed in `configs/chestxray14_aglq.yaml`.

## Experiments

Train the DenseNet121 baseline:

```bash
python scripts/train_aglq.py --config configs/chestxray14_aglq.yaml --mode train --model densenet121
```

Train the first AG-LQ experiment, DenseNet121 plus adaptive label queries:

```bash
python scripts/train_aglq.py --config configs/chestxray14_aglq.yaml --mode train --model densenet121_aglq
```

Evaluate a saved AG-LQ checkpoint:

```bash
python scripts/train_aglq.py --config configs/chestxray14_aglq.yaml --mode test --model densenet121_aglq
```

The AG-LQ checkpoint is saved as `outputs/checkpoints/best_densenet121_aglq.pth`, and test results are saved under `outputs/results/`.
