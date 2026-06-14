# DCGNet Chest X-ray

Implementation for:

**Disease Co-occurrence Guided Network for Multi-Label Chest X-ray Diagnosis**

The project trains DenseNet121_DCGNet and baseline models on NIH ChestXray14.

## Structure

- `configs/`: experiment configuration files
- `baselines/`: baseline model definitions
- `dcgnet/`: data, model, losses, metrics, and training code
- `scripts/`: runnable entry points
- `datasets/ChestXray14/`: ChestXray14 images and split CSV files
- `outputs/`: checkpoints and result CSV files

## Dataset Format

CSV files must contain:

- `image_path`
- one binary column for each disease label

The 14 labels are listed in `configs/chestxray14_dcgnet.yaml`.

## Train DCG-Net

```bash
python scripts/train_dcgnet.py --config configs/chestxray14_dcgnet.yaml --model DenseNet121_DCGNet
```

## Test DCG-Net

```bash
python scripts/train_dcgnet.py --config configs/chestxray14_dcgnet.yaml --mode test --model DenseNet121_DCGNet
```

Expected outputs:

- `outputs/checkpoints/best_densenet121_dcgnet.pth`
- `outputs/results/densenet121_dcgnet_test_results.csv`

## Baselines

```bash
python scripts/train_dcgnet.py --config configs/chestxray14_dcgnet.yaml --model DenseNet121
python scripts/train_dcgnet.py --config configs/chestxray14_dcgnet.yaml --model ResNet50
python scripts/train_dcgnet.py --config configs/chestxray14_dcgnet.yaml --model EfficientNet_B3
python scripts/train_dcgnet.py --config configs/chestxray14_dcgnet.yaml --model Query2Label
```
