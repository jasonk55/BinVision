# BinVision

Binary waste image classifier that distinguishes **Organic** from **Recyclable** items. Trains and benchmarks five models — from logistic regression to EfficientNet-B0 — then evaluates cross-dataset generalization on TrashNet.

## Models

### Baselines (sklearn)

Raw pixels flattened to a 49,152-dimensional feature vector (128×128×3).

- **Logistic Regression** — SAGA solver, C=1.0, max_iter=500
- **Random Forest** — 100 trees, max_depth=20, balanced class weights
- **SVM** — StandardScaler → PCA(300) → CalibratedClassifierCV(LinearSVC)

### Custom CNN

Four convolutional blocks (32→64→128→256 channels) with BatchNorm, ReLU, MaxPool, and Dropout. 616K parameters. Trained 20 epochs with AdamW + CosineAnnealingLR and class-weighted CrossEntropyLoss.

### EfficientNet-B0 (Transfer Learning)

Two-phase fine-tuning on ImageNet pretrained weights:

1. **Head only** (10 epochs, lr=1e-3) — backbone frozen
2. **Partial unfreeze** (10 epochs, lr=1e-4) — last 3 feature blocks + classifier unfrozen

## Cross-Dataset Inference (TrashNet)

Models trained on the primary dataset are evaluated on [TrashNet](https://github.com/garythung/trashnet) to test generalization. Label mapping: `trash → Non-recyclable (0)`, all other classes → `Recyclable (1)`.

## Project Structure

```
BinVision/
├── BinVision.ipynb               # EDA, training, and evaluation
├── inference_trashnet.ipynb      # Cross-dataset inference on TrashNet
├── DATASET/                      # Primary dataset (TRAIN/TEST)
├── saved/                        # Model checkpoints and artifacts
│   ├── cnn_best.pt
│   ├── efficientnet_best.pt
│   ├── lr_model.joblib
│   ├── rf_model.joblib
│   ├── svm_pipe.joblib
│   ├── flat_features.npz
│   └── preprocessing_config.json
└── *.png                         # Training curves, confusion matrices, EDA plots
```

## Setup

```bash
pip install torch torchvision scikit-learn numpy pandas matplotlib seaborn Pillow joblib
```

## Usage

1. Download the primary dataset from [Kaggle — Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data) and place it under `DATASET/`
2. Run `BinVision.ipynb` end-to-end to train all models and save artifacts to `saved/`.
3. To evaluate on TrashNet, download the resized dataset and run `inference_trashnet.ipynb`.
