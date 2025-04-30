# Hybrid ConvNeXt-Base + SwinV2-Base Pipeline Documentation

This document provides **comprehensive**, **detailed**, and **clear** documentation for the Hybrid ConvNeXt-Base + SwinV2-Base pipeline—**`SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py`**—including configuration, installation, usage, module breakdowns, and best practices.

Use the **`SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py`** file which is extensively commented with “what” and “why” comments on every line:
This script is ready-to-run, thoroughly commented, and aligned with industry standards (and this documentation is for **`SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py`**.

**Highlights of “what & why” comments:**
- **Shebang & version check** to enforce Python 3.11+
- **Imports** annotated by purpose (config, logging, data, modeling, explainability)
- **Default_CONFIG** keys explained
- **CLI/config loading** logic for JSON/YAML
- **`set_seed`** for reproducibility
- **Transforms** for augmentation & normalization
- **Model**: two backbones fused, optional freezing
- **Training**: AMP mixed-precision best practice
- **Evaluation**: metrics and raw outputs
- **Explainability**: Captum IG & Grad-CAM usage
- **Plotting**: quick ROC & confusion matrix
- **Main**: data splitting, CV loop, logging, checkpoint directory management


---

## Table of Contents
1. [Overview](#overview)
2. [Installation & Requirements](#installation--requirements)
3. [Configuration](#configuration)
4. [Command-Line Interface (CLI)](#command-line-interface-cli)
5. [Pipeline Modules](#pipeline-modules)
   - [1. `set_seed`](#1-set_seed)
   - [2. `get_transforms`](#2-get_transforms)
   - [3. `HybridModel` class](#3-hybridmodel-class)
   - [4. `train_one_epoch`](#4-train_one_epoch)
   - [5. `evaluate`](#5-evaluate)
   - [6. `explain_image`](#6-explain_image)
   - [7. `plot_metrics`](#7-plot_metrics)
6. [Data Loading & Splitting](#data-loading--splitting)
7. [Training & Cross-Validation Loop](#training--cross-validation-loop)
8. [Explainability & Visualization](#explainability--visualization)
9. [Extending & Customizing](#extending--customizing)
10. [Logging & Reproducibility](#logging--reproducibility)
11. [Usage Example](#usage-example)
12. [FAQ](#faq)

---

## Overview

This pipeline implements a hybrid transfer-learning architecture combining:
- **ConvNeXt-Base** (pretrained on ImageNet-22K) for rich local feature extraction
- **SwinV2-Base** (pretrained on ImageNet-22K→1K) for hierarchical self-attention

Features are fused by concatenating pooled representations from both backbones and passing through a dropout+linear head. It supports:
- Stratified 70/30 train/test split
- 5-fold cross-validation on training data
- Mixed-precision training with initial backbone freezing
- Captum-based explainability (Integrated Gradients & Grad-CAM)
- JSON/YAML config overrides, structured logging, and shebang/runtime checks for Python 3.11+

---

## Installation & Requirements

**Python 3.11+**

Install dependencies:
```bash
pip install torch torchvision timm captum scikit-learn matplotlib seaborn pyyaml
```

Ensure you have a CUDA‑enabled GPU and matching CUDA toolkit for best performance.

**Code Requirements:**
- Add shebang at the top of `SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py`:
  ```python
  #!/usr/bin/env python3.11
  ```
- Enforce Python version at runtime by adding at script start:
  ```python
  import sys
  if sys.version_info < (3, 11):
      sys.exit("ERROR: Python 3.11 or higher is required.")
  ```

---

## Configuration

All settings live in the `DEFAULT_CONFIG` dict and can be overridden via a JSON or YAML file passed with `--config`.

| Key               | Description                                     | Default                            |
|-------------------|-------------------------------------------------|------------------------------------|
| `seed`            | Random seed                                     | 42                                 |
| `image_size`      | Input H×W                                       | 224                                |
| `batch_size`      | Samples per batch                               | 16                                 |
| `num_workers`     | DataLoader workers                              | 4                                  |
| `cnn_arch`        | ConvNeXt backbone name                          | `convnext_base`                    |
| `swin_arch`       | Swin Transformer backbone name                  | `swinv2_base_window8_256`          |
| `num_classes`     | Output classes                                  | 2                                  |
| `freeze_epochs`   | Epochs to freeze pretrained weights             | 5                                  |
| `num_epochs`      | Total training epochs                           | 30                                 |
| `learning_rate`   | Initial LR                                      | 3e-4                               |
| `weight_decay`    | L2 regularization                                | 1e-2                               |
| `step_size`       | LR scheduler step interval (epochs)             | 8                                  |
| `gamma`           | LR decay factor                                 | 0.85                               |
| `mixed_precision` | Enable AMP mixed-precision                      | true                               |
| `num_folds`       | K-fold CV folds                                 | 5                                  |
| `test_size`       | Fraction for final test split                   | 0.3                                |
| `save_dir`        | Checkpoint save directory                       | `./checkpoints`                    |

**Example `config.yaml`:**
```yaml
batch_size: 32
learning_rate: 1e-3
test_size: 0.25
mixed_precision: false
```  
Run:
```bash
python SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py --config config.yaml
```

---

## Command-Line Interface (CLI)

```bash
usage: SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py [--config CONFIG]

options:
  --config CONFIG   Path to JSON/YAML config file
```

---

## Pipeline Modules

### 1. `set_seed(seed: int) -> None`
**What:** Sets seeds for Python, NumPy, and PyTorch (CPU & GPU).
**Why:** Guarantees reproducible data splits, augmentations, and training.

### 2. `get_transforms(img_size: int) -> Dict[str, Compose]`
**What:** Returns `train` and `test` transform pipelines:
- **Train:** random crop, flip, color jitter, rotate, normalize
- **Test:** resize→center crop, normalize
**Why:** Augmentation for generalization; normalization matches pretrained stats.

### 3. `HybridModel` class
```python
self.cnn = timm.create_model(cnn_name, pretrained=True, num_classes=0)
self.swin = timm.create_model(swin_name, pretrained=True, num_classes=0)
# optional freeze
dim = cnn.num_features + swin.num_features
torch.nn.Sequential(Dropout, Linear(dim, num_classes))
```
**What:** Loads two backbones without heads, optionally freezes, concatenates features, applies classification head.
**Why:** Fusion of convolutional and attention pathways captures multi-scale subtle features.

### 4. `train_one_epoch(...) -> float`
**What:** Trains one epoch with AMP:
- Zero gradients, forward, compute loss
- Scaled backward, optimizer step, scaler update
- Returns average loss
**Why:** Encapsulates best-practice mixed-precision training workflow.

### 5. `evaluate(...) -> (acc, f1, auc, ys, ps, pr)`
**What:** Runs inference on loader, returns:
- **accuracy**, **F1**, **ROC AUC**
- raw lists: true labels `ys`, preds `ps`, positive-class probs `pr`
**Why:** Clean separation for metric computation and plotting.

### 6. `explain_image(...) -> (ig_map, gc_map)`
**What:** Uses Captum to compute:
- **Integrated Gradients** attributions
- **Grad-CAM** heatmap on last Swin stage
**Why:** Visualize regions influencing the model’s decision on test samples.

### 7. `plot_metrics(...)`
**What:** Plots ROC curve with AUC and confusion matrix heatmap.
**Why:** Quick diagnostics per fold.

---

## Data Loading & Splitting

In `main()`, replace placeholder with your loader:
```python
# images: np.ndarray (N, H, W, 3); labels: np.ndarray (N,)
# e.g., images = np.load('images.npy')
```  
Then pipeline does:
```python
X_train, X_test, y_train, y_test = train_test_split(..., stratify=labels)
```  
Wrap as `TensorDataset(torch.tensor(...))` and permute channels.

---

## Training & Cross-Validation Loop

1. **Stratified split**: 70% train (504 samples), 30% test (216 samples).
2. **K-Fold**: 5 folds on training set.
3. **Per fold**:
   - DataLoaders for train, validation, test.
   - Initialize `HybridModel` (`freeze_backbones=True`).
   - Train for `num_epochs`, unfreeze after `freeze_epochs`.
   - Eval on test, log metrics, plot.
   - Explain one test image with IG & Grad-CAM.
4. **Aggregate** average test accuracy across folds.

---

## Explainability & Visualization

For each fold, displays:
- **Integrated Gradients** blended heatmap
- **Grad-CAM** overlay on original image

Use Captum’s `visualize_image_attr` for clean figures.

---

## Extending & Customizing

- Swap backbones via `cnn_arch`/`swin_arch` config.
- Add new CLI args for hyperparameters.
- Integrate CI with `pytest`, `flake8`, `black`.
- Dockerize with a `Dockerfile` for environment reproducibility.
- Export model via TorchScript or ONNX for serving.

---

## Logging & Reproducibility

- Structured `logging` outputs per fold/epoch.
- Seeds set for full reproducibility.
- Checkpoints saved to `save_dir`.

---

## Usage Example

Run with defaults:
```bash
python SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py
```
With YAML config:
```bash
python SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py --config config.yaml
```

---

## FAQ

**Q1: Can I change to PyTorch Lightning?**  
A: Yes, wrap modules and loops into `pl.LightningModule` and `Trainer`.

**Q2: How to add more attribution methods?**  
A: Extend `explain_image` with Captum’s `DeepLift`, `FeatureAblation`, etc.

**Q3: My dataset has 3 classes—how to adapt?**  
A: Set `num_classes: 3` in config; update label encoding accordingly.

---
## Caveates in using the CNN and SWIN-Transformer hybrid model
For a moderate‐sized medical dataset, a **pure Swin-Transformer** will often **outperform** a naive **hybrid** (Swin + CNN) in raw generalization, unless you take extra steps to counteract the hybrid’s increased capacity:

i. **Model capacity vs. overfitting**  
   - **Hybrid (ConvNeXt + SwinV2)** ,for example (the one used here), has on the order of 175 M parameters. With <1K training images, that high capacity can easily memorize noise and subtle artifacts, hurting test performance unless you apply very aggressive regularization or domain-specific pretraining.  
   - **Pure Swin-Tiny/Base** (≈28–86 M parameters) is large enough to learn rich features but small enough to still generalize on limited data.

ii. **Feature overlap**  
   - Both ConvNeXt and Swin already learn convolutional (local) and self-attention (global) patterns. When you fuse them, you often add **redundant** information rather than complementary signals—so the hybrid’s extra features don’t buy you proportionally more discriminative power.

iii. **Training stability & efficiency**  
   - A single‐backbone Swin is simpler to train (no freezing/unfreezing schedule across two networks) and converges more stably under mixed-precision and moderate batch sizes.  
   - The hybrid requires careful warm-up, larger batch size, longer schedules, and eats 2–3× more GPU memory and time for only marginal gains.

iv. **Empirical benchmarks**  
   - In the literature on small-to-medium medical image tasks (e.g., histopathology, dermoscopy), single Transformer or EfficientNet backbones **routinely match or exceed** composite architectures, unless the dataset is much larger (>2 k images per class) or you have extensive self-supervised pretraining on domain data.

---

### Hence:

- **If you need maximal feature richness** (which is the case here -- to distinguish normal from subtley different precancerous) and are prepared to invest in heavy regularization (dropout, augmentation, weight decay), large-scale self-supervised pretraining, or ensembling, the **hybrid** can edge out a pure Swin by capturing a few extra subtle cues.

- **Otherwise**, for a <1k colposcopy dataset, a **standalone Swin-Transformer** (e.g. SwinV2-Base or even Swin-Tiny) will likely yield **better test performance** with less tuning, faster training, and lower overfitting risk.
---
**Contact & Support**  
For questions or contributions, please raise an issue or pull request in the repository.




