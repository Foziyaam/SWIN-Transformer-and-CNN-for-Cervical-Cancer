# SWIN-Transformer-and-CNN-for-Cervical-Cancer

# Hybrid Model Pipeline Documentation

This document provides detailed documentation for the hybrid ConvNeXt‑Base + SwinV2‑Base transfer‑learning pipeline designed for moderate‑size colposcopy image classification (normal vs. precancerous). It covers installation, configuration, data preparation, architecture, training, evaluation, and customization.

---

## 1. Overview

- **Goal:** Leverage complementary strengths of a convolutional backbone (ConvNeXt‑Base) and a Transformer backbone (SwinV2‑Base) via feature fusion, to capture both fine‑grained textures and global context in medical images.
- **Dataset Size:** ~720 images total (360 normal, 360 precancerous), with a 70/30 train/test split and 5‑fold CV on training data.
- **Key Features:**
  - Two pretrained backbones from `timm` (ImageNet‑22K → ImageNet‑1K).  
  - Warm‑up freeze of backbone weights followed by full fine‑tuning.  
  - Mixed‑precision training for efficiency.  
  - Stratified split ensures balanced classes in train/test.

---

## 2. Requirements & Installation

**Python 3.8+**

Install required packages:
```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn
```  
Ensure CUDA (11.x) is configured for GPU acceleration.

---

## 3. Configuration Parameters

All hyperparameters and architecture choices live in the `CONFIG` dictionary at the top of `hybrid_model_pipeline.py`:

| Key                         | Description                                                 | Example       |
|-----------------------------|-------------------------------------------------------------|--------------:|
| `seed`                      | Random seed for reproducibility                              | 42            |
| `image_size`                | Input size for cropping/resizing                             | 224           |
| `batch_size`                | Mini‑batch size                                              | 16            |
| `num_workers`               | Number of DataLoader workers                                | 4             |
| `cnn_arch`                  | CNN backbone name in `timm`                                  | `convnext_base` |
| `swin_arch`                 | Swin backbone name in `timm`                                 | `swinv2_base_window8_256` |
| `freeze_backbones_epochs`   | Epochs to keep backbones frozen before full fine‑tuning      | 5             |
| `num_epochs`                | Total training epochs                                       | 30            |
| `learning_rate`             | Initial learning rate for AdamW                             | 3e-4          |
| `weight_decay`              | L2 regularization                                           | 1e-2          |
| `step_size`                 | LR scheduler step interval (in epochs)                      | 8             |
| `gamma`                     | LR decay factor                                             | 0.85          |
| `mixed_precision`           | Enable AMP mixed‑precision                                  | `True`        |
| `num_folds`                 | Number of CV folds on training split                        | 5             |
| `test_size`                 | Fraction of data reserved for test (stratified)             | 0.3           |
| `save_dir`                  | Directory to save checkpoints                               | `./checkpoints` |

---

## 4. Data Preparation

1. **Load Data:** Your images should be a NumPy array of shape `(720, H, W, 3)` and labels an array of length `720` with values `0` (normal) or `1` (precancerous).
2. **Stratified Split:** The code uses `train_test_split(..., stratify=labels)` to allocate 70% of samples (504) for training and 30% (216) for final testing while maintaining class balance.
3. **Transforms:** Two pipelines in `get_data_transforms`:
   - **Train:** RandomResizedCrop, horizontal flip, jitter, rotation, normalization.  
   - **Test:** Resize→CenterCrop, normalization.
4. **TensorDataset:** Images are converted to `torch.float32` and permuted to `(C,H,W)`; labels to `torch.long`.

---

## 5. Model Architecture

```python
class HybridCNN_Swin(nn.Module):
    def __init__(self, cfg, freeze_backbones=False):
        # Load pretrained ConvNeXt-Base and SwinV2-Base without classifier heads
        # Optionally freeze all backbone parameters
        # Build a fusion head: Dropout + Linear(cnn_dim + swin_dim → 2 classes)

    def forward(self, x):
        # Extract CNN features → f1
        # Extract Swin features → f2
        # Concatenate [f1, f2] → Dropout → Final FC → logits
        return logits
```

- **Why these backbones?** ConvNeXt-Base captures detailed local textures; SwinV2-Base offers hierarchical self‑attention to model global context—together they excel at picking subtle, multi‑scale patterns in limited data.

---

## 6. Training & Cross‑Validation

1. **5‑Fold CV:** On the 504 training samples, a `KFold(n_splits=5, shuffle=True, random_state=seed)` splits data into train/validation subsets.  
2. **Warm‑up Epochs:** Backbones are frozen for the first `freeze_backbones_epochs` epochs so that only the new fusion head learns; after that, all weights are unfrozen for full fine‑tuning.  
3. **Optimizer & Scheduler:** Uses `AdamW` with the configured lr and weight_decay; `StepLR` decays LR by `gamma` every `step_size` epochs.  
4. **Mixed Precision:** `torch.cuda.amp` accelerates training and reduces memory footprint.

**Training Loop Pseudocode:**
```text
for fold in CV:
  init model (freeze backbones)
  for epoch in range(num_epochs):
    if epoch == freeze_backbones_epochs: unfreeze backbones
    train one epoch → compute loss
    step scheduler
  evaluate on fixed test set → record metrics + save checkpoint
```

---

## 7. Evaluation & Metrics

- **Metrics:** Accuracy, F1-score, ROC AUC via `sklearn.metrics`.  
- **Fixed Test Set:** After each CV fold’s training, the model is evaluated on the same 216 test samples for consistency.  
- **Plots:** For each fold, the pipeline generates:
  - ROC curve with AUC label.  
  - Confusion matrix heatmap.

### Final Reporting
- **Average Test Accuracy:** Printed at the end as the mean across CV folds.

---

## 8. Usage Example

```bash
python hybrid_model_pipeline.py
```  
All parameters can be tweaked in the `CONFIG` dict. Ensure your dataset-loading code replaces the `# TODO` placeholder.

---

## 9. Customization & Extensions

- **Domain Pretraining:** Pretrain backbones with self‑supervised methods (e.g., DINOv2) on unlabeled colposcopy images before fine‑tuning.  
- **Backbone Variants:** Swap `cnn_arch`/`swin_arch` to lighter/heavier models based on resource constraints.  
- **Augmentation Strategies:** Experiment with MixUp, RandAugment, or CutMix for further regularization.

---

**Contact & Support**  
For questions or contributions, please raise an issue or pull request in the repository where this pipeline is hosted.

