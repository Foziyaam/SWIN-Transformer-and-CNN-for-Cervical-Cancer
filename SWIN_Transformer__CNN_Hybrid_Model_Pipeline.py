#!/usr/bin/env python3.11
"""
SWIN_Transformer__CNN_Hybrid_Model_Pipeline.py
Hybrid transfer-learning pipeline combining ConvNeXt and SwinV2 backbones.
Requires Python 3.11+.
"""
import sys
if sys.version_info < (3, 11):
    sys.exit("ERROR: Python 3.11 or higher is required.")

import argparse
import os
import random
import json
import yaml
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
import timm
from captum.attr import IntegratedGradients, LayerGradCam, visualization as viz
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict = {
    "seed": 42,
    "image_size": 224,
    "batch_size": 16,
    "num_workers": 4,
    "cnn_arch": "convnext_base",
    "swin_arch": "swinv2_base_window8_256",
    "num_classes": 2,
    "freeze_epochs": 5,
    "num_epochs": 30,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "step_size": 8,
    "gamma": 0.85,
    "mixed_precision": True,
    "num_folds": 5,
    "test_size": 0.3,
    "save_dir": "./checkpoints"
}

# CLI and config
parser = argparse.ArgumentParser(description="Hybrid ConvNeXt + SwinV2 Pipeline")
parser.add_argument("--config", type=str, help="Path to JSON/YAML config file")
args = parser.parse_args()
config = DEFAULT_CONFIG.copy()
if args.config:
    ext = os.path.splitext(args.config)[1].lower()
    with open(args.config) as f:
        user_cfg = yaml.safe_load(f) if ext in (".yaml", ".yml") else json.load(f)
    config.update(user_cfg)

os.makedirs(config['save_dir'], exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger()

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def set_seed(seed: int):
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms(size: int) -> Dict[str, transforms.Compose]:
    """Return training and testing torchvision transforms."""
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class HybridModel(nn.Module):
    """Fusion of ConvNeXt and SwinV2 features."""
    def __init__(self, cnn_name: str, swin_name: str, num_classes: int, freeze_backbones=False):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=True, num_classes=0, global_pool='avg')
        self.swin = timm.create_model(swin_name, pretrained=True, num_classes=0, global_pool='avg')
        if freeze_backbones:
            for p in self.cnn.parameters(): p.requires_grad = False
            for p in self.swin.parameters(): p.requires_grad = False
        dim = self.cnn.num_features + self.swin.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.cnn(x), self.swin(x)], dim=1))

# ----------------------------------------------------------------------------
# Training & Evaluation
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp) -> float:
    """Train model for one epoch with optional AMP."""
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float, float, List[int], List[int], List[float]]:
    """Evaluate model and return metrics and raw predictions."""
    model.eval()
    ys, ps, pr = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(1)
        ys.extend(labels.cpu().tolist())
        ps.extend(preds.cpu().tolist())
        pr.extend(probs.cpu().tolist())
    return (
        accuracy_score(ys, ps),
        f1_score(ys, ps),
        roc_auc_score(ys, pr),
        ys, ps, pr
    )

# ----------------------------------------------------------------------------
# Explainability
# ----------------------------------------------------------------------------
def explain_image(model, img, label, device) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Integrated Gradients and Grad-CAM for one sample."""
    ig = IntegratedGradients(model)
    lg = LayerGradCam(model, model.swin.layers[-1])
    batch = img.unsqueeze(0).to(device)
    ig_attr, _ = ig.attribute(batch, target=label, return_convergence_delta=True)
    gc_attr = lg.attribute(batch, target=label)
    return ig_attr.squeeze(0).cpu().numpy(), gc_attr.squeeze(0).cpu().numpy()

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_metrics(ys: List[int], ps: List[int], pr: List[float], title: str):
    """Plot ROC curve and confusion matrix."""
    fpr, tpr, _ = roc_curve(ys, pr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(ys, pr):.3f}')
    plt.plot([0, 1], [0, 1], '--')
    plt.title(f'{title} ROC')
    plt.legend(); plt.show()

    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title} Confusion Matrix'); plt.show()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(config: Dict):
    logger.info(f'Config: {config}')
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and split data (images: np.ndarray[N,H,W,3], labels: np.ndarray[N])
    transforms_dict = get_transforms(config['image_size'])
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=config['test_size'],
        stratify=labels, random_state=config['seed']
    )
    ds_train = TensorDataset(torch.tensor(X_train).permute(0,3,1,2), torch.tensor(y_train))
    ds_test  = TensorDataset(torch.tensor(X_test).permute(0,3,1,2),  torch.tensor(y_test))

    kf = KFold(n_splits=config['num_folds'], shuffle=True, random_state=config['seed'])
    test_accs = []

    for fold, (tr_idx, _) in enumerate(kf.split(ds_train), 1):
        logger.info(f'Fold {fold}/{config['num_folds']}')
        train_loader = DataLoader(Subset(ds_train, tr_idx), batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        test_loader  = DataLoader(ds_test, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

        model = HybridModel(config['cnn_arch'], config['swin_arch'], config['num_classes'], freeze_backbones=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'], weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
        scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None

        # Training
        for epoch in range(config['num_epochs']):
            if epoch == config['freeze_epochs']:
                for p in model.cnn.parameters(): p.requires_grad = True
                for p in model.swin.parameters(): p.requires_grad = True
            train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config['mixed_precision'])
            scheduler.step()

        # Evaluation
        acc, f1, auc, ys, ps, pr = evaluate(model, test_loader, device)
        test_accs.append(acc)
        plot_metrics(ys, ps, pr, f'Fold {fold}')

        # Explain sample
        img, lbl = ds_test[0]
        ig_map, gc_map = explain_image(model, img, lbl, device)
        viz.visualize_image_attr(ig_map, img.permute(1,2,0).numpy(), method='blended_heat_map'); plt.show()
        heatmap = plt.cm.jet(gc_map)[..., :3]
        overlay = 0.5 * heatmap + 0.5 * img.permute(1,2,0).numpy()
        plt.imshow(overlay); plt.show()

        logger.info(f'Fold {fold} accuracy: {acc:.4f}')

    logger.info(f'Average accuracy: {np.mean(test_accs):.4f}')

if __name__ == '__main__':
    main(config)
