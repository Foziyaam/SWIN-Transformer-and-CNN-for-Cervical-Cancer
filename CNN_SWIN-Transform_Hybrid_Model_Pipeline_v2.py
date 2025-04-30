import argparse
import os
import random
import json
import yaml  # YAML support
import logging
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
import timm
from captum.attr import IntegratedGradients, LayerGradCam
from captum.attr import visualization as viz
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
#           Default Configuration
# =====================================
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

# =====================================
#     Setup Logging and Config File
# =====================================
parser = argparse.ArgumentParser(description="Hybrid ConvNeXt+Swin Pipeline")
parser.add_argument('--config', type=str, default=None,
                    help='Path to JSON or YAML config file to override defaults')
args = parser.parse_args()
config = DEFAULT_CONFIG.copy()
if args.config:
    # Load JSON or YAML based on file extension
    ext = os.path.splitext(args.config)[1].lower()
    with open(args.config, 'r') as f:
        if ext in ['.yml', '.yaml']:
            user_cfg = yaml.safe_load(f)
        else:
            user_cfg = json.load(f)
    config.update(user_cfg)

os.makedirs(config['save_dir'], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =====================================
#      Utility: Set Random Seeds
# =====================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =====================================
#           Data Transforms
# =====================================
def get_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return {"train": train, "test": test}

# =====================================
#           Model Definition
# =====================================
class HybridModel(nn.Module):
    def __init__(
        self,
        cnn_name: str,
        swin_name: str,
        num_classes: int,
        freeze_backbones: bool = False
    ):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=True, num_classes=0, global_pool="avg")
        self.swin = timm.create_model(swin_name, pretrained=True, num_classes=0, global_pool="avg")
        if freeze_backbones:
            for p in self.cnn.parameters(): p.requires_grad = False
            for p in self.swin.parameters(): p.requires_grad = False
        dim = self.cnn.num_features + self.swin.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.cnn(x)
        f2 = self.swin(x)
        return self.head(torch.cat([f1, f2], dim=1))

# =====================================
#      Training & Evaluation
# =====================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    mixed_precision: bool
) -> float:
    model.train()
    total = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, List[int], List[int], List[float]]:
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
        ys,
        ps,
        pr
    )

# =====================================
#      Explainability: Captum
# =====================================
def explain_image(
    model: nn.Module,
    img: torch.Tensor,
    label: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    ig = IntegratedGradients(model)
    lg = LayerGradCam(model, model.swin.layers[-1])
    img_batch = img.unsqueeze(0).to(device)
    ig_attr, _ = ig.attribute(img_batch, target=label, return_convergence_delta=True)
    gc_attr = lg.attribute(img_batch, target=label)
    return ig_attr.squeeze(0).cpu().numpy(), gc_attr.squeeze(0).cpu().numpy()

# =====================================
#     Plot Utilities
# =====================================
def plot_metrics(
    ys: List[int],
    ps: List[int],
    pr: List[float],
    title: str
) -> None:
    fpr, tpr, _ = roc_curve(ys, pr)
    auc = roc_auc_score(ys, pr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.title(f"{title} ROC")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.show()
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{title} Conf Matrix")
    plt.xlabel("Pred"); plt.ylabel("True"); plt.show()

# =====================================
#             Main Routine
# =====================================
def main(config: Dict) -> None:
    logger.info(f"Loaded config: {config}")
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = get_transforms(config['image_size'])

    # TODO: load images (720,H,W,3) & labels (720,)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=config['test_size'],
        stratify=labels,
        random_state=config['seed']
    )
    make_ds = lambda X, y: TensorDataset(
        torch.tensor(X, dtype=torch.float32).permute(0,3,1,2),
        torch.tensor(y, dtype=torch.long)
    )
    ds_train, ds_test = make_ds(X_train, y_train), make_ds(X_test, y_test)

    kf = KFold(n_splits=config['num_folds'], shuffle=True, random_state=config['seed'])
    test_accs = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(ds_train), 1):
        logger.info(f"Starting fold {fold}/{config['num_folds']}")
        train_dl = DataLoader(Subset(ds_train, tr_idx), batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        val_dl   = DataLoader(Subset(ds_train, val_idx), batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        test_dl  = DataLoader(ds_test, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

        model = HybridModel(
            config['cnn_arch'], config['swin_arch'], config['num_classes'], freeze_backbones=True
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'], weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
        scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None

        for epoch in range(config['num_epochs']):
            if epoch == config['freeze_epochs']:
                for p in model.cnn.parameters(): p.requires_grad = True
                for p in model.swin.parameters(): p.requires_grad = True
            loss = train_one_epoch(model, train_dl, criterion, optimizer, scaler, device, config['mixed_precision'])
            scheduler.step()
            logger.info(f"Fold {fold} Epoch {epoch+1}/{config['num_epochs']} - Loss: {loss:.4f}" )

        acc, f1, auc, ys, ps, pr = evaluate(model, test_dl, device)
        test_accs.append(acc)
        logger.info(f"Fold {fold} - Test Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        plot_metrics(ys, ps, pr, f"Fold {fold}")

        img, lbl = ds_test[0]
        ig_map, gc_map = explain_image(model, img, lbl, device)
        viz.visualize_image_attr(ig_map, img.permute(1,2,0).numpy(), method='blended_heat_map')
        plt.title(f"Fold {fold} IG"); plt.show()
        heatmap = plt.cm.jet(gc_map)[..., :3]
        overlay = 0.5 * heatmap + 0.5 * img.permute(1,2,0).numpy()
        plt.imshow(overlay); plt.title(f"Fold {fold} Grad-CAM"); plt.axis('off'); plt.show()

    logger.info(f"Average Test Accuracy: {np.mean(test_accs):.4f}")

if __name__ == "__main__":
    main(config)
