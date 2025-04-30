#!/usr/bin/env python3.11
# Enforce Python 3.11+ runtime; exit otherwise
import sys
if sys.version_info < (3, 11):
    sys.exit("ERROR: Python 3.11 or higher is required.")

# Core libs
import argparse  # Parse command-line arguments (config file)
import os        # File/directory operations
import random    # Python RNG for reproducibility
import json      # JSON config parsing
import yaml      # YAML config parsing
import logging   # Structured logging
from typing import Dict, Tuple, List  # Type hints

# Numerical & ML libraries
import numpy as np  # Array ops, RNG
import torch         # PyTorch core
import torch.nn as nn       # Neural network modules
import torch.optim as optim # Optimizers
from torch.utils.data import DataLoader, TensorDataset, Subset
#    DataLoader: batching & shuffling
#    TensorDataset: wrap arrays
#    Subset: split dataset by indices
from torchvision import transforms  # Image augmentations & normalization
import timm  # Pretrained vision backbones

# Explainability
from captum.attr import IntegratedGradients, LayerGradCam
from captum.attr import visualization as viz  # Captum plotting

# Model evaluation & splitting
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)

# Plotting
import matplotlib.pyplot as plt  # Standard plotting
import seaborn as sns            # Statistical visualization

# --- Default Configuration ---
DEFAULT_CONFIG: Dict = {
    "seed": 42,                    # RNG seed
    "image_size": 224,             # Input size (H=W)
    "batch_size": 16,              # Samples per GPU batch
    "num_workers": 4,              # DataLoader parallel workers
    "cnn_arch": "convnext_base",   # CNN backbone in timm
    "swin_arch": "swinv2_base_window8_256",  # Swin backbone
    "num_classes": 2,              # Binary classification
    "freeze_epochs": 5,            # Warm-up freeze
    "num_epochs": 30,              # Total epochs
    "learning_rate": 3e-4,         # AdamW initial LR
    "weight_decay": 1e-2,          # AdamW L2 regularization
    "step_size": 8,                # LR scheduler step interval
    "gamma": 0.85,                 # LR decay factor
    "mixed_precision": True,       # Use torch.cuda.amp
    "num_folds": 5,                # K-Fold CV splits
    "test_size": 0.3,              # Holdout fraction
    "save_dir": "./checkpoints"    # Checkpoint directory
}

# --- CLI & Config Loading ---
parser = argparse.ArgumentParser(description="Hybrid ConvNeXt+Swin Pipeline")
parser.add_argument(
    "--config", type=str, default=None,
    help="Path to JSON/YAML config file to override defaults"
)
args = parser.parse_args()
config = DEFAULT_CONFIG.copy()
if args.config:
    ext = os.path.splitext(args.config)[1].lower()
    with open(args.config, "r") as f:
        user_cfg = yaml.safe_load(f) if ext in [".yaml", ".yml"] else json.load(f)
    config.update(user_cfg)

# Ensure checkpoint directory exists
os.makedirs(config["save_dir"], exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Utility: Reproducibility ---
def set_seed(seed: int) -> None:
    """Set RNG seeds for Python, NumPy, and PyTorch (CPU & GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Data Transforms ---
def get_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    """Create train/test transform pipelines."""
    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # Random crop+resize
        transforms.RandomHorizontalFlip(),                         # Flip
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),               # Color jitter
        transforms.RandomRotation(15),                            # Rotate ±15°
        transforms.ToTensor(),                                    # PIL→Tensor
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # ImageNet norm
    ])
    test = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),                  # Resize shorter side
        transforms.CenterCrop(img_size),                          # Center crop
        transforms.ToTensor(),                                    # PIL→Tensor
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # ImageNet norm
    ])
    return {"train": train, "test": test}

# --- Model Definition ---
class HybridModel(nn.Module):
    """Concatenate ConvNeXt and Swin features, then classify."""
    def __init__(
        self,
        cnn_name: str,
        swin_name: str,
        num_classes: int,
        freeze_backbones: bool = False
    ):
        super().__init__()
        # Load ConvNeXt-Base without its head
        self.cnn = timm.create_model(
            cnn_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        # Load SwinV2-Base without its head
        self.swin = timm.create_model(
            swin_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        # Optionally freeze backbones for warm-up
        if freeze_backbones:
            for p in self.cnn.parameters(): p.requires_grad = False
            for p in self.swin.parameters(): p.requires_grad = False
        # Fusion head
        dim = self.cnn.num_features + self.swin.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.cnn(x)  # CNN features
        f2 = self.swin(x)  # Swin features
        return self.head(torch.cat([f1, f2], dim=1))  # Fuse and classify

# --- Training Function ---
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    mixed_precision: bool
) -> float:
    """Train model for one epoch with optional AMP."""
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

# --- Evaluation Function ---
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, List[int], List[int], List[float]]:
    """Evaluate model, return acc, F1, AUC, and raw ys, ps, pr."""
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

# --- Explainability Function ---
def explain_image(
    model: nn.Module,
    img: torch.Tensor,
    label: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Integrated Gradients and Grad-CAM attributions."""
    ig = IntegratedGradients(model)
    lg = LayerGradCam(model, model.swin.layers[-1])
    batch = img.unsqueeze(0).to(device)
    ig_attr, _ = ig.attribute(batch, target=label, return_convergence_delta=True)
    gc_attr = lg.attribute(batch, target=label)
    return ig_attr.squeeze(0).cpu().numpy(), gc_attr.squeeze(0).cpu().numpy()

# --- Plotting Function ---
def plot_metrics(
    ys: List[int],
    ps: List[int],
    pr: List[float],
    title: str
) -> None:
    """Plot ROC curve with AUC and confusion matrix."""
    fpr, tpr, _ = roc_curve(ys, pr)
    auc = roc_auc_score(ys, pr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title(f"{title} ROC")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.show()
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{title} Conf Matrix")
    plt.xlabel("Pred"); plt.ylabel("True"); plt.show()

# --- Main Routine ---
def main(config: Dict) -> None:
    """Load data, run training/CV loops, evaluate, and visualize explanations."""
    logger.info(f"Loaded config: {config}")
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = get_transforms(config["image_size"])

    # TODO: load images (720,H,W,3) & labels (720,)
    # e.g., images = np.load("images.npy"); labels = np.load("labels.npy")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=config["test_size"],
        stratify=labels,
        random_state=config["seed"]
    )
    make_ds = lambda X, y: TensorDataset(
        torch.tensor(X, dtype=torch.float32).permute(0,3,1,2),
        torch.tensor(y, dtype=torch.long)
    )
    ds_train, ds_test = make_ds(X_train, y_train), make_ds(X_test, y_test)

    # Cross-validation
    kf = KFold(
        n_splits=config["num_folds"],
        shuffle=True,
        random_state=config["seed"]
    )
    test_accs = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(ds_train), 1):
        logger.info(f"Starting fold {fold}/{config['num_folds']}")
        train_loader = DataLoader(
            Subset(ds_train, tr_idx),
            batch_size=config["batch_size"], shuffle=True,
            num_workers=config["num_workers"]
        )
        val_loader = DataLoader(
            Subset(ds_train, val_idx),
            batch_size=config["batch_size"], shuffle=False,
            num_workers=config["num_workers"]
        )
        test_loader = DataLoader(
            ds_test,
            batch_size=config["batch_size"], shuffle=False,
            num_workers=config["num_workers"]
        )

        # Model init
        model = HybridModel(
            config["cnn_arch"],
            config["swin_arch"],
            config["num_classes"],
            freeze_backbones=True
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["step_size"],
            gamma=config["gamma"]
        )
        scaler = torch.cuda.amp.GradScaler() if config["mixed_precision"] else None

        # Training loop
        for epoch in range(config["num_epochs"]):
            if epoch == config["freeze_epochs"]:
                # Unfreeze backbones after warm-up
                for p in model.cnn.parameters(): p.requires_grad = True
                for p in model.swin.parameters(): p.requires_grad = True
            loss = train_one_epoch(
                model, train_loader, criterion, optimizer,
                scaler, device, config["mixed_precision"]
            )
            scheduler.step()
            logger.info(f"Fold {fold} Epoch {epoch+1}/{config['num_epochs']} - Loss: {loss:.4f}")

        # Evaluation
        acc, f1, auc, ys, ps, pr = evaluate(model, test_loader, device)
        test_accs.append(acc)
        logger.info(f"Fold {fold} - Test Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        plot_metrics(ys, ps, pr, f"Fold {fold}")

        # Explain sample
        img, lbl = ds_test[0]
        ig_map, gc_map = explain_image(model, img, lbl, device)
        viz.visualize_image_attr(
            ig_map, img.permute(1,2,0).numpy(),
            method="blended_heat_map"
        )
        plt.title(f"Fold {fold} IG"); plt.show()
        heatmap = plt.cm.jet(gc_map)[..., :3]
        overlay = 0.5 * heatmap + 0.5 * img.permute(1,2,0).numpy()
        plt.imshow(overlay); plt.title(f"Fold {fold} Grad-CAM"); plt.axis("off"); plt.show()

    logger.info(f"Average Test Accuracy: {np.mean(test_accs):.4f}")

if __name__ == "__main__":
    main(config)
