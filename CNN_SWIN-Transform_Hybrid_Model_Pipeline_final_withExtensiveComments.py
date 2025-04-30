import argparse  # Parse command-line arguments (allows config file path input)
import os        # File system operations (create dirs, path handling)
import random    # Setting Python RNG seed for reproducibility
import json      # Load JSON config files
import yaml      # Load YAML config files
import logging   # Structured logging
from typing import Dict, Tuple, List  # Type annotations for clarity

import numpy as np  # Array operations, random seeds
import torch         # Core PyTorch
import torch.nn as nn       # Neural network modules (layers, losses)
import torch.optim as optim # Optimizers (e.g., AdamW)
from torch.utils.data import DataLoader, TensorDataset, Subset
# - DataLoader: batching, shuffling
# - TensorDataset: wrap arrays
# - Subset: split dataset indices
from torchvision import transforms  # Image augmentations and preprocessing
import timm  # Pretrained vision backbones library
from captum.attr import IntegratedGradients, LayerGradCam
# Captum: model interpretability methods
from captum.attr import visualization as viz  # Captum plotting helper
from sklearn.model_selection import KFold, train_test_split
# - KFold: cross-validation
# - train_test_split: stratified train/test split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)  # Evaluation metrics
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Statistical visualization

# =====================================
#           Default Configuration
# =====================================
# Dictionary of default hyperparameters and settings
DEFAULT_CONFIG: Dict = {
    "seed": 42,                    # Random seed for reproducibility
    "image_size": 224,             # Input image resolution
    "batch_size": 16,              # Number of samples per batch
    "num_workers": 4,              # DataLoader CPU processes
    "cnn_arch": "convnext_base", # CNN backbone in timm
    "swin_arch": "swinv2_base_window8_256",  # Transformer backbone
    "num_classes": 2,              # Binary classification
    "freeze_epochs": 5,            # Epochs to freeze pretrained backbones
    "num_epochs": 30,              # Total training epochs
    "learning_rate": 3e-4,         # Initial LR for optimizer
    "weight_decay": 1e-2,          # L2 regularization
    "step_size": 8,                # LR scheduler step interval
    "gamma": 0.85,                 # LR decay factor
    "mixed_precision": True,       # Use AMP for speed/memory
    "num_folds": 5,                # Number of CV folds
    "test_size": 0.3,              # Fraction of data for final test set
    "save_dir": "./checkpoints"  # Directory to save checkpoints
}

# =====================================
#     Setup Logging and Config File
# =====================================
parser = argparse.ArgumentParser(description="Hybrid ConvNeXt+Swin Pipeline")
parser.add_argument('--config', type=str, default=None,
                    help='Path to JSON or YAML config file to override defaults')
args = parser.parse_args()  # Parse CLI arguments
config = DEFAULT_CONFIG.copy()  # Start with defaults
if args.config:
    # Determine file type by extension
    ext = os.path.splitext(args.config)[1].lower()
    with open(args.config, 'r') as f:
        if ext in ['.yml', '.yaml']:
            user_cfg = yaml.safe_load(f)  # Load YAML config
        else:
            user_cfg = json.load(f)  # Load JSON config
    config.update(user_cfg)  # Override defaults with user settings

# Ensure checkpoint directory exists
os.makedirs(config['save_dir'], exist_ok=True)
# Configure global logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)  # Module-level logger

# =====================================
#      Utility: Set Random Seeds
# =====================================
def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)               # Python random
    np.random.seed(seed)            # NumPy random
    torch.manual_seed(seed)         # PyTorch CPU random
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU random

# =====================================
#           Data Transforms
# =====================================
def get_transforms(img_size: int) -> Dict[str, transforms.Compose]:
    """
    Returns training and test transform pipelines.
    """
    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # Random crop+resize
        transforms.RandomHorizontalFlip(),                           # Horizontal flip
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),                # Color jitter
        transforms.RandomRotation(15),                              # Rotate ±15°
        transforms.ToTensor(),                                      # Convert PIL→Tensor
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    test = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),                   # Resize shorter side
        transforms.CenterCrop(img_size),                           # Center crop
        transforms.ToTensor(),                                      # Convert PIL→Tensor
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    return {"train": train, "test": test}

# =====================================
#           Model Definition
# =====================================
class HybridModel(nn.Module):
    """
    Hybrid architecture: concatenates ConvNeXt and Swin features.
    """
    def __init__(
        self,
        cnn_name: str,
        swin_name: str,
        num_classes: int,
        freeze_backbones: bool = False
    ):
        super().__init__()
        # Load ConvNeXt without classifier head
        self.cnn = timm.create_model(cnn_name, pretrained=True, num_classes=0, global_pool="avg")
        # Load Swin without classifier head
        self.swin = timm.create_model(swin_name, pretrained=True, num_classes=0, global_pool="avg")
        if freeze_backbones:
            # Freeze backbone parameters for warm-up
            for p in self.cnn.parameters(): p.requires_grad = False
            for p in self.swin.parameters(): p.requires_grad = False
        # Fusion head: dropout + linear
        dim = self.cnn.num_features + self.swin.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.cnn(x)  # CNN features
        f2 = self.swin(x)  # Swin features
        return self.head(torch.cat([f1, f2], dim=1))  # Fuse and classify

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
    """
    Train model for one epoch with optional AMP.
    """
    model.train()    # Enable training-specific layers
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()  # Clear gradients
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            loss = criterion(model(imgs), labels)  # Forward + loss
        scaler.scale(loss).backward()  # Backprop with scaling
        scaler.step(optimizer)         # Optimizer step
        scaler.update()                # Update scaler
        total_loss += loss.item()
    return total_loss / len(loader)  # Mean loss

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, List[int], List[int], List[float]]:
    """
    Evaluate model, returning metrics and raw preds/probs.
    """
    model.eval()   # Disable dropout, batchnorm
    ys, ps, pr = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]  # Positive-class probs
        preds = logits.argmax(1)                   # Predicted classes
        ys.extend(labels.cpu().tolist())
        ps.extend(preds.cpu().tolist())
        pr.extend(probs.cpu().tolist())
    return (
        accuracy_score(ys, ps),  # Accuracy
        f1_score(ys, ps),        # F1 score
        roc_auc_score(ys, pr),   # ROC AUC
        ys, ps, pr               # Raw data for plotting
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
    """
    Compute IG and Grad-CAM attributions for one image.
    """
    ig = IntegratedGradients(model)  # IG explainer
    lg = LayerGradCam(model, model.swin.layers[-1])  # Grad-CAM on last Swin layer
    img_batch = img.unsqueeze(0).to(device)  # Add batch dimension
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
    """
    Plot ROC curve and confusion matrix for given data.
    """
    # ROC
    fpr, tpr, _ = roc_curve(ys, pr)
    auc = roc_auc_score(ys, pr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], '--')
    plt.title(f"{title} ROC")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.show()
    # Confusion matrix
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{title} Conf Matrix")
    plt.xlabel("Pred"); plt.ylabel("True"); plt.show()

# =====================================
#             Main Routine
# =====================================
def main(config: Dict) -> None:
    """
    Main training, evaluation, and explanation loop.
    """
    logger.info(f"Loaded config: {config}")
    set_seed(config['seed'])  # Reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = get_transforms(config['image_size'])

    # Load and split data (replace TODO with actual loading code)
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

    # Cross-validation on training set
    kf = KFold(n_splits=config['num_folds'], shuffle=True, random_state=config['seed'])
    test_accs = []  # collect test accuracies

    for fold, (tr_idx, val_idx) in enumerate(kf.split(ds_train), 1):
        logger.info(f"Starting fold {fold}/{config['num_folds']}")
        # DataLoaders for current fold
        train_dl = DataLoader(Subset(ds_train, tr_idx), batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        val_dl   = DataLoader(Subset(ds_train, val_idx), batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        test_dl  = DataLoader(ds_test, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

        # Model init with backbones frozen
        model = HybridModel(
            config['cnn_arch'], config['swin_arch'], config['num_classes'], freeze_backbones=True
        ).to(device)
        criterion = nn.CrossEntropyLoss()  # Classification loss
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['learning_rate'], weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
        scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None

        # Training epochs
        for epoch in range(config['num_epochs']):
            if epoch == config['freeze_epochs']:
                # Unfreeze backbones after warm-up
                for p in model.cnn.parameters(): p.requires_grad = True
                for p in model.swin.parameters(): p.requires_grad = True
            loss = train_one_epoch(model, train_dl, criterion, optimizer, scaler, device, config['mixed_precision'])
            scheduler.step()  # Decay LR
            logger.info(f"Fold {fold} Epoch {epoch+1}/{config['num_epochs']} - Loss: {loss:.4f}")

        # Evaluate on test set
        acc, f1, auc, ys, ps, pr = evaluate(model, test_dl, device)
        test_accs.append(acc)
        logger.info(f"Fold {fold} - Test Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        plot_metrics(ys, ps, pr, f"Fold {fold}")

        # Explain one sample with IG and Grad-CAM
        img, lbl = ds_test[0]
        ig_map, gc_map = explain_image(model, img, lbl, device)
        viz.visualize_image_attr(ig_map, img.permute(1,2,0).numpy(), method='blended_heat_map')  # IG heatmap
        plt.title(f"Fold {fold} IG"); plt.show()
        heatmap = plt.cm.jet(gc_map)[..., :3]
        overlay = 0.5 * heatmap + 0.5 * img.permute(1,2,0).numpy()
        plt.imshow(overlay); plt.title(f"Fold {fold} Grad-CAM"); plt.axis('off'); plt.show()

    logger.info(f"Average Test Accuracy: {np.mean(test_accs):.4f}")

if __name__ == "__main__":
    main(config)
