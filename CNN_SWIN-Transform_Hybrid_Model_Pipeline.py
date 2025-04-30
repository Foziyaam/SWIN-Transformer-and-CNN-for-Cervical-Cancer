import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
import timm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
#        Architecture Selection
# =====================================
# Optimal transfer-learning backbones for moderate medical datasets (~360 images/class):
# - CNN: ConvNeXt-Base (convnext_base pretrained on ImageNet-22K) for high accuracy and efficient training
# - Transformer: SwinV2-Base (swinv2_base_window8_256 pretrained on ImageNet22K→1K) for robust hierarchical attention
# Together they balance capacity and overfitting risk on limited data.

# =====================================
#           Configuration
# =====================================
CONFIG = {
    'seed': 42,
    'image_size': 224,
    'batch_size': 16,
    'num_workers': 4,
    'cnn_arch': 'convnext_base',
    'swin_arch': 'swinv2_base_window8_256',
    'num_classes': 2,
    'freeze_backbones_epochs': 5,
    'num_epochs': 30,
    'learning_rate': 3e-4,
    'weight_decay': 1e-2,
    'step_size': 8,
    'gamma': 0.85,
    'mixed_precision': True,
    'num_folds': 5,
    'test_size': 0.3,
    'save_dir': './checkpoints'
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# Reproducibility
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])

# =====================================
#           Data Transforms
# =====================================
def get_data_transforms(sz):
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(sz, scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(int(sz*1.14)),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    }

# =====================================
#         Model Definition
# =====================================
class HybridCNN_Swin(nn.Module):
    def __init__(self, cfg, freeze_backbones=False):
        super().__init__()
        # Load ConvNeXt-Base without head
        self.cnn = timm.create_model(cfg['cnn_arch'], pretrained=True, num_classes=0, global_pool='avg')
        # Load SwinV2-Base without head
        self.swin = timm.create_model(cfg['swin_arch'], pretrained=True, num_classes=0, global_pool='avg')
        if freeze_backbones:
            for p in self.cnn.parameters(): p.requires_grad = False
            for p in self.swin.parameters(): p.requires_grad = False
        cnn_dim = self.cnn.num_features
        swin_dim = self.swin.num_features
        # Fusion head
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(cnn_dim + swin_dim, cfg['num_classes'])

    def forward(self, x):
        f1 = self.cnn(x)
        f2 = self.swin(x)
        fused = torch.cat([f1, f2], dim=1)
        return self.fc(self.dropout(fused))

# =====================================
#    Training & Evaluation Functions
# =====================================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=CONFIG['mixed_precision']):
            logits = model(imgs)
            loss = criterion(logits, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, pr = [], [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:,1]
        preds = torch.argmax(logits, dim=1)
        ys.extend(lbls.cpu().tolist())
        ps.extend(preds.cpu().tolist())
        pr.extend(probs.cpu().tolist())
    return accuracy_score(ys,ps), f1_score(ys,ps), roc_auc_score(ys,pr), ys, ps, pr

# =====================================
#           Plot Utilities
# =====================================
def plot_roc(ys, pr, title):
    fpr, tpr, _ = roc_curve(ys, pr)
    aucv = roc_auc_score(ys, pr)
    plt.plot(fpr, tpr, label=f'AUC={aucv:.3f}')
    plt.plot([0,1], [0,1], '--')
    plt.title(title)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.show()


def plot_cm(ys, ps, title):
    cm = confusion_matrix(ys, ps)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()

# =====================================
#             Main Routine
# =====================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = get_data_transforms(CONFIG['image_size'])

    # TODO: Load your 720 colposcopy images and labels as NumPy arrays
    # images: shape (720, H, W, 3); labels: shape (720,)

    # Stratified split (70/30 => 504 train, 216 test)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=CONFIG['test_size'],
        stratify=labels,
        random_state=CONFIG['seed']
    )

    # Dataset wrappers
    def make_ds(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32).permute(0,3,1,2),
            torch.tensor(y, dtype=torch.long)
        )

    ds_train = make_ds(X_train, y_train)
    ds_test  = make_ds(X_test,  y_test)

    # 5-fold CV on training set
    kf = KFold(n_splits=CONFIG['num_folds'], shuffle=True, random_state=CONFIG['seed'])
    test_accs = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(ds_train)):
        tr_ds = Subset(ds_train, tr_idx)
        vl_ds = Subset(ds_train, val_idx)
        loaders = {
            'train': DataLoader(tr_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers']),
            'val':   DataLoader(vl_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers']),
            'test':  DataLoader(ds_test, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        }

        # Model initialization
        model = HybridCNN_Swin(CONFIG, freeze_backbones=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay']
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG['step_size'], gamma=CONFIG['gamma'])
        scaler = torch.cuda.amp.GradScaler() if CONFIG['mixed_precision'] else None

        for epoch in range(CONFIG['num_epochs']):
            if epoch == CONFIG['freeze_backbones_epochs']:
                for p in model.cnn.parameters(): p.requires_grad = True
                for p in model.swin.parameters(): p.requires_grad = True

            train_epoch(model, loaders['train'], criterion, optimizer, scaler, device)
            scheduler.step()

        acc, _, _, ys, ps, pr = evaluate(model, loaders['test'], device)
        test_accs.append(acc)

        # Save & plot
        ckpt_path = os.path.join(CONFIG['save_dir'], f'fold{fold+1}.pth')
        torch.save(model.state_dict(), ckpt_path)
        plot_roc(ys, pr, f'Fold{fold+1} ROC')
        plot_cm(ys, ps, f'Fold{fold+1} CM')

    print(f"Average Test Accuracy: {np.mean(test_accs):.4f}")

if __name__ == '__main__':
    main()


'''


