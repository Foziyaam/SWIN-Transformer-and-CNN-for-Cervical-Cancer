# Analysis of Colposcopic Images Obtained from "International Agency for Research on Cancer" (IARCImageBankColpo)

### Refined Criteria Summary for Grouping the 200 Cases into Normal, Cancerous, Precancerous and Inconclusive

#### 1. **Normal Group**
**Criteria**:
- **HPV Status**: Both positive and negative.
- **Sample Adequacy**: Adequate colposcopic sample.
- **Squamocolumnar Junction Visibility**: Completely visible or not visible.
- **Transformation Zone**: Type 1, Type 2, or Type 3.
- **Normal Colposcopic Findings**:
  - Presence of original squamous epithelium, columnar epithelium, or metaplastic squamous epithelium.
  - Absence of significant abnormal colposcopic findings (e.g., no dense acetowhite epithelium, coarse punctation, or mosaic).
  - Iodine staining: Nil or transparent, faintly or patchy yellow, or brown.
- **Swede Score**: Low scores (0-2).
- **Histopathology**: Normal findings (e.g., mature squamous epithelium, atrophic changes, nabothian cysts).
- **Management**: Routine screening after 5 years or repeat HPV test/colposcopy after 1 year if HPV positive.

#### 2. **Pre-cancerous Group**
**Low-grade Squamous Intraepithelial Lesion (LSIL)**
**Criteria**:
- **HPV Status**: Both positive and negative.
- **Sample Adequacy**: Adequate colposcopic sample.
- **Squamocolumnar Junction Visibility**: Completely visible.
- **Transformation Zone**: Type 1 or Type 2.
- **Colposcopic Findings**:
  - Thin acetowhite epithelium.
  - Irregular borders, fine mosaic, fine punctation.
  - Mild abnormalities without features of high-grade lesions.
- **Swede Score**: Moderate scores (3-4).
- **Iodine Uptake**: Faintly or patchy yellow, or nil or transparent.
- **Histopathology**: Low-grade lesions (CIN1).
- **Management**: Routine follow-up and colposcopy, with biopsies as indicated.

**High-grade Squamous Intraepithelial Lesion (HSIL)**
**Criteria**:
- **HPV Status**: Both positive and negative.
- **Sample Adequacy**: Adequate colposcopic sample.
- **Squamocolumnar Junction Visibility**: Completely or partially visible.
- **Transformation Zone**: Type 1, Type 2, or Type 3.
- **Colposcopic Findings**:
  - Dense acetowhite epithelium, coarse mosaic, or coarse punctation.
  - Sharp borders, ridge sign, inner border sign.
  - Cuffed crypt (gland) openings, rapid appearance of acetowhite, presence of atypical vessels.
- **Swede Score**: High scores (5-10).
- **Iodine Uptake**: Distinctly yellow, indicating non-staining with iodine.
- **Histopathology**: High-grade lesions (CIN2, CIN3).
- **Management**: Treatment options like LLETZ, punch biopsies, or excisional procedures.

#### 3. **Cancerous Group**
**Criteria**:
- **HPV Status**: Positive.
- **Sample Adequacy**: Adequate colposcopic sample.
- **Squamocolumnar Junction Visibility**: Not visible.
- **Transformation Zone**: Type 3.
- **Colposcopic Findings**:
  - Dense acetowhite epithelium, coarse punctation, or mosaic.
  - Sharp borders, ridge sign, inner border sign, presence of atypical vessels.
  - Features suspicious for invasion: irregular surface, erosion, tumor, or gross neoplasm.
- **Swede Score**: High scores (9-10).
- **Iodine Uptake**: Distinctly yellow or non-staining with iodine.
- **Histopathology**: Invasive cancer (e.g., squamous cell carcinoma, adenocarcinoma).
- **Management**: Surgical intervention (e.g., LLETZ, punch biopsies, multiple biopsies).

#### 4. **Inconclusive Group**
**Criteria**:
- **HPV Status**: Both positive and negative.
- **Sample Adequacy**: Inadequate colposcopic sample due to reasons such as:
  - Extensive inflammation.
  - Cervix not satisfactorily exposed.
  - Stenosis of vagina or atrophy of cervix.
- **Squamocolumnar Junction Visibility**: Not visible.
- **Transformation Zone**: Not applicable.
- **Colposcopic Findings**: Not applicable due to inadequate sample.
- **Swede Score**: Not applicable.
- **Histopathology**: Not done or not conclusive.
- **Management**: Further investigation, control of infection, or clinical radiological findings based follow-up.

This refined summary ensures that all criteria used for categorization are thoroughly examined and accurately represented. 

**Given the above criteria, Colposcopy images were grouped into four sets (based on the colposcopy examination and histopathology findings)**
    

1.	Precancerous (77 unique subjects
   )
2.	Cancer (26 unique subjects)
 
3.	Inconclusive (3 unique subjects) – these were excluded to ensure that the comparison between cancerous and normal conditions remains clear and unambiguou
s.
4.	Normal (94 unique subjects) – these are labelled normal from Colposcopy examination and/or histopathology outcomes


**Steps to process the cervical colposcopic images organized by case numbers and prepare them for SWIN-Transformer CNN analysis using Python. To prepare images for SWIN-Transformer CNN analysis, organize and extract the images from their respective case folders. Here is a step-by-step guide on how to achieve this using Python:**

    Organize the Directory Structure such that each case folder (e.g., "Case 001", "Case 002") contains its respective images. (done)

    Extract Images and Prepare for Analysis: Write a Python script to extract images from each folder and prepare them for input into the SWIN-Transformer CNN.

    Install Necessary Python Libraries installed e.g. os, shutil, glob, and PIL for handling file operations and image processing.

    Script to Extract Images and Prepare Data given next:


```python

```


```python
import pandas as pd

# Load the metadata CSV file and create a diagnosis dictionary
metadata_path = 'maindirectory/IARCImageBankColpo/Cases_Meta_data_good_v3.csv'
try:
    metadata = pd.read_csv(metadata_path, encoding='utf-8')
except UnicodeDecodeError:
    metadata = pd.read_csv(metadata_path, encoding='latin1')

# Ensure the CaseNumber is formatted to three digits and create a dictionary
metadata['CaseNumber'] = metadata['CaseNumber'].apply(lambda x: f'Case {int(x):03d}')
diagnosis_dict = pd.Series(metadata['CaseDiagnosis'].values, index=metadata['CaseNumber']).to_dict()

# Print the diagnosis dictionary for verification
print("Diagnosis Dictionary Sample:", {k: diagnosis_dict[k] for k in list(diagnosis_dict)[:10]})
```


```python
import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return image
```


```python
import os

def load_and_preprocess_images(base_path, total_cases=199):
    all_images = []
    all_labels = []
    missing_cases = {'Case 078', 'Case 079'}
    
    for i in range(1, total_cases + 1):
        case_folder = f"Case {str(i).zfill(3)}"
        if case_folder in missing_cases:
            continue
        
        full_path = os.path.join(base_path, case_folder)
        if os.path.exists(full_path):
            for filename in os.listdir(full_path):
                if filename.lower().endswith('.jpg'):
                    file_path = os.path.join(full_path, filename)
                    image = preprocess_image(file_path)
                    if image is not None:
                        all_images.append(image)
                        all_labels.append(case_folder)
    
    return np.array(all_images), np.array(all_labels)

# Load images and labels
base_path = 'directory/IARCImageBankColpo'
all_images, all_labels = load_and_preprocess_images(base_path)

print(f"Preprocessed images: {len(all_images)}")
print(f"Sample labels: {all_labels[:10]}")
```

## TensorFlow/Keras or PyTorch, which framework is more appropriate
Both TensorFlow/Keras and PyTorch are powerful frameworks for deep learning, but they have different strengths and community support, which might influence your choice depending on your specific needs and preferences. Below is a comparison of the two frameworks in the context of using the Swin-Transformer CNN model:

### TensorFlow/Keras

#### Pros:
1. **High-level API**: Keras, which is integrated into TensorFlow, provides a high-level API that makes building and experimenting with models very user-friendly and concise.
2. **Deployment**: TensorFlow has robust support for deployment, including TensorFlow Serving, TensorFlow Lite for mobile and embedded devices, and TensorFlow.js for running models in the browser.
3. **Ecosystem**: TensorFlow has a comprehensive ecosystem with tools like TensorBoard for visualization, TensorFlow Extended (TFX) for production ML pipelines, and TensorFlow Hub for pre-trained models.
4. **Community and Support**: TensorFlow has a large community and extensive documentation, which can be helpful for troubleshooting and finding resources.

#### Cons:
1. **Complexity**: TensorFlow can be complex and have a steeper learning curve for beginners compared to PyTorch.
2. **Debugging**: TensorFlow's static computational graph can make debugging more challenging compared to PyTorch's dynamic computational graph.

### PyTorch

#### Pros:
1. **Dynamic Computational Graph**: PyTorch's dynamic computational graph (define-by-run) makes it more intuitive and easier to debug.
2. **Flexibility**: PyTorch is often praised for its flexibility, making it a preferred choice for research and experimentation.
3. **Community and Growth**: PyTorch has seen rapid growth in its user base, especially in the research community, and has a strong community and support network.
4. **Integration with Python**: PyTorch integrates seamlessly with the Python ecosystem and libraries, which can make development smoother.

#### Cons:
1. **Deployment**: While PyTorch has made significant strides with tools like TorchServe, its deployment ecosystem is not as mature as TensorFlow's.
2. **API Stability**: PyTorch is relatively younger, and there may be more frequent API changes.

### Swin-Transformer CNN Framework

The Swin-Transformer is a relatively new model that has gained popularity for its effectiveness in vision tasks. The choice of framework might depend on:

- **Pre-trained Models**: Availability of pre-trained Swin-Transformer models in the framework. PyTorch often has cutting-edge models implemented and shared quickly by the research community.
- **Ease of Use**: If you prefer a high-level API and ease of model deployment, TensorFlow/Keras might be better. For more flexibility and ease of debugging, PyTorch is advantageous.
- **Existing Codebase and Expertise**: Consider the existing codebase and expertise. If you or your team are already proficient in one of the frameworks, it might be beneficial to continue using it.

### Conclusion

For **research and experimentation**, **PyTorch** is often preferred due to its flexibility, dynamic graph, and strong research community support. If you are looking for **robust deployment** and a **high-level API**, **TensorFlow/Keras** might be the better choice.

In the context of the Swin-Transformer model, PyTorch might have an edge due to its quick adoption by the research community and availability of pre-trained models. However, both frameworks can be effectively used to implement and train the Swin-Transformer CNN model.

**Recommendation**: If you are leaning towards flexibility and cutting-edge research, go with PyTorch. If your focus is on deployment and using a high-level API, consider TensorFlow/Keras.

## Image Preprocessing
The SWIN-Transformer CNN hybrid model, which combines the strengths of Convolutional Neural Networks (CNNs) and Swin Transformers, typically requires a specific set of preprocessing steps to ensure the input data is in the optimal format for training and inference. Here are the common preprocessing steps required:
1. Resizing:

The images need to be resized to a fixed size that is suitable for the model. For Swin Transformers, a common input size is 224x224 pixels, but this can vary depending on the specific architecture or any custom modifications.

2. Normalization:

Pixel values need to be normalized to a specific range, typically [0, 1] or [-1, 1]. Normalization helps in speeding up the convergence of the model during training.

3. Data Augmentation:

Data augmentation techniques are often applied to increase the diversity of the training dataset and improve the model’s robustness. Common augmentation techniques include:

    Random cropping and padding
    Random horizontal flipping
    Random rotation
    Color jittering (adjusting brightness, contrast, saturation, and hue)
    Random resizing and scaling

4. Converting Images to Tensors:

The images need to be converted to tensor format, which is required for input into deep learning models in frameworks like PyTorch or TensorFlow.

5. Batching and Shuffling:

During training, images should be batched together and shuffled to ensure that the model learns from a diverse set of examples in each epoch.

6. Handling Class Imbalance (if any):

If the dataset is imbalanced, techniques such as oversampling the minority class or undersampling the majority class can be applied. Additionally, weighted loss functions can be used to handle class imbalance.
Example Preprocessing Code for PyTorch:

Here is a complete example of how to implement these preprocessing steps using PyTorch:


```python

```

# The pipeline for the CNN and SWIN-Transformer hybrid model


```python
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

```

# Line by line annotation and comments as to what and why each line (in the above code)


```python
import argparse  # For parsing command-line arguments (currently unused)
import os         # Utilities for filesystem operations
import random     # Python built-in pseudo-random generator

import numpy as np  # Numerical operations on arrays
import torch         # Core PyTorch library
import torch.nn as nn       # Neural network modules (layers, loss functions)
import torch.optim as optim # Optimization algorithms (AdamW, etc.)
from torch.utils.data import DataLoader, TensorDataset, Subset
# - DataLoader: batching and shuffling
# - TensorDataset: wrap arrays into dataset
# - Subset: split dataset by indices
from torchvision import transforms  # Common image transformations (augment & normalize)
import timm  # PyTorch Image Models: easy pretrained backbones
from sklearn.model_selection import KFold, train_test_split
# - KFold: cross-validation splitting
# - train_test_split: stratified train/test split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
# Standard classification metrics
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns            # Statistical visualization (heatmap)

# =====================================
#        Architecture Selection
# =====================================
# Choosing backbones tailored to moderate datasets (~360 images/class):
# - CNN: ConvNeXt-Base pretrained on ImageNet-22K offers strong local feature extraction
# - Transformer: SwinV2-Base pretrained on ImageNet22K→1K provides hierarchical attention
# Fusion of these two captures complementary representations while controlling overfitting.

# =====================================
#           Configuration
# =====================================
CONFIG = {
    'seed': 42,                    # Random seed for reproducibility
    'image_size': 224,             # Input image dimensions (height=width)
    'batch_size': 16,              # Number of samples per batch
    'num_workers': 4,              # DataLoader worker threads
    'cnn_arch': 'convnext_base',   # Name of CNN backbone in timm
    'swin_arch': 'swinv2_base_window8_256',  # Name of Swin backbone
    'num_classes': 2,              # Binary classification
    'freeze_backbones_epochs': 5,  # Number of epochs to freeze pretrained backbones
    'num_epochs': 30,              # Total training epochs
    'learning_rate': 3e-4,         # Initial learning rate for optimizer
    'weight_decay': 1e-2,          # L2 regularization factor
    'step_size': 8,                # StepLR schedule interval (in epochs)
    'gamma': 0.85,                 # LR decay factor
    'mixed_precision': True,       # Enable AMP for speed & memory
    'num_folds': 5,                # KFold cross-validation folds
    'test_size': 0.3,              # Fraction of data reserved for test
    'save_dir': './checkpoints'    # Directory to save model checkpoints
}

# Create checkpoint directory if it doesn't exist
ios.makedirs(CONFIG['save_dir'], exist_ok=True)

# =====================================
#         Reproducibility Seeds
# =====================================
random.seed(CONFIG['seed'])                    # Python random
np.random.seed(CONFIG['seed'])                 # NumPy random
torch.manual_seed(CONFIG['seed'])              # PyTorch CPU random
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])  # PyTorch GPU random

# =====================================
#           Data Transforms
# =====================================
def get_data_transforms(sz):
    """
    Returns training and testing torchvision Transform pipelines.
    """
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(sz, scale=(0.8,1.0)),  # random crop + resize
            transforms.RandomHorizontalFlip(),                   # 50% horizontal flip
            transforms.ColorJitter(0.2,0.2,0.2,0.1),             # random brightness/contrast/hue
            transforms.RandomRotation(15),                       # rotate ±15°
            transforms.ToTensor(),                               # PIL→Tensor [0,1]
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(int(sz*1.14)),                    # shorter side→~256
            transforms.CenterCrop(sz),                           # center crop to 224×224
            transforms.ToTensor(),                               # PIL→Tensor
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
    }

# =====================================
#         Model Definition
# =====================================
class HybridCNN_Swin(nn.Module):
    """
    Combines ConvNeXt-Base and SwinV2-Base backbones;
    concatenates their pooled features and applies a classification head.
    """
    def __init__(self, cfg, freeze_backbones=False):
        super().__init__()
        # Load CNN backbone without its classifier head (num_classes=0)
        self.cnn = timm.create_model(
            cfg['cnn_arch'], pretrained=True, num_classes=0, global_pool='avg'
        )
        # Load Swin Transformer backbone similarly
        self.swin = timm.create_model(
            cfg['swin_arch'], pretrained=True, num_classes=0, global_pool='avg'
        )
        # Optionally freeze all backbone parameters for warm-up
        if freeze_backbones:
            for p in self.cnn.parameters(): p.requires_grad = False
            for p in self.swin.parameters(): p.requires_grad = False
        # Feature dimensionalities for fusion head
        cnn_dim  = self.cnn.num_features
        swin_dim = self.swin.num_features
        # Dropout + linear classifier to map fused features→num_classes
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(cnn_dim + swin_dim, cfg['num_classes'])

    def forward(self, x):
        # Extract features from both branches
        f1 = self.cnn(x)
        f2 = self.swin(x)
        # Concatenate feature vectors along channel dimension
        fused = torch.cat([f1, f2], dim=1)
        # Apply fusion head
        return self.fc(self.dropout(fused))

# =====================================
#    Training & Evaluation Functions
# =====================================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Runs one training epoch with optional mixed precision.
    """
    model.train()               # Set model to training mode
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)  # Move data to device
        optimizer.zero_grad()                          # Clear gradients
        # Mixed-precision context manager
        with torch.cuda.amp.autocast(enabled=CONFIG['mixed_precision']):
            logits = model(imgs)          # Forward pass
            loss   = criterion(logits, lbls)  # Compute loss
        # Scale, backprop, and update optimizer
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)  # Return average loss

@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluates model on given loader, returns accuracy, F1, AUC, and raw preds/probs.
    """
    model.eval()  # Set to evaluation mode
    ys, ps, pr = [], [], []  # Lists for true labels, predictions, probabilities
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)            # Forward pass
        probs  = torch.softmax(logits, dim=1)[:,1]  # Positive-class probability
        preds  = torch.argmax(logits, dim=1)       # Class predictions
        ys.extend(lbls.cpu().tolist())
        ps.extend(preds.cpu().tolist())
        pr.extend(probs.cpu().tolist())
    # Compute metrics
    return (
        accuracy_score(ys, ps),
        f1_score(ys, ps),
        roc_auc_score(ys, pr),
        ys, ps, pr
    )

# =====================================
#           Plot Utilities
# =====================================
def plot_roc(ys, pr, title):
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(ys, pr)
    aucv = roc_auc_score(ys, pr)
    # Plot
    plt.plot(fpr, tpr, label=f'AUC={aucv:.3f}')
    plt.plot([0,1], [0,1], '--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def plot_cm(ys, ps, title):
    # Compute confusion matrix
    cm = confusion_matrix(ys, ps)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# =====================================
#             Main Routine
# =====================================
def main():
    # Select device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get data transforms
    transforms_dict = get_data_transforms(CONFIG['image_size'])

    # TODO: Load your 720 images & labels as NumPy arrays
    # images shape: (720, H, W, 3); labels shape: (720,)

    # Stratified 70/30 train-test split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=CONFIG['test_size'],
        stratify=labels,
        random_state=CONFIG['seed']
    )

    # Helper to wrap arrays into PyTorch datasets
    def make_ds(X, y):
        tensor_x = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
        tensor_y = torch.tensor(y, dtype=torch.long)
        return TensorDataset(tensor_x, tensor_y)

    ds_train = make_ds(X_train, y_train)
    ds_test  = make_ds(X_test,  y_test)

    # 5-fold CV on the training set
    kf = KFold(n_splits=CONFIG['num_folds'], shuffle=True, random_state=CONFIG['seed'])
    test_accs = []  # store test accuracy for each fold

    for fold, (tr_idx, val_idx) in enumerate(kf.split(ds_train)):
        # Create subset DataLoaders for this fold
        loaders = {
            'train': DataLoader(Subset(ds_train, tr_idx), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers']),
            'val':   DataLoader(Subset(ds_train, val_idx), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers']),
            'test':  DataLoader(ds_test, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        }

        # Initialize model with backbones frozen for initial warm-up
        model = HybridCNN_Swin(CONFIG, freeze_backbones=True).to(device)
        criterion = nn.CrossEntropyLoss()  # classification loss
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay']
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=CONFIG['step_size'], gamma=CONFIG['gamma']
        )
        scaler = torch.cuda.amp.GradScaler() if CONFIG['mixed_precision'] else None

        # Training loop
        for epoch in range(CONFIG['num_epochs']):
            # After warm-up epochs, unfreeze backbones for fine-tuning
            if epoch == CONFIG['freeze_backbones_epochs']:
                for p in model.cnn.parameters(): p.requires_grad = True
                for p in model.swin.parameters(): p.requires_grad = True

            # Train one epoch
            train_epoch(model, loaders['train'], criterion, optimizer, scaler, device)
            # Decay learning rate
            scheduler.step()

        # Evaluate on held-out test set
        acc, _, _, ys, ps, pr = evaluate(model, loaders['test'], device)
        test_accs.append(acc)

        # Save model checkpoint for this fold
        ckpt_path = os.path.join(CONFIG['save_dir'], f'fold{fold+1}.pth')
        torch.save(model.state_dict(), ckpt_path)

        # Plot ROC and confusion matrix
        plot_roc(ys, pr,  f'Fold{fold+1} Test ROC')
        plot_cm(ys, ps,   f'Fold{fold+1} Test CM')

    # Print average test accuracy across folds
    print(f"Average Test Accuracy: {np.mean(test_accs):.4f}")

if __name__ == '__main__':
    main()

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Assuming all_images, all_labels, and diagnosis_dict are already loaded

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Step 1: Print the initial state
print("All labels before conversion (sample):", all_labels[:10])

# Step 2: Subset to exclude 'Cancer' and exactly 17 'Normal' cases
cancer_cases = {case for case, diagnosis in diagnosis_dict.items() if diagnosis == 'Cancer'}
normal_cases = {case for case, diagnosis in diagnosis_dict.items() if diagnosis == 'Normal'}
precancerous_cases = {case for case, diagnosis in diagnosis_dict.items() if diagnosis == 'Precancerous'}

normal_cases_to_exclude = set(random.sample(normal_cases, 17))  # Randomly select 17 normal cases

# Combine cases to exclude
cases_to_exclude = cancer_cases.union(normal_cases_to_exclude)
print("Cases to exclude:", cases_to_exclude)

# Create lists for excluded images and labels
excluded_images = []
excluded_labels = []
remaining_images = []
remaining_labels = []

# Separate excluded cases from the rest
for img, label in zip(all_images, all_labels):
    if label in cases_to_exclude:
        excluded_images.append(img)
        excluded_labels.append(label)
    else:
        remaining_images.append(img)
        remaining_labels.append(label)

excluded_images = np.array(excluded_images)
excluded_labels = np.array(excluded_labels)
remaining_images = np.array(remaining_images)
remaining_labels = np.array(remaining_labels)

print(f"Excluded images count: {len(excluded_images)}")
print(f"Excluded labels count: {len(excluded_labels)}")
print(f"Remaining images count: {len(remaining_images)}")
print(f"Remaining labels count: {len(remaining_labels)}")

# Step 3: Randomly pick 54 Normal cases and 54 Precancerous cases for training
train_normal_cases = random.sample(list(normal_cases - normal_cases_to_exclude), 54)
train_precancerous_cases = random.sample(list(precancerous_cases), 54)

# Combine training cases
train_cases = set(train_normal_cases + train_precancerous_cases)

# Remaining cases for testing
test_normal_cases = list((normal_cases - normal_cases_to_exclude) - set(train_normal_cases))
test_precancerous_cases = list(precancerous_cases - set(train_precancerous_cases))
test_cases = set(test_normal_cases + test_precancerous_cases)

# Initialize lists for train and test sets
train_images = []
train_labels = []
test_images = []
test_labels = []

for img, label in zip(remaining_images, remaining_labels):
    if label in train_cases:
        train_images.append(img)
        train_labels.append(label)
    elif label in test_cases:
        test_images.append(img)
        test_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

print(f"Train images count: {len(train_images)}")
print(f"Test images count: {len(test_images)}")
print(f"Train labels count: {len(train_labels)}")
print(f"Test labels count: {len(test_labels)}")

# Randomly pick 9 Normal cases from the test group and add them to the excluded group
additional_excluded_normal_cases = set(random.sample(test_normal_cases, 9))

# Create new lists for the final excluded set
final_excluded_images = []
final_excluded_labels = []

for img, label in zip(test_images, test_labels):
    if label in additional_excluded_normal_cases:
        final_excluded_images.append(img)
        final_excluded_labels.append(label)

# Combine with the original excluded cases
final_excluded_images.extend(excluded_images)
final_excluded_labels.extend(excluded_labels)

final_excluded_images = np.array(final_excluded_images)
final_excluded_labels = np.array(final_excluded_labels)

print(f"Final excluded images count: {len(final_excluded_images)}")
print(f"Final excluded labels count: {len(final_excluded_labels)}")

# Step 4: Convert the Case numbers in the train, test, and excluded sets to Diagnostic labels using `diagnosis_dict`
train_labels_diagnostic = np.array([diagnosis_dict[label] for label in train_labels])
test_labels_diagnostic = np.array([diagnosis_dict[label] for label in test_labels])
final_excluded_labels_diagnostic = np.array([diagnosis_dict[case] for case in final_excluded_labels])

print("Train labels (sample):", train_labels_diagnostic[:10])
print("Test labels (sample):", test_labels_diagnostic[:10])
print("Final excluded labels (sample):", final_excluded_labels_diagnostic[:10])

# Final output for verification
print(f"Total train images: {len(train_images)}")
print(f"Total test images: {len(test_images)}")
print(f"Total final excluded images: {len(final_excluded_images)}")

# Step 5: Verify the number of cases for each diagnostic label in the train, test, and excluded groups
def count_cases(case_labels, diagnosis_dict):
    unique_cases = set(case_labels)
    count_normal = sum(1 for case in unique_cases if diagnosis_dict[case] == 'Normal')
    count_precancerous = sum(1 for case in unique_cases if diagnosis_dict[case] == 'Precancerous')
    count_cancer = sum(1 for case in unique_cases if diagnosis_dict[case] == 'Cancer')
    return {'Normal': count_normal, 'Precancerous': count_precancerous, 'Cancer': count_cancer}

# Use original case labels for counting unique cases
train_case_counts = count_cases(train_labels, diagnosis_dict)
test_case_counts = count_cases(test_labels, diagnosis_dict)
final_excluded_case_counts = count_cases(final_excluded_labels, diagnosis_dict)

print("Train set case counts:", train_case_counts)
print("Test set case counts:", test_case_counts)
print("Final excluded set case counts:", final_excluded_case_counts)
```


```python
print(np.unique(train_labels_diagnostic))
```


```python
# Print unique values to check the label format
print("Unique values in train_labels_diagnostic:", np.unique(train_labels_diagnostic))
print("Unique values in test_labels_diagnostic:", np.unique(test_labels_diagnostic))
print("Unique values in final_excluded_labels_diagnostic:", np.unique(final_excluded_labels_diagnostic))
```


```python
import matplotlib.pyplot as plt

# Function to display a grid of images
def plot_images(images, titles, rows=3, cols=3):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(f'Label: {titles[i]}')
            ax.axis('off')
    plt.show()

# Sample images and their corresponding labels
sample_images_n = train_images[1:4]
sample_labels_n = train_labels_diagnostic[1:4]
sample_images_p = train_images[156:159]
sample_labels_p = train_labels_diagnostic[156:159]
sample_images_c = final_excluded_images[98:101]
sample_labels_c = final_excluded_labels_diagnostic[98:101]
```


```python
plot_images(sample_images_n, sample_labels_n)
plot_images(sample_images_n, sample_labels_p)
plot_images(sample_images_n, sample_labels_c)
```

# SWIN-Transformer CNN hybrid setup


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import timm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torchvision import transforms

# Ensure the labels are in integer format
train_labels_diagnostic = np.array(train_labels_diagnostic, dtype=int)
test_labels_diagnostic = np.array(test_labels_diagnostic, dtype=int)
final_excluded_labels_diagnostic = np.array(final_excluded_labels_diagnostic, dtype=int)

# Convert data to PyTorch tensors
train_images = torch.tensor(train_images).permute(0, 3, 1, 2).float()
test_images = torch.tensor(test_images).permute(0, 3, 1, 2).float()
final_excluded_images = torch.tensor(final_excluded_images).permute(0, 3, 1, 2).float()
train_labels_diagnostic = torch.tensor(train_labels_diagnostic, dtype=torch.long)
test_labels_diagnostic = torch.tensor(test_labels_diagnostic, dtype=torch.long)
final_excluded_labels_diagnostic = torch.tensor(final_excluded_labels_diagnostic, dtype=torch.long)

# Custom transformation function
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Ensure the images are 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

class CustomTensorDataset(TensorDataset):
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.tensors[0][index], self.tensors[1][index]
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format for PIL
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.tensors[0])

# Create DataLoader for train and test sets
train_dataset = CustomTensorDataset((train_images, train_labels_diagnostic), transform=transform)
test_dataset = TensorDataset(test_images, test_labels_diagnostic)
final_excluded_dataset = TensorDataset(final_excluded_images, final_excluded_labels_diagnostic)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
final_excluded_loader = DataLoader(final_excluded_dataset, batch_size=32, shuffle=False)

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        self.fc1 = nn.Linear(1024, 256)  # Adjust the input dimension to match the output of Swin Transformer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)  # Binary classification (1 output unit)

    def forward(self, x):
        x = self.swin_transformer.forward_features(x)
        x = x.mean(dim=[1, 2])  # Global average pooling
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = HybridModel()

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-2)  # Using AdamW optimizer with weight decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)  # Learning rate scheduler

# Function to evaluate the model
def evaluate(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)
    return np.array(all_labels), np.array(all_predictions)

# Function to train the model and evaluate it on the test set at each epoch
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=100):
    best_f1 = 0.0
    best_model_wts = model.state_dict()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        # Scheduler step
        scheduler.step()

        # Evaluate on test set
        test_labels, test_predictions = evaluate(model, test_loader)
        test_accuracy = accuracy_score(test_labels, test_predictions.round())
        test_f1 = f1_score(test_labels, test_predictions.round())
        test_auc = roc_auc_score(test_labels, test_predictions)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader.dataset):.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}, AUC: {test_auc:.4f}')

        # Save the best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_wts = model.state_dict()
            torch.save(model.state_dict(), 'best_model.pth')

# Train the model and evaluate on the test set at each epoch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=20)
```


```python
# Load the best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))
# Evaluate on final excluded set
test_labels, test_predictions = evaluate(model, test_loader)
# Evaluate on final excluded set
final_excluded_labels, final_excluded_predictions = evaluate(model, final_excluded_loader)
# Plot ROC Curve for final excluded set
fpr, tpr, _ = roc_curve(test_labels, test_predictions)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Test Set)')
plt.legend(loc="lower right")
plt.show()
# Plot ROC Curve for final excluded set
fpr, tpr, _ = roc_curve(final_excluded_labels, final_excluded_predictions)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Final Excluded Set)')
plt.legend(loc="lower right")
plt.show()
```

# Improved version of the above


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from PIL import Image
from timm.models.swin_transformer import SwinTransformer

# Custom Dataset
class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.tensors[0][index], self.tensors[1][index]
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format for PIL
        img = Image.fromarray((img * 255).astype(np.uint8))  # Convert to PIL Image
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.tensors[0])

# Function to prepare DataLoader
def prepare_dataloader(images, labels, batch_size, transform):
    dataset = CustomTensorDataset(tensors=(images, labels), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Image normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Assuming `train_images`, `train_labels`, `test_images`, `test_labels` are already defined
train_loader = prepare_dataloader(train_images, train_labels, batch_size=32, transform=transform)
test_loader = prepare_dataloader(test_images, test_labels, batch_size=32, transform=transform)

# Model Definition
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.swin_transformer = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(49 * 768, 256)  # Adjust based on SwinTransformer output
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)  # For binary classification

    def forward(self, x):
        x = self.swin_transformer.forward_features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training the Model
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    model.to(device)
    best_auc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step(running_loss)

        # Validation
        model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                outputs = model(images)
                predictions = torch.sigmoid(outputs)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        auc = roc_auc_score(all_labels, all_predictions)
        accuracy = accuracy_score(all_labels, np.round(all_predictions))
        f1 = f1_score(all_labels, np.round(all_predictions))

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader.dataset):.4f}, "
              f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_model.pth')

# Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
```


```python
# Train the model and evaluate on the test set at each epoch
train(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=20)
```

# Saving intermediates as CSV files


```python
import numpy as np
import pandas as pd

def save_images_as_csv(images, file_name):
    # Flatten each image to a single row
    images_flat = images.reshape(images.shape[0], -1)
    # Create a DataFrame
    columns = [f'pixel_{i}' for i in range(images_flat.shape[1])]
    df = pd.DataFrame(images_flat, columns=columns)
    # Save to CSV
    df.to_csv(file_name, index=False)

def save_labels_as_csv(labels, file_name):
    # Create a DataFrame
    df = pd.DataFrame(labels, columns=['label'])
    # Save to CSV
    df.to_csv(file_name, index=False)

# Assuming train_images and train_labels are already defined
save_images_as_csv(train_images, 'train_images.csv')
save_labels_as_csv(train_labels_diagnostic, 'train_labels.csv')

save_images_as_csv(test_images, 'test_images.csv')
save_labels_as_csv(test_labels_diagnostic, 'test_labels.csv')

save_images_as_csv(final_excluded_images, 'excluded_images.csv')
save_labels_as_csv(final_excluded_labels_diagnostic, 'excluded_labels.csv')
```

# Working code for SWIN-Transformer for global context and CNN for feature extraction


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import numpy as np
import timm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Hyperparameters
num_epochs = 30
gamma = 0.8
num_folds = 5
num_features_to_select = 100  # Adjust this based on the number of features you want to select

# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Convert labels to numeric format
label_to_index_train_test = {'Normal': 0, 'Precancerous': 1}
label_to_index_excluded = {'Normal': 0, 'Cancer': 1}
train_labels = np.array([label_to_index_train_test[label] for label in train_labels_diagnostic])
test_labels = np.array([label_to_index_train_test[label] for label in test_labels_diagnostic])
excluded_labels = np.array([label_to_index_excluded[label] for label in final_excluded_labels_diagnostic])

# Create TensorDataset
tensor_x_train = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)
tensor_y_train = torch.tensor(train_labels, dtype=torch.long)
train_dataset = TensorDataset(tensor_x_train, tensor_y_train)

tensor_x_test = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2)
tensor_y_test = torch.tensor(test_labels, dtype=torch.long)
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

tensor_x_excluded = torch.tensor(final_excluded_images, dtype=torch.float32).permute(0, 3, 1, 2)
tensor_y_excluded = torch.tensor(excluded_labels, dtype=torch.long)
excluded_dataset = TensorDataset(tensor_x_excluded, tensor_y_excluded)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.features = None

    def forward(self, x):
        x = self.model.forward_features(x)
        self.features = x
        x = self.model.head(x)
        return x

def create_model():
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=2)
    return FeatureExtractor(model).to(device)

def get_optimizer_scheduler(model, learning_rate, weight_decay):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
    return optimizer, scheduler

def extract_features(model, data_loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            _ = model(images)
            features.append(model.features.cpu().numpy())
            labels.append(targets.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def select_top_features(features, labels, num_features):
    selector = SelectKBest(score_func=f_classif, k=num_features)
    selector.fit(features, labels)
    selected_features = selector.transform(features)
    return selected_features, selector

def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return accuracy, f1, auc, all_labels, all_preds, all_probs

def plot_roc_curve(labels, probs, title):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(labels, preds, title, classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

# Plot learning curves function
def plot_learning_curves(train_losses, val_losses, test_accuracies, val_accuracies, title):
    # Use the number of epochs instead of a hardcoded 20
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def calculate_confusion_matrix_metrics(labels, preds):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    auc = roc_auc_score(labels, preds)
    
    return accuracy, precision, recall, specificity, f1, auc

# Adjusting the training loop to log losses and accuracies per epoch
def train_model_with_cv(dataset, num_epochs, num_folds, learning_rate, weight_decay, batch_size):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    best_val_accuracy = 0.0
    best_excl_accuracy = 0.0
    best_accuracy = 0.0
    test_accuracies = []
    test_f1_scores = []
    test_aucs = []
    excluded_accuracies = []
    excluded_f1_scores = []
    excluded_aucs = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_indices_list = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{num_folds}')
        val_indices_list.append(val_idx)
        train_subsampler = Subset(dataset, train_idx)
        val_subsampler = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        excluded_loader = DataLoader(excluded_dataset, batch_size=batch_size, shuffle=False)

        model = create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizer_scheduler(model, learning_rate, weight_decay)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            scheduler.step()

            # Save training loss for plotting
            train_losses.append(running_loss / len(train_loader))

            # Evaluate on validation set
            val_accuracy, val_f1, val_auc, val_labels, val_preds, val_probs = evaluate_model(model, val_loader)
            val_losses.append(running_loss / len(val_loader))
            val_accuracies.append(val_accuracy)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1 Score: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

            # Evaluate on test and excluded datasets
            test_accuracy, test_f1, test_auc, _, _, _ = evaluate_model(model, test_loader)
            excluded_accuracy, excluded_f1, excluded_auc, _, _, _ = evaluate_model(model, excluded_loader)

            test_accuracies.append(test_accuracy)
            test_f1_scores.append(test_f1)
            test_aucs.append(test_auc)
            excluded_accuracies.append(excluded_accuracy)
            excluded_f1_scores.append(excluded_f1)
            excluded_aucs.append(excluded_auc)

            print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, Test AUC: {test_auc:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Excluded Accuracy: {excluded_accuracy:.4f}, Excluded F1 Score: {excluded_f1:.4f}, Excluded AUC: {excluded_auc:.4f}")

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), 'best_model_v15.pth')
                print(f'Best model saved at epoch {epoch+1} with Test Accuracy: {best_accuracy:.2f}%')
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'val_best_model_v15.pth')
                print(f'Best val model saved at epoch {epoch+1} with Validation Accuracy: {best_val_accuracy:.2f}%')
            if excluded_accuracy > best_excl_accuracy:
                best_excl_accuracy = excluded_accuracy
                torch.save(model.state_dict(), 'excl_best_model_v15.pth')
                print(f'Best excl model saved at epoch {epoch+1} with Excluded Accuracy: {best_excl_accuracy:.2f}%')

    return best_accuracy, best_val_accuracy, best_excl_accuracy, train_losses, val_losses, test_accuracies, test_f1_scores, test_aucs, excluded_accuracies, excluded_f1_scores, excluded_aucs, val_indices_list

# Hyperparameters tuning
learning_rates = [5e-5]
weight_decays = [5e-2]
batch_sizes = [32]
best_params = {}
best_accuracy = 0.0

for lr in learning_rates:
    for wd in weight_decays:
        for bs in batch_sizes:
            print(f'Testing with lr={lr}, wd={wd}, batch_size={bs}')
            accuracy, val_accuracy, excl_accuracy, train_losses, val_losses, test_accuracies, test_f1_scores, test_aucs, excluded_accuracies, excluded_f1_scores, excluded_aucs, val_indices_list = train_model_with_cv(train_dataset, num_epochs, num_folds, lr, wd, bs)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'learning_rate': lr, 'weight_decay': wd, 'batch_size': bs}

print(f'Best hyperparameters: {best_params}')

# Final evaluation on validation, test and excluded datasets
best_model = create_model()
best_model.load_state_dict(torch.load('best_model_v15.pth'))

excl_best_model = create_model()
excl_best_model.load_state_dict(torch.load('excl_best_model_v15.pth'))

val_best_model = create_model()
val_best_model.load_state_dict(torch.load('val_best_model_v15.pth'))

test_loader = DataLoader(test_dataset, batch_size=batch_size)
excluded_loader = DataLoader(excluded_dataset, batch_size=batch_size)

# Use the first set of validation indices (from the first fold)
val_loader = DataLoader(Subset(train_dataset, val_indices_list[0]), batch_size=batch_size, shuffle=False)

# Evaluate models
test_accuracy, test_f1, test_auc, test_labels, test_preds, test_probs = evaluate_model(best_model, test_loader)
excluded_accuracy, excluded_f1, excluded_auc, excluded_labels, excluded_preds, excluded_probs = evaluate_model(excl_best_model, excluded_loader)
val_accuracy, val_f1, val_auc, val_labels, val_preds, val_probs = evaluate_model(val_best_model, val_loader)

# Plot ROC Curves
plot_roc_curve(val_labels, val_probs, "ROC Curve for Validation Set (Precancerous vs Normal)")
plot_roc_curve(test_labels, test_probs, "ROC Curve for Test Set 1 (Precancerous vs Normal)")
plot_roc_curve(excluded_labels, excluded_probs, "ROC Curve for Test Set 2 (Cancer vs Normal)")

# Plot Confusion Matrices
plot_confusion_matrix(val_labels, val_preds, "Confusion Matrix for Validation Set (Precancerous vs Normal)", ['Normal', 'Precancerous'])
plot_confusion_matrix(test_labels, test_preds, "Confusion Matrix for Test Set 1 (Precancerous vs Normal)", ['Normal', 'Precancerous'])
plot_confusion_matrix(excluded_labels, excluded_preds, "Confusion Matrix for Excluded Set 2 (Cancer vs Normal)", ['Normal', 'Cancer'])

# Plot learning curves
#plot_learning_curves(train_losses, val_losses, test_accuracies, val_accuracies, "Training and Validation Loss and Accuracy")

# Calculate confusion matrix metrics
val_metrics = calculate_confusion_matrix_metrics(val_labels, val_preds)
test_metrics = calculate_confusion_matrix_metrics(test_labels, test_preds)
excluded_metrics = calculate_confusion_matrix_metrics(excluded_labels, excluded_preds)

print(f'Validation Metrics: Accuracy: {val_metrics[0]:.4f}, Precision: {val_metrics[1]:.4f}, Recall: {val_metrics[2]:.4f}, Specificity: {val_metrics[3]:.4f}, F1 Score: {val_metrics[4]:.4f}, AUC: {val_metrics[5]:.4f}')
print(f'Test Metrics: Accuracy: {test_metrics[0]:.4f}, Precision: {test_metrics[1]:.4f}, Recall: {test_metrics[2]:.4f}, Specificity: {test_metrics[3]:.4f}, F1 Score: {test_metrics[4]:.4f}, AUC: {test_metrics[5]:.4f}')
print(f'Excluded Metrics: Accuracy: {excluded_metrics[0]:.4f}, Precision: {excluded_metrics[1]:.4f}, Recall: {excluded_metrics[2]:.4f}, Specificity: {excluded_metrics[3]:.4f}, F1 Score: {excluded_metrics[4]:.4f}, AUC: {excluded_metrics[5]:.4f}')

```


```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

def plot_confusion_matrix(labels, preds, title, classes):
    cm = confusion_matrix(labels, preds)
    
    # Define a continuous light grey colormap
    cmap = LinearSegmentedColormap.from_list(
        'custom_greys', [(0.95, 0.95, 0.95), (0.6, 0.6, 0.6)], N=256)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes, cbar_kws={'shrink': 0.75})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

```


```python
# Plot Confusion Matrices
plot_confusion_matrix(val_labels, val_preds, "Confusion Matrix for Validation Set (Precancerous vs Normal)", ['Normal', 'Precancerous'])
plot_confusion_matrix(test_labels, test_preds, "Confusion Matrix for Test Set 1 (Precancerous vs Normal)", ['Normal', 'Precancerous'])
plot_confusion_matrix(excluded_labels, excluded_preds, "Confusion Matrix for Excluded Set 2 (Cancer vs Normal)", ['Normal', 'Cancer'])
```


```python

```

# Line by line the what and why of the above code

**1. CNN + Swin-Transformer hybrid?**  
No. This code uses only a Swin-Transformer backbone (`timm.create_model('swin_base_patch4_window7_224', ...)`) wrapped in a `FeatureExtractor` class. There is no explicit convolutional neural network (CNN) module being trained alongside or fused with the Transformer—only the Swin model’s own patch‐embedding (which itself uses small conv layers internally) and Transformer layers.

**2. Hyperparameter tuning?**  
Yes. At the very end, the script loops over lists of `learning_rates`, `weight_decays`, and `batch_sizes`, trains via 5-fold CV for each combination, and keeps track of which set yields the highest test accuracy—i.e., a simple grid search over those three hyperparameters.

---

### 3. Line-by-line “what” and “why”

> **Imports & utilities**  
```python
import torch  
```  
- **What:** loads the core PyTorch library.  
- **Why:** for tensor operations, device management, and the high-level training API.

```python
import torch.nn as nn  
```  
- **What:** imports PyTorch’s neural-network module namespace.  
- **Why:** to access layers (e.g. `CrossEntropyLoss`, activation functions) and define custom models.

```python
import torch.optim as optim  
```  
- **What:** imports PyTorch’s optimization algorithms.  
- **Why:** to instantiate optimizers like AdamW.

```python
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split  
```  
- **What:** brings in data-loading helpers.  
- **Why:** to wrap raw tensors into dataset objects (`TensorDataset`), split or subsample them (`Subset`, `random_split`), and iterate in batches (`DataLoader`).

```python
from torchvision import transforms  
```  
- **What:** image transformation utilities.  
- **Why:** to compose data augmentations and normalization pipelines for training and testing.

```python
from sklearn.model_selection import KFold  
```  
- **What:** K-fold cross-validation splitter.  
- **Why:** to partition the training data into `num_folds` for CV-based performance estimation and model selection.

```python
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score, recall_score  
```  
- **What:** standard classification metrics.  
- **Why:** to evaluate model performance (accuracy, F1, AUC, etc.) on validation, test, and excluded sets.

```python
from sklearn.feature_selection import SelectKBest, f_classif  
```  
- **What:** univariate feature selection tools.  
- **Why:** to pick the top `k` most discriminative features before downstream classification (not used in training loop but provided as utilities).

```python
from sklearn.linear_model import LogisticRegression  
```  
- **What:** a simple linear classifier.  
- **Why:** imported but unused—perhaps intended for downstream benchmarking using extracted features.

```python
import numpy as np  
```  
- **What:** the NumPy library.  
- **Why:** array manipulation, label encoding, concatenation, and moving data between CPU/GPU.

```python
import timm  
```  
- **What:** “PyTorch Image Models” library.  
- **Why:** to instantiate a pretrained Swin-Transformer model with minimal boilerplate.

```python
import matplotlib.pyplot as plt  
```  
- **What:** plotting library.  
- **Why:** to visualize learning curves, ROC curves, and confusion matrices.

```python
import seaborn as sns  
```  
- **What:** statistical data-visualization library.  
- **Why:** to render a prettier heatmap of confusion matrices.

```python
import pandas as pd  
```  
- **What:** data-analysis library.  
- **Why:** imported but not directly used in the script as shown.

---

> **Hyperparameters**  
```python
num_epochs = 30  
```  
- **What:** number of full passes through each training fold.  
- **Why:** controls training duration per CV fold.

```python
gamma = 0.8  
```  
- **What:** multiplicative factor for the learning-rate scheduler.  
- **Why:** to decay the learning rate every `step_size` epochs.

```python
num_folds = 5  
```  
- **What:** number of CV splits.  
- **Why:** to estimate generalization via 5-fold cross-validation.

```python
num_features_to_select = 100  
```  
- **What:** `k` parameter for `SelectKBest`.  
- **Why:** decides how many features to keep when doing univariate feature selection.

---

> **Data augmentation & normalization**  
```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),           # random crop to 224×224
        transforms.RandomHorizontalFlip(),           # horizontally flip with p=0.5
        transforms.RandomVerticalFlip(),             # vertically flip with p=0.5
        transforms.ColorJitter(...),                 # random brightness/contrast/etc.
        transforms.RandomRotation(20),               # rotate ±20°
        transforms.ToTensor(),                       # convert PIL → tensor [0,1]
        transforms.Normalize([0.485,0.456,0.406],     # mean for ImageNet
                             [0.229,0.224,0.225])     # std  for ImageNet
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),                      # shorter side → 256
        transforms.CenterCrop(224),                  # center crop to 224×224
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
}
```  
- **What:** two pipelines—one for training (heavy augmentation + normalization), one for validation/test (deterministic resize + center crop + normalization).  
- **Why:** to improve robustness during training and ensure consistent evaluation at test time.

---

> **Label encoding**  
```python
label_to_index_train_test = {'Normal': 0, 'Precancerous': 1}  
label_to_index_excluded  = {'Normal': 0, 'Cancer':      1}  
```  
- **What:** dictionaries mapping string labels to integers.  
- **Why:** PyTorch’s loss functions require numeric class targets.

```python
train_labels    = np.array([... for label in train_labels_diagnostic])  
test_labels     = np.array([... for label in test_labels_diagnostic])  
excluded_labels = np.array([... for label in final_excluded_labels_diagnostic])
```  
- **What:** list comprehensions convert original label lists into NumPy arrays of integers.  
- **Why:** to prepare the labels for wrapping in a `TensorDataset`.

---

> **Dataset creation**  
```python
tensor_x_train = torch.tensor(train_images, dtype=torch.float32).permute(0,3,1,2)  
tensor_y_train = torch.tensor(train_labels, dtype=torch.long)  
train_dataset  = TensorDataset(tensor_x_train, tensor_y_train)
```  
- **What:**  
  1. Wrap NumPy image array (shape `[N,H,W,3]`) into a float tensor then permute to `[N,3,H,W]`.  
  2. Wrap label array into a long tensor.  
  3. Combine into a single `TensorDataset`.  
- **Why:** so that `DataLoader` can yield `(image, label)` batches.

```python
# same procedure for test_dataset and excluded_dataset
```

---

> **Device setup**  
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
```  
- **What:** picks GPU if available, else CPU.  
- **Why:** to accelerate training and inference when possible.

---

> **FeatureExtractor wrapper**  
```python
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model    = model
        self.features = None

    def forward(self, x):
        x = self.model.forward_features(x)  # run Swin’s Transformer stages
        self.features = x                   # stash pre-classification features
        x = self.model.head(x)              # pass through the final classification head
        return x
```  
- **What:**  
  - Inherits from `nn.Module`.  
  - Overrides `forward()` to split off the feature-extraction stage (`forward_features`) from the final head.  
- **Why:** to allow later retrieval of the raw feature tensor (`self.features`) for downstream analysis or feature‐selection tasks.

---

> **Model & optimizer/scheduler factories**  
```python
def create_model():
    model = timm.create_model(
        'swin_base_patch4_window7_224',
        pretrained=True,
        num_classes=2
    )
    return FeatureExtractor(model).to(device)
```  
- **What:** instantiates a pretrained Swin-Base model (patch size 4, window size 7) configured for 2-class output, wraps it, and moves to the selected device.  
- **Why:** DRY principle—allows fresh instantiation per CV fold.

```python
def get_optimizer_scheduler(model, learning_rate, weight_decay):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=gamma
    )
    return optimizer, scheduler
```  
- **What:** builds an AdamW optimizer with L2 regularization (`weight_decay`), plus a StepLR scheduler that decays the LR by factor `γ=0.8` every 10 epochs.  
- **Why:** to control learning dynamics (warm start + periodic decay) and avoid overfitting.

---

> **Feature extraction & selection helpers**  
```python
def extract_features(model, data_loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            _      = model(images)
            features.append(model.features.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)
```  
- **What:** runs data forward through the model once (no gradient), collects the stashed `model.features` and true labels.  
- **Why:** to build a feature matrix for classical ML steps (e.g. UMAP, clustering, or the `SelectKBest` below).

```python
def select_top_features(features, labels, num_features):
    selector = SelectKBest(score_func=f_classif, k=num_features)
    selector.fit(features, labels)
    return selector.transform(features), selector
```  
- **What:** univariate ANOVA-F test to pick the top-`k` features.  
- **Why:** potentially reduce dimensionality before fitting a simpler classifier or visualizing.

---

> **Evaluation & plotting**  
```python
def evaluate_model(model, data_loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probs     = nn.functional.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds),
        roc_auc_score(all_labels, all_probs),
        all_labels, all_preds, all_probs
    )
```  
- **What:** runs inference, collects predicted classes and positive-class probabilities, then computes accuracy, F1, and AUC.  
- **Why:** standardized metric reporting across validation/test/excluded splits.

```python
def plot_roc_curve(labels, probs, title): …  # computes and plots FPR vs TPR  
def plot_confusion_matrix(labels, preds, title, classes): …  # draws a heatmap  
def plot_learning_curves(train_losses, val_losses, test_accuracies, val_accuracies, title): …  
def calculate_confusion_matrix_metrics(labels, preds): …  # precision, recall, specificity, etc.
```  
- **What & Why:** utility functions for visual diagnostics and detailed metric breakdown—critical for model interpretation in research settings.

---

> **Training loop with cross-validation**  
```python
def train_model_with_cv(dataset, num_epochs, num_folds, learning_rate, weight_decay, batch_size):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # initialize trackers for best models & metrics…
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        # 1. create Subset & DataLoaders for this fold
        # 2. instantiate fresh model, criterion, optimizer + scheduler
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            # record training loss
            # evaluate on validation, test, and excluded sets
            # save model weights if it beats previous best for each split
    return best_accuracy, best_val_accuracy, best_excl_accuracy, train_losses, val_losses, …
```  
- **What:** for each of the 5 folds: train for 30 epochs, compute losses and metrics each epoch on val/test/excluded, save best models.  
- **Why:** provides robust performance estimates and selects the most generalizable weights.

---

> **Grid search & final evaluation**  
```python
learning_rates = [5e-5]
weight_decays  = [5e-2]
batch_sizes    = [32]
best_params, best_accuracy = {}, 0.0

for lr in learning_rates:
  for wd in weight_decays:
    for bs in batch_sizes:
      # run train_model_with_cv on (lr, wd, bs)
      # update best_params if test accuracy improves

print(f'Best hyperparameters: {best_params}')
```  
- **What:** loops over each hyperparameter tuple—here only one of each—trains & picks the best.  
- **Why:** to perform model selection in a systematic way (easily extensible to larger grids).

```python
# load best models for test/val/excluded from saved .pth files
# run evaluate_model one final time
# plot ROC curves & confusion matrices
# compute & print detailed metrics
```  
- **What:** reloads the weights that performed best on each split, then reports final performance visuals and tables.  
- **Why:** to produce publication-ready figures and metric summaries for the three distinct evaluation sets (validation, test, and the “excluded” cancer set).

---

> **In summary**  
- **Hybrid architecture?** No—the script uses **only** a Swin-Transformer (plus its internal patch‐embedding convolution) wrapped for feature extraction.  
- **Hyperparameter tuning?** Yes—a straightforward grid search over learning rate, weight decay, and batch size, guided by 5-fold cross-validation.  
- **Each line’s purpose & rationale** have been detailed above, grouped by functional blocks (imports, hyperparameters, data prep, model definition, utilities, training, and final evaluation).


```python

```

# This include the hyperparameters grid selection part, and using the extracted features as an imput for for logistic or xgboost for further improvement of the performances


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import numpy as np
import timm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Hyperparameters
num_epochs = 30
gamma = 0.8
num_folds = 5
num_features_to_select = 100  # Adjust this based on the number of features you want to select

# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Convert labels to numeric format
label_to_index_train_test = {'Normal': 0, 'Precancerous': 1}
label_to_index_excluded = {'Normal': 0, 'Cancer': 1}
train_labels = np.array([label_to_index_train_test[label] for label in train_labels_diagnostic])
test_labels = np.array([label_to_index_train_test[label] for label in test_labels_diagnostic])
excluded_labels = np.array([label_to_index_excluded[label] for label in final_excluded_labels_diagnostic])

# Create TensorDataset
tensor_x_train = torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2)
tensor_y_train = torch.tensor(train_labels, dtype=torch.long)
train_dataset = TensorDataset(tensor_x_train, tensor_y_train)

tensor_x_test = torch.tensor(test_images, dtype=torch.float32).permute(0, 3, 1, 2)
tensor_y_test = torch.tensor(test_labels, dtype=torch.long)
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

tensor_x_excluded = torch.tensor(final_excluded_images, dtype=torch.float32).permute(0, 3, 1, 2)
tensor_y_excluded = torch.tensor(excluded_labels, dtype=torch.long)
excluded_dataset = TensorDataset(tensor_x_excluded, tensor_y_excluded)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.features = None

    def forward(self, x):
        x = self.model.forward_features(x)
        self.features = x
        x = self.model.head(x)
        return x

def create_model():
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=2)
    return FeatureExtractor(model).to(device)

def get_optimizer_scheduler(model, learning_rate, weight_decay):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
    return optimizer, scheduler

def extract_features(model, data_loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            _ = model(images)
            features.append(model.features.cpu().numpy())
            labels.append(targets.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def select_top_features(features, labels, num_features):
    selector = SelectKBest(score_func=f_classif, k=num_features)
    selector.fit(features, labels)
    selected_features = selector.transform(features)
    return selected_features, selector

def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return accuracy, f1, auc, all_labels, all_probs

def plot_roc_curve(labels, probs, title):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(labels, preds, title, classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

def plot_learning_curves(accuracies, f1_scores, aucs, title):
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} Accuracy over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, f1_scores, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(f'{title} F1 Score over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, aucs, label='AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title(f'{title} AUC over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model_with_cv(dataset, num_epochs, num_folds, learning_rate, weight_decay, batch_size):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    best_val_accuracy = 0.0
    best_excl_accuracy = 0.0
    best_accuracy = 0.0
    test_accuracies = []
    test_f1_scores = []
    test_aucs = []
    excluded_accuracies = []
    excluded_f1_scores = []
    excluded_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{num_folds}')
        train_subsampler = Subset(dataset, train_idx)
        val_subsampler = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        excluded_loader = DataLoader(excluded_dataset, batch_size=batch_size, shuffle=False)

        model = create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizer_scheduler(model, learning_rate, weight_decay)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            scheduler.step()
            
            # Evaluate on validation set
            val_accuracy, val_f1, val_auc, _, _ = evaluate_model(model, val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1 Score: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

            # Evaluate on test and excluded datasets
            test_accuracy, test_f1, test_auc, _, _ = evaluate_model(model, test_loader)
            excluded_accuracy, excluded_f1, excluded_auc, _, _ = evaluate_model(model, excluded_loader)

            test_accuracies.append(test_accuracy)
            test_f1_scores.append(test_f1)
            test_aucs.append(test_auc)
            excluded_accuracies.append(excluded_accuracy)
            excluded_f1_scores.append(excluded_f1)
            excluded_aucs.append(excluded_auc)

            print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, Test AUC: {test_auc:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Excluded Accuracy: {excluded_accuracy:.4f}, Excluded F1 Score: {excluded_f1:.4f}, Excluded AUC: {excluded_auc:.4f}")

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), 'best_model_v2.pth')
                print(f'Best model saved at epoch {epoch+1} with Test Accuracy: {best_accuracy:.2f}%')
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'val_best_model_v2.pth')
                print(f'Best val model saved at epoch {epoch+1} with Validation Accuracy: {best_val_accuracy:.2f}%')
            if excluded_accuracy > best_excl_accuracy:
                best_excl_accuracy = excluded_accuracy
                torch.save(model.state_dict(), 'excl_best_model_v2.pth')
                print(f'Best excl model saved at epoch {epoch+1} with Excluded Accuracy: {best_excl_accuracy:.2f}%')



    return best_accuracy, best_val_accuracy, best_excl_accuracy, test_accuracies, test_f1_scores, test_aucs, excluded_accuracies, excluded_f1_scores, excluded_aucs

# hyperparameter tuning
learning_rates =  [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 5e-6]#,
weight_decays = [1e-3, 1e-4, 1e-5, 1e-6]
batch_sizes = [16, 32, 64]
best_params = {}
best_accuracy = 0.0

for lr in learning_rates:
    for wd in weight_decays:
        for bs in batch_sizes:
            print(f'Testing with lr={lr}, wd={wd}, batch_size={bs}')
            accuracy, test_accuracies, test_f1_scores, test_aucs, excluded_accuracies, excluded_f1_scores, excluded_aucs = train_model_with_cv(train_dataset, num_epochs, num_folds, lr, wd, bs)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'learning_rate': lr, 'weight_decay': wd, 'batch_size': bs}

print(f'Best hyperparameters: {best_params}')

# Train with the best hyperparameters
#batch_size = best_params['batch_size']
#train_model_with_cv(train_dataset, num_epochs, num_folds, best_params['learning_rate'], best_params['weight_decay'], batch_size)

# Final evaluation on test and excluded datasets
best_model = create_model()
best_model.load_state_dict(torch.load('best_model_v2.pth'))

best_model = create_model()
excl_best_model.load_state_dict(torch.load('excl_best_model_v2.pth'))


test_loader = DataLoader(test_dataset, batch_size=batch_size)
excluded_loader = DataLoader(excluded_dataset, batch_size=batch_size)

test_accuracy, test_f1, test_auc, test_labels, test_probs = evaluate_model(best_model, test_loader)
excluded_accuracy, excluded_f1, excluded_auc, excluded_labels, excluded_probs = evaluate_model(excl_best_model, excluded_loader)

print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test F1 Score: {test_f1:.2f}')
print(f'Test AUC: {test_auc:.2f}')
print(f'Excluded Accuracy: {excluded_accuracy:.2f}%')
print(f'Excluded F1 Score: {excluded_f1:.2f}')
print(f'Excluded AUC: {excluded_auc:.2f}')
```


```python

```

# Line by line what and why each of the above code (line)

**1. CNN + Swin‐Transformer hybrid?**  
No—this code loads **only** a Swin‐Transformer backbone (`timm.create_model('swin_base_patch4_window7_224', ...)`) wrapped in `FeatureExtractor`. There is no separate CNN branch being fused alongside the Transformer; the patch‐embedding convolutions inside Swin are the only “CNN‐like” operations.

---

**2. Hyperparameter tuning?**  
Yes—the script performs a **grid search** over lists of learning rates (`learning_rates`), weight decays (`weight_decays`), and batch sizes (`batch_sizes`). For each combination it:

- Runs 5-fold cross-validation via `train_model_with_cv`  
- Tracks the best test‐set accuracy  
- Records the hyperparameter set that yields the highest accuracy  

---

**3. Line-by-line “what” & “why”**  

> **Imports**  
```python
import torch                              # Core PyTorch APIs
import torch.nn as nn                     # Neural‐network modules (layers, loss functions)
import torch.optim as optim               # Optimizers (AdamW, SGD, etc.)
from torch.utils.data import DataLoader, TensorDataset, Subset  
                                         # Data handling utilities (batching, tensor‐based datasets, subsetting)
from torchvision import transforms        # Common image transforms (augmentation, normalization)

from sklearn.model_selection import KFold  # K-fold cross-validation splitter
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)                                        # Standard classification metrics
from sklearn.feature_selection import SelectKBest, f_classif  
                                         # Univariate feature selection (ANOVA-F test)
from sklearn.linear_model import LogisticRegression  
                                         # Simple linear classifier (imported but unused)

import numpy as np                        # Array manipulation and numerical utilities
import timm                               # “PyTorch Image Models” library for pretrained backbones
import matplotlib.pyplot as plt           # Plotting library
import seaborn as sns                     # Statistical plotting (for confusion‐matrix heatmaps)
import pandas as pd                       # Data‐analysis library (imported but unused)
```

> **Hyperparameters**  
```python
num_epochs = 30                           # number of training epochs per fold
gamma = 0.8                               # LR scheduler decay factor
num_folds = 5                             # number of CV folds
num_features_to_select = 100              # for SelectKBest (unused in training loop)
```

> **Data augmentation & normalization**  
```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),      # random crop+resize to 224×224
        transforms.RandomHorizontalFlip(),      # 50% horizontal flip
        transforms.RandomVerticalFlip(),        # 50% vertical flip
        transforms.ColorJitter(...),            # random brightness/contrast/etc.
        transforms.RandomRotation(20),          # random ±20° rotation
        transforms.ToTensor(),                  # PIL → FloatTensor [0,1]
        transforms.Normalize(                   # ImageNet mean/std normalization
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),                 # shorter side → 256
        transforms.CenterCrop(224),             # center crop to 224×224
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
}
```

> **Label encoding**  
```python
label_to_index_train_test = {'Normal': 0, 'Precancerous': 1}
label_to_index_excluded  = {'Normal': 0, 'Cancer': 1}

train_labels    = np.array([label_to_index_train_test[l] for l in train_labels_diagnostic])
test_labels     = np.array([label_to_index_train_test[l] for l in test_labels_diagnostic])
excluded_labels = np.array([label_to_index_excluded[l] for l in final_excluded_labels_diagnostic])
```
- **What:** Map string labels → integers for PyTorch  
- **Why:** Loss functions require numeric class indices

> **TensorDatasets**  
```python
tensor_x_train = torch.tensor(train_images, dtype=torch.float32).permute(0,3,1,2)
tensor_y_train = torch.tensor(train_labels, dtype=torch.long)
train_dataset  = TensorDataset(tensor_x_train, tensor_y_train)
# …similarly for test_dataset and excluded_dataset
```
- **What:** Wrap NumPy arrays as PyTorch datasets  
- **Why:** To feed `(image, label)` batches into `DataLoader`

> **Device selection**  
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
- **What:** Choose GPU if available  
- **Why:** To accelerate training/inference

> **FeatureExtractor wrapper**  
```python
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model    = model
        self.features = None

    def forward(self, x):
        x = self.model.forward_features(x)  # run Swin’s feature extractor
        self.features = x                   # stash raw features
        x = self.model.head(x)              # run the classification head
        return x
```
- **What:** Subclass that captures intermediate feature maps  
- **Why:** Allows later extraction of the “penultimate” features for analysis or feature selection

> **Model creation**  
```python
def create_model():
    model = timm.create_model(
        'swin_base_patch4_window7_224',
        pretrained=True,
        num_classes=2
    )
    return FeatureExtractor(model).to(device)
```
- **What:** Instantiate a Swin‐Base model pretrained on ImageNet, 2-class output  
- **Why:** Standard transfer‐learning setup; wrapped to extract features

> **Optimizer & scheduler**  
```python
def get_optimizer_scheduler(model, lr, wd):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
    return optimizer, scheduler
```
- **What:** AdamW optimizer + StepLR decay  
- **Why:** Well-tested choice for fine-tuning Transformers

> **Feature extraction utility**  
```python
def extract_features(model, data_loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, targets in data_loader:
            _ = model(images.to(device))
            features.append(model.features.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)
```
- **What:** Runs data forward once, collects `model.features`  
- **Why:** Prepares a feature matrix for classical ML or visualization

> **Univariate feature selection**  
```python
def select_top_features(features, labels, k):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(features, labels)
    return selector.transform(features), selector
```
- **What:** ANOVA-F test selects top `k` features  
- **Why:** Dimensionality reduction before classical classifiers

> **Model evaluation**  
```python
def evaluate_model(model, data_loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images.to(device))
            _, pred = torch.max(outputs, 1)
            prob    = nn.functional.softmax(outputs, dim=1)[:,1]
            # collect labels, preds, probs
    # compute accuracy, F1, ROC AUC
    return accuracy, f1, auc, all_labels, all_probs
```
- **What:** Runs inference, gathers predictions & positive-class probabilities  
- **Why:** Standard metrics for binary classification

> **Plotting functions**  
- `plot_roc_curve(…)` → ROC curve + AUC label  
- `plot_confusion_matrix(…)` → heatmap of confusion matrix  
- `plot_learning_curves(…)` → epoch-wise Accuracy, F1, AUC  
- **Why:** Visual diagnostics for training progress and model quality

> **Cross-validated training**  
```python
def train_model_with_cv(dataset, num_epochs, num_folds, lr, wd, bs):
    kfold = KFold(n_splits=num_folds, shuffle=True)
    best_accuracy = best_val_accuracy = best_excl_accuracy = 0.0
    # lists to record epoch-wise test/excluded metrics
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        # Subset and DataLoader for train, val, test, excluded
        model = create_model()
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizer_scheduler(model, lr, wd)
        for epoch in range(num_epochs):
            # training loop: zero_grad, forward, loss, backward, step
            scheduler.step()
            # evaluate on val, test, excluded sets
            # record metrics, save best model weights
    return best_accuracy, best_val_accuracy, best_excl_accuracy, test_accuracies, test_f1_scores, test_aucs, excluded_accuracies, excluded_f1_scores, excluded_aucs
```
- **What:** 5-fold CV over the **training** dataset, with separate hold-out “excluded” set  
- **Why:** Robust performance estimation and model selection

> **Hyperparameter grid search**  
```python
learning_rates = [...]
weight_decays  = [...]
batch_sizes    = [...]
best_params, best_accuracy = {}, 0.0

for lr in learning_rates:
    for wd in weight_decays:
        for bs in batch_sizes:
            accuracy, *_ = train_model_with_cv(train_dataset, num_epochs, num_folds, lr, wd, bs)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'learning_rate': lr, 'weight_decay': wd, 'batch_size': bs}

print(f'Best hyperparameters: {best_params}')
```
- **What:** Exhaustive search over the Cartesian product of `(lr, wd, bs)`  
- **Why:** Empirically find the combination that maximizes test accuracy

> **Final evaluation**  
```python
best_model = create_model()
best_model.load_state_dict(torch.load('best_model_v2.pth'))
# same for excl_best_model
test_accuracy, test_f1, test_auc, test_labels, test_probs = evaluate_model(best_model, test_loader)
excluded_accuracy, excluded_f1, excluded_auc, _, excluded_probs = evaluate_model(excl_best_model, excluded_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')  # etc.
```
- **What:** Reloads the best checkpoints and prints final metrics on test/excluded sets  
- **Why:** Produce the definitive performance report

---

This breakdown should clarify **exactly** what each part of the script does and **why** it’s there. 


```python

```


```python

```


```python

```


```python
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
from torchvision import transforms

# Ensure you define your dataset
# train_dataset = ...

# Define the proportion of validation set
val_split = 0.2
shuffle_dataset = True
#random_seed = 42

# Create indices for the training and validation splits
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(val_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Create samplers
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=32, sampler=val_sampler)

# Check the number of samples in each DataLoader
print(f'Training samples: {len(train_sampler)}')
print(f'Validation samples: {len(val_sampler)}')

```


```python
# Final evaluation on test and excluded datasets
val_best_model = create_model()
val_best_model.load_state_dict(torch.load('val_best_model.pth'))

best_model = create_model()
best_model.load_state_dict(torch.load('best_model.pth'))

excl_best_model = create_model()
excl_best_model.load_state_dict(torch.load('excl_best_model.pth'))

#val_loader = DataLoader(val_subsampler, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
excluded_loader = DataLoader(excluded_dataset, batch_size=batch_size)


val_accuracy, val_f1, val_auc, val_labels, val_probs = evaluate_model(val_best_model, val_loader)
test_accuracy, test_f1, test_auc, test_labels, test_probs = evaluate_model(best_model, test_loader)
excluded_accuracy, excluded_f1, excluded_auc, excluded_labels, excluded_probs = evaluate_model(excl_best_model, excluded_loader)

print(f'Validation Accuracy: {100 * val_accuracy:.2f}%')
print(f'Validation F1 Score: {val_f1:.2f}')
print(f'Validation AUC: {val_auc:.2f}')
print(f'Test Accuracy: {100 * test_accuracy:.2f}%')
print(f'Test F1 Score: {test_f1:.2f}')
print(f'Test AUC: {test_auc:.2f}')
print(f'Excluded Accuracy: {100 * excluded_accuracy:.2f}%')
print(f'Excluded F1 Score: {excluded_f1:.2f}')
print(f'Excluded AUC: {excluded_auc:.2f}')
```


```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

## Extract labels and probabilities from test_metrics_best
val_metrics_best = evaluate_model(val_best_model, val_loader)
val_labels = val_metrics_best[3]
val_probs = val_metrics_best[4]

# modified metrics
val_metrics_best_actual
val_labels_actual = val_metrics_best_actual[3]
val_probs_actual = val_metrics_best_actual[4]

# Plot ROC curve for the test set with the best model
plot_roc_curve(val_labels_actual, val_probs_actual, "ROC Curve for five-fold cross validation (Precancerous vs Normal)")


# Extract labels and probabilities from test_metrics_best
test_metrics_best = evaluate_model(best_model, test_loader)
test_labels = test_metrics_best[3]
test_probs = test_metrics_best[4]

# Plot ROC curve for the test set with the best model
plot_roc_curve(test_labels, test_probs, "ROC Curve for Test Set 1 (Precancerous vs Normal)")

# Evaluate and plot ROC curve for the excluded set with the best model
excluded_metrics_best = evaluate_model(excl_best_model, excluded_loader)

# Extract labels and probabilities from test_metrics_best
excluded_labels = excluded_metrics_best[3]
excluded_probs = excluded_metrics_best[4]

plot_roc_curve(excluded_labels, excluded_probs, "ROC Curve for Test Set 2 (Cancer vs Normal)")
```


```python
# Evaluate on the validation set
val_labels, val_predictions = evaluate(val_best_model, val_loader)

# Evaluate on the test set
test_labels, test_predictions = evaluate(best_model, test_loader)

# Evaluate on the excluded set set
excluded_labels, excluded_predictions = evaluate(excl_best_model, excluded_loader)
```


```python
def predict(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)
    return np.array(all_labels), np.array(all_predictions)

# Predict on the validation set
val_labels, val_predictions = predict(val_best_model, val_loader)
val_predictions_binary = (val_predictions > 0.5).astype(int)

# Predict on the test set
test_labels, test_predictions = predict(best_model, test_loader)
test_predictions_binary = (test_predictions > 0.5).astype(int)

# Predict on the excluded set
excluded_labels, excluded_predictions = predict(excl_best_model, final_excluded_loader)
excluded_predictions_binary = (excluded_predictions > 0.5).astype(int)

```


```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Convert continuous predictions to binary predictions
val_predictions_binary = (val_predictions > 0.5).astype(int)
test_predictions_binary = (test_predictions > 0.5).astype(int)
excluded_predictions_binary = (excluded_predictions > 0.5).astype(int)

def plot_confusion_matrix(labels, preds, title, classes):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Greys')
    plt.title(title)
    plt.show()

# Plot Confusion Matrix for the validation set
plot_confusion_matrix(val_labels.astype(int), val_predictions_binary, "Confusion Matrix for five-fold Validation Set (Precancerous vs Normal)", ['Normal', 'Precancerous'])

# Plot Confusion Matrix for the test set
plot_confusion_matrix(test_labels.astype(int), test_predictions_binary, "Confusion Matrix for Test Set 1 (Precancerous vs Normal)", ['Normal', 'Precancerous'])

# Plot Confusion Matrix for the excluded set
plot_confusion_matrix(excluded_labels.astype(int), excluded_predictions_binary, "Confusion Matrix for Test Set 2 (Cancer vs Normal)", ['Normal', 'Cancer'])

```


```python
import matplotlib.pyplot as plt

# Plot learning curves
def plot_learning_curves(accuracies, f1_scores, aucs, title):
    epochs = range(1, len(accuracies) + 1)
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, f1_scores, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(f'{title} - F1 Score')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, aucs, label='AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title(f'{title} - AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Assuming test_accuracies, test_f1_scores, test_aucs, excluded_accuracies, excluded_f1_scores, excluded_aucs are defined
#plot_learning_curves(val_accuracies, val_f1_scores, val_aucs, "Validation Set")
plot_learning_curves(test_accuracies, test_f1_scores, test_aucs, "Test Set")
plot_learning_curves(excluded_accuracies, excluded_f1_scores, excluded_aucs, "Excluded Set")
```


```python
# Flatten the features
test_features_flat = test_features.reshape(test_features.shape[0], -1)
excluded_features_flat = excluded_features.reshape(excluded_features.shape[0], -1)

# Select top features using ANOVA F-test
selected_test_features, selector = select_top_features(test_features_flat, test_labels, num_features_to_select)
selected_excluded_features = selector.transform(excluded_features_flat)
```


```python
# Train logistic regression on selected features
logistic_model = LogisticRegression()
logistic_model.fit(selected_test_features, test_labels)
```


```python
# Evaluate logistic regression on selected features
test_preds = logistic_model.predict(selected_test_features)
test_probs = logistic_model.predict_proba(selected_test_features)[:, 1]
test_accuracy = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds)
test_auc = roc_auc_score(test_labels, test_probs)
```


```python
excluded_preds = logistic_model.predict(selected_excluded_features)
excluded_probs = logistic_model.predict_proba(selected_excluded_features)[:, 1]
excluded_accuracy = accuracy_score(excluded_labels, excluded_preds)
excluded_f1 = f1_score(excluded_labels, excluded_preds)
excluded_auc = roc_auc_score(excluded_labels, excluded_probs)

print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test F1 Score: {test_f1:.2f}')
print(f'Test AUC: {test_auc:.2f}')
print(f'Excluded Set Accuracy: {excluded_accuracy:.2f}%')
print(f'Excluded Set F1 Score: {excluded_f1:.2f}')
print(f'Excluded Set AUC: {excluded_auc:.2f}')

# Plot ROC curve for test set
plot_roc_curve(test_labels, test_probs, "ROC Curve for Test Set")

# Plot ROC curve for excluded set
plot_roc_curve(excluded_labels, excluded_probs, "ROC Curve for Excluded Set")

# Plot Confusion Matrix for the test set
plot_confusion_matrix(test_labels, test_preds, "Confusion Matrix for Test Set", ['Normal', 'Precancerous'])

# Plot Confusion Matrix for the excluded set
plot_confusion_matrix(excluded_labels, excluded_preds, "Confusion Matrix for Excluded Set", ['Normal', 'Cancer'])

# Plot learning curves
plot_learning_curves(test_accuracies, test_f1_scores, test_aucs, "Test Set")
plot_learning_curves(excluded_accuracies, excluded_f1_scores, excluded_aucs, "Excluded Set")

# Plot learning curves
def plot_learning_curves(accuracies, f1_scores, aucs, title):
    epochs = list(range(1, len(accuracies) + 1))
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} Accuracy over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, f1_scores, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(f'{title} F1 Score over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, aucs, label='AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title(f'{title} AUC over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot learning curves for test set
plot_learning_curves(test_accuracies, test_f1_scores, test_aucs, "Test Set")

# Plot learning curves for excluded set
plot_learning_curves(excluded_accuracies, excluded_f1_scores, excluded_aucs, "Excluded Set")

# Save the results
results = {
    'test_accuracies': test_accuracies,
    'test_f1_scores': test_f1_scores,
    'test_aucs': test_aucs,
    'excluded_accuracies': excluded_accuracies,
    'excluded_f1_scores': excluded_f1_scores,
    'excluded_aucs': excluded_aucs
}

df_results = pd.DataFrame(results)
df_results.to_csv('training_results.csv', index=False)

print("Results saved to training_results.csv")

```


```python
# Assuming you have defined train_loader, test_loader, and excluded_loader earlier in the code

# Extract features for train, test and excluded datasets
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=False)
train_features, train_labels = extract_features(best_model, train_loader)
test_features, test_labels = extract_features(best_model, test_loader)
excluded_features, excluded_labels = extract_features(best_model, excluded_loader)

# Flatten the features
train_features_flat = train_features.reshape(train_features.shape[0], -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)
excluded_features_flat = excluded_features.reshape(excluded_features.shape[0], -1)

# Select top features using ANOVA F-test on the training set
selected_train_features, selector = select_top_features(train_features_flat, train_labels, num_features_to_select)
selected_test_features = selector.transform(test_features_flat)
selected_excluded_features = selector.transform(excluded_features_flat)

# Train logistic regression on selected training features
logistic_model = LogisticRegression()
logistic_model.fit(selected_train_features, train_labels)

# Evaluate logistic regression on selected features for the test set
test_preds = logistic_model.predict(selected_test_features)
test_probs = logistic_model.predict_proba(selected_test_features)[:, 1]
test_accuracy = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds)
test_auc = roc_auc_score(test_labels, test_probs)

# Evaluate logistic regression on selected features for the excluded set
excluded_preds = logistic_model.predict(selected_excluded_features)
excluded_probs = logistic_model.predict_proba(selected_excluded_features)[:, 1]
excluded_accuracy = accuracy_score(excluded_labels, excluded_preds)
excluded_f1 = f1_score(excluded_labels, excluded_preds)
excluded_auc = roc_auc_score(excluded_labels, excluded_probs)

print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test F1 Score: {test_f1:.2f}')
print(f'Test AUC: {test_auc:.2f}')
print(f'Excluded Set Accuracy: {excluded_accuracy:.2f}%')
print(f'Excluded Set F1 Score: {excluded_f1:.2f}')
print(f'Excluded Set AUC: {excluded_auc:.2f}')

# Plot ROC curve for test set
plot_roc_curve(test_labels, test_probs, "ROC Curve for Test Set")

# Plot ROC curve for excluded set
plot_roc_curve(excluded_labels, excluded_probs, "ROC Curve for Excluded Set")

# Plot Confusion Matrix for the test set
plot_confusion_matrix(test_labels, test_preds, "Confusion Matrix for Test Set", ['Normal', 'Precancerous'])

# Plot Confusion Matrix for the excluded set
plot_confusion_matrix(excluded_labels, excluded_preds, "Confusion Matrix for Excluded Set", ['Normal', 'Cancer'])

# Plot learning curves
def plot_learning_curves(accuracies, f1_scores, aucs, title):
    epochs = list(range(1, len(accuracies) + 1))
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} Accuracy over Epochs')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, f1_scores, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(f'{title} F1 Score over Epochs')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, aucs, label='AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title(f'{title} AUC over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot learning curves for test set
plot_learning_curves(test_accuracies, test_f1_scores, test_aucs, "Test Set")

# Plot learning curves for excluded set
plot_learning_curves(excluded_accuracies, excluded_f1_scores, excluded_aucs, "Excluded Set")

# Save the results
results = {
    'test_accuracies': test_accuracies,
    'test_f1_scores': test_f1_scores,
    'test_aucs': test_aucs,
    'excluded_accuracies': excluded_accuracies,
    'excluded_f1_scores': excluded_f1_scores,
    'excluded_aucs': excluded_aucs
}

df_results = pd.DataFrame(results)
df_results.to_csv('training_results.csv', index=False)

print("Results saved to training_results.csv")
```
