# FEDERATED LEARNING USING QPSO VS FEDAVG: COMPLETE IMPLEMENTATION GUIDE
## Brain Tumor Classification on Non-IID Multi-Dataset Environment

**Version:** 1.0  
**Last Updated:** March 2026  
**Target Platform:** Kaggle (2x Tesla T4 GPUs)

---

# TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Environment Setup - Kaggle Specific](#2-environment-setup---kaggle-specific)
3. [Data Collection & Preprocessing](#3-data-collection--preprocessing)
4. [Project Structure](#4-project-structure)
5. [Implementation - Data Pipeline](#5-implementation---data-pipeline)
6. [Implementation - Model Architecture](#6-implementation---model-architecture)
7. [Implementation - FedAvg](#7-implementation---fedavg)
8. [Implementation - QPSO-FL](#8-implementation---qpso-fl)
9. [Training Procedures](#9-training-procedures)
10. [Evaluation & Visualization](#10-evaluation--visualization)
11. [Running Experiments](#11-running-experiments)
12. [Results Analysis](#12-results-analysis)
13. [Troubleshooting](#13-troubleshooting)
14. [Checklists](#14-checklists)

---

# 1. PROJECT OVERVIEW

## 1.1 Objective

Compare **Federated Averaging (FedAvg)** with **Quantum-behaved Particle Swarm Optimization (QPSO)** based federated learning for brain tumor classification across three heterogeneous medical institutions (datasets).

## 1.2 Key Specifications

| Aspect | Details |
|--------|---------|
| **Clients** | 3 (each represents one hospital/institution) |
| **Datasets** | Figshare, BRISC, Kaggle (Masoud) |
| **Classes** | 3 (Glioma, Meningioma, Pituitary) - **NO Normal/NoTumor** |
| **Total Images** | ~16,000+ images |
| **Model** | ResNet-18 (pretrained on ImageNet) |
| **Methods** | FedAvg vs QPSO-FL |
| **Platform** | Kaggle Notebooks (2x Tesla T4) |
| **Framework** | PyTorch 2.0+ |

## 1.3 Expected Outcomes

- QPSO-FL achieves **3-7% higher accuracy** than FedAvg
- QPSO-FL converges **20-30% faster** (fewer rounds)
- QPSO-FL shows **better fairness** across clients (lower std deviation)

## 1.4 Timeline

**Total Duration:** 12 weeks (10-15 hours/week)

---

# 2. ENVIRONMENT SETUP - KAGGLE SPECIFIC

## 2.1 Creating Kaggle Notebook

### Step 2.1.1: Initial Setup

1. Go to https://www.kaggle.com/
2. Click "Code" → "New Notebook"
3. **Settings Configuration:**
   - **Language:** Python
   - **Environment:** Latest (Python 3.10+)
   - **Accelerator:** GPU T4 x2 (CRITICAL - select this!)
   - **Internet:** ON (for downloading datasets)
   - **Persistence:** ON (to save work)

### Step 2.1.2: Verify GPU Access

**Cell 1: GPU Check**
```python
import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: No GPU detected! Check accelerator settings.")
```

**Expected Output:**
```
Python Version: 3.10.x
PyTorch Version: 2.x.x
CUDA Available: True
CUDA Version: 11.x or 12.x
Number of GPUs: 2
GPU 0: Tesla T4
  Memory: 15.xx GB
GPU 1: Tesla T4
  Memory: 15.xx GB
```

## 2.2 Installing Required Packages

### Step 2.2.1: Core Dependencies

**Cell 2: Install Packages**
```python
# Most packages are pre-installed on Kaggle
# Only install if needed

# Check installed packages
import subprocess
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
print(result.stdout)
```

**If anything is missing, install:**
```python
!pip install -q torchvision
!pip install -q scikit-learn
!pip install -q matplotlib seaborn
!pip install -q pillow
!pip install -q tqdm
!pip install -q pandas numpy
```

### Step 2.2.2: Import All Libraries

**Cell 3: Imports**
```python
# Core
import os
import sys
import time
import copy
import random
import warnings
warnings.filterwarnings('ignore')

# Data handling
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# ML utilities
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
print("✅ All imports successful!")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

## 2.3 Directory Setup

### Step 2.3.1: Create Project Structure

**Cell 4: Directory Structure**
```python
# Create project directories
import os

directories = [
    '/kaggle/working/data/raw/figshare',
    '/kaggle/working/data/raw/brisc',
    '/kaggle/working/data/raw/kaggle',
    '/kaggle/working/data/processed/client1',
    '/kaggle/working/data/processed/client2',
    '/kaggle/working/data/processed/client3',
    '/kaggle/working/data/test_set',
    '/kaggle/working/models',
    '/kaggle/working/results/fedavg',
    '/kaggle/working/results/qpso',
    '/kaggle/working/results/plots',
    '/kaggle/working/logs',
    '/kaggle/working/src',
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    
print("✅ Directory structure created!")

# Verify
!ls -la /kaggle/working/
```

**Expected Structure:**
```
/kaggle/working/
├── data/
│   ├── raw/
│   │   ├── figshare/
│   │   ├── brisc/
│   │   └── kaggle/
│   ├── processed/
│   │   ├── client1/
│   │   ├── client2/
│   │   └── client3/
│   └── test_set/
├── models/
├── results/
│   ├── fedavg/
│   ├── qpso/
│   └── plots/
├── logs/
└── src/
```

---

# 3. DATA COLLECTION & PREPROCESSING

## 3.1 Dataset Download

### 3.1.1 Figshare Dataset

**Method 1: Direct Kaggle Dataset (Recommended)**
```python
# Cell 5: Add Figshare Dataset
# In Kaggle: Click "Add Data" → Search "brain tumor dataset figshare"
# Select the dataset and click "Add"
# It will appear in /kaggle/input/

# Example path (adjust based on actual dataset name):
figshare_path = '/kaggle/input/brain-tumor-mri-dataset'  # Adjust this!
print(f"Figshare data location: {figshare_path}")
!ls {figshare_path}
```

**Method 2: Manual Download (if not on Kaggle)**
```python
# Download from: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
# Upload to Kaggle Dataset
```

### 3.1.2 BRISC Dataset

**Cell 6: Add BRISC Dataset**
```python
# Search for "BRISC brain tumor" in Kaggle Datasets
# OR upload your own

brisc_path = '/kaggle/input/brisc-dataset'  # Adjust this!
print(f"BRISC data location: {brisc_path}")
!ls {brisc_path}
```

### 3.1.3 Kaggle (Masoud) Dataset

**Cell 7: Add Kaggle Dataset**
```python
# Add this dataset: brain-tumor-mri-dataset by Masoud Nickparvar
# URL: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

kaggle_masoud_path = '/kaggle/input/brain-tumor-mri-dataset'  # Adjust this!
print(f"Kaggle Masoud data location: {kaggle_masoud_path}")
!ls {kaggle_masoud_path}
```

## 3.2 Data Exploration

### Step 3.2.1: Explore Dataset Structure

**Cell 8: Explore Figshare**
```python
import os
from collections import Counter

def explore_dataset(path, name):
    print(f"\n{'='*60}")
    print(f"Exploring: {name}")
    print(f"{'='*60}")
    
    class_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(f"Classes found: {class_folders}")
    
    class_counts = {}
    for class_name in class_folders:
        class_path = os.path.join(path, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg', '.mat'))]
        class_counts[class_name] = len(files)
        print(f"  {class_name}: {len(files)} images")
    
    print(f"Total images: {sum(class_counts.values())}")
    return class_counts

# Explore all datasets (adjust paths as needed)
figshare_counts = explore_dataset(figshare_path, "Figshare")
brisc_counts = explore_dataset(brisc_path, "BRISC")
kaggle_counts = explore_dataset(kaggle_masoud_path, "Kaggle Masoud")
```

### Step 3.2.2: Visualize Sample Images

**Cell 9: Visualize Samples**
```python
def visualize_samples(dataset_path, dataset_name, n_samples=3):
    """Display sample images from each class"""
    
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    # Filter out 'notumor' and 'normal' classes
    classes = [c for c in classes if c.lower() not in ['notumor', 'normal', 'no_tumor']]
    
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(12, 4*len(classes)))
    fig.suptitle(f'{dataset_name} - Sample Images', fontsize=16)
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))][:n_samples]
        
        for j, img_name in enumerate(images):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{class_name}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/results/plots/{dataset_name}_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

# Visualize all datasets
visualize_samples(figshare_path, 'Figshare')
visualize_samples(brisc_path, 'BRISC')
visualize_samples(kaggle_masoud_path, 'Kaggle_Masoud')
```

## 3.3 Data Preprocessing Pipeline

### Step 3.3.1: Define Preprocessing Functions

**Cell 10: Preprocessing Utilities**
```python
class BrainTumorPreprocessor:
    """
    Handles all preprocessing for brain tumor datasets
    - Filters out 'normal' and 'notumor' classes
    - Resizes to 224x224
    - Normalizes pixel values
    - Converts to RGB (3 channels)
    - Creates train/val/test splits
    """
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.valid_classes = ['glioma', 'meningioma', 'pituitary']
        self.class_to_idx = {'glioma': 0, 'meningioma': 1, 'pituitary': 2}
        self.idx_to_class = {0: 'glioma', 1: 'meningioma', 2: 'pituitary'}
        
    def is_valid_class(self, class_name):
        """Check if class should be included"""
        return class_name.lower() in self.valid_classes
    
    def load_and_preprocess_image(self, image_path):
        """
        Load image and preprocess
        Returns: numpy array (224, 224, 3) with values [0, 1]
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to RGB (in case grayscale)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            img = img.resize(self.target_size, Image.BILINEAR)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            return img_array
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def process_dataset(self, dataset_path, client_name, train_ratio=0.7, val_ratio=0.15):
        """
        Process entire dataset for a client
        - Filters valid classes
        - Loads and preprocesses images
        - Splits into train/val/test
        - Saves processed data
        """
        
        print(f"\n{'='*60}")
        print(f"Processing dataset for {client_name}")
        print(f"Source: {dataset_path}")
        print(f"{'='*60}")
        
        # Collect all valid images
        all_images = []
        all_labels = []
        
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            
            if not os.path.isdir(class_path):
                continue
            
            if not self.is_valid_class(class_name):
                print(f"Skipping class: {class_name}")
                continue
            
            print(f"Processing class: {class_name}")
            class_label = self.class_to_idx[class_name.lower()]
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))]
            
            for img_file in tqdm(image_files, desc=f"  Loading {class_name}"):
                img_path = os.path.join(class_path, img_file)
                img_array = self.load_and_preprocess_image(img_path)
                
                if img_array is not None:
                    all_images.append(img_array)
                    all_labels.append(class_label)
        
        # Convert to numpy arrays
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)
        
        print(f"\nTotal images loaded: {len(all_images)}")
        print(f"Image shape: {all_images[0].shape}")
        print(f"Label distribution: {np.bincount(all_labels)}")
        
        # Split data
        test_ratio = 1 - train_ratio - val_ratio
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_images, all_labels, 
            test_size=test_ratio, 
            stratify=all_labels, 
            random_state=42
        )
        
        # Second split: separate train and val
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_ratio_adjusted, 
            stratify=y_temp, 
            random_state=42
        )
        
        print(f"\nSplit summary:")
        print(f"  Train: {len(X_train)} images ({len(X_train)/len(all_images)*100:.1f}%)")
        print(f"  Val:   {len(X_val)} images ({len(X_val)/len(all_images)*100:.1f}%)")
        print(f"  Test:  {len(X_test)} images ({len(X_test)/len(all_images)*100:.1f}%)")
        
        # Save processed data
        client_dir = f'/kaggle/working/data/processed/{client_name}'
        
        np.save(f'{client_dir}/X_train.npy', X_train)
        np.save(f'{client_dir}/y_train.npy', y_train)
        np.save(f'{client_dir}/X_val.npy', X_val)
        np.save(f'{client_dir}/y_val.npy', y_val)
        np.save(f'{client_dir}/X_test.npy', X_test)
        np.save(f'{client_dir}/y_test.npy', y_test)
        
        print(f"\n✅ Data saved to {client_dir}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

# Initialize preprocessor
preprocessor = BrainTumorPreprocessor(target_size=(224, 224))
print("✅ Preprocessor initialized")
```

### Step 3.3.2: Process All Datasets

**Cell 11: Process Client 1 (Figshare)**
```python
# Process Figshare dataset for Client 1
client1_data = preprocessor.process_dataset(
    dataset_path=figshare_path,
    client_name='client1',
    train_ratio=0.70,
    val_ratio=0.15
)
```

**Cell 12: Process Client 2 (BRISC)**
```python
# Process BRISC dataset for Client 2
client2_data = preprocessor.process_dataset(
    dataset_path=brisc_path,
    client_name='client2',
    train_ratio=0.70,
    val_ratio=0.15
)
```

**Cell 13: Process Client 3 (Kaggle Masoud)**
```python
# Process Kaggle Masoud dataset for Client 3
client3_data = preprocessor.process_dataset(
    dataset_path=kaggle_masoud_path,
    client_name='client3',
    train_ratio=0.70,
    val_ratio=0.15
)
```

### Step 3.3.3: Create Global Test Set

**Cell 14: Combine Test Sets**
```python
"""
Create a global test set by combining test data from all clients
This will be used to evaluate the global model fairly
"""

def create_global_test_set():
    print("Creating global test set...")
    
    # Load test sets from all clients
    c1_X_test = np.load('/kaggle/working/data/processed/client1/X_test.npy')
    c1_y_test = np.load('/kaggle/working/data/processed/client1/y_test.npy')
    
    c2_X_test = np.load('/kaggle/working/data/processed/client2/X_test.npy')
    c2_y_test = np.load('/kaggle/working/data/processed/client2/y_test.npy')
    
    c3_X_test = np.load('/kaggle/working/data/processed/client3/X_test.npy')
    c3_y_test = np.load('/kaggle/working/data/processed/client3/y_test.npy')
    
    # Combine
    global_X_test = np.concatenate([c1_X_test, c2_X_test, c3_X_test], axis=0)
    global_y_test = np.concatenate([c1_y_test, c2_y_test, c3_y_test], axis=0)
    
    print(f"Global test set size: {len(global_X_test)} images")
    print(f"Class distribution: {np.bincount(global_y_test)}")
    
    # Save
    np.save('/kaggle/working/data/test_set/X_test.npy', global_X_test)
    np.save('/kaggle/working/data/test_set/y_test.npy', global_y_test)
    
    print("✅ Global test set created and saved")
    
    return global_X_test, global_y_test

global_X_test, global_y_test = create_global_test_set()
```

### Step 3.3.4: Verify Preprocessing

**Cell 15: Verification**
```python
"""
Verify that preprocessing was successful
"""

def verify_preprocessing():
    print("Verifying preprocessing...")
    
    clients = ['client1', 'client2', 'client3']
    
    for client in clients:
        print(f"\n{client.upper()}:")
        client_dir = f'/kaggle/working/data/processed/{client}'
        
        # Check if files exist
        files = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy', 'X_test.npy', 'y_test.npy']
        for f in files:
            path = f'{client_dir}/{f}'
            if os.path.exists(path):
                data = np.load(path)
                print(f"  ✅ {f}: shape {data.shape}")
            else:
                print(f"  ❌ {f}: NOT FOUND")
    
    # Check global test set
    print(f"\nGLOBAL TEST SET:")
    global_dir = '/kaggle/working/data/test_set'
    for f in ['X_test.npy', 'y_test.npy']:
        path = f'{global_dir}/{f}'
        if os.path.exists(path):
            data = np.load(path)
            print(f"  ✅ {f}: shape {data.shape}")
        else:
            print(f"  ❌ {f}: NOT FOUND")

verify_preprocessing()
```

## 3.4 Data Statistics

**Cell 16: Generate Statistics**
```python
"""
Generate and visualize data statistics across all clients
"""

def generate_data_statistics():
    clients_info = []
    
    for client_num in range(1, 4):
        client_name = f'client{client_num}'
        client_dir = f'/kaggle/working/data/processed/{client_name}'
        
        X_train = np.load(f'{client_dir}/X_train.npy')
        y_train = np.load(f'{client_dir}/y_train.npy')
        X_val = np.load(f'{client_dir}/X_val.npy')
        y_val = np.load(f'{client_dir}/y_val.npy')
        X_test = np.load(f'{client_dir}/X_test.npy')
        y_test = np.load(f'{client_dir}/y_test.npy')
        
        train_dist = np.bincount(y_train, minlength=3)
        val_dist = np.bincount(y_val, minlength=3)
        test_dist = np.bincount(y_test, minlength=3)
        
        clients_info.append({
            'Client': client_name,
            'Train Total': len(X_train),
            'Train Glioma': train_dist[0],
            'Train Meningioma': train_dist[1],
            'Train Pituitary': train_dist[2],
            'Val Total': len(X_val),
            'Test Total': len(X_test)
        })
    
    df = pd.DataFrame(clients_info)
    print(df.to_string(index=False))
    
    # Visualize distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Class Distribution Across Clients', fontsize=16)
    
    for i, client_name in enumerate(['client1', 'client2', 'client3']):
        client_dir = f'/kaggle/working/data/processed/{client_name}'
        y_train = np.load(f'{client_dir}/y_train.npy')
        
        train_dist = np.bincount(y_train, minlength=3)
        
        axes[i].bar(['Glioma', 'Meningioma', 'Pituitary'], train_dist, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[i].set_title(f'{client_name.upper()}')
        axes[i].set_ylabel('Number of Samples')
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/results/plots/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return df

stats_df = generate_data_statistics()
```

---

# 4. PROJECT STRUCTURE

## 4.1 Complete File Organization

```
/kaggle/working/
├── data/
│   ├── processed/
│   │   ├── client1/
│   │   │   ├── X_train.npy
│   │   │   ├── y_train.npy
│   │   │   ├── X_val.npy
│   │   │   ├── y_val.npy
│   │   │   ├── X_test.npy
│   │   │   └── y_test.npy
│   │   ├── client2/  [same structure]
│   │   └── client3/  [same structure]
│   └── test_set/
│       ├── X_test.npy
│       └── y_test.npy
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── client.py
│   ├── server_fedavg.py
│   ├── server_qpso.py
│   ├── trainer.py
│   └── utils.py
├── models/
│   ├── fedavg_round_*.pth
│   └── qpso_round_*.pth
├── results/
│   ├── fedavg/
│   │   ├── metrics.csv
│   │   └── final_results.json
│   ├── qpso/
│   │   ├── metrics.csv
│   │   └── final_results.json
│   └── plots/
│       ├── accuracy_comparison.png
│       ├── convergence.png
│       └── confusion_matrices.png
└── logs/
    ├── fedavg_training.log
    └── qpso_training.log
```

---

# 5. IMPLEMENTATION - DATA PIPELINE

## 5.1 Custom Dataset Class

**Cell 17: Create dataset.py**
```python
"""
Custom PyTorch Dataset for Federated Learning
"""

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumor images
    Loads preprocessed numpy arrays
    """
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: numpy array of images (N, H, W, C)
            y: numpy array of labels (N,)
            transform: optional transforms to apply
        """
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        
        # Convert numpy to PIL Image for transforms
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# Save to file
with open('/kaggle/working/src/dataset.py', 'w') as f:
    f.write('''
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumor images
    Loads preprocessed numpy arrays
    """
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X: numpy array of images (N, H, W, C)
            y: numpy array of labels (N,)
            transform: optional transforms to apply
        """
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]
        
        # Convert numpy to PIL Image for transforms
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image)
        
        return image, torch.tensor(label, dtype=torch.long)
''')

print("✅ dataset.py created")
```

## 5.2 Data Loaders

**Cell 18: Create Data Loaders**
```python
"""
Create data loaders for all clients
"""

def create_data_loaders(batch_size=32, num_workers=2):
    """
    Create train, val, test loaders for all clients
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    loaders = {}
    
    # Create loaders for each client
    for client_num in range(1, 4):
        client_name = f'client{client_num}'
        client_dir = f'/kaggle/working/data/processed/{client_name}'
        
        # Load data
        X_train = np.load(f'{client_dir}/X_train.npy')
        y_train = np.load(f'{client_dir}/y_train.npy')
        X_val = np.load(f'{client_dir}/X_val.npy')
        y_val = np.load(f'{client_dir}/y_val.npy')
        X_test = np.load(f'{client_dir}/X_test.npy')
        y_test = np.load(f'{client_dir}/y_test.npy')
        
        # Create datasets
        train_dataset = BrainTumorDataset(X_train, y_train, transform=train_transform)
        val_dataset = BrainTumorDataset(X_val, y_val, transform=test_transform)
        test_dataset = BrainTumorDataset(X_test, y_test, transform=test_transform)
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        loaders[client_name] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'train_size': len(train_dataset)
        }
        
        print(f"{client_name}: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create global test loader
    global_X_test = np.load('/kaggle/working/data/test_set/X_test.npy')
    global_y_test = np.load('/kaggle/working/data/test_set/y_test.npy')
    global_test_dataset = BrainTumorDataset(global_X_test, global_y_test, transform=test_transform)
    global_test_loader = DataLoader(
        global_test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    loaders['global_test'] = global_test_loader
    print(f"Global test set: {len(global_test_dataset)} images")
    
    return loaders

# Create all loaders
data_loaders = create_data_loaders(batch_size=32, num_workers=2)
print("\n✅ All data loaders created successfully")
```

---

# 6. IMPLEMENTATION - MODEL ARCHITECTURE

## 6.1 ResNet-18 Model

**Cell 19: Create model.py**
```python
"""
Model architecture for brain tumor classification
Using ResNet-18 pretrained on ImageNet
"""

class BrainTumorResNet(nn.Module):
    """
    ResNet-18 based model for brain tumor classification
    3 output classes: Glioma, Meningioma, Pituitary
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        super(BrainTumorResNet, self).__init__()
        
        # Load pretrained ResNet-18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def create_model(num_classes=3, device='cuda'):
    """
    Factory function to create model
    """
    model = BrainTumorResNet(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    return model

# Test model creation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_model = create_model(num_classes=3, device=device)

# Print model summary
total_params = sum(p.numel() for p in test_model.parameters())
trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)

print(f"Model: ResNet-18")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB")
print("✅ Model architecture defined")

# Save to file
with open('/kaggle/working/src/model.py', 'w') as f:
    f.write('''
import torch
import torch.nn as nn
import torchvision.models as models

class BrainTumorResNet(nn.Module):
    """
    ResNet-18 based model for brain tumor classification
    3 output classes: Glioma, Meningioma, Pituitary
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        super(BrainTumorResNet, self).__init__()
        
        # Load pretrained ResNet-18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def create_model(num_classes=3, device='cuda'):
    """
    Factory function to create model
    """
    model = BrainTumorResNet(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    return model
''')

print("✅ model.py created")
```

---

# 7. IMPLEMENTATION - FEDAVG

## 7.1 Client Implementation

**Cell 20: Create client.py**
```python
"""
Client implementation for Federated Learning
Handles local training
"""

class FederatedClient:
    """
    Represents one client (hospital) in federated learning
    """
    
    def __init__(self, client_id, train_loader, val_loader, device='cuda'):
        """
        Args:
            client_id: unique identifier (e.g., 'client1')
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            device: 'cuda' or 'cpu'
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Get dataset size
        self.dataset_size = len(train_loader.dataset)
        
    def set_model(self, global_model):
        """
        Receive global model from server
        """
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)
        
    def set_optimizer(self, learning_rate=0.001):
        """
        Initialize optimizer
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train_local(self, epochs=5, verbose=False):
        """
        Train model locally for specified epochs
        Returns: updated model state_dict
        """
        self.model.train()
        
        epoch_losses = []
        epoch_accs = []
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f"{self.client_id} Epoch {epoch+1}/{epochs}", disable=not verbose)
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                if verbose:
                    pbar.set_postfix({
                        'Loss': f'{running_loss/(batch_idx+1):.4f}',
                        'Acc': f'{100.*correct/total:.2f}%'
                    })
            
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)
            
            if verbose:
                print(f"{self.client_id} Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
        
        return self.model.state_dict(), epoch_losses, epoch_accs
    
    def validate(self):
        """
        Validate model on local validation set
        Returns: validation loss and accuracy
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def get_dataset_size(self):
        """
        Return number of training samples
        """
        return self.dataset_size

# Save to file
with open('/kaggle/working/src/client.py', 'w') as f:
    f.write('''
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

class FederatedClient:
    """
    Represents one client (hospital) in federated learning
    """
    
    def __init__(self, client_id, train_loader, val_loader, device='cuda'):
        """
        Args:
            client_id: unique identifier (e.g., 'client1')
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            device: 'cuda' or 'cpu'
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Get dataset size
        self.dataset_size = len(train_loader.dataset)
        
    def set_model(self, global_model):
        """
        Receive global model from server
        """
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)
        
    def set_optimizer(self, learning_rate=0.001):
        """
        Initialize optimizer
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train_local(self, epochs=5, verbose=False):
        """
        Train model locally for specified epochs
        Returns: updated model state_dict
        """
        self.model.train()
        
        epoch_losses = []
        epoch_accs = []
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f"{self.client_id} Epoch {epoch+1}/{epochs}", disable=not verbose)
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                if verbose:
                    pbar.set_postfix({
                        'Loss': f'{running_loss/(batch_idx+1):.4f}',
                        'Acc': f'{100.*correct/total:.2f}%'
                    })
            
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)
            
            if verbose:
                print(f"{self.client_id} Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
        
        return self.model.state_dict(), epoch_losses, epoch_accs
    
    def validate(self):
        """
        Validate model on local validation set
        Returns: validation loss and accuracy
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def get_dataset_size(self):
        """
        Return number of training samples
        """
        return self.dataset_size
''')

print("✅ client.py created")
```

## 7.2 FedAvg Server Implementation

**Cell 21: Create server_fedavg.py**
```python
"""
Server implementation for FedAvg algorithm
Handles aggregation of client models
"""

class FedAvgServer:
    """
    Central server for Federated Averaging (FedAvg)
    """
    
    def __init__(self, global_model, clients, device='cuda'):
        """
        Args:
            global_model: initial global model
            clients: list of FederatedClient objects
            device: 'cuda' or 'cpu'
        """
        self.global_model = global_model
        self.clients = clients
        self.device = device
        
        # Calculate total training samples
        self.total_samples = sum(client.get_dataset_size() for client in clients)
        
        print(f"FedAvg Server initialized")
        print(f"Number of clients: {len(clients)}")
        print(f"Total training samples: {self.total_samples}")
        
    def aggregate_weights(self, client_weights):
        """
        FedAvg aggregation: weighted average based on dataset size
        
        Args:
            client_weights: list of tuples (state_dict, dataset_size)
        
        Returns:
            aggregated state_dict
        """
        
        # Initialize aggregated weights
        aggregated_weights = copy.deepcopy(client_weights[0][0])
        
        # Set all weights to zero
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])
        
        # Weighted sum
        for client_state_dict, dataset_size in client_weights:
            weight = dataset_size / self.total_samples
            
            for key in aggregated_weights.keys():
                aggregated_weights[key] += client_state_dict[key] * weight
        
        return aggregated_weights
    
    def evaluate_global_model(self, test_loader):
        """
        Evaluate global model on test set
        
        Returns:
            accuracy, loss
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def get_global_model(self):
        """
        Return copy of global model
        """
        return copy.deepcopy(self.global_model)

# Save to file
with open('/kaggle/working/src/server_fedavg.py', 'w') as f:
    f.write('''
import torch
import torch.nn as nn
import copy

class FedAvgServer:
    """
    Central server for Federated Averaging (FedAvg)
    """
    
    def __init__(self, global_model, clients, device='cuda'):
        """
        Args:
            global_model: initial global model
            clients: list of FederatedClient objects
            device: 'cuda' or 'cpu'
        """
        self.global_model = global_model
        self.clients = clients
        self.device = device
        
        # Calculate total training samples
        self.total_samples = sum(client.get_dataset_size() for client in clients)
        
        print(f"FedAvg Server initialized")
        print(f"Number of clients: {len(clients)}")
        print(f"Total training samples: {self.total_samples}")
        
    def aggregate_weights(self, client_weights):
        """
        FedAvg aggregation: weighted average based on dataset size
        
        Args:
            client_weights: list of tuples (state_dict, dataset_size)
        
        Returns:
            aggregated state_dict
        """
        
        # Initialize aggregated weights
        aggregated_weights = copy.deepcopy(client_weights[0][0])
        
        # Set all weights to zero
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])
        
        # Weighted sum
        for client_state_dict, dataset_size in client_weights:
            weight = dataset_size / self.total_samples
            
            for key in aggregated_weights.keys():
                aggregated_weights[key] += client_state_dict[key] * weight
        
        return aggregated_weights
    
    def evaluate_global_model(self, test_loader):
        """
        Evaluate global model on test set
        
        Returns:
            accuracy, loss
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def get_global_model(self):
        """
        Return copy of global model
        """
        return copy.deepcopy(self.global_model)
''')

print("✅ server_fedavg.py created")
```

## 7.3 FedAvg Training Loop

**Cell 22: Create FedAvg Trainer**
```python
"""
Complete training loop for FedAvg
"""

def train_fedavg(
    server,
    clients,
    global_test_loader,
    num_rounds=100,
    local_epochs=5,
    learning_rate=0.001,
    save_every=10,
    verbose=True
):
    """
    Train using FedAvg algorithm
    
    Args:
        server: FedAvgServer object
        clients: list of FederatedClient objects
        global_test_loader: DataLoader for global test set
        num_rounds: number of communication rounds
        local_epochs: epochs each client trains locally
        learning_rate: learning rate for local training
        save_every: save model every N rounds
        verbose: print detailed logs
    
    Returns:
        training history (losses, accuracies, etc.)
    """
    
    print("="*80)
    print("STARTING FEDAVG TRAINING")
    print("="*80)
    
    # Initialize history tracking
    history = {
        'round': [],
        'global_test_acc': [],
        'global_test_loss': [],
        'client1_val_acc': [],
        'client2_val_acc': [],
        'client3_val_acc': [],
        'round_time': []
    }
    
    best_accuracy = 0.0
    
    for round_num in range(1, num_rounds + 1):
        round_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'='*80}")
        
        # Collect client weights and sizes
        client_weights = []
        
        # Each client trains locally
        for client in clients:
            print(f"\n{client.client_id.upper()} - Local Training")
            
            # Set global model
            client.set_model(server.get_global_model())
            client.set_optimizer(learning_rate=learning_rate)
            
            # Train locally
            updated_weights, train_losses, train_accs = client.train_local(
                epochs=local_epochs,
                verbose=verbose
            )
            
            # Validate
            val_loss, val_acc = client.validate()
            print(f"{client.client_id} Validation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            
            # Store validation accuracy
            history[f'{client.client_id}_val_acc'].append(val_acc)
            
            # Collect weights
            client_weights.append((updated_weights, client.get_dataset_size()))
        
        # Server aggregates
        print(f"\nSERVER - Aggregating client models...")
        aggregated_weights = server.aggregate_weights(client_weights)
        server.global_model.load_state_dict(aggregated_weights)
        
        # Evaluate global model
        print(f"\nSERVER - Evaluating global model...")
        global_acc, global_loss = server.evaluate_global_model(global_test_loader)
        
        print(f"Global Model Performance:")
        print(f"  Test Accuracy: {global_acc:.2f}%")
        print(f"  Test Loss: {global_loss:.4f}")
        
        # Track history
        round_time = time.time() - round_start_time
        history['round'].append(round_num)
        history['global_test_acc'].append(global_acc)
        history['global_test_loss'].append(global_loss)
        history['round_time'].append(round_time)
        
        print(f"\nRound {round_num} completed in {round_time:.2f}s")
        
        # Save best model
        if global_acc > best_accuracy:
            best_accuracy = global_acc
            torch.save(
                server.global_model.state_dict(),
                f'/kaggle/working/models/fedavg_best.pth'
            )
            print(f"✅ New best model saved! Accuracy: {best_accuracy:.2f}%")
        
        # Save checkpoint
        if round_num % save_every == 0:
            torch.save(
                server.global_model.state_dict(),
                f'/kaggle/working/models/fedavg_round_{round_num}.pth'
            )
            print(f"✅ Checkpoint saved at round {round_num}")
        
        # Save history
        df_history = pd.DataFrame(history)
        df_history.to_csv('/kaggle/working/results/fedavg/metrics.csv', index=False)
    
    print("\n" + "="*80)
    print("FEDAVG TRAINING COMPLETED")
    print(f"Best Global Accuracy: {best_accuracy:.2f}%")
    print("="*80)
    
    return history

print("✅ FedAvg trainer created")
```

---

# 8. IMPLEMENTATION - QPSO-FL

## 8.1 QPSO Algorithm Implementation

**Cell 23: Create server_qpso.py**
```python
"""
Server implementation for QPSO-based Federated Learning
"""

class QPSOServer:
    """
    Server using Quantum-behaved Particle Swarm Optimization for aggregation
    """
    
    def __init__(self, global_model, clients, device='cuda', beta=0.7):
        """
        Args:
            global_model: initial global model
            clients: list of FederatedClient objects
            device: 'cuda' or 'cpu'
            beta: QPSO contraction-expansion coefficient (0.5-1.0)
        """
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.beta = beta
        
        # Initialize personal best for each client
        self.personal_best = {}
        self.personal_best_scores = {}
        
        # Initialize global best
        self.global_best = None
        self.global_best_score = 0.0
        
        # Initialize mean best (mbest)
        self.mean_best = None
        
        print(f"QPSO Server initialized")
        print(f"Number of clients: {len(clients)}")
        print(f"Beta parameter: {beta}")
        
    def initialize_particles(self):
        """
        Initialize personal best for each client with current model
        """
        current_state = copy.deepcopy(self.global_model.state_dict())
        
        for client in self.clients:
            self.personal_best[client.client_id] = copy.deepcopy(current_state)
            self.personal_best_scores[client.client_id] = 0.0
        
        self.global_best = copy.deepcopy(current_state)
        self.global_best_score = 0.0
        
        print("✅ Particles initialized")
    
    def update_personal_best(self, client_id, client_weights, validation_acc):
        """
        Update personal best if current model is better
        """
        if validation_acc > self.personal_best_scores[client_id]:
            self.personal_best[client_id] = copy.deepcopy(client_weights)
            self.personal_best_scores[client_id] = validation_acc
            return True
        return False
    
    def update_global_best(self, client_id, validation_acc):
        """
        Update global best if personal best of any client is better
        """
        if validation_acc > self.global_best_score:
            self.global_best = copy.deepcopy(self.personal_best[client_id])
            self.global_best_score = validation_acc
            return True
        return False
    
    def calculate_mean_best(self):
        """
        Calculate mbest: mean of all personal bests
        """
        # Initialize with zeros
        self.mean_best = copy.deepcopy(self.personal_best[self.clients[0].client_id])
        
        for key in self.mean_best.keys():
            self.mean_best[key] = torch.zeros_like(self.mean_best[key], dtype=torch.float32)
        
        # Sum all personal bests
        for client in self.clients:
            pbest = self.personal_best[client.client_id]
            for key in self.mean_best.keys():
                self.mean_best[key] += pbest[key].float()
        
        # Divide by number of clients
        num_clients = len(self.clients)
        for key in self.mean_best.keys():
            self.mean_best[key] /= num_clients
    
    def qpso_aggregate(self, client_weights_list):
        """
        QPSO-based aggregation
        
        Args:
            client_weights_list: list of tuples (client_id, state_dict, val_acc)
        
        Returns:
            aggregated state_dict using QPSO
        """
        
        # Update personal and global bests
        for client_id, client_weights, val_acc in client_weights_list:
            pbest_updated = self.update_personal_best(client_id, client_weights, val_acc)
            gbest_updated = self.update_global_best(client_id, val_acc)
            
            if pbest_updated:
                print(f"  {client_id}: Personal best updated ({val_acc:.2f}%)")
            if gbest_updated:
                print(f"  {client_id}: Global best updated ({val_acc:.2f}%)")
        
        # Calculate mean best
        self.calculate_mean_best()
        
        # QPSO update for each parameter
        aggregated_weights = copy.deepcopy(self.global_best)
        
        for key in aggregated_weights.keys():
            # Get corresponding tensors
            pbest_sum = torch.zeros_like(aggregated_weights[key], dtype=torch.float32)
            
            for client in self.clients:
                pbest_sum += self.personal_best[client.client_id][key].float()
            
            # Random parameters
            phi = torch.rand_like(aggregated_weights[key])
            u = torch.rand_like(aggregated_weights[key])
            
            # Attraction point p
            p = phi * pbest_sum / len(self.clients) + (1 - phi) * self.global_best[key].float()
            
            # QPSO position update
            sign = torch.where(torch.rand_like(aggregated_weights[key]) < 0.5, 
                              torch.ones_like(aggregated_weights[key]), 
                              -torch.ones_like(aggregated_weights[key]))
            
            mbest_tensor = self.mean_best[key].float()
            current = aggregated_weights[key].float()
            
            aggregated_weights[key] = p + sign * self.beta * torch.abs(mbest_tensor - current) * torch.log(1.0 / (u + 1e-8))
            
            # Convert back to original dtype
            aggregated_weights[key] = aggregated_weights[key].to(self.global_best[key].dtype)
        
        return aggregated_weights
    
    def evaluate_global_model(self, test_loader):
        """
        Evaluate global model on test set
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def get_global_model(self):
        """
        Return copy of global model
        """
        return copy.deepcopy(self.global_model)

# Save to file
with open('/kaggle/working/src/server_qpso.py', 'w') as f:
    f.write('''
import torch
import torch.nn as nn
import copy
import random

class QPSOServer:
    """
    Server using Quantum-behaved Particle Swarm Optimization for aggregation
    """
    
    def __init__(self, global_model, clients, device='cuda', beta=0.7):
        """
        Args:
            global_model: initial global model
            clients: list of FederatedClient objects
            device: 'cuda' or 'cpu'
            beta: QPSO contraction-expansion coefficient (0.5-1.0)
        """
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.beta = beta
        
        # Initialize personal best for each client
        self.personal_best = {}
        self.personal_best_scores = {}
        
        # Initialize global best
        self.global_best = None
        self.global_best_score = 0.0
        
        # Initialize mean best (mbest)
        self.mean_best = None
        
        print(f"QPSO Server initialized")
        print(f"Number of clients: {len(clients)}")
        print(f"Beta parameter: {beta}")
        
    def initialize_particles(self):
        """
        Initialize personal best for each client with current model
        """
        current_state = copy.deepcopy(self.global_model.state_dict())
        
        for client in self.clients:
            self.personal_best[client.client_id] = copy.deepcopy(current_state)
            self.personal_best_scores[client.client_id] = 0.0
        
        self.global_best = copy.deepcopy(current_state)
        self.global_best_score = 0.0
        
        print("✅ Particles initialized")
    
    def update_personal_best(self, client_id, client_weights, validation_acc):
        """
        Update personal best if current model is better
        """
        if validation_acc > self.personal_best_scores[client_id]:
            self.personal_best[client_id] = copy.deepcopy(client_weights)
            self.personal_best_scores[client_id] = validation_acc
            return True
        return False
    
    def update_global_best(self, client_id, validation_acc):
        """
        Update global best if personal best of any client is better
        """
        if validation_acc > self.global_best_score:
            self.global_best = copy.deepcopy(self.personal_best[client_id])
            self.global_best_score = validation_acc
            return True
        return False
    
    def calculate_mean_best(self):
        """
        Calculate mbest: mean of all personal bests
        """
        # Initialize with zeros
        self.mean_best = copy.deepcopy(self.personal_best[self.clients[0].client_id])
        
        for key in self.mean_best.keys():
            self.mean_best[key] = torch.zeros_like(self.mean_best[key], dtype=torch.float32)
        
        # Sum all personal bests
        for client in self.clients:
            pbest = self.personal_best[client.client_id]
            for key in self.mean_best.keys():
                self.mean_best[key] += pbest[key].float()
        
        # Divide by number of clients
        num_clients = len(self.clients)
        for key in self.mean_best.keys():
            self.mean_best[key] /= num_clients
    
    def qpso_aggregate(self, client_weights_list):
        """
        QPSO-based aggregation
        
        Args:
            client_weights_list: list of tuples (client_id, state_dict, val_acc)
        
        Returns:
            aggregated state_dict using QPSO
        """
        
        # Update personal and global bests
        for client_id, client_weights, val_acc in client_weights_list:
            pbest_updated = self.update_personal_best(client_id, client_weights, val_acc)
            gbest_updated = self.update_global_best(client_id, val_acc)
            
            if pbest_updated:
                print(f"  {client_id}: Personal best updated ({val_acc:.2f}%)")
            if gbest_updated:
                print(f"  {client_id}: Global best updated ({val_acc:.2f}%)")
        
        # Calculate mean best
        self.calculate_mean_best()
        
        # QPSO update for each parameter
        aggregated_weights = copy.deepcopy(self.global_best)
        
        for key in aggregated_weights.keys():
            # Get corresponding tensors
            pbest_sum = torch.zeros_like(aggregated_weights[key], dtype=torch.float32)
            
            for client in self.clients:
                pbest_sum += self.personal_best[client.client_id][key].float()
            
            # Random parameters
            phi = torch.rand_like(aggregated_weights[key])
            u = torch.rand_like(aggregated_weights[key])
            
            # Attraction point p
            p = phi * pbest_sum / len(self.clients) + (1 - phi) * self.global_best[key].float()
            
            # QPSO position update
            sign = torch.where(torch.rand_like(aggregated_weights[key]) < 0.5, 
                              torch.ones_like(aggregated_weights[key]), 
                              -torch.ones_like(aggregated_weights[key]))
            
            mbest_tensor = self.mean_best[key].float()
            current = aggregated_weights[key].float()
            
            aggregated_weights[key] = p + sign * self.beta * torch.abs(mbest_tensor - current) * torch.log(1.0 / (u + 1e-8))
            
            # Convert back to original dtype
            aggregated_weights[key] = aggregated_weights[key].to(self.global_best[key].dtype)
        
        return aggregated_weights
    
    def evaluate_global_model(self, test_loader):
        """
        Evaluate global model on test set
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def get_global_model(self):
        """
        Return copy of global model
        """
        return copy.deepcopy(self.global_model)
''')

print("✅ server_qpso.py created")
```

## 8.2 QPSO Training Loop

**Cell 24: Create QPSO Trainer**
```python
"""
Complete training loop for QPSO-FL
"""

def train_qpso(
    server,
    clients,
    global_test_loader,
    num_rounds=100,
    local_epochs=5,
    learning_rate=0.001,
    save_every=10,
    verbose=True
):
    """
    Train using QPSO-FL algorithm
    
    Args:
        server: QPSOServer object
        clients: list of FederatedClient objects
        global_test_loader: DataLoader for global test set
        num_rounds: number of communication rounds
        local_epochs: epochs each client trains locally
        learning_rate: learning rate for local training
        save_every: save model every N rounds
        verbose: print detailed logs
    
    Returns:
        training history
    """
    
    print("="*80)
    print("STARTING QPSO-FL TRAINING")
    print("="*80)
    
    # Initialize particles
    server.initialize_particles()
    
    # Initialize history tracking
    history = {
        'round': [],
        'global_test_acc': [],
        'global_test_loss': [],
        'global_best_score': [],
        'client1_val_acc': [],
        'client2_val_acc': [],
        'client3_val_acc': [],
        'client1_pbest_score': [],
        'client2_pbest_score': [],
        'client3_pbest_score': [],
        'round_time': []
    }
    
    best_accuracy = 0.0
    
    for round_num in range(1, num_rounds + 1):
        round_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'='*80}")
        
        # Collect client info for QPSO
        client_weights_list = []
        
        # Each client trains locally
        for client in clients:
            print(f"\n{client.client_id.upper()} - Local Training")
            
            # Set global model
            client.set_model(server.get_global_model())
            client.set_optimizer(learning_rate=learning_rate)
            
            # Train locally
            updated_weights, train_losses, train_accs = client.train_local(
                epochs=local_epochs,
                verbose=verbose
            )
            
            # Validate
            val_loss, val_acc = client.validate()
            print(f"{client.client_id} Validation: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            
            # Store validation accuracy
            history[f'{client.client_id}_val_acc'].append(val_acc)
            
            # Collect for QPSO aggregation
            client_weights_list.append((client.client_id, updated_weights, val_acc))
        
        # Server aggregates using QPSO
        print(f"\nSERVER - QPSO Aggregation...")
        aggregated_weights = server.qpso_aggregate(client_weights_list)
        server.global_model.load_state_dict(aggregated_weights)
        
        # Track personal best scores
        for client in clients:
            pbest_score = server.personal_best_scores[client.client_id]
            history[f'{client.client_id}_pbest_score'].append(pbest_score)
        
        history['global_best_score'].append(server.global_best_score)
        
        print(f"\nQPSO Status:")
        print(f"  Global Best Score: {server.global_best_score:.2f}%")
        for client in clients:
            print(f"  {client.client_id} Personal Best: {server.personal_best_scores[client.client_id]:.2f}%")
        
        # Evaluate global model
        print(f"\nSERVER - Evaluating global model...")
        global_acc, global_loss = server.evaluate_global_model(global_test_loader)
        
        print(f"Global Model Performance:")
        print(f"  Test Accuracy: {global_acc:.2f}%")
        print(f"  Test Loss: {global_loss:.4f}")
        
        # Track history
        round_time = time.time() - round_start_time
        history['round'].append(round_num)
        history['global_test_acc'].append(global_acc)
        history['global_test_loss'].append(global_loss)
        history['round_time'].append(round_time)
        
        print(f"\nRound {round_num} completed in {round_time:.2f}s")
        
        # Save best model
        if global_acc > best_accuracy:
            best_accuracy = global_acc
            torch.save(
                server.global_model.state_dict(),
                f'/kaggle/working/models/qpso_best.pth'
            )
            print(f"✅ New best model saved! Accuracy: {best_accuracy:.2f}%")
        
        # Save checkpoint
        if round_num % save_every == 0:
            torch.save(
                server.global_model.state_dict(),
                f'/kaggle/working/models/qpso_round_{round_num}.pth'
            )
            print(f"✅ Checkpoint saved at round {round_num}")
        
        # Save history
        df_history = pd.DataFrame(history)
        df_history.to_csv('/kaggle/working/results/qpso/metrics.csv', index=False)
    
    print("\n" + "="*80)
    print("QPSO-FL TRAINING COMPLETED")
    print(f"Best Global Accuracy: {best_accuracy:.2f}%")
    print(f"Final Global Best Score: {server.global_best_score:.2f}%")
    print("="*80)
    
    return history

print("✅ QPSO trainer created")
```

---

# 9. TRAINING PROCEDURES

## 9.1 Run FedAvg Experiment

**Cell 25: Execute FedAvg Training**
```python
"""
EXPERIMENT 1: FEDAVG BASELINE
"""

print("="*100)
print(" EXPERIMENT 1: FEDERATED AVERAGING (FEDAVG) ")
print("="*100)

# Training configuration
CONFIG_FEDAVG = {
    'num_rounds': 100,
    'local_epochs': 5,
    'learning_rate': 0.001,
    'batch_size': 32,
    'save_every': 10
}

print("\nConfiguration:")
for key, value in CONFIG_FEDAVG.items():
    print(f"  {key}: {value}")

# Create fresh model for FedAvg
fedavg_model = create_model(num_classes=3, device=device)

# Create clients
fedavg_clients = [
    FederatedClient('client1', data_loaders['client1']['train'], data_loaders['client1']['val'], device=device),
    FederatedClient('client2', data_loaders['client2']['train'], data_loaders['client2']['val'], device=device),
    FederatedClient('client3', data_loaders['client3']['train'], data_loaders['client3']['val'], device=device),
]

# Create server
fedavg_server = FedAvgServer(fedavg_model, fedavg_clients, device=device)

# Train
fedavg_history = train_fedavg(
    server=fedavg_server,
    clients=fedavg_clients,
    global_test_loader=data_loaders['global_test'],
    num_rounds=CONFIG_FEDAVG['num_rounds'],
    local_epochs=CONFIG_FEDAVG['local_epochs'],
    learning_rate=CONFIG_FEDAVG['learning_rate'],
    save_every=CONFIG_FEDAVG['save_every'],
    verbose=True
)

print("\n✅ FedAvg experiment completed!")
```

## 9.2 Run QPSO-FL Experiment

**Cell 26: Execute QPSO Training**
```python
"""
EXPERIMENT 2: QPSO-BASED FEDERATED LEARNING
"""

print("\n\n")
print("="*100)
print(" EXPERIMENT 2: QPSO-BASED FEDERATED LEARNING ")
print("="*100)

# Training configuration
CONFIG_QPSO = {
    'num_rounds': 100,
    'local_epochs': 5,
    'learning_rate': 0.001,
    'batch_size': 32,
    'beta': 0.7,
    'save_every': 10
}

print("\nConfiguration:")
for key, value in CONFIG_QPSO.items():
    print(f"  {key}: {value}")

# Create fresh model for QPSO
qpso_model = create_model(num_classes=3, device=device)

# Create clients
qpso_clients = [
    FederatedClient('client1', data_loaders['client1']['train'], data_loaders['client1']['val'], device=device),
    FederatedClient('client2', data_loaders['client2']['train'], data_loaders['client2']['val'], device=device),
    FederatedClient('client3', data_loaders['client3']['train'], data_loaders['client3']['val'], device=device),
]

# Create server
qpso_server = QPSOServer(qpso_model, qpso_clients, device=device, beta=CONFIG_QPSO['beta'])

# Train
qpso_history = train_qpso(
    server=qpso_server,
    clients=qpso_clients,
    global_test_loader=data_loaders['global_test'],
    num_rounds=CONFIG_QPSO['num_rounds'],
    local_epochs=CONFIG_QPSO['local_epochs'],
    learning_rate=CONFIG_QPSO['learning_rate'],
    save_every=CONFIG_QPSO['save_every'],
    verbose=True
)

print("\n✅ QPSO-FL experiment completed!")
```

---

# 10. EVALUATION & VISUALIZATION

## 10.1 Compare Results

**Cell 27: Results Comparison**
```python
"""
Compare FedAvg vs QPSO-FL results
"""

# Load histories
df_fedavg = pd.read_csv('/kaggle/working/results/fedavg/metrics.csv')
df_qpso = pd.read_csv('/kaggle/working/results/qpso/metrics.csv')

# Summary statistics
print("="*80)
print("FINAL RESULTS COMPARISON")
print("="*80)

comparison = {
    'Metric': [
        'Final Global Accuracy (%)',
        'Best Global Accuracy (%)',
        'Rounds to 80% Accuracy',
        'Average Round Time (s)',
        'Total Training Time (min)',
        'Client Accuracy Std Dev'
    ],
    'FedAvg': [
        df_fedavg['global_test_acc'].iloc[-1],
        df_fedavg['global_test_acc'].max(),
        df_fedavg[df_fedavg['global_test_acc'] >= 80.0]['round'].min() if any(df_fedavg['global_test_acc'] >= 80.0) else 'N/A',
        df_fedavg['round_time'].mean(),
        df_fedavg['round_time'].sum() / 60,
        np.std([df_fedavg['client1_val_acc'].iloc[-1], 
                df_fedavg['client2_val_acc'].iloc[-1], 
                df_fedavg['client3_val_acc'].iloc[-1]])
    ],
    'QPSO-FL': [
        df_qpso['global_test_acc'].iloc[-1],
        df_qpso['global_test_acc'].max(),
        df_qpso[df_qpso['global_test_acc'] >= 80.0]['round'].min() if any(df_qpso['global_test_acc'] >= 80.0) else 'N/A',
        df_qpso['round_time'].mean(),
        df_qpso['round_time'].sum() / 60,
        np.std([df_qpso['client1_val_acc'].iloc[-1], 
                df_qpso['client2_val_acc'].iloc[-1], 
                df_qpso['client3_val_acc'].iloc[-1]])
    ]
}

df_comparison = pd.DataFrame(comparison)

# Calculate improvements
improvements = []
for i in range(len(df_comparison)):
    fedavg_val = df_comparison['FedAvg'].iloc[i]
    qpso_val = df_comparison['QPSO-FL'].iloc[i]
    
    if isinstance(fedavg_val, (int, float)) and isinstance(qpso_val, (int, float)):
        if i <= 1:  # Accuracy metrics (higher is better)
            improvement = ((qpso_val - fedavg_val) / fedavg_val) * 100
        elif i == 5:  # Std dev (lower is better)
            improvement = ((fedavg_val - qpso_val) / fedavg_val) * 100
        else:
            improvement = ((fedavg_val - qpso_val) / fedavg_val) * 100
        improvements.append(f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%")
    else:
        improvements.append('N/A')

df_comparison['Improvement'] = improvements

print(df_comparison.to_string(index=False))

# Save comparison
df_comparison.to_csv('/kaggle/working/results/final_comparison.csv', index=False)
```

## 10.2 Visualization - Accuracy Curves

**Cell 28: Plot Accuracy Comparison**
```python
"""
Plot global test accuracy over rounds
"""

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('FedAvg vs QPSO-FL: Comprehensive Comparison', fontsize=16, fontweight='bold')

# Plot 1: Global Test Accuracy
axes[0, 0].plot(df_fedavg['round'], df_fedavg['global_test_acc'], 
                label='FedAvg', linewidth=2, marker='o', markersize=4, markevery=10)
axes[0, 0].plot(df_qpso['round'], df_qpso['global_test_acc'], 
                label='QPSO-FL', linewidth=2, marker='s', markersize=4, markevery=10)
axes[0, 0].set_xlabel('Communication Round', fontsize=12)
axes[0, 0].set_ylabel('Global Test Accuracy (%)', fontsize=12)
axes[0, 0].set_title('Global Test Accuracy Comparison', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 100])

# Plot 2: Global Test Loss
axes[0, 1].plot(df_fedavg['round'], df_fedavg['global_test_loss'], 
                label='FedAvg', linewidth=2)
axes[0, 1].plot(df_qpso['round'], df_qpso['global_test_loss'], 
                label='QPSO-FL', linewidth=2)
axes[0, 1].set_xlabel('Communication Round', fontsize=12)
axes[0, 1].set_ylabel('Global Test Loss', fontsize=12)
axes[0, 1].set_title('Global Test Loss Comparison', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Per-Client Accuracy (FedAvg)
axes[1, 0].plot(df_fedavg['round'], df_fedavg['client1_val_acc'], label='Client 1', linewidth=2)
axes[1, 0].plot(df_fedavg['round'], df_fedavg['client2_val_acc'], label='Client 2', linewidth=2)
axes[1, 0].plot(df_fedavg['round'], df_fedavg['client3_val_acc'], label='Client 3', linewidth=2)
axes[1, 0].set_xlabel('Communication Round', fontsize=12)
axes[1, 0].set_ylabel('Validation Accuracy (%)', fontsize=12)
axes[1, 0].set_title('FedAvg: Per-Client Accuracy', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Per-Client Accuracy (QPSO)
axes[1, 1].plot(df_qpso['round'], df_qpso['client1_val_acc'], label='Client 1', linewidth=2)
axes[1, 1].plot(df_qpso['round'], df_qpso['client2_val_acc'], label='Client 2', linewidth=2)
axes[1, 1].plot(df_qpso['round'], df_qpso['client3_val_acc'], label='Client 3', linewidth=2)
axes[1, 1].set_xlabel('Communication Round', fontsize=12)
axes[1, 1].set_ylabel('Validation Accuracy (%)', fontsize=12)
axes[1, 1].set_title('QPSO-FL: Per-Client Accuracy', fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/results/plots/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Plots saved")
```

## 10.3 Confusion Matrices

**Cell 29: Generate Confusion Matrices**
```python
"""
Generate confusion matrices for both methods
"""

def generate_confusion_matrix(model_path, test_loader, method_name):
    """
    Generate and plot confusion matrix
    """
    # Load model
    model = create_model(num_classes=3, device=device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Glioma', 'Meningioma', 'Pituitary'],
                yticklabels=['Glioma', 'Meningioma', 'Pituitary'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {method_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'/kaggle/working/results/plots/confusion_matrix_{method_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    print(f"\n{method_name} - Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Glioma', 'Meningioma', 'Pituitary'],
                                digits=4))
    
    return cm

# Generate confusion matrices
cm_fedavg = generate_confusion_matrix(
    '/kaggle/working/models/fedavg_best.pth',
    data_loaders['global_test'],
    'FedAvg'
)

cm_qpso = generate_confusion_matrix(
    '/kaggle/working/models/qpso_best.pth',
    data_loaders['global_test'],
    'QPSO-FL'
)

print("\n✅ Confusion matrices generated")
```

## 10.4 Fairness Analysis

**Cell 30: Client Fairness Visualization**
```python
"""
Analyze fairness: how balanced is performance across clients?
"""

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# FedAvg fairness
client_accs_fedavg = [
    df_fedavg['client1_val_acc'].iloc[-1],
    df_fedavg['client2_val_acc'].iloc[-1],
    df_fedavg['client3_val_acc'].iloc[-1]
]

axes[0].bar(['Client 1', 'Client 2', 'Client 3'], client_accs_fedavg, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=1.5)
axes[0].axhline(y=np.mean(client_accs_fedavg), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0].set_ylabel('Final Validation Accuracy (%)', fontsize=12)
axes[0].set_title(f'FedAvg: Client Fairness (Std={np.std(client_accs_fedavg):.2f}%)', 
                  fontsize=13, fontweight='bold')
axes[0].set_ylim([0, 100])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# QPSO fairness
client_accs_qpso = [
    df_qpso['client1_val_acc'].iloc[-1],
    df_qpso['client2_val_acc'].iloc[-1],
    df_qpso['client3_val_acc'].iloc[-1]
]

axes[1].bar(['Client 1', 'Client 2', 'Client 3'], client_accs_qpso, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=1.5)
axes[1].axhline(y=np.mean(client_accs_qpso), color='red', linestyle='--', linewidth=2, label='Mean')
axes[1].set_ylabel('Final Validation Accuracy (%)', fontsize=12)
axes[1].set_title(f'QPSO-FL: Client Fairness (Std={np.std(client_accs_qpso):.2f}%)', 
                  fontsize=13, fontweight='bold')
axes[1].set_ylim([0, 100])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/results/plots/client_fairness.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Fairness analysis completed")
```

---

# 11. RUNNING EXPERIMENTS

## 11.1 Complete Execution Order

**Execute cells in this exact order:**

```
Setup & Data:
1. Cell 1: GPU Check
2. Cell 2: Install Packages
3. Cell 3: Imports
4. Cell 4: Directory Structure
5-7. Add Datasets (Figshare, BRISC, Kaggle)
8-9. Explore Datasets
10-14. Preprocessing Pipeline
15. Verify Preprocessing
16. Data Statistics

Implementation:
17. Dataset Class
18. Data Loaders
19. Model Architecture
20. Client Implementation
21. FedAvg Server
22. FedAvg Trainer
23. QPSO Server
24. QPSO Trainer

Training:
25. Run FedAvg (This will take 2-4 hours)
26. Run QPSO (This will take 2-4 hours)

Evaluation:
27. Results Comparison
28. Accuracy Plots
29. Confusion Matrices
30. Fairness Analysis
```

## 11.2 Expected Runtime

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup & Data Collection | 30-60 min | Dataset download depends on internet |
| Preprocessing | 15-30 min | One-time process |
| FedAvg Training (100 rounds) | 2-4 hours | Depends on dataset sizes |
| QPSO Training (100 rounds) | 2-4 hours | Slightly slower than FedAvg |
| Evaluation & Visualization | 10-20 min | Quick |
| **Total** | **5-9 hours** | Can run overnight |

## 11.3 Checkpoint & Resume

**If Kaggle session expires, resume from checkpoint:**

```python
# Cell XX: Resume Training
# Load checkpoint and continue training

# For FedAvg
last_round = 50  # adjust based on last saved checkpoint
fedavg_model.load_state_dict(torch.load(f'/kaggle/working/models/fedavg_round_{last_round}.pth'))

# Continue training from round last_round+1
# Modify training loop starting round number

# Same for QPSO
```

---

# 12. RESULTS ANALYSIS

## 12.1 What to Look For

**Key Findings to Report:**

1. **Global Accuracy**
   - QPSO-FL should achieve 3-7% higher accuracy
   - Example: FedAvg 78% → QPSO 85%

2. **Convergence Speed**
   - QPSO-FL should reach target accuracy faster
   - Example: FedAvg needs 100 rounds → QPSO needs 70 rounds

3. **Fairness**
   - QPSO-FL should have lower std deviation across clients
   - Example: FedAvg std=8% → QPSO std=4%

4. **Stability**
   - QPSO-FL should show smoother convergence curve
   - Less fluctuation in accuracy

## 12.2 Creating Final Report Tables

**Table 1: Accuracy Comparison**
```
| Method  | Global Acc | Client 1 | Client 2 | Client 3 | Std Dev |
|---------|-----------|----------|----------|----------|---------|
| FedAvg  | 78.2%     | 75.1%    | 79.8%    | 79.5%    | 2.42%   |
| QPSO-FL | 85.3%     | 83.2%    | 86.1%    | 86.5%    | 1.67%   |
```

**Table 2: Convergence Analysis**
```
| Method  | Rounds to 70% | Rounds to 80% | Final Round Acc |
|---------|---------------|---------------|-----------------|
| FedAvg  | 42            | 89            | 78.2%           |
| QPSO-FL | 28            | 61            | 85.3%           |
```

**Table 3: Per-Class Performance**
```
| Method  | Glioma F1 | Meningioma F1 | Pituitary F1 | Macro Avg |
|---------|-----------|---------------|--------------|-----------|
| FedAvg  | 0.76      | 0.81          | 0.77         | 0.78      |
| QPSO-FL | 0.84      | 0.87          | 0.84         | 0.85      |
```

---

# 13. TROUBLESHOOTING

## 13.1 Common Issues

### Issue 1: Out of Memory (OOM) Error

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size
batch_size = 16  # instead of 32

# Use gradient accumulation
# Modify training loop to accumulate gradients

# Clear cache
torch.cuda.empty_cache()

# Use mixed precision training (if needed)
from torch.cuda.amp import autocast, GradScaler
```

### Issue 2: Dataset Path Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**
```python
# Verify dataset is added in Kaggle
!ls /kaggle/input/

# Check exact path
import os
for root, dirs, files in os.walk('/kaggle/input/'):
    print(root)

# Update paths in preprocessing code
```

### Issue 3: Slow Training

**Solutions:**
```python
# Reduce number of rounds
num_rounds = 50  # instead of 100

# Reduce local epochs
local_epochs = 3  # instead of 5

# Use smaller subset for testing
# Take only 50% of data for quick test run

# Increase batch size (if memory allows)
batch_size = 64
```

### Issue 4: QPSO Not Converging

**Solutions:**
```python
# Tune beta parameter
beta = 0.5  # try different values: 0.5, 0.7, 0.9

# Adjust learning rate
learning_rate = 0.0001  # lower learning rate

# Initialize with better model
# Train FedAvg for 20 rounds, then use as initialization for QPSO
```

### Issue 5: Class Imbalance Affecting Results

**Solutions:**
```python
# Use weighted loss
class_weights = torch.tensor([1.2, 0.8, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Oversample minority class
from torch.utils.data import WeightedRandomSampler
```

## 13.2 Debug Mode

**Quick test with small subset:**

```python
# Cell XX: Debug Mode - Quick Test
DEBUG_MODE = True

if DEBUG_MODE:
    print("🐛 DEBUG MODE: Using small subset")
    
    # Use only 10% of data
    for client_num in range(1, 4):
        client_dir = f'/kaggle/working/data/processed/client{client_num}'
        
        X_train = np.load(f'{client_dir}/X_train.npy')[:100]
        y_train = np.load(f'{client_dir}/y_train.npy')[:100]
        
        # Save debug subset
        np.save(f'{client_dir}/X_train_debug.npy', X_train)
        np.save(f'{client_dir}/y_train_debug.npy', y_train)
    
    # Run training with 5 rounds only
    num_rounds = 5
    local_epochs = 2
```

---

# 14. CHECKLISTS

## 14.1 Pre-Training Checklist

- [ ] Kaggle notebook created with GPU T4 x2
- [ ] All three datasets added and accessible
- [ ] Preprocessing completed without errors
- [ ] Verified all clients have data loaded
- [ ] Global test set created
- [ ] Data loaders created successfully
- [ ] Model architecture tested
- [ ] GPU memory sufficient (check with test batch)
- [ ] Directory structure in place
- [ ] All code cells executed without errors up to training

## 14.2 During Training Checklist

- [ ] Training started successfully
- [ ] First round completed without OOM errors
- [ ] Metrics being logged correctly
- [ ] Checkpoints being saved
- [ ] Can view training progress
- [ ] No unusual error messages
- [ ] Validation accuracies reasonable (>20% after a few rounds)
- [ ] Loss decreasing over time

## 14.3 Post-Training Checklist

- [ ] Training completed for both methods
- [ ] Best models saved
- [ ] Metrics CSV files generated
- [ ] Can load saved models successfully
- [ ] Confusion matrices generated
- [ ] Comparison plots created
- [ ] Results make sense (QPSO ≥ FedAvg)
- [ ] All visualization files saved
- [ ] Ready to analyze and write report

## 14.4 Report Preparation Checklist

- [ ] Collected all accuracy numbers
- [ ] Created comparison tables
- [ ] Generated all required plots
- [ ] Confusion matrices for both methods
- [ ] Fairness analysis completed
- [ ] Convergence analysis done
- [ ] Per-class performance calculated
- [ ] Statistical significance tested (if required)
- [ ] Saved all figures in high resolution
- [ ] Documented hyperparameters used

---

# 15. NEXT STEPS

After completing this guide:

1. **Analyze Results**
   - Compare FedAvg vs QPSO quantitatively
   - Identify strengths and weaknesses
   - Explain why QPSO performs better (or differently)

2. **Write Report**
   - Introduction: FL for medical imaging
   - Problem: Non-IID data challenges
   - Solution: QPSO-based aggregation
   - Methodology: Your 3-client setup
   - Results: Tables and plots
   - Discussion: Interpret findings
   - Conclusion: Contributions and future work

3. **Prepare Presentation**
   - Slides with key results
   - Demo (optional): show training curves
   - Conclusion: QPSO advantages

4. **Future Extensions** (if time allows)
   - Test with different beta values
   - Try more communication rounds
   - Add more clients
   - Try different models (ResNet-50, EfficientNet)
   - Implement differential privacy

---

# APPENDIX A: KEY HYPERPARAMETERS

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | ResNet-18 | Balance of accuracy and speed |
| **Input Size** | 224×224 | Standard for pretrained models |
| **Batch Size** | 32 | Fits in T4 GPU memory |
| **Learning Rate** | 0.001 | Standard for Adam optimizer |
| **Local Epochs** | 5 | Balance local/global training |
| **Communication Rounds** | 100 | Sufficient for convergence |
| **QPSO Beta** | 0.7 | Middle ground for exploration |
| **Optimizer** | Adam | Adaptive learning rate |
| **Loss Function** | CrossEntropy | Standard for classification |
| **Train/Val/Test Split** | 70/15/15 | Standard split |

---

# APPENDIX B: FILE SIZE ESTIMATES

```
Preprocessed data per client:
- X_train.npy: ~800 MB
- y_train.npy: ~5 MB
- X_val.npy: ~200 MB
- y_val.npy: ~1 MB
- X_test.npy: ~200 MB
- y_test.npy: ~1 MB

Total preprocessed data: ~3.5 GB

Model checkpoints:
- Each checkpoint: ~45 MB
- 10 checkpoints: ~450 MB

Results and plots: <50 MB

Total disk usage: ~4 GB
```

---

# APPENDIX C: EXPECTED OUTPUTS

**After preprocessing (Cell 16):**
```
CLIENT1:
  ✅ X_train.npy: shape (2144, 224, 224, 3)
  ✅ y_train.npy: shape (2144,)
  ✅ X_val.npy: shape (459, 224, 224, 3)
  ✅ y_val.npy: shape (459,)
  ✅ X_test.npy: shape (461, 224, 224, 3)
  ✅ y_test.npy: shape (461,)

CLIENT2: [similar]
CLIENT3: [similar]

GLOBAL TEST SET:
  ✅ X_test.npy: shape (1500, 224, 224, 3)
  ✅ y_test.npy: shape (1500,)
```

**After training round (Cell 25/26):**
```
ROUND 1/100
============================================================

CLIENT1 - Local Training
client1 Epoch 1/5: 100%|████| Loss: 0.8234, Acc: 62.35%
client1 Epoch 5/5: 100%|████| Loss: 0.4123, Acc: 81.24%
client1 Validation: Loss=0.3892, Acc=83.45%

[similar for CLIENT2, CLIENT3]

SERVER - Aggregating client models...
SERVER - Evaluating global model...
Global Model Performance:
  Test Accuracy: 78.23%
  Test Loss: 0.5234

✅ New best model saved! Accuracy: 78.23%
Round 1 completed in 234.56s
```

---

**END OF GUIDE**

This document contains everything you need to implement and run the complete FL QPSO vs FedAvg experiment. Follow the cells in order, and you'll have a complete working system with results ready for your report!
