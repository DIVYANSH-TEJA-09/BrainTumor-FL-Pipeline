# Chapter 5: Implementation

## 5.1 Development Environment Setup

### 5.1.1 Infrastructure Requirements

**Hardware:**
- GPU: NVIDIA Tesla P100/T4 (16GB+ VRAM) recommended
- RAM: 32GB system memory (for full dataset preloading)
- Storage: 100GB+ (raw data + models + results)
- Network: ≥ 10Mbps (for dataset downloads)

**Software Stack:**
- Python 3.10+
- PyTorch 2.0+
- MONAI 1.3+ (medical imaging framework)
- scikit-learn 1.3+
- NumPy 1.24+, Pandas
- Matplotlib/Seaborn (visualization)
- Kaggle API (dataset access)

**Recommended Platforms:**
- Kaggle Notebooks (free GPU, simplified setup)
- Google Colab (free T4 GPU)
- Local machine with NVIDIA GPU + CUDA 11.8/12.1
- Docker container (reproducible environment)

---

## 5.2 Module 1: Federated Classification Implementation

### 5.2.1 Model Definition: ResNet-18 Brain Tumor Classifier

**File:** `federated_learning/src/model.py`

```python
"""
ResNet-18 model for brain tumor classification.
Pretrained on ImageNet; final FC layer replaced for 3-class output.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class BrainTumorResNet(nn.Module):
    """
    ResNet-18 with custom FC head for brain tumor classification.
    Input: (B, 3, 224, 224) RGB images
    Output: (B, num_classes) logits
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        
        # Replace final FC layer
        num_features = self.model.fc.in_features  # 512
        self.model.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)

def create_model(num_classes=3, device="cuda"):
    """Instantiate model, move to device, print summary"""
    model = BrainTumorResNet(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb = total_params * 4 / (1024 ** 2)
    
    print(f"Model: ResNet-18 | Params: {total_params:,} | "
          f"Trainable: {trainable_params:,} | Size: ~{mb:.1f}MB")
    return model
```

---

### 5.2.2 Federated Client Implementation

**File:** `federated_learning/src/client.py`

```python
"""
FederatedClient: Handles local training for one hospital/client.
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

class FederatedClient:
    """
    Represents one hospital/silo in federated setting.
    Typical round:
        client.set_model(global_model)
        client.set_optimizer(lr=0.001)
        weights, losses, accs = client.train_local(epochs=5)
        val_loss, val_acc = client.validate()
    """
    def __init__(self, client_id, train_loader, val_loader, device="cuda"):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.dataset_size = len(train_loader.dataset)
    
    def set_model(self, global_model):
        """Deep-copy server's global model to local client"""
        self.model = copy.deepcopy(global_model)
        self.model.to(self.device)
    
    def set_optimizer(self, learning_rate=0.001):
        """Initialize optimizer with client's model parameters"""
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train_local(self, epochs=5, verbose=False):
        """
        Local training on private hospital data.
        Returns: (state_dict, train_losses, train_accs)
        """
        self.model.train()
        epoch_losses, epoch_accs = [], []
        
        for ep in range(epochs):
            running_loss = 0.0
            correct = total = 0
            
            pbar = tqdm(self.train_loader, 
                       desc=f"{self.client_id} Epoch {ep+1}/{epochs}",
                       disable=not verbose, leave=False)
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            ep_loss = running_loss / len(self.train_loader)
            ep_acc = 100.0 * correct / total
            epoch_losses.append(ep_loss)
            epoch_accs.append(ep_acc)
        
        return self.model.state_dict(), epoch_losses, epoch_accs
    
    def validate(self):
        """Evaluate on local validation set. Returns (loss, accuracy%)"""
        self.model.eval()
        val_loss = 0.0
        correct = total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100.0 * correct / total
        return val_loss, val_acc
```

---

### 5.2.3 QPSO Server Implementation (Core Aggregation)

**File:** `federated_learning/src/server_qpso.py` (excerpt)

```python
"""
QPSOServer: Quantum Particle Swarm Optimization for FL aggregation.
Maintains pbest (personal best), gbest (global best), mbest (mean best).
"""
import copy
import torch
import math

class QPSOServer:
    """Central server using QPSO-based aggregation strategy."""
    
    def __init__(self, global_model, clients, device="cuda", beta=0.7):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.beta = beta
        
        # Per-client personal bests and scores
        self.personal_best = {}
        self.personal_best_scores = {}
        
        # Global best
        self.global_best = None
        self.global_best_score = 0.0
        
        # Mean best (computed each round)
        self.mean_best = None
        print(f"QPSO Server initialized | Clients={len(clients)} | β={beta}")
    
    def initialize_particles(self):
        """Set all pbest and gbest to current global model"""
        state = copy.deepcopy(self.global_model.state_dict())
        for c in self.clients:
            self.personal_best[c.client_id] = copy.deepcopy(state)
            self.personal_best_scores[c.client_id] = 0.0
        self.global_best = copy.deepcopy(state)
        self.global_best_score = 0.0
        print("✅ QPSO particles initialized")
    
    def update_personal_best(self, client_id, weights, val_acc):
        """Update client's personal best if new accuracy is better"""
        if val_acc > self.personal_best_scores[client_id]:
            self.personal_best[client_id] = copy.deepcopy(weights)
            self.personal_best_scores[client_id] = val_acc
            return True
        return False
    
    def update_global_best(self, client_id, val_acc):
        """Update global best if this client is better"""
        if val_acc > self.global_best_score:
            self.global_best = copy.deepcopy(self.personal_best[client_id])
            self.global_best_score = val_acc
            return True
        return False
    
    def calculate_mean_best(self):
        """Compute element-wise mean of all personal-best state_dicts"""
        first_id = self.clients[0].client_id
        self.mean_best = copy.deepcopy(self.personal_best[first_id])
        
        # Average all pbests
        for key in self.mean_best:
            sum_tensor = torch.zeros_like(self.mean_best[key])
            for client in self.clients:
                sum_tensor += self.personal_best[client.client_id][key]
            self.mean_best[key] = sum_tensor / len(self.clients)
    
    def aggregate_qpso(self):
        """
        QPSO aggregation step:
        For each parameter: x_new = p ± β * |mbest - x| * ln(1/u)
        where p = φ*mbest + (1-φ)*gbest is the attraction point
        """
        self.calculate_mean_best()
        w_new = copy.deepcopy(self.global_best)
        
        for param_name in w_new:
            # Per-parameter quantum update
            phi = torch.rand(1).item()  # ∈ [0,1]
            u = torch.rand(1).item()  # ∈ [0,1], then clamp to [0.3, 1.0]
            u = max(u, 0.3)
            
            # Attraction point
            p = (phi * self.mean_best[param_name] + 
                 (1 - phi) * self.global_best[param_name])
            
            # Quantum position update
            if u > 0:
                ln_term = math.log(1.0 / u)
                perturbation = (self.beta * 
                               torch.abs(self.mean_best[param_name] - 
                                        self.global_best[param_name]) * 
                               ln_term)
                
                # Clamp perturbation for stability
                perturbation = torch.clamp(perturbation, -0.1, 0.1)
                
                # Random sign
                sign = 1 if torch.rand(1).item() > 0.5 else -1
                w_new[param_name] = p + sign * perturbation
        
        # Load aggregated weights into global model
        self.global_model.load_state_dict(w_new)
        return self.global_best_score
    
    def federated_round(self, round_num, test_loader):
        """Execute one complete FL round"""
        # Broadcast global model to all clients
        for client in self.clients:
            client.set_model(self.global_model)
        
        # Local training
        client_weights = {}
        for client in self.clients:
            _, _, _ = client.train_local(epochs=5, verbose=False)
            val_loss, val_acc = client.validate()
            
            client_weights[client.client_id] = client.model.state_dict()
            self.update_personal_best(client.client_id, 
                                     client.model.state_dict(), 
                                     val_acc)
            self.update_global_best(client.client_id, val_acc)
        
        # QPSO Aggregation
        best_score = self.aggregate_qpso()
        
        # Evaluate on global test set
        test_acc = self.evaluate_on_test_set(test_loader)
        
        return best_score, test_acc
    
    def evaluate_on_test_set(self, test_loader):
        """Evaluate global model on balanced test set"""
        self.global_model.eval()
        correct = total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.global_model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100.0 * correct / total
```

---

## 5.3 Module 2: Segmentation Implementation

### 5.3.1 Data Loading and Preprocessing (MONAI)

```python
"""
Segmentation preprocessing using MONAI framework.
Handles 3D MRI volumes with 4 modalities (T1, T1ce, T2, FLAIR).
"""
from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd, 
    Orientationd, Spacingd, RandCropByPosNegLabel,
    EnsureChannelFirstd, EnsureTyped
)
from monai.data import Dataset, DataLoader

def get_preprocessing_transforms(training=True):
    """Define preprocessing pipeline for BraTS data"""
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        
        # Reorient to RAS standard
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Resample to 1mm³ isotropic
        Spacingd(keys=["image", "label"], 
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")),
        
        # Normalize intensity (nonzero voxels only)
        NormalizeIntensityd(keys=["image"], 
                           nonzero=True, 
                           channel_wise=True),
        
        # Random crop during training (96³ patches, balanced pos/neg)
        RandCropByPosNegLabel(keys=["image", "label"],
                             label_key="label",
                             spatial_size=(96, 96, 96),
                             pos=1.0,  # 50% positive (tumor) samples
                             neg=1.0,  # 50% negative (background)
                             num_samples=4) if training else None,
    ])
    
    # Filter None
    return Compose([t for t in transforms if t is not None])

# Load dataset
train_transforms = get_preprocessing_transforms(training=True)
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
```

### 5.3.2 Inference: Sliding Window with Model

```python
"""
3D Segmentation inference using sliding window approach.
Handles large volumes by processing overlapping patches.
"""
from monai.inferers import sliding_window_inference

def segment_patient(model, patient_image_path, device):
    """
    Segment one patient's 3D MRI volume.
    Args:
        model: Trained AttentionUnet
        patient_image_path: Path to .nii.gz file
        device: torch device (cuda/cpu)
    Returns:
        predictions: (1, 3, H, W, D) segmentation mask
    """
    # Load and preprocess
    sample = train_transforms({"image": patient_image_path, 
                              "label": patient_image_path})
    image = sample["image"].unsqueeze(0).to(device)  # (1, 4, H, W, D)
    
    model.eval()
    with torch.no_grad():
        # Sliding window inference: process in 96³ patches
        outputs = sliding_window_inference(
            inputs=image,
            roi_size=(96, 96, 96),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5  # 50% overlap between patches
        )
        
        # Apply sigmoid + threshold
        predictions = (outputs.sigmoid() > 0.5).float()
    
    return predictions  # (1, 3, H, W, D)

# Example usage
model = AttentionUnet(spatial_dims=3, in_channels=4, out_channels=3)
model.load_state_dict(torch.load("best_model.pth", map_location=device))

pred = segment_patient(model, "patient_T1.nii.gz", device)
print(f"Segmentation shape: {pred.shape}")
print(f"Tumor voxels: TC={pred[0,0].sum():.0f}, "
      f"WT={pred[0,1].sum():.0f}, ET={pred[0,2].sum():.0f}")
```

---

## 5.4 Module 3: Progression Implementation

### 5.4.1 Mathematical Model Fitting

**File:** `progression/src/01_mathematical_models.py` (excerpt)

```python
"""
Fit mathematical models to tumor volume time series.
"""
import numpy as np
from scipy.optimize import curve_fit

class MathematicalModels:
    """Classical tumor growth models"""
    
    @staticmethod
    def logistic(t, V0, K, r):
        """V(t) = K / (1 + ((K-V0)/V0)*exp(-r*t))"""
        return K / (1 + ((K - V0) / V0) * np.exp(-r * t))
    
    @staticmethod
    def gompertz(t, V0, a, b):
        """V(t) = V0 * exp((a/b)*(1-exp(-b*t)))"""
        return V0 * np.exp((a / b) * (1 - np.exp(-b * t)))
    
    @staticmethod
    def exponential(t, V0, lam):
        """V(t) = V0 * exp(λ*t)"""
        return V0 * np.exp(lam * t)

def fit_models_per_patient(times, volumes, grade="HGG"):
    """
    Fit all models to patient data.
    Args:
        times: Array of time points (days from first scan)
        volumes: Array of tumor volumes (mm³)
        grade: 'HGG' or 'LGG' for model parameter initialization
    Returns:
        best_model_name, best_r2, fitted_params
    """
    results = {}
    
    # Logistic fit
    try:
        V0 = volumes[0]
        K = volumes.max() * 1.5
        r = 0.01
        popt, _ = curve_fit(MathematicalModels.logistic, 
                           times, volumes,
                           p0=[V0, K, r],
                           bounds=([V0*0.5, K, 0.0001], 
                                   [V0*1.5, K*2, 1.0]))
        pred = MathematicalModels.logistic(times, *popt)
        r2 = 1 - np.sum((volumes - pred)**2) / np.sum((volumes - volumes.mean())**2)
        results['logistic'] = {'params': popt, 'r2': r2, 'pred': pred}
    except:
        pass
    
    # Select best model
    best_model = max(results, key=lambda k: results[k]['r2'])
    return best_model, results[best_model]['r2'], results[best_model]['params']
```

### 5.4.2 LSTM Hybrid Training

**File:** `progression/src/07_hybrid_lstm_training.py` (excerpt)

```python
"""
Train LSTM to learn residuals from mathematical model.
"""
import torch
import torch.nn as nn

class ResidualLSTM(nn.Module):
    """LSTM learns where mathematical model fails"""
    
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """x: (batch, seq_len, 1) residual sequence"""
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        x = self.relu(self.fc1(last_hidden))
        x = self.relu(self.fc2(x))
        residual_correction = self.fc3(x)
        
        return residual_correction

def train_hybrid_lstm(model, train_loader, val_loader, device, epochs=50):
    """Train LSTM on residual sequences"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

```

---

## 5.5 Key Implementation Details

### 5.5.1 Data Privacy in FL

**How FL achieves privacy:**

```python
# PRIVATE (on Hospital Client)
hospital_data = load_private_patient_data()  # Raw images
hospital_model.train(hospital_data)  # Train locally
updated_weights = hospital_model.state_dict()  # Extract weights only

# TRANSMITTED (Hospital → Server)
# Only send: updated_weights  (~45 MB)
# Never send: hospital_data, intermediate activations, gradients

# SERVER receives only weights, aggregates them
global_weights = aggregate([hospital1_weights, hospital2_weights, hospital3_weights])

# Result: Hospital data never leaves premises ✓
```

### 5.5.2 Federated Round Pseudocode

```python
# Main FL Training Loop
for round_t in range(1, 101):
    # 1. Server broadcasts to all clients
    for client in clients:
        client.receive_global_model(global_model)
    
    # 2. Clients train locally (parallel)
    client_weights = []
    for client in clients:
        weights, losses, accs = client.train_local(epochs=5)
        val_loss, val_acc = client.validate()
        client_weights.append((weights, val_acc))
    
    # 3. Server aggregates
    if strategy == "FedAvg":
        global_weights = fedavg_aggregate(client_weights)
    elif strategy == "QPSO":
        global_weights = qpso_aggregate(client_weights)
    
    # 4. Evaluate on global test set
    test_acc = evaluate(global_model, test_loader)
    
    # Log metrics
    log(round=round_t, test_acc=test_acc, fairness_sigma=fairness_metric)

# Save final model
torch.save(global_model.state_dict(), "final_fl_model.pth")
```

---

## 5.6 Performance Optimizations

### 5.6.1 GPU Memory Optimization

- **Gradient Checkpointing:** Save only select activations, recompute others during backprop
- **Mixed Precision:** Use float16 for forward pass, float32 for loss
- **Batch Size Tuning:** Max batch size before OOM; found to be 32 for ResNet-18 on 16GB GPU
- **Data Prefetching:** Load next batch while GPU processes current batch

### 5.6.2 Communication Efficiency

- **Model Quantization (Future):** Compress weights from float32 → int8 (4x reduction)
- **Gradient Sparsification (Future):** Send only top-k gradients
- **Synchronous SGD (Current):** All clients must finish before aggregation (simple but suboptimal)

---

## 5.7 Chapter Summary

| Component | Implementation | Status |
|---|---|---|
| **FL Classification** | FedAvg, FedProx, QPSO aggregation | ✅ Complete |
| **ResNet-18 Training** | Local client training + global aggregation | ✅ Complete |
| **3D Segmentation** | MONAI U-Net + sliding window inference | ✅ Complete |
| **Progression Forecasting** | Math models + LSTM hybrid | ✅ Complete |
| **Privacy** | Structural (weights-only FL) | ✅ Complete |
| **Code Organization** | Modular src/, models/, results/ | ✅ Complete |

---

**Next:** Proceed to Chapter 6 for comprehensive testing and validation.
