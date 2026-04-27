# Chapter 4: System Design

## 4.1 Overall System Architecture

### 4.1.1 High-Level System Diagram

The system comprises three independent modules operating in sequence, with federated learning applied exclusively to Module 1:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATIENT MRI SCANS                            │
│              T1, T1ce, T2, FLAIR Modalities                     │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│  MODULE 1: FEDERATED CLASSIFICATION (Privacy-Preserved)         │
│  ├─ Hospital A (ResNet-18) ─┐                                   │
│  ├─ Hospital B (ResNet-18) ─┼─→ QPSO/FedAvg/FedProx ─→ Tumor   │
│  └─ Hospital C (ResNet-18) ─┘    Aggregation Server     Class   │
│  Output: Glioma / Meningioma / No Tumor / Pituitary             │
└─────────────────┬───────────────────────────────────────────────┘
                  │
        ┌─────────▼─────────┐
        │  Is Glioma?       │
        │  /           \    │
       YES             NO   │ [Stop for other tumor types]
        │               └────
        │
┌───────▼──────────────────────────────────────────────────────────┐
│  MODULE 2: 3D SEGMENTATION (Local Hospital Processing)          │
│  ├─ MONAI Preprocessing: RAS orientation, 1mm³ voxel size       │
│  ├─ 3D Attention U-Net: 4 in → 3 out (WT, TC, ET)              │
│  └─ Output: Tumor Masks + Volume Calculations                   │
│  Privacy: LOCAL ONLY (no data leaves hospital)                  │
└───────┬──────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────┐
│  MODULE 3: PROGRESSION FORECASTING (Local Hospital Processing)  │
│  ├─ Math Models: Logistic, Gompertz, Exponential, Linear       │
│  ├─ LSTM Hybrid: V_hybrid = V_math + LSTM_residual_correction  │
│  ├─ Grade-Stratified: Separate HGG and LGG models              │
│  └─ Output: 6-month Volume Prediction + Confidence              │
│  Privacy: LOCAL ONLY (no data leaves hospital)                  │
└───────┬──────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────┐
│         CLINICAL DECISION SUPPORT DASHBOARD                      │
│  ├─ Diagnosis (Classification + Confidence)                     │
│  ├─ Volumetric Analysis (Segmentation + Regions)               │
│  ├─ Prognosis (Growth Prediction + Uncertainty)                │
│  └─ Treatment Recommendations                                   │
└───────────────────────────────────────────────────────────────────┘
```

**See also:** `01_system_architecture.png` in `/diagrams/rendered/`

---

### 4.1.2 Federated Learning Communication Flow

The federated learning component (Module 1) follows a standard synchronous FL protocol:

```
ROUND t=1 to 100:
  ┌─────────────────────────────────────────────┐
  │ Server: Broadcast Global Model W_global^(t) │
  └────────────┬────────────────────────────────┘
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
  Client 1  Client 2  Client 3
  Train 5   Train 5   Train 5
  epochs    epochs    epochs
  on local  on local  on local
  private   private   private
  data      data      data
      │        │        │
      └────────┼────────┘
               ▼
  ┌─────────────────────────────────────────────┐
  │ Server: Receive W1, W2, W3 from clients    │
  │ Perform QPSO/FedAvg/FedProx Aggregation    │
  │ Evaluate on global test set                │
  │ W_global^(t+1) = Aggregate(W1, W2, W3)    │
  └─────────────────────────────────────────────┘
```

**See also:** `04_fl_sequence.png` (Mermaid Sequence Diagram) in `/diagrams/rendered/`

---

## 4.2 Module 1: Federated Classification Architecture

### 4.2.1 Client-Server Architecture

**Role: Central Server**
- Maintains global ResNet-18 model (11.2M parameters)
- Coordinates 100 federated rounds
- Implements aggregation strategy (FedAvg, FedProx, or QPSO)
- Evaluates on balanced global test set
- Logs metrics and checkpoints

**Role: Hospital Clients (3 instances)**
- Receive broadcast global model from server
- Perform local training on private hospital data (5 epochs per round)
- Send updated model weights back to server
- Receive updated global model for next round
- Compute local validation metrics (tracking fairness)

### 4.2.2 Model Architecture: ResNet-18 for Brain Tumor Classification

```
INPUT (224 × 224 × 3)
  ↓
[Conv 7×7, 64 filters, stride 2]
[BatchNorm + ReLU]
[MaxPool 3×3, stride 2]
  ↓
[ResNet Block 1: 64 filters × 2 blocks]
  ↓
[ResNet Block 2: 128 filters × 2 blocks]
  ↓
[ResNet Block 3: 256 filters × 2 blocks]
  ↓
[ResNet Block 4: 512 filters × 2 blocks]
  ↓
[GlobalAvgPool → 512-dim feature vector]
  ↓
[FC Layer: 512 → 3 classes]
  (Glioma, Meningioma, Pituitary)
  
OUTPUT: Logits (3-dim)
  ↓
[Softmax + Cross-Entropy Loss]
  ↓
Predicted Class + Confidence
```

**Configuration:**
- **Pretrained Weights:** ImageNet (transfer learning)
- **Total Parameters:** 11,178,051 (all trainable)
- **Memory:** ~45 MB model size
- **Input:** 224×224 RGB images
- **Output:** 3-class or 4-class probabilities (depending on experimental setup)

---

### 4.2.3 Aggregation Strategies

#### **Strategy 1: Federated Averaging (FedAvg)**

```python
# Pseudocode
for round t in 1..100:
    # Server broadcasts
    send w_global to all clients
    
    # Clients train locally
    for each client c:
        w_c ← local_training(data_c, w_global, epochs=5)
        send w_c to server
    
    # Server aggregates (weighted by dataset size)
    n_total ← sum of all client dataset sizes
    w_global ← (0, 0, 0)
    for each client c:
        w_global ← w_global + (n_c / n_total) * w_c
```

**Characteristics:**
- **Deterministic:** Same input always produces same output
- **Fast:** O(n_clients) computation, minimal memory
- **Non-memory:** No tracking of previous solutions
- **Prone to averaging:** May lose distinct features learned by individual clients

#### **Strategy 2: Federated Proximal (FedProx)**

```python
# Server side: SAME as FedAvg
# Client side: MODIFIED local training objective

for round t in 1..100:
    send w_global to all clients
    
    for each client c:
        # FedProx modification: Add proximal term
        for epoch in 1..5:
            for batch (x, y) in train_loader_c:
                pred ← model(x)
                loss_data ← CrossEntropy(pred, y)
                
                # Proximal regularization prevents drift
                loss_prox ← (μ / 2) * ||w_local - w_global||²
                loss_total ← loss_data + loss_prox
                
                optimizer.step(loss_total)
        
        w_c ← model.state_dict()
        send w_c to server
    
    w_global ← weighted_average(w_1, w_2, w_3)
```

**Characteristics:**
- **Regularized:** Proximal term penalizes large client divergence
- **μ = 0.01:** Tunable hyperparameter (higher = more constraint)
- **Intent:** Stabilize learning under non-IID data
- **Risk:** Can over-constrain, preventing beneficial local adaptation

#### **Strategy 3: Quantum Particle Swarm Optimization FL (QPSO-FL)** - Novel Contribution

```python
# Server-side aggregation strategy using QPSO

class QPSOServer:
    def __init__(self, clients, beta=0.7):
        self.personal_best = {}        # per-client best weights found
        self.personal_best_scores = {} # per-client best validation accuracy
        self.global_best = None        # best weights across all clients
        self.global_best_score = 0.0
        self.mean_best = None          # element-wise mean of pbests
        
    def initialize_particles(self):
        """Initialize all pbests and gbest to current global model"""
        for client in self.clients:
            self.personal_best[client.id] = deepcopy(self.global_model)
            self.personal_best_scores[client.id] = 0.0
        self.global_best = deepcopy(self.global_model)
    
    def aggregate_qpso(self, client_weights, client_val_accs):
        """
        QPSO Aggregation Step:
        1. Update personal best (pbest) and global best (gbest)
        2. Compute mean best (mbest)
        3. Apply quantum position update
        """
        
        # Step 1: Update pbest and gbest
        for client_id, weights in client_weights.items():
            val_acc = client_val_accs[client_id]
            
            if val_acc > self.personal_best_scores[client_id]:
                self.personal_best[client_id] = weights
                self.personal_best_scores[client_id] = val_acc
            
            if val_acc > self.global_best_score:
                self.global_best = weights
                self.global_best_score = val_acc
        
        # Step 2: Compute mean best (element-wise mean of all pbests)
        self.mean_best = compute_mean_state_dict(
            [self.personal_best[c_id] for c_id in self.clients]
        )
        
        # Step 3: Quantum position update for each parameter
        for param_name in self.global_model.state_dict():
            # Per-parameter vectors
            φ = random.uniform(0, 1)  # per-client attraction weight
            u = random.uniform(0.3, 1.0)  # quantum randomness
            
            # Attraction point: weighted combination of pbest and gbest
            p = φ * mean_best[param_name] + (1 - φ) * gbest[param_name]
            
            # Quantum update: stochastic jump with bounded magnitude
            perturbation = β * abs(mean_best[param_name] - gbest[param_name]) * ln(1/u)
            perturbation = clip(perturbation, -0.1, 0.1)  # Safety bounds
            
            new_position = p + random.choice([-1, 1]) * perturbation
            
            w_new[param_name] = new_position
        
        return w_new
```

**QPSO Advantages:**
- **Memory-Based:** Tracks pbest (personal best) and gbest (global best)
- **Stochastic Exploration:** Quantum jumps escape local optima
- **Adaptive:** Attraction point balances individual and global solutions
- **Fairness:** Validates based on loss surface (implicitly penalizes unfair models)

**Parameters:**
- **β = 0.7:** Contraction-expansion coefficient (exploration vs. exploitation)
- **u ∈ [0.3, 1.0]:** Quantum randomness range
- **Perturbation clamp [-0.1, 0.1]:** Safety constraint (prevents divergence)

**See also:** `05_aggregation_strategies.png` (comparison of all 3 strategies)

---

## 4.3 Module 2: 3D Brain Tumor Segmentation Architecture

### 4.3.1 Model Architecture: 3D Attention U-Net

```
INPUT: (1, 4, 128, 128, 128)  [batch, channels, depth, height, width]
       4 MRI modalities (T1, T1ce, T2, FLAIR)
  │
  ├─ ENCODER (Contracting Path)
  │  ├─ Conv3D (4→16) + ReLU + Conv3D (16→16)  [spatial res: 128³]
  │  │  │ ↓ MaxPool3D(2)
  │  ├─ Conv3D (16→32) + ReLU + Conv3D (32→32) [spatial res: 64³]
  │  │  │ ↓ MaxPool3D(2)
  │  ├─ Conv3D (32→64) + ReLU + Conv3D (64→64) [spatial res: 32³]
  │  │  │ ↓ MaxPool3D(2)
  │  ├─ Conv3D (64→128) + ReLU + Conv3D (128→128) [spatial res: 16³]
  │  │  │ ↓ MaxPool3D(2)
  │  └─ Conv3D (128→256) + ReLU + Conv3D (256→256) [spatial res: 8³]
  │
  ├─ BOTTLENECK
  │  └─ Conv3D (256→256)  [spatial res: 8³]
  │
  ├─ DECODER (Expanding Path)
  │  ├─ Upsample3D → Concat with encoder features [16³]
  │  ├─ AttentionGate(256, 128) → Conv3D (128+128→128)
  │  │
  │  ├─ Upsample3D → Concat [32³]
  │  ├─ AttentionGate(128, 64) → Conv3D (64+64→64)
  │  │
  │  ├─ Upsample3D → Concat [64³]
  │  ├─ AttentionGate(64, 32) → Conv3D (32+32→32)
  │  │
  │  ├─ Upsample3D → Concat [128³]
  │  └─ AttentionGate(32, 16) → Conv3D (16+16→16)
  │
  └─ OUTPUT HEAD
     └─ Conv3D (16→3) + Sigmoid
        Output: (1, 3, 128, 128, 128)
        3 channels: Tumor Core, Whole Tumor, Enhancing Tumor
```

**Attention Gate Mechanism (per skip connection):**
```
Encoder Features (from contracting path)  [spatial res: H×W×D, features: C_enc]
       │
       └─→ AttentionGate ←─ Decoder Features [spatial res: H×W×D, features: C_dec]
           │
           ├─ Conv(C_enc → C_agg)
           ├─ Conv(C_dec → C_agg) + Add → ReLU
           ├─ Conv(C_agg → 1) → Sigmoid  [Attention weights: H×W×D]
           │
           └─ Output: Encoder Features × Attention Weights (channel-wise)
                      ↓
           Refined Encoder Features (suppresses irrelevant regions)
```

**Configuration:**
- **Channels:** 16, 32, 64, 128, 256 (per MONAI standard)
- **Strides:** (2, 2, 2, 2) depth-wise
- **Patch Size (during training):** 96³ voxels
- **Loss Function:** Generalized Dice Loss (handles class imbalance)
- **Optimizer:** Adam (lr=1e-4, weight_decay=1e-5)
- **Training Epochs:** 20
- **Batch Size:** 1 (volumetric processing)

**See also:** `03_unet_architecture.png` (detailed layer structure)

### 4.3.2 Preprocessing Pipeline (MONAI)

```
Raw NIfTI File (multi-modality)
  ↓
[LoadImage] → Load all 4 modalities separately
  ↓
[Orientation] → Reorient to RAS (Right-Anterior-Superior) standard
  ↓
[Spacing] → Resample to 1.0 × 1.0 × 1.0 mm³ isotropic
  ↓
[NormalizeIntensity] → Z-score normalization (nonzero voxels only)
  ↓
[RandCropByPosNegLabel] → Extract 96³ patches (balanced pos/neg samples)
  ↓
Preprocessed Tensor (1, 4, 96, 96, 96)
  ↓
[Model Forward Pass]
  ↓
Segmentation Output (1, 3, 96, 96, 96)
  ↓
[Inverse Transforms] → Map back to original space
  ↓
Final Segmentation Mask (same space as input)
```

---

## 4.4 Module 3: Progression Forecasting Architecture

### 4.4.1 Mathematical Models Pipeline

```
Historical MRI Scans (t0, t1, t2, ...)
  ├─ Segmentation (Module 2) or External Annotation
  ├─ Extract Tumor Volume (mm³) per timepoint
  ├─ Compute Real time intervals (Δt in days/months)
  │
  └─→ FIT Mathematical Models (per-patient)
      │
      ├─ Exponential: V(t) = V0 * exp(λ*t)
      │  └─ Fit using scipy.optimize.curve_fit
      │
      ├─ Gompertz: V(t) = V0 * exp((a/b)*(1-exp(-b*t)))
      │  └─ Fit using scipy.optimize.curve_fit
      │
      ├─ Logistic: V(t) = K / (1 + ((K-V0)/V0)*exp(-r*t))
      │  └─ Fit using scipy.optimize.curve_fit
      │
      └─ Linear: V(t) = V0 + v*t
         └─ Fit using scipy.stats.linregress
      
      ↓
  EVALUATE Models (on test timepoints)
  ├─ Mean Absolute Error (MAE)
  ├─ Root Mean Squared Error (RMSE)
  ├─ R² (coefficient of determination)
  │
  └─→ SELECT Best-Fit Model (highest R²)
      │
      └─ Baseline Prediction: V_math(t_future)
```

**Grade-Stratified Fitting:**
- **HGG (High-Grade):** 89 patients, aggressive growth (exponential-like)
- **LGG (Low-Grade):** 22 patients, slow growth (logistic/gompertz)

---

### 4.4.2 LSTM Hybrid Architecture

```
Baseline Predictions V_math from Mathematical Models
  ├─ Per-timepoint: V_math(t0), V_math(t1), V_math(t2), ...
  │
  └─→ Compute Residuals (Learning Targets)
      residual(ti) = V_actual(ti) - V_math(ti)
      
      ↓
  Residual Time Series (padded/normalized)
  │
  └─→ LSTM Hybrid Architecture
      │
      INPUT: Sequence of residuals [T_lookback = 3 or 4 timepoints]
      │
      ├─ LSTM Layer (hidden_size=32)
      │  ├─ Input Gate: What information to add?
      │  ├─ Forget Gate: What information to discard?
      │  └─ Output Gate: What to expose?
      │     ↓
      │  Output: Hidden state (1, 32) + Cell state (1, 32)
      │
      ├─ Attention Layer (optional)
      │  ├─ Compute attention weights over LSTM hidden states
      │  ├─ Weighted sum of hidden states
      │  └─ Output: Context vector (32-dim)
      │
      ├─ Fully Connected Layers
      │  ├─ FC(32 → 64) + ReLU
      │  ├─ FC(64 → 32) + ReLU
      │  └─ FC(32 → 1)  [Output: Residual correction]
      │
      └─→ LSTM Prediction: ŷ = residual_correction
      
      ↓
  Hybrid Prediction
  └─ V_hybrid = V_math + LSTM_correction
```

**Loss Function:** Mean Squared Error (MSE) on residuals

**Training Configuration:**
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Learning Rate Scheduler:** ReduceLROnPlateau (factor=0.5, patience=10)
- **Epochs:** 50-100 (until convergence)
- **Batch Size:** 8 (sequences per batch)
- **Train/Test Split:** 80/20 per grade

---

### 4.4.3 Prediction Pipeline

```
FOR EACH Patient (with ≥ 2 historical timepoints):
  
  1. EXTRACT Historical Volumes
     └─ V_t0, V_t1, V_t2, ... (from segmentation/annotation)
  
  2. FIT Mathematical Models
     └─ Exponential, Gompertz, Logistic, Linear
     └─ Select best-fit using R²
  
  3. BASELINE PREDICTION (Mathematical)
     └─ V_baseline(t_future) = Best_fit_model(t_future)
  
  4. LSTM CORRECTION
     └─ Input: Residuals of past timepoints
     └─ Output: residual_correction(t_future)
  
  5. HYBRID PREDICTION
     └─ V_hybrid(t_future) = V_baseline(t_future) + residual_correction
  
  6. CONFIDENCE INTERVAL
     └─ Compute std dev of predictions across cross-validation folds
     └─ Return: [V_hybrid - 2*std, V_hybrid, V_hybrid + 2*std]
     └─ Interpretation: ~95% confidence interval
```

**See also:** `07_progression_pipeline.png` (detailed flowchart)

---

## 4.5 Cross-Module Data Flow

### 4.5.1 Complete End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PATIENT MRI DATA                               │
│          (T1, T1ce, T2, FLAIR per timepoint)                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   MODULE 1: FL      │
                │ CLASSIFICATION      │
                │ Glioma vs Others    │
                └──────────┬──────────┘
                           │
              ┌────────────▼────────────┐
              │  Is Glioma?             │
              │  /                 \    │
             YES                   NO   │
              │                     └────
              │
        ┌─────▼────────────────────┐
        │   MODULE 2:              │
        │   SEGMENTATION           │
        │   (Attention U-Net)      │
        │   WT, TC, ET masks       │
        │   + Volume(mm³)          │
        └─────┬────────────────────┘
              │
        ┌─────▼────────────────────────┐
        │   MODULE 3:                  │
        │   PROGRESSION FORECASTING    │
        │   (Math + LSTM Hybrid)       │
        │   6-month prediction         │
        └─────┬────────────────────────┘
              │
        ┌─────▼────────────────────────┐
        │  CLINICAL DECISION ENGINE    │
        │  Diagnosis + Prognosis       │
        │  Recommendations             │
        └──────────────────────────────┘

See also: `02_data_flow.png` (Mermaid flow diagram)
          `09_integration_architecture.png` (detailed integration)
```

---

## 4.6 Error Handling and Data Validation

### 4.6.1 Input Validation (All Modules)

**Classification Module:**
- Check image dimensions: must be 224×224
- Check color channels: must be 3 (RGB)
- Check pixel range: must be [0, 1] after normalization
- Reject images with >10% black borders (non-brain)

**Segmentation Module:**
- Check 3D volume dimensions: typically 155×188×155 (BraTS standard)
- Check 4 modalities present: T1, T1ce, T2, FLAIR
- Check pixel range after preprocessing: typically [-3, 3] (Z-norm)
- Reject volumes with >50% missing slices

**Progression Module:**
- Check ≥ 2 timepoints per patient (minimum for fitting)
- Check temporal ordering: timepoints must be chronologically ordered
- Check volume continuity: no abrupt 10x jumps (likely segmentation error)
- Flag suspicious outliers for manual review

### 4.6.2 Model Robustness

**Federated Learning:**
- If 1 client unavailable: Continue with 2 clients (degraded fairness evaluation)
- If validation loss diverges: Early stopping at round t
- Model checkpointing: Save best model per 10 rounds

**Segmentation:**
- Catch GPU out-of-memory: Reduce patch size dynamically
- Handle 1-2 modality inputs: Pad missing modalities with zeros (with warning)

**Progression:**
- Handle missing timepoints: Interpolate (with uncertainty flag)
- Handle contradictory models: Use ensemble prediction (average all model predictions)

---

## 4.7 System Configuration and Hyperparameters

### 4.7.1 Module 1: Federated Classification

| Parameter | Value | Rationale |
|---|---|---|
| **Global Rounds** | 100 | Sufficient for convergence; validated empirically |
| **Local Epochs** | 5 | Balance between local adaptation and communication overhead |
| **Batch Size** | 32 | GPU-friendly; good gradient estimates |
| **Learning Rate** | 0.001 (Adam) | Standard for ResNet fine-tuning |
| **QPSO β** | 0.7 | Balance exploration/exploitation |
| **QPSO u range** | [0.3, 1.0] | Bounded quantum randomness |
| **FedProx μ** | 0.01 | Moderate regularization |
| **Image Size** | 224×224 | ResNet standard input |

### 4.7.2 Module 2: Segmentation

| Parameter | Value | Rationale |
|---|---|---|
| **Training Epochs** | 20 | Fast convergence; generalized from Phase 1 |
| **Batch Size** | 1 | Volumetric 3D processing (memory-intensive) |
| **Learning Rate** | 1e-4 (Adam) | Conservative for fine-tuning |
| **Weight Decay** | 1e-5 | L2 regularization (avoid overfitting) |
| **Patch Size** | 96³ voxels | Balance receptive field and memory |
| **Voxel Size** | 1.0 × 1.0 × 1.0 mm³ | Isotropic resampling (standard) |
| **Loss Function** | Generalized Dice | Handles class imbalance well |

### 4.7.3 Module 3: Progression

| Parameter | Value | Rationale |
|---|---|---|
| **LSTM Hidden Size** | 32 | Lightweight (few parameters) |
| **LSTM Layers** | 1 | Single layer sufficient for residuals |
| **Attention** | Optional | Improves interpretability |
| **FC Layers** | 2 (64→32→1) | Simple, avoids overfitting |
| **Optimizer** | Adam (lr=0.001) | Standard for LSTM |
| **Loss Function** | MSE | Minimizes prediction error magnitude |
| **Epochs** | 50-100 | Until validation loss plateaus |

---

## 4.8 Chapter Summary

| Component | Design Decision | Rationale |
|---|---|---|
| **Module Separation** | FL only for Classification | Efficiency + Privacy + Clinical feasibility |
| **Aggregation** | QPSO over FedAvg/FedProx | Superior fairness (72-81% max-min gap reduction) |
| **Classification** | ResNet-18 + Transfer Learning | Proven on medical imaging; good accuracy-efficiency trade-off |
| **Segmentation** | 3D Attention U-Net + MONAI | State-of-the-art; handles multi-modality well |
| **Progression** | Hybrid Math + LSTM | Interpretability + Flexibility |
| **Hyperparameters** | Empirically validated | Balanced for convergence vs. computational cost |

---

**Next:** Proceed to Chapter 5 for implementation details and code snippets.
