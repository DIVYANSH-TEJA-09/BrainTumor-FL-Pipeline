# Chapter 3: System Analysis

## 3.1 Functional Requirements

### 3.1.1 Module 1: Federated Classification (FL Component)

**Requirement FL-C1: Privacy-Preserving Distributed Training**
- System shall enable 3 hospitals to train a shared classification model without sharing raw patient MRI images
- Only model weights (state_dict) shall be transmitted between clients and central server
- Raw patient data shall never leave hospital premises

**Requirement FL-C2: Multi-Aggregation Strategy Support**
- System shall support three aggregation algorithms:
  - FedAvg: Standard weighted averaging baseline
  - FedProx: Proximal regularization variant for non-IID data
  - QPSO-FL: Quantum-inspired particle swarm optimization (novel method)
- Each strategy shall be independently executable with same clients/data for fair comparison

**Requirement FL-C3: Classification Accuracy**
- System shall classify brain tumors into 4 classes: Glioma, Meningioma, No Tumor, Pituitary
- Minimum global test accuracy target: ≥ 95%
- Per-class performance shall be tracked (precision, recall, F1, ROC-AUC)

**Requirement FL-C4: Fairness Under Data Heterogeneity**
- System shall maintain minimum acceptable accuracy for smallest/most data-deprived client (≥ 80% under moderate label skew)
- Fairness metric: Standard deviation of per-client accuracies shall be minimized
- System shall explicitly protect minority clients from being sidelined by majority

**Requirement FL-C5: Convergence and Communication Efficiency**
- Training shall complete in ≤ 100 communication rounds
- Each round shall involve ≤ 5 local training epochs per client
- Total training time shall be ≤ 2 hours on standard GPU infrastructure

---

### 3.1.2 Module 2: Segmentation (Local, Non-Federated)

**Requirement SEG-1: Volumetric Segmentation**
- System shall segment 3D multimodal MRI volumes (4 input channels: T1, T1ce, T2, FLAIR)
- Output shall identify 3 tumor sub-regions: Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET)
- Segmentation shall be pixel-accurate (voxel-level classification)

**Requirement SEG-2: Clinical-Grade Accuracy**
- Mean Dice Score: ≥ 0.75 (clinical-grade threshold)
- Tumor Core Dice: ≥ 0.80 (clinically critical region)
- Enhancing Tumor Dice: ≥ 0.75

**Requirement SEG-3: Automated Attention**
- Model shall use Attention Gates to automatically focus on tumor regions
- Healthy tissue suppression shall reduce false positives in non-tumor areas

**Requirement SEG-4: 3D Volumetric Processing**
- System shall process full 3D volumes, not 2D slices
- Spatial context in 3D shall inform segmentation decisions

**Requirement SEG-5: Local Processing Only**
- Segmentation shall occur entirely at individual hospitals
- No federated learning, weight sharing, or central aggregation
- Each hospital maintains full autonomy over segmentation results

**KEY DISTINCTION:** Module 2 is **completely separate** from Module 1. Segmentation does NOT use federated learning.

---

### 3.1.3 Module 3: Progression Forecasting (Local, Non-Federated)

**Requirement PROG-1: Multi-Model Forecasting**
- System shall fit 4 mathematical models to longitudinal tumor volume data:
  - Exponential growth
  - Gompertz (S-curve with deceleration)
  - Logistic (bounded growth)
  - Linear (constant rate baseline)
- System shall automatically select best-fitting model per patient using R² metric

**Requirement PROG-2: Hybrid Deep Learning**
- System shall train LSTM networks to learn residuals (actual - mathematical prediction)
- Hybrid model: V_hybrid = V_math + LSTM_correction
- Hybrid model shall improve prediction accuracy by ≥ 5% over baseline

**Requirement PROG-3: Grade-Stratified Models**
- Separate models for HGG (High-Grade, aggressive growth) and LGG (Low-Grade, slow growth)
- Hyperparameters optimized per grade

**Requirement PROG-4: Prediction Interval and Confidence**
- System shall provide point predictions AND uncertainty intervals
- 6-month prediction horizon

**Requirement PROG-5: Local Processing Only**
- Progression forecasting occurs entirely at individual hospitals
- No federated components
- Clinical interpretability prioritized over black-box predictions

**KEY DISTINCTION:** Module 3 is **completely separate** from Module 1. Progression forecasting does NOT use federated learning.

---

## 3.2 Workflow and Clinical Process Flow

### 3.2.1 Correct Execution Order: Classify → Segment → Forecast

```
START
  ↓
[Patient MRI Scans: T1, T1ce, T2, FLAIR]
  ↓
┌──────────────────────────────────────────────────────────┐
│ MODULE 1: FEDERATED CLASSIFICATION (Privacy-Preserved) │
│ - 3 Hospitals → Central Server (QPSO Aggregation)       │
│ - Output: Tumor Type (Glioma/Meningioma/No Tumor/Pit) │
│ - Federated Learning: YES                                │
│ - Data Sharing: Weights only (no raw images)            │
└────────────────┬─────────────────────────────────────────┘
                 ↓
    Is Classification = Glioma? 
           /             \
          YES             NO
           ↓               ↓
    ┌──────────────┐   [STOP]
    │ CONTINUE     │   (Other tumor types do not
    │ to Seg.      │    require downstream
    │              │    segmentation/progression
    └──────┬───────┘    in this pipeline)
           ↓
┌──────────────────────────────────────────────────────────┐
│ MODULE 2: LOCAL 3D SEGMENTATION (Hospital-Only)         │
│ - Attention U-Net on BraTS-2021 standard                │
│ - Output: Tumor masks (WT, TC, ET volumes)             │
│ - Federated Learning: NO                                │
│ - Data Sharing: None (all processing local)            │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│ MODULE 3: LOCAL PROGRESSION FORECASTING (Hospital-Only) │
│ - Math models + LSTM hybrid on historical scans        │
│ - Output: 6-month volume prediction + confidence       │
│ - Federated Learning: NO                                │
│ - Data Sharing: None (all processing local)            │
└────────────────┬─────────────────────────────────────────┘
                 ↓
[Clinical Decision: Diagnosis + Prognosis + Treatment Plan]
                 ↓
END
```

**CRITICAL POINT:** Non-Glioma tumors exit at Module 1. This is clinically appropriate:
- Meningiomas: Usually extradural, less complex segmentation requirements
- Pituitary: Small, standardized treatment often does not require volumetric analysis
- No Tumor: No further analysis needed

---

### 3.2.2 Data Flow Diagram (DFD)

**Level 0 (Context Diagram):**

```
     Hospital A              Hospital B              Hospital C
    (~1,200 imgs)           (~3,900 imgs)           (~4,200 imgs)
         │                       │                        │
         ├──────────────────────┬┴────────────────────────┤
                               ↓
                     ┌─────────────────────┐
                     │ CENTRAL FL SERVER   │
                     │ QPSO Aggregator     │
                     └──────────┬──────────┘
                               ↓
                    [Converged Global Model]
                               ↓
                    ┌──────────┴──────────┐
                    ↓                     ↓
              Hospital A           Hospital B            Hospital C
          [Segmentation]       [Segmentation]       [Segmentation]
          [Progression]        [Progression]        [Progression]
              ↓                     ↓                    ↓
          [Diagnosis +          [Diagnosis +        [Diagnosis +
           Prognosis]           Prognosis]          Prognosis]
```

**See also:** Diagram `02_data_flow.png` and `09_integration_architecture.png` in `/diagrams/rendered/`

---

## 3.3 Data Requirements

### 3.3.1 Module 1: Classification Data

| Parameter | Specification |
|-----------|---|
| **Modality** | 2D MRI slices (preprocessed, brain-extracted) |
| **Input Size** | 224 × 224 pixels (per image) |
| **Color Channels** | 3 (converted from grayscale or multi-channel) |
| **Classes** | 4 (Glioma=0, Meningioma=1, No Tumor=2, Pituitary=3) |
| **Client 1 Dataset** | Masoud Brain Tumor (Test split) ≈ 1,200 images |
| **Client 2 Dataset** | BRISC 2025 ≈ 3,900 images |
| **Client 3 Dataset** | Masoud Brain Tumor (Train split) ≈ 4,200 images |
| **Total** | ≈ 9,300 images across 3 clients |
| **Train/Val/Test Split** | 70% / 15% / 15% per client |
| **Global Test Set** | Balanced combination from all 3 clients ≈ 1,400 images |
| **Preprocessing** | Resize to 224×224, normalize to [0,1], standardize per batch |
| **Augmentation** | Rotation (±15°), horizontal flip (p=0.5), brightness/contrast jitter |

**Privacy Consideration:** Data never leaves client boundaries. Each hospital maintains full control over its data.

---

### 3.3.2 Module 2: Segmentation Data

| Parameter | Specification |
|-----------|---|
| **Modality** | 3D Multimodal MRI (4 channels) |
| **Channels** | T1, T1ce (T1 contrast-enhanced), T2, FLAIR |
| **Voxel Size** | 1.0 × 1.0 × 1.0 mm³ (isotropic after resampling) |
| **Volume Size** | ~155 × 188 × 155 voxels (BraTS standard) |
| **Dataset** | BraTS 2021 challenge dataset |
| **Patients** | 369 total (both HGG and LGG) |
| **Regions** | 3 output channels (WT, TC, ET) |
| **File Format** | NIfTI (.nii.gz) |
| **Preprocessing** | RAS orientation, Z-norm, patch extraction (96³) during training |
| **Preprocessing Framework** | MONAI (Medical Open Network for AI) |

**Local Processing:** All segmentation data remains hospital-local. No sharing with federated server.

---

### 3.3.3 Module 3: Progression Data

| Parameter | Specification |
|-----------|---|
| **Modality** | Longitudinal 3D MRI volumes (time-series) |
| **Dataset** | MU-Glioma-Post (TCIA) |
| **Patients** | 203 total (111 successfully modeled) |
| **Subtypes** | HGG (89 patients), LGG (22 patients) |
| **Timepoints per Patient** | 3–6 real MRI scans with scan dates |
| **Total Timepoints** | 791 across all patients |
| **Time Interval** | Real clinical follow-up (days → years) |
| **Extracted Feature** | Tumor volume (mm³) per timepoint |
| **Extraction Method** | 3D U-Net segmentation + voxel counting |
| **Prediction Horizon** | 6 months forward |
| **Models** | Logistic, Gompertz, Exponential, Linear (baseline) |
| **Secondary Models** | LSTM hybrid (learns residuals) |

**Local Processing:** Progression forecasting occurs entirely at individual hospitals.

---

## 3.4 Non-Functional Requirements

### 3.4.1 Performance Requirements

| Requirement | Specification | Rationale |
|---|---|---|
| **Training Time (Module 1)** | ≤ 2 hours for 100 FL rounds | Practical for iterative research |
| **Inference Latency (Classification)** | ≤ 100ms per image | Near real-time clinical use |
| **Inference Latency (Segmentation)** | ≤ 30 seconds per 3D volume | Acceptable for clinical workflow |
| **Inference Latency (Progression)** | ≤ 5 seconds per patient | Acceptable for dashboard |

### 3.4.2 Privacy Requirements

| Requirement | Specification | Mechanism |
|---|---|---|
| **Structural Privacy** | Raw patient data never leaves hospital | FL: weights-only communication |
| **Audit Trail** | Log all model weight transmissions | Server-side logging |
| **Model Interpretability** | Support Grad-CAM for classification | Explains model decisions per image |
| **Local Autonomy** | Hospitals can opt-out of federation | FL protocol supports client dropout |

### 3.4.3 Reliability Requirements

| Requirement | Specification |
|---|---|
| **Model Robustness** | System maintains ≥ 80% accuracy under label skew |
| **Fault Tolerance** | FL continues if 1 client unavailable (degrades gracefully) |
| **Data Validation** | Input validation on all MRI dimensions and ranges |
| **Model Versioning** | Track all model checkpoints per FL round |

### 3.4.4 Maintainability Requirements

| Requirement | Specification |
|---|---|
| **Code Organization** | Modular structure: `src/`, `models/`, `results/`, `tests/` |
| **Documentation** | Inline code comments + docstrings on all classes/functions |
| **Testing** | Unit tests for data loaders, model forward pass, aggregation logic |
| **Reproducibility** | Fixed random seeds, versioned dependencies, logged hyperparameters |

---

## 3.5 System Constraints

### 3.5.1 Technical Constraints

**Federated Learning Architecture:**
- Synchronous updates: All clients must complete local training before aggregation
- Central server orchestration: Single server coordinates rounds (simplified vs. peer-to-peer)
- No client-to-client communication (simplified privacy model)

**Scalability Constraints:**
- Currently tested with 3 clients only
- Scaling to 5–10 hospitals untested (computational overhead increases ~O(n))
- Communication bandwidth assumes typical hospital internet (≤ model file size ≤ 50MB)

**Model Capacity Constraints:**
- ResNet-18: 11.2M parameters (standard, not ultra-lightweight)
- Local training assumes ≥ 1 GPU per client or shared GPU time

### 3.5.2 Data Constraints

**Classification:**
- Limited to 2D MRI slices (volumetric processing deferred to Module 2)
- Only supports brain-extracted images (preprocessing required)
- Fixed input resolution: 224×224

**Segmentation:**
- Full 3D volumes required (memory-intensive)
- Assumes 4-modality input (some hospitals may only have 1-2 modalities initially)

**Progression:**
- Requires ≥ 2 longitudinal timepoints per patient (some patients may have only 1 scan)
- Temporal gaps vary (not uniformly spaced)

### 3.5.3 Regulatory Constraints

- **HIPAA Compliance:** De-identification required; direct identifiers stripped
- **Institutional Review:** Requires IRB approval at each participating hospital
- **Data Retention:** Comply with retention policies at each institution

---

## 3.6 System Dependencies

### 3.6.1 Software Dependencies

| Component | Version | Rationale |
|---|---|---|
| **Python** | 3.10+ | Modern async support, type hints |
| **PyTorch** | 2.0+ | Latest optimization, CUDA 11.8/12.1 support |
| **MONAI** | 1.3+ | Medical imaging preprocessing (3D transformations) |
| **NumPy** | 1.24+ | Efficient numerical operations |
| **scikit-learn** | 1.3+ | Metrics, train-test-split utilities |
| **Matplotlib/Seaborn** | Latest | Visualization (plots, confusion matrices) |

### 3.6.2 Hardware Dependencies

| Component | Minimum | Recommended |
|---|---|---|
| **GPU Memory** | 8 GB VRAM | 16 GB VRAM (segmentation needs 15+ GB) |
| **CPU** | 4 cores | 8+ cores (multi-threaded data loading) |
| **RAM** | 16 GB | 32+ GB (full dataset preloading) |
| **Storage** | 50 GB | 100+ GB (raw data + models + results) |

### 3.6.3 Deployment Dependencies

- **Kaggle Notebooks:** For initial research (free GPU, simplified setup)
- **Local GPU Servers:** For reproducible experiments (NVIDIA GPU + CUDA drivers)
- **Streamlit:** Web dashboard for visualization (Python 3.8+)
- **Docker:** Optional containerization for multi-hospital deployment

---

## 3.7 Design Rationale: Why This Architecture?

### 3.7.1 Why Separate FL Only for Classification?

**Question:** Why not federate all three modules?

**Answers:**

1. **Privacy-Utility Trade-off:**
   - Classification: Direct output is tumor type (low cardinality, easy to anonymize)
   - Segmentation masks: Can leak anatomical patient information (risky to share)
   - Progression data: Time-series volumes are sensitive (unique patient signatures)
   - FL overhead only justified for high-privacy-value components

2. **Computational Efficiency:**
   - 3D segmentation requires sliding-window inference (expensive)
   - Federated 3D segmentation would require shipping massive 3D volumes or intermediate features (communication bottleneck)
   - Local segmentation is more efficient

3. **Clinical Workflow:**
   - Hospitals already have local segmentation/progression infrastructure
   - Classification is the novel collaborative component
   - Minimal disruption to existing hospital workflows

4. **Regulatory Simplicity:**
   - Federal learning adds complexity to compliance
   - Hospitals comfort with local-only processing for segments/forecasts
   - Federated classification is a pilot, proving value before broader adoption

### 3.7.2 Why QPSO Over FedAvg/FedProx?

**Research Question:** What aggregation strategy minimizes clinical inequity?

**QPSO Advantages (from literature review + project experiments):**

| Aspect | FedAvg | FedProx | QPSO-FL |
|---|---|---|---|
| **Under Label Skew** | 90.56% accuracy | 93.02% accuracy | 92.09% accuracy |
| **Client Fairness** | σ=12.82% (high disparity) | σ=6.05% (moderate) | **σ=3.65% (best)** |
| **Smallest Client Protection** | Client 1: 60.42% (fail) | Client 1: 77.50% (marginal) | **Client 1: 80.00% (viable)** |
| **Convergence Speed** | 6 rounds to 80% | 8 rounds to 80% | **2 rounds to 80%** |
| **Computational Overhead** | ~8s/round | ~10s/round | ~75s/round (acceptable for offline research) |

**Conclusion:** QPSO trades ~1% global accuracy for massive fairness gains. Clinically, protecting the weakest hospital is more important than marginal global accuracy improvements.

---

## 3.8 Chapter Summary

| Aspect | Specification |
|---|---|
| **Workflow Order** | Classify (FL) → IF Glioma → Segment → Forecast |
| **FL Scope** | Classification ONLY; Segmentation & Progression are local |
| **Clients** | 3 hospitals with heterogeneous datasets |
| **Fairness Focus** | QPSO minimizes client performance disparity |
| **Data Privacy** | Structural privacy (FL) for classification; local-only for others |
| **Target Accuracy** | ≥ 95% global, ≥ 80% for all clients (fairness) |

---

**Next:** Proceed to Chapter 4 for detailed system design and architecture diagrams.
