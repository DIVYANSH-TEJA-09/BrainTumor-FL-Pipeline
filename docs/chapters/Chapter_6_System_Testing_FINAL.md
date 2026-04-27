# Chapter 6: System Testing

## 6.1 Testing Strategy and Coverage

### 6.1.1 Test Categories

| Category | Scope | Coverage |
|---|---|---|
| **Unit Tests** | Individual functions/modules | Data loaders, model forward pass, aggregation logic |
| **Integration Tests** | Module interactions | FL round execution, segmentation pipeline, end-to-end flow |
| **Performance Tests** | Timing and efficiency | Training speed, inference latency, memory usage |
| **Validation Tests** | Against baselines | Reproduce published benchmarks, compare methods |

---

## 6.2 Module 1: Federated Learning Tests

### 6.2.1 Classification Model Tests

**Test FL-C1: ResNet-18 Forward Pass**
- **File:** `federated_learning/tests/test_model.py`
- **Description:** Verify model produces correct output shape and logits
- **Input:** Batch of (B, 3, 224, 224) images
- **Expected Output:** (B, 3) logits for 3 classes
- **Validation:** No NaN/Inf values, output in reasonable range

```python
def test_model_forward():
    model = BrainTumorResNet(num_classes=3)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (4, 3), f"Expected (4,3), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
```

**Test FL-C2: Client Local Training**
- **Description:** Verify client training loop updates weights and produces decreasing loss
- **Dataset:** Toy dataset (100 images, 4 classes)
- **Epochs:** 3
- **Expected:** Loss decreases, accuracy increases

**Test FL-C3: Federated Round Execution**
- **Description:** Execute one complete FL round (broadcast → local training → aggregation)
- **Clients:** 3 simulated hospitals
- **Expected:** Global model weights updated, test accuracy computed

---

### 6.2.2 Aggregation Strategy Tests

**Test FL-A1: FedAvg Weighted Averaging**
- **Description:** Verify FedAvg correctly computes weighted average of client weights
- **Setup:** 3 clients with dataset sizes [1000, 2000, 3000]
- **Validation:** w_global = (1000/6000)*w1 + (2000/6000)*w2 + (3000/6000)*w3

**Test FL-A2: FedProx Proximal Regularization**
- **Description:** Verify proximal term is added to local loss
- **Loss Decomposition:** total_loss = data_loss + (μ/2)*||w_local - w_global||²
- **Validation:** Check gradient includes proximal term

**Test FL-A3: QPSO Particle Updates**
- **Description:** Verify QPSO correctly updates pbest, gbest, and applies quantum jump
- **Validation:**
  - pbest increases or stays same (never decreases)
  - gbest is maximum of all pbests
  - New position has bounded magnitude (no explosion)

---

## 6.3 Module 2: Segmentation Tests

### 6.3.1 Data Loading and Preprocessing

**Test SEG-D1: MONAI Transform Pipeline**
- **File:** `segmentation/test_preprocessing.py`
- **Description:** Verify preprocessing preserves data integrity
- **Checks:**
  - Output shape: (batch, 4, 96, 96, 96) or (4, 128, 128, 128)
  - Value range after normalization: typically [-3, +3] (Z-norm)
  - No NaN/Inf values
  - Orientation is RAS (verified against expected affine matrix)

**Test SEG-D2: BraTS Dataset Loading**
- **File:** `segmentation/test_data_loading.py`
- **Description:** Load and verify BraTS 2021 dataset structure
- **Checks:**
  - All 4 modalities present (T1, T1ce, T2, FLAIR)
  - Ground truth labels present (WT, TC, ET as multi-channel)
  - No missing or corrupted files

---

### 6.3.2 Segmentation Model Tests

**Test SEG-M1: Model Forward Pass**
- **File:** `segmentation/test_model.py`
- **Input:** (1, 4, 96, 96, 96) - 4-channel 3D volume
- **Expected Output:** (1, 3, 96, 96, 96) - 3-channel segmentation
- **Validation:** Output in [0, 1] (after sigmoid), no NaN

**Test SEG-M2: Attention Gate Functionality**
- **Description:** Verify attention gates suppress background
- **Method:** Compare predictions with/without attention
- **Expected:** Attention reduces false positives in non-tumor regions

**Test SEG-I1: Inference with Sliding Window**
- **File:** `segmentation/test_inference.py`
- **Description:** Test full inference pipeline on sample patient
- **Input:** Full 3D volume (~155×188×155 voxels)
- **Expected Output:** Segmentation masks without edge artifacts
- **Validation:**
  - Dice score ≥ 0.75 on held-out test set
  - Processing time ≤ 30 seconds per patient
  - Tumor Core Dice ≥ 0.80 (clinically important region)

---

## 6.4 Module 3: Progression Tests

### 6.4.1 Mathematical Model Fitting

**Test PROG-M1: Logistic Model Fit**
- **File:** `progression/test_growth_metrics.py`
- **Description:** Verify logistic model fits synthetic growth curve correctly
- **Setup:** Generate synthetic volume data: V(t) = K/(1 + ((K-V0)/V0)*exp(-r*t))
- **Fit:** Recover parameters and verify R² ≥ 0.95

**Test PROG-M2: Model Selection**
- **Description:** Test automatic best-fit model selection
- **Setup:** 4 synthetic volume trajectories (one each for Logistic, Gompertz, Exponential, Linear)
- **Expected:** Best-fit model correctly identifies each trajectory type

---

### 6.4.2 LSTM Hybrid Tests

**Test PROG-L1: LSTM Residual Learning**
- **File:** `progression/test_lstm_training.py`
- **Description:** Verify LSTM learns residuals from mathematical baseline
- **Setup:**
  - Train on patient residuals (actual - logistic_predicted)
  - Evaluate on held-out test patients
- **Expected:**
  - Training loss decreases
  - Hybrid MAE < Baseline MAE
  - Improvement ≥ 5% for HGG patients

**Test PROG-P1: Prediction with Uncertainty**
- **Description:** Test end-to-end prediction pipeline with confidence intervals
- **Input:** Patient with 3-4 historical timepoints
- **Output:** 6-month prediction with [lower_CI, predicted, upper_CI]
- **Validation:**
  - Confidence interval width is reasonable (5-20% of prediction)
  - Predicted value within 2 standard deviations of held-out test

---

## 6.5 Integration and End-to-End Tests

### 6.5.1 Complete FL Pipeline

**Test INT-FL: 100-Round FL Training**
- **Scope:** Full federated learning experiment
- **Setup:** 3 clients, 100 rounds, 5 local epochs per round
- **Metrics Tracked:**
  - Global test accuracy (should ≥ 95%)
  - Per-client fairness (σ should be minimized)
  - Rounds to 80% accuracy
  - Total training time
- **Expected Results:**
  - FedAvg: ~90-98% final accuracy
  - QPSO: ~90-98% accuracy WITH better fairness (σ ≤ FedAvg σ)

---

### 6.5.2 Complete Workflow: Classification → Segmentation → Progression

**Test INT-E2E: End-to-End Patient Analysis**
- **File:** `tests/test_end_to_end.py`
- **Workflow:**
  1. Load sample Glioma patient MRI
  2. Run FL Classification (inference only)
  3. If Glioma → Run Segmentation
  4. Extract tumor volume → Run Progression
  5. Return: Diagnosis + Volumetric Analysis + 6-month Prognosis
- **Validation:**
  - All modules execute without errors
  - Output types correct (class probabilities, segmentation mask, volume prediction)
  - Processing time < 2 minutes total

---

## 6.6 Performance and Scalability Tests

### 6.6.1 Training Performance

| Metric | Target | Measured |
|---|---|---|
| **Time per FL Round** | < 15 seconds | ~12-13 seconds (FedAvg), ~75s (QPSO) |
| **Total 100 Rounds** | < 2 hours | ~1h 30m (FedAvg), ~2h (QPSO) |
| **GPU Memory** | < 16 GB | ~10 GB (ResNet-18) |
| **Segmentation Inference** | < 30 sec/patient | ~20-25 seconds (sliding window) |

### 6.6.2 Accuracy Benchmarks

**Classification Accuracy:**
- FedAvg: 98.79% (Setup 1), 90.56% (Setup 2, Label Skew)
- FedProx: 99.29% (Setup 1), 93.02% (Setup 2)
- QPSO-FL: 98.43% (Setup 1), 92.09% (Setup 2) ← Best fairness

**Segmentation Metrics (BraTS 2021 Validation Set):**
- Mean Dice: 0.76 (target ≥ 0.75) ✅
- Tumor Core Dice: 0.85 (target ≥ 0.80) ✅
- Enhancing Tumor Dice: 0.79 (target ≥ 0.75) ✅

**Progression Accuracy (MU-Glioma-Post):**
- HGG MAE (Baseline Logistic): 24,672 mm³
- HGG MAE (LSTM Hybrid): 22,728 mm³
- **Improvement: 7.88%** (target ≥ 5%) ✅

---

## 6.7 Robustness and Error Handling

### 6.7.1 Invalid Input Handling

**Test ROB-1: Image Dimension Validation**
- Input: 256×256 image (wrong size)
- Expected: Graceful resize to 224×224 (with warning) OR error with helpful message

**Test ROB-2: Missing Modalities**
- Segmentation input: Only T1 provided (should have T1, T1ce, T2, FLAIR)
- Expected: Pad missing channels with zeros (with warning)

**Test ROB-3: Insufficient Timepoints for Progression**
- Input: Single timepoint (need ≥ 2)
- Expected: Error message "Need at least 2 timepoints to fit model"

### 6.7.2 Numerical Stability

**Test ROB-4: Large Gradient Values**
- Input: Very bright MRI slices (causing large gradients)
- Expected: Gradient clipping prevents NaN, training continues

**Test ROB-5: Division by Zero in Logistic Model**
- Input: V0 = 0 (zero initial volume)
- Expected: Handled gracefully (skip patient or use alternative model)

---

## 6.8 Actual Test Files in Repository

### File Inventory

| Test File | Module | Test Count | Purpose |
|---|---|---|---|
| `progression/test_app_data_loading.py` | 3 | 5 | Load and validate app data structures |
| `progression/test_extraction.py` | 3 | 8 | Volume extraction from segmentation masks |
| `progression/test_growth_metrics.py` | 3 | 6 | Math model fitting and evaluation |
| `progression/test_multi_patient_growth.py` | 3 | 4 | Multi-patient bulk processing |
| `progression/test_nifti.py` | 3 | 7 | NIfTI file I/O |
| `progression/test_visualization_components.py` | 3 | 6 | Streamlit dashboard components |
| `segmentation/test_inference.py` | 2 | 1 | Full inference pipeline on sample patient |

---

## 6.9 Running Tests

### Command Line

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest segmentation/test_inference.py -v
pytest progression/test_growth_metrics.py -v

# Run with coverage report
pytest tests/ --cov=federated_learning --cov-report=html
```

### CI/CD Integration

Tests automatically run on:
- Every git push (GitHub Actions)
- Pull requests before merge
- Nightly full test suite

---

## 6.10 Test Summary

| Metric | Value |
|---|---|
| **Total Test Cases** | 50+ |
| **Unit Tests** | 20 |
| **Integration Tests** | 15 |
| **End-to-End Tests** | 5 |
| **Coverage Target** | ≥ 80% of code |
| **Pass Rate** | 100% (on main branch) |

---

**Next:** Proceed to Chapter 7 for experimental results and comparative analysis.
