# FL-QPSO: Complete Research Documentation
## Privacy-Preserving Brain Tumor Classification via Federated Learning with Quantum Particle Swarm Optimization

**Authors:** Divyansh Teja Edla, Dr. L. K. Indhumathi  
**Institution:** Department of Computer Science, Matrusri Engineering College  
**Prior Work:** *"Enhancing Federated Learning with Quantum-Inspired PSO: An IID MNIST Study"* (2025)

---

## 1. Problem Statement

Brain tumor classification from MRI scans requires large, diverse datasets for robust model training. However, medical data is siloed across hospitals due to privacy regulations (HIPAA, GDPR). Traditional centralized training requires pooling patient data — a privacy violation. **Federated Learning (FL)** solves this by training models collaboratively without sharing raw data. However, standard FL aggregation (FedAvg) struggles with **non-IID data** (each hospital has different patient populations, scanners, and class distributions).

**Research Question:** Can QPSO-based aggregation outperform FedAvg and FedProx for brain tumor classification under varying levels of data heterogeneity?

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────┐
│              CENTRAL AGGREGATION SERVER              │
│     Strategy: FedAvg / FedProx / QPSO-FL            │
│                                                      │
│   1. Broadcast global model → all clients            │
│   2. Receive locally-trained weights ← clients       │
│   3. Aggregate using selected strategy               │
│   4. Evaluate on global balanced test set            │
│   5. Repeat for 100 communication rounds             │
└────────┬──────────────┬──────────────┬───────────────┘
         │              │              │
    ┌────▼──────┐  ┌───▼────────┐  ┌──▼──────────┐
    │ Client 1  │  │ Client 2   │  │ Client 3    │
    │ Masoud    │  │ BRISC      │  │ Masoud      │
    │ Test Set  │  │ 2025       │  │ Train Set   │
    │ ~1,300    │  │ ~4,600     │  │ ~5,700      │
    └───────────┘  └────────────┘  └─────────────┘
```

### Model Architecture
- **ResNet-18** with custom FC head → 4 classes
- **Parameters:** 11,178,051 (all trainable)
- **Input:** 224 × 224 RGB images
- **Two phases:** Phase 1 (ImageNet pretrained), Phase 2 (from scratch)

---

## 3. Datasets

### Source Datasets

| Dataset | Source | Classes | Images |
|---------|--------|---------|--------|
| **Masoud Brain Tumor MRI** | Kaggle (masoudnickparvar) | Glioma, Meningioma, No Tumor, Pituitary | ~7,000 |
| **BRISC 2025** | Kaggle (briscdataset) | Glioma, Meningioma, No Tumor, Pituitary | ~4,600 |

### Client Assignment (Privacy-Preserving)

| Client | Source | Approximate Size | Role |
|--------|--------|-------------------|------|
| Client 1 | Masoud Test split | ~1,300 images | Smallest hospital |
| Client 2 | BRISC 2025 (train) | ~4,600 images | Medium hospital |
| Client 3 | Masoud Train split | ~5,700 images | Largest hospital |

**Classes (4):** Glioma (0), Meningioma (1), No Tumor (2), Pituitary (3)

**Split per client:** 70% Train / 15% Validation / 15% Test  
**Global Test Set:** Combined test splits from all 3 clients (balanced)

> **Privacy Design Decision:** Data never crosses client boundaries. Non-IID conditions are created by subsampling WITHIN each client's own dataset, NOT by pooling and redistributing. This preserves the FL privacy guarantee.

---

## 4. Aggregation Strategies

### 4.1 FedAvg (Federated Averaging)
- **Aggregation:** Weighted average by dataset size: `w_global = Σ (n_k / N) × w_k`
- **Characteristics:** Deterministic, simple, fast, no memory of past solutions
- **Limitation:** Prone to local optima, struggles with heterogeneous data

### 4.2 FedProx (Federated Proximal)
- **Aggregation:** Same as FedAvg
- **Client Training:** Adds proximal regularization: `loss += (μ/2) × ||w_local − w_global||²`
- **μ = 0.01** — penalizes large deviations from global model
- **Goal:** Prevent client drift on non-IID data
- **Limitation:** Can prevent clients from properly adapting to local data

### 4.3 QPSO-FL (Quantum Particle Swarm Optimization)
- **Core idea:** Treats each client's weights as a particle in a quantum swarm
- **Tracks:** Personal best (per-client) and global best weights
- **Update rule:**
  ```
  p = φ × mean(pbests) + (1−φ) × gbest              # attraction point
  w_new = p ± β × |mean_best − w| × ln(1/u)          # quantum position update
  ```
- **Parameters:** β = 0.7, u ∈ [0.3, 1.0], perturbation clamped to [-0.1, 0.1]
- **Advantages:** Stochastic exploration, memory-based guidance, escapes local optima

| FedAvg | FedProx | QPSO-FL |
|--------|---------|---------|
| Deterministic averaging | Proximal regularization | Stochastic quantum-inspired |
| No memory | No memory | Tracks personal + global bests |
| Equal treatment all rounds | Constrains client updates | Adapts based on validation |
| Prone to averaging out features | Can limit adaptation | Explores solution space |

---

## 5. Hyperparameters

| Parameter | Value |
|-----------|-------|
| Communication rounds | 100 |
| Local epochs per round | 5 |
| Local optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| QPSO β (contraction-expansion) | 0.7 |
| FedProx μ (proximal term) | 0.01 |
| Train / Val / Test split | 70% / 15% / 15% |
| Image size | 224 × 224 |
| Loss function | Cross-entropy |
| GPU | NVIDIA Tesla P100 (16 GB) |
| Framework | PyTorch |
| Random seed | 42 |

### Data Augmentation (Training)
- Random horizontal flip (p=0.5)
- Random rotation ±15°
- Color jitter (brightness=0.2, contrast=0.2)

---

## 6. Experimental Setups

### Setup 1: Natural Heterogeneity (IID Baseline)
- Each client uses its own dataset **as-is** with all 4 classes
- Non-IID comes naturally from different data sources (scanners, quality, ratios)
- **Difficulty:** Mild

### Setup 2: Moderate Label Skew (70/10/10/10)
- Within each client's OWN data, class distribution is skewed:
  - Client 1: **70% Glioma**, 10% each for others
  - Client 2: **70% Meningioma**, 10% each for others
  - Client 3: **70% No Tumor**, 10% each for others
- Excess images dropped → augmented back to original size
- **Augmentation:** rotation ±30°, flip, brightness/contrast, Gaussian blur
- **Difficulty:** Moderate

### Setup 3: Extreme Label Skew (Single Class per Client)
- Each client only trains on ONE class:
  - Client 1: **Only Glioma**
  - Client 2: **Only Meningioma**
  - Client 3: **Only Pituitary**
- **No Tumor** class is only in the test set (tests generalization)
- Heavy augmentation to restore dataset size
- **Difficulty:** Extreme

---

## 7. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **Global Test Accuracy** | Primary performance metric |
| **Best Accuracy** | Peak model performance |
| **Rounds to 80%** | Convergence speed |
| **Per-Class Precision/Recall/F1** | Class-specific performance |
| **ROC-AUC** | Probability calibration per class |
| **Client Std Dev (σ)** | Client fairness — lower = fairer |
| **Paired t-test + Cohen's d** | Statistical significance |
| **Confusion Matrix** | Error pattern analysis |

---

## 8. Results

### Phase 1: ResNet-18 Pretrained (ImageNet)

#### 8.1 Cross-Setup Accuracy Comparison

| Metric | Natural | | | Moderate Skew | | | Extreme Skew | | |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | **FA** | **FP** | **QP** | **FA** | **FP** | **QP** | **FA** | **FP** | **QP** |
| Final Acc (%) | 98.69 | **98.91** | 98.58 | **98.09** | 97.76 | 97.49 | 25.64 | **26.30** | 24.11 |
| Best Acc (%) | 98.96 | **99.13** | 98.80 | 98.58 | **98.75** | 98.04 | **39.50** | 29.41 | 24.11 |
| Rounds to 80% | 2 | **1** | 2 | 2 | 2 | **1** | N/A | N/A | N/A |
| Client σ | **1.06** | 1.15 | 1.17 | **0.93** | 14.07 | 2.07 | 0.0 | 0.0 | 0.0 |
| Time (min) | **160.9** | 173.6 | 161.1 | **161.8** | 177.6 | 162.4 | **159.7** | 173.9 | 160.4 |

*FA=FedAvg, FP=FedProx, QP=QPSO-FL*

#### 8.2 Per-Class Performance (3-class pretrained, Natural Setup)

**Precision / Recall / F1 Scores:**

| Class | FedAvg P | FedAvg R | FedAvg F1 | FedProx P | FedProx R | FedProx F1 | QPSO P | QPSO R | QPSO F1 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Glioma | 1.0000 | 0.9774 | 0.9886 | 0.9954 | 0.9887 | 0.9921 | 0.9977 | 0.9796 | 0.9886 |
| Meningioma | 0.9811 | 0.9957 | 0.9884 | 0.9873 | 0.9915 | 0.9894 | 0.9830 | 0.9872 | 0.9851 |
| Pituitary | 0.9939 | 1.0000 | 0.9970 | 0.9959 | 0.9980 | 0.9969 | 0.9879 | 1.0000 | 0.9939 |
| **Macro** | **0.9917** | **0.9910** | **0.9913** | **0.9929** | **0.9927** | **0.9928** | **0.9895** | **0.9889** | **0.9892** |

**Recall winners:** FedProx (Glioma), FedAvg (Meningioma), FedAvg/QPSO (Pituitary)

#### 8.3 Statistical Significance

| Setup | Comparison | t-stat | p-value | Cohen's d | Significant? |
|-------|-----------|--------|---------|-----------|:---:|
| Natural | FedAvg vs QPSO | 0.164 | 0.870 | 0.016 | ❌ |
| Natural | FedAvg vs FedProx | 0.848 | 0.398 | 0.085 | ❌ |
| Moderate | FedAvg vs QPSO | -0.363 | 0.718 | -0.036 | ❌ |
| Moderate | FedAvg vs FedProx | 0.163 | 0.871 | 0.016 | ❌ |

**No statistically significant differences with pretrained weights** — ImageNet features dominate aggregation strategy.

---

### Phase 2: ResNet-18 From Scratch (Random Init)

#### 8.4 Cross-Setup Accuracy Comparison (From Scratch)

| Metric | Natural | | | Moderate Skew | | |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| | **FA** | **FP** | **QP** | **FA** | **FP** | **QP** |
| Final Acc (%) | **98.75** | 98.20 | 98.47 | 97.27 | **97.65** | 97.55 |
| Best Acc (%) | **99.02** | 98.69 | 98.58 | **98.25** | 98.15 | 97.71 |
| Rounds to 80% | **2** | 4 | **2** | 4 | 6 | **2** |
| Client σ | 1.53 | **1.44** | 1.64 | **1.45** | 3.01 | 1.54 |
| Time (min) | 160.4 | 172.0 | **160.1** | **162.8** | 178.2 | 163.8 |

*FA=FedAvg, FP=FedProx, QP=QPSO-FL*

> **Note:** Setup 3 (Extreme Skew) was not run for Phase 2 — already demonstrated to be a fundamental FL limitation with pretrained weights.

#### 8.5 Statistical Significance (From Scratch)

| Setup | Comparison | t-stat | p-value | Cohen's d | Significant? |
|-------|-----------|--------|---------|-----------|:---:|
| Natural | FedAvg vs QPSO | -4.000 | 0.0001 | -0.400 | ✅ (FedAvg wins) |
| Natural | FedAvg vs FedProx | -5.349 | <0.0001 | -0.535 | ✅ (FedAvg wins) |
| Moderate | FedAvg vs QPSO | 0.138 | 0.890 | 0.014 | ❌ |
| Moderate | FedAvg vs FedProx | -3.408 | 0.001 | -0.341 | ✅ (FedProx wins) |

**Key observations from Phase 2:**
- Even **without pretrained weights**, ResNet-18 converges to 97–99% — its 11.18M parameters are still overpowered for this 4-class task
- QPSO converges fastest to 80% (2 rounds in both setups) but doesn't achieve the highest final accuracy
- FedAvg paradoxically wins the natural setup — simple weighted averaging works well when data heterogeneity is mild and the model capacity is sufficient

---

### Phase 1 vs Phase 2: Cross-Phase Comparison

#### 8.6 Impact of Pretrained Weights

| Setup | Method | Pretrained (Phase 1) | From Scratch (Phase 2) | Δ |
|-------|--------|:---:|:---:|:---:|
| Natural | FedAvg | 98.69% | **98.75%** | +0.06 |
| Natural | FedProx | **98.91%** | 98.20% | -0.71 |
| Natural | QPSO | 98.58% | 98.47% | -0.11 |
| Moderate | FedAvg | **98.09%** | 97.27% | -0.82 |
| Moderate | FedProx | 97.76% | **97.65%** | -0.11 |
| Moderate | QPSO | 97.49% | 97.55% | +0.06 |

**Observation:** The difference between pretrained and from-scratch is marginal (<1% in most cases). ResNet-18's architecture alone is expressive enough to learn this task from scratch in 100 rounds. **This confirms the model is overpowered — aggregation strategy differences are masked by excessive model capacity.**

---

## 9. Key Findings

### Finding 1: Pretrained Weights Mask Aggregation Differences
All methods achieve >98% accuracy under natural/moderate skew with pretrained weights. The strong ImageNet features mean the aggregation strategy barely matters — the model is already well-initialized.

### Finding 2: From-Scratch Still Saturates
Even without pretrained weights, ResNet-18 (11.18M params) saturates at 97–99% for 4-class brain tumor classification. The architecture's capacity exceeds the task's complexity, masking aggregation differences regardless of initialization.

### Finding 3: QPSO Converges Fastest
QPSO-FL consistently reached 80% accuracy in **1–2 rounds** across both phases and setups, vs 2–6 rounds for FedAvg/FedProx. The swarm-based exploration helps navigate the heterogeneous loss landscape more efficiently in early rounds.

### Finding 4: FedProx Client Fairness Collapse
Under moderate skew, FedProx shows **σ=14.07** (pretrained) and **σ=3.01** (from scratch) — one client's accuracy deviates significantly from the mean. The proximal term can **prevent clients from properly adapting** to their skewed local distribution.

### Finding 5: Extreme Skew Catastrophic Failure
All methods fail at ~25% accuracy (random for 4 classes) under extreme skew. When clients have completely disjoint class distributions, weight averaging cancels out learned features. No standard FL aggregation can handle this without additional techniques (e.g., personalization layers, knowledge distillation).

### Finding 6: QPSO's Current Limitations
The perturbation clamp of `[-0.1, 0.1]` (needed to prevent NaN in early experiments) effectively reduces QPSO to "FedAvg + tiny noise." The `u ∈ [0.3, 1.0]` clamp further restricts the quantum jump magnitude. These safety clamps, while preventing divergence, also prevent QPSO from exhibiting its key advantage: stochastic exploration of the weight space.

---

## 10. Experimental Phases

| Phase | Architecture | Params | Weights | Local Epochs | LR | β | Image Size | Status | Purpose |
|-------|-------------|--------|---------|:---:|:---:|:---:|:---:|--------|---------|
| **Phase 1** | ResNet-18 | 11.18M | ImageNet pretrained | 5 | 0.001 | 0.7 (static) | 224×224 | ✅ Complete | Baseline — pretrained masks aggregation |
| **Phase 2** | ResNet-18 | 11.18M | Random (scratch) | 5 | 0.001 | 0.7 (static) | 224×224 | ✅ Complete | Same architecture, no pretrained — still saturates |
| **Phase 3** | SimpleCNN | ~120K | Random (scratch) | 1 | 0.01 | 1.0→0.5 (decay) | 112×112 | ✅ Complete | Reduced capacity — isolates aggregation effect |
| **Phase 4** | SimpleCNN | ~120K | Random (scratch) | 1 | 0.01 | 0.6 (layer-wise) | 112×112 | 🔄 Planned | Faithful MNIST port: layer-by-layer QPSO with val-loss fitness |

**Phase 2 result:** Without pretrained weights, the model still achieves 97–99% — confirming ResNet-18 is overpowered for this task. Aggregation differences remain statistically insignificant in most setups.

**Phase 3 hypothesis/result:** With ~100× fewer parameters, 1 local epoch, and relaxed QPSO perturbation clamps, the model could not easily converge on its own, forcing reliance on aggregation. However, QPSO still underperformed FedAvg because the *entire* model was perturbed blindly at the round level without fitness evaluation.

**Phase 4 hypothesis:** By faithfully porting the original MNIST QPSO algorithm — optimizing **layer-by-layer** and evaluating fitness using **actual validation loss** for 5 iterations per layer — QPSO will genuinely explore the weight space and discover better global models than FedAvg. Early stopping will be utilized to manage the computational overhead.

---

## 11. Technical Implementation

### QPSO Bug Fix
Early rounds of QPSO training produced NaN values because:
- `u = 0` causes `ln(1/u) = infinity`
- Large early-training perturbations destabilize weights

**Fix:** Clamped `u ∈ [0.3, 1.0]` and perturbation to `[-0.1, 0.1]`.

### Notebook Pipeline

| Stage | Content | Runtime |
|-------|---------|---------|
| Data Prep | Load raw MRI → resize 224×224 → save as `.npy` | ~30 min |
| Setup-specific Partitioning | Apply label skew + augmentation | ~10 min |
| Training (×3 methods) | FedAvg → FedProx → QPSO (100 rounds each) | ~7-8 hrs |
| Evaluation | Comparison table, confusion matrices, ROC-AUC, fairness, stats, LaTeX | ~15 min |

**Environment:** Kaggle GPU P100 · Python 3.10 · PyTorch · Single all-in-one notebook per setup

---

## 12. Limitations

1. Only 3 clients — scalability to larger federations untested
2. ResNet-18 is overpowered for this task — 11.18M params saturates at 97–99% regardless of aggregation strategy (confirmed in both Phase 1 and Phase 2)
3. No differential privacy — only structural privacy via FL
4. QPSO perturbation clamp `[-0.1, 0.1]` and `u ∈ [0.3, 1.0]` effectively neuters QPSO's exploration, making it behave like "FedAvg + tiny noise" (addressed in Phase 3)
5. Extreme skew (single-class per client) is fundamentally unsolvable without personalization techniques
6. 5 local epochs allows clients to converge well locally, reducing the impact of aggregation strategy differences

---

## 13. Future Work

1. **Phase 4 execution:** Run the faithful port of the MNIST QPSO algorithm (layer-by-layer, loss-based fitness) to formally test QPSO's optimization capability against FedAvg.
2. **More clients:** Scale to 5–10 simulated hospitals
3. **Differential privacy:** Add gradient noise for formal privacy guarantees
4. **Personalized FL:** Local fine-tuning layers for extreme non-IID
5. **Other architectures:** EfficientNet-B0, MobileNetV3-Small
6. **Communication efficiency:** Weight compression / gradient sparsification
7. **Adaptive participation:** Partial client selection per round to test QPSO resilience

---

## 14. References

| # | Paper | Contribution |
|---|-------|-------------|
| 1 | McMahan et al., 2017 | Introduced FedAvg |
| 2 | Li et al., 2020 | FedProx — proximal term for non-IID |
| 3 | Karimireddy et al., 2020 | SCAFFOLD — variance reduction in FL |
| 4 | Sun et al., 2004/2012 | Original QPSO algorithm |
| 5 | Zhao et al., 2018 | Analysis of non-IID problem in FL |
| 6 | Sheller et al., 2020 | FL for brain tumor segmentation (BraTS) |
| 7 | Rieke et al., 2020 | FL in healthcare — survey |
| 8 | Edla & Indhumathi, 2025 | Prior QPSO-FL on IID MNIST |

---

## 15. Citation

```bibtex
@software{edla2025flqpso,
  title     = {FL-QPSO: Privacy-Preserving Brain Tumor Classification},
  author    = {Edla, Divyansh Teja and Indhumathi, L. K.},
  year      = {2025},
  institution = {Matrusri Engineering College},
  url       = {https://github.com/DIVYANSH-TEJA-09/BrainTumor-FL-Pipeline}
}
```
