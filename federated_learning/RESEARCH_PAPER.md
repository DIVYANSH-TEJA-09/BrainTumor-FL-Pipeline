# Research Paper Reference — FL QPSO vs FedAvg

> Quick-reference notes for writing the paper. Covers motivation, methodology, experimental setup, expected results structure, and key talking points.

---

## Paper Title (Suggested)

**"QPSO-Enhanced Federated Learning for Privacy-Preserving Brain Tumor Classification Across Heterogeneous Medical Institutions"**

Alternative titles:
- "Quantum-Inspired Aggregation for Federated Brain Tumor Classification on Non-IID Multi-Source MRI Data"
- "Beyond FedAvg: QPSO-Based Model Aggregation for Robust Federated Medical Image Classification"

---

## Abstract Outline

1. **Problem**: FL for medical imaging faces accuracy degradation from Non-IID data heterogeneity across hospitals
2. **Solution**: QPSO-based aggregation replaces naive weighted averaging with quantum-inspired stochastic exploration
3. **Setup**: 3 clients (2 datasets), ResNet-18, brain tumor classification (3 classes)
4. **Result**: QPSO-FL achieves X% higher accuracy, Y% faster convergence, and Z% better client fairness than FedAvg
5. **Significance**: Demonstrates QPSO's effectiveness in realistic Non-IID medical setting (extending prior IID-only work)

---

## 1. Introduction — Key Points

- **Federated Learning** enables collaborative model training without sharing patient data → critical for healthcare privacy (HIPAA, GDPR)
- **FedAvg limitation**: Weighted averaging is deterministic, prone to local optima, and struggles with heterogeneous data distributions
- **Our contribution**: QPSO brings stochastic, quantum-inspired exploration to FL aggregation, enabling the global model to escape local optima
- **Extension from prior work**: Previous study (your IID MNIST paper) showed QPSO improved FedAvg from 41% → 81% on IID data. This work extends to the harder **Non-IID medical imaging** setting

### Key References to Cite
- McMahan et al., 2017 — FedAvg original paper
- Zhao et al., 2018 — "Federated Learning with Non-IID Data"
- Karimireddy et al., 2020 — SCAFFOLD for FL
- Sun et al., 2004 / 2012 — Original QPSO algorithm
- Your IID MNIST paper — Edla & Indhumathi

---

## 2. Methodology

### 2.1 Dataset Description

| Property | Client 1 | Client 2 | Client 3 |
|----------|----------|----------|----------|
| **Source** | Masoud Nickparvar (Testing split) | BRISC 2025 (Classification train) | Masoud Nickparvar (Training split) |
| **Glioma** | 400 | ~1,147 | 1,400 |
| **Meningioma** | 400 | ~1,329 | 1,400 |
| **Pituitary** | 400 | ~1,457 | 1,400 |
| **No Tumor** | ~~400~~ excluded | ~~1,067~~ excluded | ~~1,400~~ excluded |
| **Total (used)** | ~1,200 | ~3,933 | ~4,200 |

> **Non-IID Nature**: Clients have different dataset sizes, different acquisition protocols, different class ratios (BRISC is imbalanced vs Masoud is balanced), and originate from different institutions.

### 2.2 Model Architecture

- **Base**: ResNet-18 pretrained on ImageNet
- **Head**: Final FC layer → 3 classes (glioma, meningioma, pituitary)
- **Input**: 224 × 224 RGB
- **Parameters**: ~11.2M total, all trainable
- **Augmentation**: Random horizontal flip, rotation (±15°), color jitter

### 2.3 FedAvg Algorithm

```
For each round r = 1, ..., R:
    For each client k = 1, ..., K:
        w_k ← LocalTrain(w_global, D_k, E epochs)
    w_global ← Σ (n_k / n_total) · w_k
```

### 2.4 QPSO-FL Algorithm

```
Initialize: pbest_k = gbest = w_global,  ∀k
For each round r = 1, ..., R:
    For each client k:
        w_k ← LocalTrain(w_global, D_k, E epochs)
        if val_acc(w_k) > val_acc(pbest_k): pbest_k ← w_k
        if val_acc(w_k) > val_acc(gbest):   gbest ← w_k
    
    mbest ← mean(pbest_1, ..., pbest_K)   # mean best position
    
    For each parameter θ:
        φ ~ U(0,1),  u ~ U(0,1)
        p = φ · mean(pbests) + (1−φ) · gbest     # attraction point
        θ_new = p ± β · |mbest − θ| · ln(1/u)    # quantum update
    
    w_global ← θ_new
```

**Key parameters**:
- β = 0.7 (contraction-expansion coefficient)
- Controls exploration (β → 1) vs exploitation (β → 0.5)

### 2.5 Why QPSO Works Better Than FedAvg

| FedAvg | QPSO-FL |
|--------|---------|
| Deterministic averaging | Stochastic quantum-inspired updates |
| Single point estimate | Explores solution space probabilistically |
| No memory of past solutions | Tracks personal bests and global best |
| Equal treatment of all rounds | Adapts based on validation performance |
| Prone to averaging out learned features | Preserves best-performing model components |

---

## 3. Experimental Setup

### Hyperparameters Table (for paper)

| Parameter | Value |
|-----------|-------|
| Communication rounds | 100 |
| Local epochs per round | 5 |
| Local optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| QPSO β | 0.7 |
| Train / Val / Test split | 70% / 15% / 15% |
| Image size | 224 × 224 |
| Pretrained weights | ImageNet |
| Loss function | Cross-entropy |
| GPU | NVIDIA P100 (16 GB) |
| Framework | PyTorch |
| Random seed | 42 |

### Evaluation Metrics

1. **Global Test Accuracy** — primary metric
2. **Per-Class F1 Score** — precision/recall per tumor type
3. **Convergence Speed** — rounds to reach 80% accuracy
4. **Client Fairness** — std deviation of per-client validation accuracies (lower = fairer)
5. **Statistical Significance** — paired t-test + Cohen's d

---

## 4. Results Tables (Templates)

### Table 1: Overall Comparison

| Metric | FedAvg | QPSO-FL | Improvement |
|--------|--------|---------|-------------|
| Final Global Accuracy (%) | — | — | +X.X% |
| Best Global Accuracy (%) | — | — | +X.X% |
| Rounds to 80% Accuracy | — | — | −Y rounds |
| Client Accuracy Std Dev | — | — | −Z.Z% |
| Total Training Time (min) | — | — | — |

### Table 2: Per-Client Final Accuracy

| Client | Dataset | FedAvg | QPSO-FL |
|--------|---------|--------|---------|
| Client 1 | Masoud Test (1,200) | —% | —% |
| Client 2 | BRISC (3,933) | —% | —% |
| Client 3 | Masoud Train (4,200) | —% | —% |
| **Mean** | | —% | —% |
| **Std Dev** | | —% | —% |

### Table 3: Per-Class Performance (Best Model)

| Class | FedAvg P | FedAvg R | FedAvg F1 | QPSO P | QPSO R | QPSO F1 |
|-------|---------|---------|----------|--------|--------|---------|
| Glioma | — | — | — | — | — | — |
| Meningioma | — | — | — | — | — | — |
| Pituitary | — | — | — | — | — | — |
| **Macro Avg** | — | — | — | — | — | — |

---

## 5. Figures Checklist

- [ ] **Fig 1**: System architecture diagram (FL topology with 3 clients + server)
- [ ] **Fig 2**: QPSO vs FedAvg aggregation flow diagram
- [ ] **Fig 3**: Class distribution across clients (bar chart) — shows Non-IID nature
- [ ] **Fig 4**: Global test accuracy over rounds (FedAvg vs QPSO curves)
- [ ] **Fig 5**: Global test loss over rounds
- [ ] **Fig 6**: Per-client accuracy over rounds (side by side)
- [ ] **Fig 7**: Confusion matrices (FedAvg vs QPSO, side by side)
- [ ] **Fig 8**: Client fairness bar chart with std deviation

---

## 6. Discussion Points

### Why QPSO Outperforms FedAvg
1. **Escapes local optima**: Stochastic perturbation via `ln(1/u)` term enables exploration beyond weighted average
2. **Memory-based guidance**: Personal + global bests steer aggregation toward historically good solutions
3. **Adaptive aggregation**: Attraction point `p` dynamically balances local expertise vs global consensus
4. **Non-IID resilience**: QPSO doesn't blindly average divergent client updates — it selectively explores near the best-performing regions

### Limitations to Acknowledge
- Computationally slightly more expensive per round (QPSO update step)
- β sensitivity — different values may be needed for different datasets
- Only tested with 3 clients — scalability to many clients needs investigation
- No differential privacy added (future work)

---

## 7. Conclusion Talking Points

1. Successfully extended QPSO-FL from IID (prior paper) to **Non-IID medical imaging** setting
2. QPSO achieves higher accuracy with better convergence on heterogeneous brain tumor data
3. Improved fairness across clients — important for healthcare equity
4. Opens door for QPSO-FL in other privacy-sensitive medical domains (cardiac, pulmonary, dermatology)

---

## 8. Future Work Ideas

- **Differential Privacy**: Add noise to client updates for formal privacy guarantees
- **Adaptive β scheduling**: Decay β over rounds (explore early → exploit late)
- **More clients**: Scale to 5-10 clients with more diverse datasets
- **Other models**: EfficientNet, Vision Transformer
- **Real Non-IID**: Dirichlet-based label distribution skew
- **Communication efficiency**: Compress model updates before QPSO aggregation

---

## 9. Citation for Your Previous Paper

```bibtex
@article{edla2025qpsofl,
  title={Enhancing Federated Learning With Quantum-Inspired Particle Swarm Optimization: An IID MNIST Study},
  author={Edla, Divyansh Teja and Indhumathi, L. K.},
  year={2025},
  institution={Department of Computer Science, Matrusri Engineering College}
}
```

---

## 10. Related Work — Key Papers

| Paper | Contribution |
|-------|-------------|
| McMahan et al., 2017 | Introduced FedAvg |
| Li et al., 2020 (FedProx) | Proximal term for Non-IID robustness |
| Karimireddy et al., 2020 (SCAFFOLD) | Variance reduction in FL |
| Sun et al., 2004 | Original QPSO algorithm |
| Zhao et al., 2018 | Analysis of Non-IID problem in FL |
| Sheller et al., 2020 | FL for brain tumor segmentation (BraTS) |
| Rieke et al., 2020 | FL in healthcare — survey |
