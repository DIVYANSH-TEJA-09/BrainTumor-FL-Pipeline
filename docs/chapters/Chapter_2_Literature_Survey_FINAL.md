# Chapter 2: Literature Survey

## 2.1 Introduction to Literature Context

Developing a privacy-preserving brain tumor classification system requires understanding three interconnected research domains:

1. **Federated Learning (FL):** Algorithms, optimization strategies, and non-IID challenges
2. **Medical Image Analysis:** Brain tumor classification, segmentation, and progression forecasting
3. **Privacy and Security in Healthcare:** Regulatory frameworks and technical implementations

This chapter synthesizes foundational work, identifies research gaps, and positions this project's contributions within the broader context.

---

## 2.2 Federated Learning: Fundamentals and Evolution

### 2.2.1 Foundational Algorithm: Federated Averaging (FedAvg)

**McMahan et al. (2017)** introduced Federated Averaging, the foundational algorithm that revolutionized distributed learning. In their seminal paper, they demonstrated that training a global model across multiple decentralized devices (clients) through weighted averaging could achieve comparable accuracy to centralized training while preserving privacy.

**Key Contributions:**
- **Algorithm:** Simple weighted averaging of client-side model weights by dataset size
- **Convergence Proof:** Theoretical convergence guarantees under convex loss functions
- **Practical Impact:** Enabled large-scale distributed training on mobile devices, IoT sensors, and healthcare institutions

**Formula:**
```
w_global^(t+1) = Σ (n_k / N) × w_k^(t)
```
where `n_k` is Client k's dataset size, `N` is total data across all clients.

**Limitations Identified:** FedAvg performs poorly when client data is non-IID (heterogeneous class distributions, different data qualities across institutions).

### 2.2.2 Non-IID Challenge and Proximal Solutions

**Li et al. (2020)** extended FedAvg by introducing **Federated Proximal (FedProx)**, addressing the core challenge of data heterogeneity in federated settings.

**Problem:** When clients have different data distributions (non-IID), naively averaging weights from diverse clients can result in:
- Rapid divergence of client models from the global optimum
- Slower convergence
- Bias toward clients with larger datasets

**FedProx Solution:**
- Added a **proximal regularization term** to local client training loss
- Prevents clients from drifting too far from the global model during local updates
- Mathematically: `loss_local = loss_data + (μ/2) × ||w_local - w_global||²`

**Clinical Relevance:** In multi-hospital settings where each hospital has different patient populations (e.g., different tumor incidence rates, imaging protocols), FedProx helps maintain model agreement while allowing local adaptation.

**Limitation:** Proximal term can be too restrictive, preventing clients from learning local-specific patterns (explored in this project through fairness analysis).

### 2.2.3 Variance Reduction: SCAFFOLD

**Karimireddy et al. (2020)** introduced SCAFFOLD (Stochastic Controlled Averaging for Federated Learning), addressing client-drift through variance reduction:

- Maintains client control variates to track divergence from global model
- Reduces impact of heterogeneous data on convergence
- Provides improved convergence guarantees for non-convex functions (relevant for neural networks)

**Not used in this project** but important context: SCAFFOLD demonstrates that sophisticated variance-based methods can improve FL robustness.

### 2.2.4 Non-IID Analysis and Characterization

**Zhao et al. (2018)** conducted foundational analysis of non-IID data in federated settings, categorizing heterogeneity types:

| Heterogeneity Type | Definition | Example in Hospitals |
|---|---|---|
| **Feature Distribution Skew** | Different feature distributions across clients | Different MRI scanner models, protocols |
| **Label Distribution Skew (Class Imbalance)** | Different class proportions per client | Hospital A: 70% Glioma, Hospital B: 50% Glioma |
| **Quantity Skew** | Unequal dataset sizes | Hospital A: 1,200 images, Hospital C: 4,200 images |

**This Project:** Implements both natural heterogeneity (Setup 1) and moderate label skew (Setup 2) based on Zhao et al.'s taxonomy.

---

## 2.3 Quantum-Inspired Optimization: QPSO

### 2.3.1 Classical Particle Swarm Optimization (PSO)

**Kennedy & Eberhart (1995)** introduced Particle Swarm Optimization, a bio-inspired optimization algorithm mimicking bird flocking behavior. Each "particle" represents a potential solution, iteratively improving through:
- Personal best memory (pbest): best position each particle has found
- Global best memory (gbest): best position found by any particle
- Velocity update rules balancing exploration and exploitation

**PSO Formula:**
```
v(t+1) = w×v(t) + c1×r1×(pbest - x(t)) + c2×r2×(gbest - x(t))
x(t+1) = x(t) + v(t+1)
```

### 2.3.2 Quantum-Behaved PSO (QPSO)

**Sun et al. (2004, 2012)** introduced Quantum-Behaved Particle Swarm Optimization, replacing classical velocity dynamics with quantum mechanics-inspired position updates:

**Key Innovation:** Instead of classical velocity, particles occupy quantum states with probabilistic position distributions. Position update:
```
x_new = p ± β × |mbest - x| × ln(1/u)
```
where:
- `p` is the attraction point (weighted combination of pbest and gbest)
- `β` is the contraction-expansion coefficient (exploration vs. exploitation)
- `u ∈ [0,1]` is a random variable providing stochasticity
- `mbest` is the mean of all personal bests

**Advantages Over Classical PSO:**
- **Faster convergence:** Quantum uncertainty principle enables more efficient exploration
- **Better escape from local optima:** Stochastic quantum jumps explore solution space more thoroughly
- **Fewer parameters:** Only `β` and `u` range need tuning vs. 3+ parameters in classical PSO

**Application to Federated Learning (Novel in this Project):**
- Treat each client's model weights as a "particle" in optimization space
- Track personal best (best weights client has produced) and global best (best weights overall)
- Use QPSO updates to aggregate weights, balancing stability (exploitation) with exploration of better solutions

---

## 2.4 Brain Tumor Classification and Detection

### 2.4.1 Traditional CNN Approaches

**Krizhevsky et al. (2012)** introduced AlexNet, demonstrating deep convolutional neural networks' superiority on image classification tasks. For medical imaging:

**ResNet Architecture (He et al., 2015):**
- Introduced residual connections enabling training of very deep networks
- ResNet-18: 18 layers, ~11.2M parameters
- ImageNet pretraining provides strong feature initialization for downstream tasks like brain tumor classification

**In this Project:** ResNet-18 serves as the classification backbone across all 3 federated clients, fine-tuned for 3-class (Glioma, Meningioma, Pituitary) and 4-class problems.

### 2.4.2 Transfer Learning for Medical Imaging

**Tan & Le (2019)** systematized transfer learning, showing pre-trained features from large-scale datasets (ImageNet) transfer effectively to specialized medical domains with limited data.

**Findings Relevant to Our Work:**
- Pre-trained ImageNet weights significantly accelerate medical image model training
- Fine-tuning on task-specific data improves performance
- Works well even with moderate dataset sizes (9,000+ images as in this project)

### 2.4.3 Brain Tumor Classification Datasets

**Masoud Brain Tumor MRI (Kaggle Dataset - nickparvar):**
- ~7,000 2D MRI slices
- Classes: Glioma, Meningioma, No Tumor, Pituitary
- Splits available: training/testing
- Used in Client 1 and Client 3 in federated setup

**BRISC 2025 (Brain Tumor Image Segmentation Challenge):**
- ~4,600 multimodal 3D brain MRI volumes
- Standardized preprocessing and evaluation metrics
- Used in Client 2 federated setup

**Masoud + BRISC Combination:**
- Provides natural data heterogeneity: different imaging protocols, scanner types, patient populations
- Simulates realistic multi-hospital scenario without explicit dataset redistribution

---

## 2.5 Brain Tumor Segmentation

### 2.5.1 U-Net: Foundational Architecture

**Ronneberger et al. (2015)** introduced U-Net, the seminal architecture for biomedical image segmentation:

**Architecture Features:**
- **Encoder-Decoder Structure:** Downsampling path captures context, upsampling path enables precise localization
- **Skip Connections:** Concatenate encoder features to decoder, preserving fine-grained spatial information
- **Efficiency:** Requires relatively few training samples compared to fully convolutional networks

**Extension to 3D:** MONAI framework implements 3D variants for volumetric medical imaging.

### 2.5.2 Attention Mechanisms in U-Net

**Oktay et al. (2018)** enhanced U-Net with Attention Gates (AGs), enabling the network to automatically learn which spatial regions are important:

**Attention Gate Mechanism:**
- Suppresses irrelevant regions (healthy tissue)
- Focuses network capacity on target regions (tumor)
- Mathematically: `attention_coeff = sigmoid(transform(encoder_features, decoder_features))`
- Applied channel-wise, reducing computational overhead

**In this Project:** 3D Attention U-Net processes 4-channel MRI inputs (T1, T1ce, T2, FLAIR) to segment 3 tumor sub-regions (WT, TC, ET).

### 2.5.3 BraTS Challenge and Benchmarks

**Menze et al. (2014) → Baid et al. (2021):** Multi-year Brain Tumor Segmentation Challenge establishing benchmarks and best practices:

| Aspect | BraTS Standard |
|--------|---|
| Dataset Size | 369 patients (2021) with 4-modality MRI |
| Regions | WT, TC, ET (3-channel output) |
| Evaluation Metric | Dice score, Hausdorff distance |
| Clinical Data | Mixed HGG (high-grade glioblastoma) and LGG (low-grade glioma) |

**This Project:** Uses BraTS 2021 for Module 2 (Segmentation), achieving Mean Dice 0.76 against benchmark standards (~0.75-0.78).

---

## 2.6 Tumor Progression and Growth Forecasting

### 2.6.1 Mathematical Growth Models

Classical mathematical models from population biology have been adapted for tumor growth:

**Logistic Growth Model (Verhulst, 1838 → Adapted for Tumors):**
```
V(t) = K / (1 + ((K - V0)/V0) × exp(-r×t))
```
- `K`: Carrying capacity (maximum sustainable tumor volume)
- `V0`: Initial tumor volume
- `r`: Intrinsic growth rate
- Produces S-shaped curve with growth saturation

**Gompertz Growth Model (Gompertz, 1825 → Adapted for Tumors):**
```
V(t) = V0 × exp((a/b) × (1 - exp(-b×t)))
```
- More realistic for solid tumors (deceleration over time)
- Biologically motivated: growth rate decreases as tumor size increases

**Clinical Application:** Different tumor types show different growth patterns:
- **HGG (High-Grade Glioma):** Aggressive, exponential-like growth
- **LGG (Low-Grade Glioma):** Slow, often logistic or gompertz patterns

### 2.6.2 LSTM for Time-Series Medical Data

**Hochreiter & Schmidhuber (1997):** Introduced Long Short-Term Memory networks, capable of learning long-range dependencies in sequential data.

**LSTM Architecture:**
- **Input Gate:** Controls information flow into cell state
- **Forget Gate:** Selectively retains or discards past information
- **Output Gate:** Controls what cell state information is output

**Medical Time-Series Applications:**
- Patient vital sign monitoring
- Disease progression prediction
- Treatment response forecasting

### 2.6.3 Hybrid Approaches: Mathematical + Deep Learning

**Recent Trend in Medical AI (Rajkomar et al., 2018):**
- Combine interpretable mathematical models with flexible deep learning
- Approach: Hybrid = Math_baseline + LSTM_correction
- Allows model interpretability (math component) + adaptability (LSTM)

**In this Project:** 
```
V_hybrid(t) = V_logistic(t) + LSTM_correction(residual_sequence)
```
Achieves 7.88% MAE improvement for HGG patients by learning where logistic model fails.

---

## 2.7 Privacy and Security in Healthcare AI

### 2.7.1 Regulatory Framework

**HIPAA (Health Insurance Portability and Accountability Act, USA):**
- Requires de-identification or aggregate statistical analysis for research
- Prohibits direct patient data sharing without explicit consent

**GDPR (General Data Protection Regulation, EU):**
- Right to deletion ("right to be forgotten")
- Data minimization principle
- Data processing transparency

**Challenge:** Traditional federated learning still requires hospitals to trust a central server. With GDPR, even aggregated insights may require consent.

### 2.7.2 Differential Privacy in FL

**Dwork et al. (2006):** Introduced formal definition of Differential Privacy:

**Definition:** An algorithm is ε-differentially private if removing one patient's data changes output probability by at most e^ε.

**Application to FL:**
- Add Laplace or Gaussian noise to client gradients before sending to server
- Provides formal privacy guarantees independent of computational power of adversary
- Trade-off: privacy vs. model accuracy

**Not Implemented in This Project:** Module 1 implements structural privacy (FL) but not formal DP-epsilon guarantees. Recommended future work.

### 2.7.3 Secure Aggregation

**Bonawitz et al. (2017):** Introduced secure aggregation protocol where server cannot decode individual client updates, only aggregated result.

**Key Idea:**
- Client weights encrypted before transmission
- Server performs encrypted aggregation (uses homomorphic encryption properties)
- Only aggregated result decrypted

**Implementation Complexity:** Trade-off with computational overhead. Not implemented in this project (synchronous aggregation without encryption).

---

## 2.8 Research Gaps and Project Positioning

### 2.8.1 Identified Gaps

| Gap | Significance | This Project's Contribution |
|-----|---|---|
| **Fair FL for Healthcare** | Most FL work optimizes global accuracy, ignoring minority clients | Proposes fairness-centric evaluation; QPSO preserves equity |
| **Quantum-Inspired FL** | PSO not applied to medical FL aggregation | Novel layer-by-layer QPSO with validation-loss fitness |
| **End-to-End Privacy Pipeline** | Privacy usually applied to one step | Privacy-preserved classification + local-only segmentation/progression |
| **Multi-Task Tumor Analysis** | Classification, segmentation, progression usually studied separately | Integrated 3-module pipeline with clear data flow |

### 2.8.2 Positioning

This project sits at the intersection of three research areas:

```
    ┌─────────────────┐
    │ Federated       │
    │ Learning        │ ← QPSO aggregation strategy
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Medical Image   │
    │ Analysis        │ ← 3-module pipeline (classification, segmentation, progression)
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Privacy-        │
    │ Preserving AI   │ ← Practical federated system for hospitals
    └─────────────────┘
```

**Novelty:**
- First application of QPSO to federated medical imaging
- Fairness-aware aggregation strategy proven statistically superior (p = 2.91 × 10⁻²²) in non-IID settings
- Integrated framework allowing multi-hospital brain tumor analysis with privacy guarantees

---

## 2.9 Chapter Summary

| Topic | Key Takeaway |
|-------|---|
| **FedAvg/FedProx** | Standard baselines; struggle with non-IID client data and fairness |
| **QPSO** | Quantum-inspired particle swarm optimization; unexplored for FL |
| **CNN Architectures** | ResNet-18 proven for transfer learning; fits medical imaging constraints |
| **Segmentation** | Attention U-Net with MONAI framework enables 3D volumetric analysis |
| **Progression** | Hybrid mathematical + LSTM models provide interpretability + flexibility |
| **Privacy** | FL provides structural privacy; differential privacy is future enhancement |

---

## 2.10 Key References

**Literature Cited:**

1. **McMahan et al. (2017)** - "Communication-Efficient Learning of Deep Networks from Decentralized Data"  
   *AISTATS 2017* — Introduced FedAvg algorithm

2. **Li et al. (2020)** - "Federated Optimization in Heterogeneous Networks"  
   *MLSys 2020* — Introduced FedProx for non-IID data

3. **Karimireddy et al. (2020)** - "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"  
   *ICML 2020* — Variance reduction approach

4. **Zhao et al. (2018)** - "Federated Learning with Non-IID Data"  
   *arXiv:1806.00582* — Analysis of data heterogeneity

5. **Sun et al. (2004, 2012)** - "Quantum-Behaved Particle Swarm Optimization"  
   *IJCNN 2004, IEEE Transactions 2012* — Quantum PSO algorithm

6. **He et al. (2015)** - "Deep Residual Learning for Image Recognition"  
   *CVPR 2015* — ResNet architecture

7. **Ronneberger et al. (2015)** - "U-Net: Convolutional Networks for Biomedical Image Segmentation"  
   *MICCAI 2015* — U-Net segmentation architecture

8. **Oktay et al. (2018)** - "Attention U-Net: Learning Where to Look for the Pancreas"  
   *MIDL 2018* — Attention mechanisms for medical imaging

9. **Baid et al. (2021)** - "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification"  
   *arXiv:2107.02314* — BraTS 2021 benchmark dataset

10. **Hochreiter & Schmidhuert (1997)** - "Long Short-Term Memory"  
    *Neural Computation 9(8)* — LSTM architecture

11. **Dwork et al. (2006)** - "Differential Privacy"  
    *ICALP 2006* — Formal privacy framework

12. **Bonawitz et al. (2017)** - "Towards Federated Learning at Scale"  
    *MLSys 2019* — Secure aggregation protocol

13. **Edla & Indhumathi (2025)** - "Enhancing Federated Learning with Quantum-Inspired PSO: An IID MNIST Study"  
    *Matrusri Engineering College* — Prior work on QPSO for FL

---

**Next:** Proceed to Chapter 3 for system requirements analysis and design rationale.
