# Chapter 1: Introduction

## 1.1 Background and Motivation

Brain tumors represent a significant global health challenge, requiring timely and accurate diagnosis, precise localization, and continuous monitoring. Traditional approaches to brain tumor analysis rely on centralized medical systems where patient data is pooled for collaborative model training. However, this centralized paradigm violates patient privacy and conflicts with healthcare regulations such as HIPAA (Health Insurance Portability and Accountability Act) and GDPR (General Data Protection Regulation).

**Federated Learning (FL)** offers a paradigm shift: multiple hospitals can collaboratively train a shared machine learning model *without sharing any raw patient data*. Instead, only model weights are exchanged between hospitals and a central aggregation server. This preserves privacy while enabling the development of robust, generalizable diagnostic systems.

### The Three-Stage Clinical Pipeline

This project implements a comprehensive brain tumor analysis system combining three independent modules executed in sequence:

```
MRI Scans → [Module 1: Classification (FL)] → IF Glioma → [Module 2: Segmentation] → [Module 3: Progression]
                    ↓
             Tumor Type Identified
             (Glioma/Meningioma/Pituitary)
```

**Critical Point:** Federated Learning is applied *exclusively* to **Module 1 (Classification)**. Modules 2 and 3 operate independently at individual hospitals after classification, requiring no federated infrastructure.

---

## 1.2 Research Problem and Objectives

### 1.2.1 The Core Research Question

Can **Quantum-behaved Particle Swarm Optimization (QPSO)** outperform traditional Federated Averaging (FedAvg) and Federated Proximal (FedProx) aggregation strategies for brain tumor classification under non-IID (non-Independently-and-Identically-Distributed) data conditions while maintaining clinical equity across hospitals?

### 1.2.2 Primary Objectives

1. **Module 1 - Federated Classification:**
   - Implement three FL aggregation strategies: FedAvg, FedProx, and QPSO-FL
   - Train across 3 simulated hospital clients with heterogeneous datasets
   - Classify tumors into 4 classes: Glioma, Meningioma, No Tumor, Pituitary
   - Evaluate under two data heterogeneity setups: Natural Heterogeneity and Moderate Label Skew

2. **Module 2 - 3D Brain Tumor Segmentation:**
   - Segment volumetric brain MRI (BraTS 2021) into tumor sub-regions
   - Identify Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)
   - Achieve clinical-grade segmentation (Mean Dice Score ≥ 0.75)

3. **Module 3 - Longitudinal Tumor Progression Forecasting:**
   - Predict 6-month tumor volume growth from historical MRI scans
   - Combine mathematical models (Logistic, Gompertz) with LSTM deep learning
   - Improve prediction accuracy through hybrid modeling approach

### 1.2.3 Secondary Objectives

- Compare QPSO's ability to preserve **clinical equity** (fairness across hospitals) vs. FedAvg/FedProx
- Demonstrate practical privacy preservation through federated learning
- Provide interpretable results suitable for clinical decision-making
- Build a modular, extensible pipeline for future multi-hospital deployment

---

## 1.3 Project Scope and Limitations

### 1.3.1 Scope

- **Federation Size:** 3 simulated hospitals with geographically distributed but computationally simulated clients
- **Classification Data:** ~9,300 total MRI images from Masoud (Kaggle) and BRISC 2025 datasets
- **Segmentation Data:** BraTS 2021 dataset with volumetric 3D MRI volumes and annotated tumor masks
- **Progression Data:** MU-Glioma-Post dataset from The Cancer Imaging Archive (TCIA) with 203 patients, 791 longitudinal timepoints
- **Training Infrastructure:** PyTorch, MONAI framework, Kaggle GPUs or local NVIDIA GPUs
- **Privacy Guarantee:** Structural privacy through federated learning (no raw patient data shared)

### 1.3.2 Limitations

1. **Classification Scalability:** Currently tested with 3 clients only; scalability to 5+ hospitals untested
2. **Data Distribution:** Natural heterogeneity simulated through different dataset sources rather than true geographic/institutional variation
3. **No Differential Privacy:** System implements only structural privacy (FL); formal DP-epsilon guarantees not provided
4. **Extreme Non-IID Conditions:** Single-class-per-client scenarios remain unsolved without personalization techniques
5. **Progression Sample Size:** LGG (slow-growing) sub-cohort limited to ~22 patients
6. **Model Capacity:** ResNet-18 is computationally powerful for a 4-class problem; may mask aggregation strategy differences

---

## 1.4 System Architecture Overview

### 1.4.1 Three-Module Design

The system is structured as three independent, sequential modules:

| Module | Purpose | Architecture | Data Privacy |
|--------|---------|--------------|---------------|
| **1: Classification (FL)** | Tumor type identification | ResNet-18 + 3 FL strategies | FEDERATED: Weights only exchanged |
| **2: Segmentation** | Tumor region delineation | 3D Attention U-Net (MONAI) | LOCAL: Hospital-only processing |
| **3: Progression** | Growth prediction | Math models + LSTM hybrid | LOCAL: Hospital-only processing |

### 1.4.2 Federated Learning Architecture (Module 1 Only)

```
┌─────────────────────────────────────────────────┐
│      Central Aggregation Server                 │
│      Strategy: FedAvg / FedProx / QPSO-FL      │
│                                                 │
│  1. Broadcast global model → all clients        │
│  2. Collect locally-trained weights ← clients   │
│  3. Aggregate using selected strategy           │
│  4. Evaluate on global balanced test set        │
│  5. Repeat for 100 communication rounds         │
└────────┬──────────────┬──────────────┬──────────┘
         │              │              │
    ┌────▼──────┐  ┌───▼────────┐  ┌──▼──────────┐
    │ Client 1  │  │ Client 2   │  │ Client 3    │
    │ Hospital  │  │ Hospital   │  │ Hospital    │
    │ ~1,200    │  │ ~3,900     │  │ ~4,200      │
    │ images    │  │ images     │  │ images      │
    └───────────┘  └────────────┘  └─────────────┘
```

**See also:** Diagram `01_system_architecture.png` in `/diagrams/rendered/`

---

## 1.5 Key Contributions

### 1.5.1 Federated Learning Contribution

1. **QPSO-based Aggregation for FL:**
   - Novel layer-by-layer QPSO aggregation with validation-loss fitness evaluation
   - Achieves superior clinical fairness: reduces max-min client performance gap by 72-81%
   - Statistically significant outperformance vs. FedAvg under label skew (p = 2.91 × 10⁻²²)

2. **Fairness-Focused Evaluation:**
   - Demonstrates that global accuracy alone masks clinical inequity
   - Proposes client-level fairness metric (standard deviation of per-client accuracies)
   - Shows QPSO protects smallest/most data-deprived hospital clients

### 1.5.2 Clinical Workflow Contribution

1. **End-to-End Privacy-Preserving Pipeline:**
   - Privacy-preserved tumor classification (FL) followed by local-only segmentation and progression
   - Modular design allows independent hospital adoption

2. **Multi-Modal Deep Learning Integration:**
   - 3D multimodal segmentation (T1, T1ce, T2, FLAIR)
   - Attention mechanisms for automated region-of-interest focus
   - Hybrid mathematical + deep learning progression model

---

## 1.6 Document Roadmap

This technical report is organized as follows:

| Chapter | Focus | Key Content |
|---------|-------|-------------|
| **2: Literature Survey** | State-of-the-art context | FL algorithms, brain tumor analysis, privacy in healthcare |
| **3: System Analysis** | Requirements & design rationale | Data requirements, module interactions, privacy design |
| **4: System Design** | Architectural details | Data flow, model architectures, aggregation algorithms |
| **5: Implementation** | Code & methodology | Core algorithms, training pipelines, inference mechanisms |
| **6: System Testing** | Test coverage & validation | Unit tests, integration tests, performance benchmarks |
| **7: Results & Analysis** | Experimental findings | Accuracy metrics, fairness analysis, comparative performance |
| **8: Conclusion** | Summary & future work | Key takeaways, limitations, research directions |
| **9: References** | Citations | Academic papers, official documentation, URLs |

---

## 1.7 Reading Guide for Different Audiences

- **Clinical Decision-Makers:** Start with §1.1-1.2, then skip to Chapter 7 (Results) for accuracy/performance metrics
- **ML/AI Researchers:** Follow standard document flow; focus on Chapter 4 (Algorithms) and Chapter 7 (Comparative Analysis)
- **Privacy/Security Officers:** Focus on §1.3 (Limitations), Chapter 3 (System Analysis - Privacy Design), and Chapter 5 (Implementation)
- **Developers/Engineers:** Prioritize Chapters 4-6 (Design, Implementation, Testing) for deployment guidance

---

## 1.8 Definitions and Terminology

| Term | Definition |
|------|-----------|
| **Federated Learning (FL)** | Collaborative model training where hospitals keep raw data local; only model weights are shared with a central server |
| **Non-IID Data** | Non-Independently-and-Identically-Distributed data; hospital datasets have different class distributions and data characteristics |
| **QPSO-FL** | Quantum-behaved Particle Swarm Optimization applied to Federated Learning aggregation |
| **Glioma** | Malignant brain tumor arising from glial cells; most aggressive type in dataset |
| **Whole Tumor (WT)** | Total tumor region including all sub-components in segmentation |
| **Segmentation** | Pixel-level classification to identify tumor regions in MRI |
| **Fairness (in FL)** | Equitable model performance across all participating hospitals; measured as low standard deviation of per-client accuracies |
| **MAE** | Mean Absolute Error; averaged prediction error magnitude |
| **Dice Score** | F1-like metric for segmentation; measures overlap between predicted and ground-truth masks |

---

## 1.9 Key Metrics and Success Criteria

### Module 1 (Classification - FL)

| Metric | Target | Status |
|--------|--------|--------|
| Global Accuracy (FedAvg baseline) | ≥ 95% | ✅ Achieved: 98.79% |
| QPSO Fairness (σ) | < FedAvg σ | ✅ Achieved: QPSO σ=1.47 vs FedAvg σ=1.58 |
| Client 1 Min Accuracy | ≥ 80% (clinical viability) | ✅ Achieved: 80.00% under label skew |
| Convergence Speed | Rounds to 80% acc | ✅ QPSO converges in 1-2 rounds |

### Module 2 (Segmentation)

| Metric | Target | Status |
|--------|--------|--------|
| Mean Dice Score | ≥ 0.75 (clinical grade) | ✅ Achieved: 0.76 |
| Tumor Core (TC) Dice | ≥ 0.80 | ✅ Achieved: 0.85 |
| Enhancing Tumor (ET) Dice | ≥ 0.75 | ✅ Achieved: 0.79 |

### Module 3 (Progression)

| Metric | Target | Status |
|--------|--------|--------|
| LSTM Hybrid MAE Improvement (HGG) | ≥ 5% | ✅ Achieved: 7.88% |
| Baseline R² (LGG) | ≥ 0.70 | ✅ Achieved: 0.76 |

---

**Next:** Proceed to Chapter 2 for comprehensive literature review and theoretical foundation.
