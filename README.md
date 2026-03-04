<div align="center">

# 🧠 FL-QPSO Brain Tumor Management System

**Privacy-Preserving Brain Tumor Classification, Segmentation & Progression Forecasting**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![MONAI](https://img.shields.io/badge/MONAI-1.0+-00B388)](https://monai.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://kaggle.com)

*Federated Learning with Quantum Particle Swarm Optimization for multi-institutional brain tumor analysis*

[Overview](#overview) · [Modules](#modules) · [Results](#results) · [Quick Start](#quick-start) · [Contributing](#contributing) · [Citation](#citation)

</div>

---

## Overview

This project presents an **end-to-end brain tumor management pipeline** that combines three deep learning modules while preserving patient privacy through Federated Learning:

```mermaid
graph LR
    A["MRI Scans"] --> B["3D Attention U-Net\nSegmentation"]
    A --> C["FL-QPSO\nClassification"]
    B --> D["Tumor Time Travel\nProgression"]
    B --> E["Volume\nExtraction"]
    E --> D
    C --> F["Integrated\nRisk Score"]
    D --> F
    F --> G["Clinical\nDashboard"]
```

| Module | What It Does | Key Result |
|--------|-------------|------------|
| **Segmentation** | 3D volumetric tumor parsing (WT/TC/ET) | Dice Score: **0.76** |
| **Classification** | Privacy-preserving tumor typing via FL | Accuracy: **99.29%** |
| **Progression** | 6-month growth prediction + RANO alerts | LSTM + Math models |

> **Prior Work:** Extends our published research *"Enhancing Federated Learning with Quantum-Inspired PSO: An IID MNIST Study"* (Edla & Indhumathi, 2025) to the harder Non-IID medical imaging setting.

---

## Modules

### Module 1: 3D Attention U-Net Segmentation

Volumetric segmentation of brain tumors from multimodal MRI (BraTS 2021) using MONAI.

- **Input:** 4 MRI modalities (T1, T1ce, T2, FLAIR) — 3D volumes
- **Output:** Tumor masks (Whole Tumor, Tumor Core, Enhancing Tumor)
- **Architecture:** 3D Attention U-Net with attention gates
- **Results:** Mean Dice **0.76** | TC Dice **0.85** | ET Dice **0.79**
- **Demo:** Interactive Streamlit app for 3D slice visualization

📂 [`segmentation/`](segmentation/) · 📄 [Module README](segmentation/README.md)

---

### Module 2: Federated Classification with QPSO

Privacy-preserving brain tumor classification across 3 simulated hospital nodes.

- **Model:** ResNet-18 (ImageNet pretrained, 11.17M params)
- **Classes:** Glioma, Meningioma, Pituitary
- **Aggregation:** FedAvg vs FedProx vs **QPSO-FL**
- **3 Experimental Setups:** Natural heterogeneity ✅ | Label skew | Extreme skew

```
QPSO Update Rule:
  p = φ · mean(pbests) + (1-φ) · gbest          # attraction point
  x' = p ± β · |mbest - x| · ln(1/u)            # quantum position update
```

📂 [`federated_learning/`](federated_learning/) · 📄 [Module README](federated_learning/README.md)

---

### Module 3: Tumor Time Travel (Progression)

Longitudinal growth prediction using mathematical models and LSTM deep learning.

- **Mathematical Models:** Exponential, Gompertz, Logistic, Linear
- **Deep Learning:** LSTM time-series forecaster
- **Output:** 6-month growth curves, RANO status (CR/PR/SD/PD), risk alerts
- **Datasets:** MU-Glioma-Post, LUMIERE, UCSD-PTGBM (from TCIA)

📂 [`progression/`](progression/) · 📄 [Module README](progression/README.md)

---

## Results

### Classification — Setup 1 (Natural Heterogeneity, 100 Rounds)

| Metric | FedAvg | FedProx | QPSO-FL |
|--------|--------|---------|---------|
| **Final Accuracy** | 98.79% | **99.29%** | 98.43% |
| **Best Accuracy** | 99.14% | **99.29%** | 98.93% |
| **Rounds to 80%** | 2 | **1** | **1** |
| **Client Fairness (σ)** | 1.58 | 1.70 | **1.47** |

### Segmentation — BraTS 2021

| Region | Dice Score |
|--------|-----------|
| Tumor Core (TC) | **0.85** |
| Enhancing Tumor (ET) | 0.79 |
| Whole Tumor (WT) | 0.65 |
| **Mean** | **0.76** |

---

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- [Kaggle account](https://kaggle.com) for notebook execution

### Installation

```bash
# Clone the repository
git clone https://github.com/DIVYANSH-TEJA-09/BrainTumor-FL-Pipeline.git
cd FL_QPSO_FedAvg

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Modules

#### Segmentation (Local)
```bash
cd segmentation/streamlit_app
streamlit run app.py
```

#### Federated Learning (Kaggle)
Upload notebooks from `federated_learning/notebooks/` to Kaggle:

| Notebook | Purpose | Runtime |
|----------|---------|---------|
| `notebook1_data_prep.ipynb` | Prepare client datasets | ~30 min |
| `notebook2_training.ipynb` | Train FedAvg/FedProx/QPSO | ~7-8 hrs |
| `notebook3_evaluation.ipynb` | Analyze results & plots | ~15 min |

---

## Repository Structure

```
FL_QPSO_FedAvg/
│
├── segmentation/              # Module 1: 3D Attention U-Net
│   ├── dataset_step1_refined.ipynb
│   ├── streamlit_app/         # Interactive demo app
│   ├── extract_demo_data.py
│   └── inspect_data.py
│
├── segmentation_2d/           # 2D BraTS segmentation experiments
│   ├── BinarySeg.ipynb
│   └── MulticlassSeg.ipynb
│
├── federated_learning/        # Module 2: FL-QPSO Classification
│   ├── src/                   # Core implementation
│   │   ├── model.py           # ResNet-18 classifier
│   │   ├── client.py          # Federated client
│   │   ├── server_fedavg.py   # FedAvg aggregation
│   │   ├── server_qpso.py     # QPSO aggregation
│   │   ├── trainer_fedavg.py  # FedAvg training loop
│   │   ├── trainer_qpso.py    # QPSO training loop
│   │   ├── data_loader.py     # Multi-source data loading
│   │   ├── preprocessor.py    # Image preprocessing
│   │   ├── dataset.py         # PyTorch Dataset
│   │   ├── analysis.py        # Statistical analysis
│   │   ├── visualize.py       # Plotting utilities
│   │   └── utils.py           # Helper functions
│   ├── notebooks/             # Kaggle execution notebooks
│   ├── setup1_natural/        # Setup 1: Natural heterogeneity
│   ├── setup2_label_skew/     # Setup 2: 80/10/10 skew
│   └── setup3_extreme_skew/   # Setup 3: Single-class extreme
│
├── progression/               # Module 3: Tumor Time Travel
│
├── docs/                      # Documentation
│   ├── FINAL_PROJECT_PROPOSAL.md
│   ├── INTEGRATION_GUIDE.md
│   ├── TUMOR_PROGRESSION_COMPLETE_GUIDE.md
│   ├── PPT_CONTENT_AND_DIAGRAMS.md
│   └── PROJECT_DOCUMENTS_INDEX.md
│
├── diagrams/                  # Architecture diagrams
│   ├── mermaid/               # Mermaid source files
│   └── rendered/              # Exported PNGs
│
├── presentation/              # Presentation assets
│
├── .github/                   # GitHub templates
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Project Proposal](docs/FINAL_PROJECT_PROPOSAL.md) | Abstract, objectives, architecture, research & patent potential |
| [Integration Guide](docs/INTEGRATION_GUIDE.md) | How to connect all modules into a unified pipeline |
| [Progression Guide](docs/TUMOR_PROGRESSION_COMPLETE_GUIDE.md) | Complete tumor growth prediction implementation |
| [PPT Content](docs/PPT_CONTENT_AND_DIAGRAMS.md) | Presentation content with Mermaid diagrams |
| [Research Paper](federated_learning/RESEARCH_PAPER.md) | Paper reference for FL-QPSO vs FedAvg |

---

## Research & Publications

### Prior Publication
> Edla, D.T. & Indhumathi, L.K. (2025). *"Enhancing Federated Learning with Quantum-Inspired Particle Swarm Optimization: An IID MNIST Study."* Matrusri Engineering College.

### Target Venues
- **IEEE Transactions on Medical Imaging**
- **MICCAI** (Medical Image Computing and Computer Assisted Intervention)
- **Medical Image Analysis (MedIA)**

### Key References
- McMahan et al., 2017 — FedAvg
- Li et al., 2020 — FedProx
- Sun et al., 2004/2012 — QPSO Algorithm
- Oktay et al., 2018 — Attention U-Net
- Sheller et al., 2020 — FL for Brain Tumor Segmentation

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Branching strategy and commit conventions
- Code style (PEP 8, type hints, docstrings)
- Pull request process
- Bug reports and feature requests

---

## Citation

If you use this work, please cite:

```bibtex
@software{edla2025flqpso,
  title     = {FL-QPSO: Privacy-Preserving Brain Tumor Management System},
  author    = {Edla, Divyansh Teja and Indhumathi, L. K.},
  year      = {2025},
  institution = {Matrusri Engineering College},
  url       = {https://github.com/DIVYANSH-TEJA-09/BrainTumor-FL-Pipeline}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with** ❤️ **for privacy-preserving medical AI**

</div>
