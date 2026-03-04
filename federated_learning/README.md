# Module 2: Federated Classification with QPSO

Privacy-preserving brain tumor classification using Federated Learning with Quantum Particle Swarm Optimization.

## Overview

This module trains a **ResNet-18** classifier across 3 simulated hospital nodes to classify brain tumors into **Glioma**, **Meningioma**, and **Pituitary** — without sharing any patient data.

Three aggregation strategies are benchmarked:
- **FedAvg** — Weighted averaging baseline
- **FedProx** — Proximal regularization (prevents client drift)
- **QPSO-FL** — Quantum-inspired swarm optimization (ours)

## Source Code

| File | Description |
|------|-------------|
| `src/model.py` | ResNet-18 with custom 3-class FC head |
| `src/client.py` | Federated client (local training logic) |
| `src/server_fedavg.py` | FedAvg aggregation server |
| `src/server_qpso.py` | QPSO aggregation server |
| `src/trainer_fedavg.py` | FedAvg training loop |
| `src/trainer_qpso.py` | QPSO training loop |
| `src/data_loader.py` | Multi-source dataset loading |
| `src/preprocessor.py` | Image preprocessing pipeline |
| `src/dataset.py` | PyTorch Dataset class |
| `src/analysis.py` | Statistical analysis utilities |
| `src/visualize.py` | Plotting and visualization |
| `src/utils.py` | Helper functions |

## Notebooks (Run on Kaggle)

| # | Notebook | Purpose | Runtime |
|---|----------|---------|---------|
| 1 | `notebook1_data_prep.ipynb` | Download & preprocess datasets → `.npy` | ~30 min |
| 2 | `notebook2_training.ipynb` | Train FedAvg / FedProx / QPSO-FL | ~7-8 hrs |
| 3 | `notebook3_evaluation.ipynb` | Generate plots, tables, statistics | ~15 min |

## Experimental Setups

| Setup | Description | Status |
|-------|-------------|--------|
| `setup1_natural/` | Natural heterogeneity (different data sources) | ✅ Complete |
| `setup2_label_skew/` | Moderate label skew (80/10/10 per client) | 🔄 Planned |
| `setup3_extreme_skew/` | Extreme skew (single class per client) | 🔄 Planned |

## Datasets

| Client | Source | Images | Role |
|--------|--------|--------|------|
| Client 1 | Masoud Brain Tumor MRI (Test) | ~1,200 | Smallest hospital |
| Client 2 | BRISC 2025 | ~3,900 | Medium hospital |
| Client 3 | Masoud Brain Tumor MRI (Train) | ~4,200 | Largest hospital |

**Classes:** Glioma (0), Meningioma (1), Pituitary (2)

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Rounds | 100 |
| Local epochs | 5 |
| Optimizer | Adam (lr=0.001) |
| Batch size | 32 |
| QPSO β | 0.7 |
| Image size | 224 × 224 |

## Results (Setup 1)

| Metric | FedAvg | FedProx | QPSO-FL |
|--------|--------|---------|---------|
| Final Accuracy | 98.79% | **99.29%** | 98.43% |
| Client Fairness (σ) | 1.58 | 1.70 | **1.47** |

## References

- [Research Paper Notes](RESEARCH_PAPER.md)
- [Approach & Results](APPROACH.md)
- [FL-QPSO Complete Guide](FL_QPSO_COMPLETE_GUIDE.md)
