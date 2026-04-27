# Module 3: Tumor Progression Forecasting

Longitudinal brain tumor growth prediction using mathematical models and LSTM deep learning, with interactive 3D visualization.

## Overview

This module predicts future tumor growth or shrinkage from historical MRI scans, enabling proactive clinical interventions.

```
MRI Scans → 3D U-Net Segmentation → Volume Extraction → Growth Model Fitting → Prediction → Validation
```

## Quick Start

```bash
# 1. Ensure data is ready (already done)
python src/09_generate_enhanced_viz_data.py

# 2. Launch the dashboard
python run_streamlit_app.py

# 3. Open http://localhost:8501
```

## Approaches

### Path A: Mathematical Model (Logistic Growth)
- **Formula:** `V(t) = K / (1 + ((K - V₀)/V₀) × e^(-r×t))`
- **Fitted per-patient** with clinical data
- **Interpretable:** Parameters K (max size), r (growth rate) have clinical meaning
- **Performance:** R² = 0.62 (HGG), 0.76 (LGG) per-patient fits

### Path B: LSTM Hybrid (Residual Correction)
- Learns where the math model fails
- `V_hybrid = V_logistic + LSTM_correction`
- Architecture: LSTM(32) → Attention → FC(64→32→1)
- **7.88% MAE improvement on HGG** over baseline

## Dataset

| Dataset | Patients | Timepoints | Source |
|---------|----------|------------|--------|
| MU-Glioma-Post | 203 (111 modeled) | 791 | [TCIA](https://www.cancerimagingarchive.net) |

- **HGG:** 89 patients (aggressive growth)
- **LGG:** 22 patients (slow growth)
- **3–6 timepoints** per patient with real scan dates

## Project Structure

```
progression/
├── streamlit_3d_progression.py    ← Main app (Streamlit dashboard)
├── run_streamlit_app.py           ← Launcher
├── src/
│   ├── 00_verify_and_extract.py   ← Data preprocessing
│   ├── 01_mathematical_models.py  ← Four growth models
│   ├── 02_numeric_feature_builder.py
│   ├── 03_covariate_logistic_model.py
│   ├── 04_treatment_forced_logistic.py (failed experiment)
│   ├── 05_treatment_forced_logistic_v2.py (failed experiment)
│   ├── 06_hybrid_lstm_infrastructure.py
│   ├── 07_hybrid_lstm_training.py ← LSTM training
│   ├── 08_generate_viz_data.py
│   ├── 08_volume_to_masks.py
│   ├── 09_generate_enhanced_viz_data.py ← Enhanced data generator
│   ├── extract_real_trajectories.py
│   ├── infrastructure_lstm.py     ← LSTM model architecture
│   └── data_loader.py
├── data/
│   ├── raw/mu_glioma_post/        ← 11GB MRI data
│   └── processed/                 ← Extracted features & trajectories
├── results/                       ← Model outputs & trained weights
├── streamlit_data/                ← Dashboard data
├── docs/
│   ├── METHODOLOGY.md
│   ├── RESULTS.md
│   └── CHANGELOG.md
└── README.md                      ← This file
```

## Key Results

| Metric | Math Baseline | LSTM Hybrid | Improvement |
|--------|--------------|-------------|-------------|
| HGG MAE | 24,672 mm³ | 22,728 mm³ | **+7.88%** |
| HGG R² | 0.518 | 0.592 | +0.074 |
| LGG MAE | 166,728 mm³ | 167,317 mm³ | -0.35% (neutral) |
| Overall | 48,263 mm³ | 46,942 mm³ | +2.74% |

## Status

✅ Phase 0: Data Infrastructure
✅ Phase 1: Mathematical Models (4 models, real data)
✅ Phase 2: LSTM Hybrid (residual learning)
✅ Phase 3: Dashboard (Input → Predict → Validate workflow)
