# Changelog

## Phase 3 — Dashboard Rebuild (2026-04-12)

### New
- **Complete UI rebuild** (`streamlit_3d_progression.py`): Input → Predict → Validate workflow
  - 4-panel 3D view: Input Scan, Math Prediction, LSTM Prediction, Ground Truth
  - Method explanation cards showing logistic formula, fitted parameters, LSTM architecture
  - Growth curve visualization with fitted logistic overlay and residual chart
  - Per-timepoint error comparison table + bar chart
  - Overall model performance stats (HGG, LGG, Overall)
  - Dark premium theme with glassmorphism styling
- **Enhanced data generator** (`src/09_generate_enhanced_viz_data.py`): includes real scan days, logistic parameters, residuals
- **Consolidated documentation**: 21 markdown files → README.md + docs/METHODOLOGY.md + docs/RESULTS.md + docs/CHANGELOG.md

### Fixed
- **Time spacing bug**: Replaced hardcoded 30-day assumption with actual scan dates from clinical metadata (`day_from_diagnosis_imputed`)
- **Prediction index**: Now includes real days, logistic model params (V₀, K, r, R²), and per-timepoint residuals

### Removed
- `streamlit_3d_progression_enhanced.py` (duplicate app)
- `progression/progression/streamlit_data/` (duplicate nested directory)
- Redundant MD files moved to `docs/_archive/` (kept for reference)

---

## Phase 2 — LSTM Hybrid Model (2026-04-12)

### New
- Grade-stratified LSTM trained on logistic residuals
- 7.88% MAE improvement on HGG; neutral on LGG
- 3D visualization with brain overlay and synchronized camera
- Trained model checkpoints (`.pth` files)
- 654 predictions from 111 patients

---

## Phase 1 — Mathematical Models (2026-04-12)

### New
- Real tumor volume extraction from NIfTI segmentation masks
- Per-patient logistic growth fit (R² 0.617 HGG, 0.756 LGG)
- Covariate-augmented model (underperformed baseline)
- Treatment-forced ODE experiments (both failed)

### Fixed
- Replaced synthetic trajectory generation with real data extraction
- Documented synthetic data problem in REAL_vs_SYNTHETIC_EXPLANATION.md

---

## Phase 0 — Data Infrastructure (2026-04-12)

### New
- Dataset analysis: MU-Glioma-Post selected (203 patients, 596 timepoints)
- Data download instructions and preprocessing scripts
- Grade stratification (LGG/HGG) from clinical metadata
- PyTorch data loading infrastructure
