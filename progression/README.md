# Module 3: Tumor Time Travel (Progression Forecasting)

Longitudinal brain tumor growth prediction using mathematical models and deep learning.

## Overview

This module predicts future tumor growth or shrinkage from historical MRI scans, enabling proactive clinical interventions.

**Pipeline:**
```
Longitudinal MRI → 3D U-Net Segmentation → Volume Extraction → Growth Model Fitting → 6-Month Prediction → RANO Alert
```

## Implementation Approaches

### Path A: Mathematical Models (Recommended Start)
| Model | Formula | Best For |
|-------|---------|----------|
| Exponential | V(t) = V₀ · eᵏᵗ | Early aggressive growth |
| Gompertz | V(t) = Vmax · e^(-ln(Vmax/V₀) · e^(-kt)) | Saturation plateau |
| Logistic | V(t) = Vmax / (1 + e^(-k(t-t₀))) | S-curve growth |
| Linear | V(t) = V₀ + kt | Constant-rate growth |

### Path B: LSTM Deep Learning (Advanced)
- 2-layer LSTM (hidden: 64 → 32)
- Sliding window input (seq_len=4)
- Predicts next volume in sequence

## Datasets

| Dataset | Patients | Source |
|---------|----------|--------|
| MU-Glioma-Post | 65 | [TCIA](https://www.cancerimagingarchive.net) |
| LUMIERE | 30+ | TCIA |
| UCSD-PTGBM | 50+ | TCIA |

## Output Metrics

- **RANO Status:** Complete Response / Partial Response / Stable Disease / Progressive Disease
- **Growth Rate:** AGR (cm³/month), RGR (%), Doubling Time (days)
- **Risk Level:** LOW / MODERATE / HIGH / CRITICAL

## Status

🔄 **In Progress** — See the [Complete Guide](../docs/TUMOR_PROGRESSION_COMPLETE_GUIDE.md) for implementation details.

## Getting Started

```bash
pip install -r ../requirements.txt
```

Refer to the [Tumor Progression Complete Guide](../docs/TUMOR_PROGRESSION_COMPLETE_GUIDE.md) for step-by-step implementation and code cells.
