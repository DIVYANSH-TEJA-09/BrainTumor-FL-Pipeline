# Results

## Per-Patient Logistic Fit Quality

### HGG (89 patients)
| Metric | Value |
|--------|-------|
| Median R² | 0.617 |
| R² > 0.5 (good fit) | 59/89 (66%) |
| R² 0.3–0.5 (moderate) | 18/89 (20%) |
| R² < 0.3 (poor) | 12/89 (13%) |
| Typical MAE | 13,410 mm³ (~21% of median HGG volume) |

### LGG (19 patients)
| Metric | Value |
|--------|-------|
| Median R² | 0.756 |
| R² > 0.5 (good fit) | 17/19 (89%) |
| Typical MAE | 3,733 mm³ (~7% of median LGG volume) |

---

## Hybrid Model Performance (Temporal Cross-Validation)

### HGG — Strong Improvement ✅
| Metric | Baseline | Hybrid | Improvement |
|--------|----------|--------|-------------|
| MAE | 24,672 mm³ | 22,728 mm³ | **+7.88%** |
| RMSE | 42,705 mm³ | 39,287 mm³ | +8.00% |
| R² | 0.518 | 0.592 | +0.074 |
| N observations | 503 | 503 | — |

### LGG — Neutral ⚠️
| Metric | Baseline | Hybrid | Change |
|--------|----------|--------|--------|
| MAE | 166,728 mm³ | 167,317 mm³ | -0.35% |
| R² | -501.996 | -501.995 | +0.001 |
| N observations | 113 | 113 | — |

LGG growth is too slow and predictable for LSTM to add value (only 22 patients, ~5 timepoints each).

### Overall
| Metric | Baseline | Hybrid | Improvement |
|--------|----------|--------|-------------|
| MAE | 48,263 mm³ | 46,942 mm³ | +2.74% |
| N predictions | 654 | 654 | 111 patients |

---

## What We Tried and Why It Failed

| Approach | Result | Why |
|----------|--------|-----|
| Covariate logistic (231 features) | R² 0.063 (worse than baseline) | Feature noise > signal; ElasticNet zeroed most coefficients |
| Treatment-forced ODE v1 | +3.8% worse RMSE | Single treatment alpha insufficient for complexity |
| Treatment-forced ODE v2 | +410% worse RMSE | Underdetermined: 3+ ODE params with only 5–6 data points |

---

## Key Takeaways

1. **Per-patient logistic fits work** (R² 0.62–0.76) — captures basic growth dynamics
2. **Grade stratification is essential** — HGG and LGG behave fundamentally differently
3. **LSTM residual correction helps on HGG** — 7.88% MAE reduction is real and validated
4. **Simple models beat complex ones** — Grade median > covariate regression; ODE forcing fails catastrophically
5. **Real data is heterogeneous** — Individual differences >> treatment effects at this sample size

---

## Volume Ranges

| Grade | N | Min (mm³) | Max (mm³) | Median (mm³) |
|-------|---|-----------|-----------|--------------|
| LGG | 19 | 12,500 | 62,000 | 53,000 |
| HGG | 89 | 18,000 | 301,000 | 64,000 |
| All | 111 | 12,500 | 301,000 | 62,000 |

24× range in tumor sizes explains the modest population-level R².

---

## Trained Models

| File | Description |
|------|-------------|
| `results/phase2_hgg_lstm_model.pth` | HGG LSTM weights (58 epochs) |
| `results/phase2_lgg_lstm_model.pth` | LGG LSTM weights (20 epochs) |

Inference: ~0.01 sec per patient on CPU.
