"""
Treatment-forced logistic ODE vs plain logistic (real trajectories).

Goal:
- Evaluate whether adding explicit treatment timing improves short-horizon
  progression forecasting compared with plain logistic growth.

Method:
- Per patient temporal holdout (train early points, test later points).
- Plain model: logistic(K, r) with fixed V0.
- Forced model: dV/dt = r*V*(1 - V/K) - alpha*I_treat(t)*V
  where I_treat(t)=1 during chemo/radiation windows from clinical timing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, least_squares


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "longitudinal_modeling_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"


def logistic_with_v0(t: np.ndarray, k: float, r: float, v0: float) -> np.ndarray:
    return k / (1.0 + ((k - v0) / v0) * np.exp(-r * t))


def grade_from_numeric(g: float) -> str:
    if not np.isfinite(g):
        return "UNKNOWN"
    if g in (3.0, 4.0):
        return "HGG"
    if g == 2.0:
        return "LGG"
    return "UNKNOWN"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": int(len(y_true))}


def fit_plain_logistic(days_abs: np.ndarray, vols: np.ndarray) -> Tuple[float, float, bool]:
    t = days_abs - days_abs[0]
    v0 = float(vols[0])
    vmax = float(np.max(vols))

    k0 = max(vmax * 1.5, v0 * 1.1, 1.0)
    r0 = 0.01

    try:
        popt, _ = curve_fit(
            lambda tt, kk, rr: logistic_with_v0(tt, kk, rr, v0),
            t,
            vols,
            p0=[k0, r0],
            bounds=([1.0, 1e-6], [max(vmax * 50.0, 1e7), 0.2]),
            maxfev=40000,
        )
        return float(popt[0]), float(popt[1]), True
    except Exception:
        return float(k0), float(r0), False


def windows_from_row(row: pd.Series) -> List[Tuple[float, float]]:
    wins: List[Tuple[float, float]] = []

    def get_num(col: str) -> float:
        return float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))

    # Initial chemo window
    chemo_yes = float(pd.to_numeric(row.get("initial_chemo_therapy_Yes", np.nan), errors="coerce"))
    cs = get_num("number_of_days_from_diagnosis_to_initial_chemo_therapy_start_date")
    ce = get_num("number_of_days_from_diagnosis_to_initial_chemo_therapy_end_date")
    if np.isfinite(chemo_yes) and chemo_yes == 1.0 and np.isfinite(cs):
        if not np.isfinite(ce):
            ce = cs + 180.0
        if ce < cs:
            ce = cs
        wins.append((cs, ce))

    # Radiation window
    rad_yes = float(pd.to_numeric(row.get("radiation_therapy_Yes", np.nan), errors="coerce"))
    rs = get_num("number_of_days_from_diagnosis_to_radiation_therapy_start_date")
    re = get_num("number_of_days_from_diagnosis_to_radiation_therapy_end_date")
    if np.isfinite(rad_yes) and rad_yes == 1.0 and np.isfinite(rs):
        if not np.isfinite(re):
            re = rs + 42.0
        if re < rs:
            re = rs
        wins.append((rs, re))

    return wins


def is_treatment_on(t_abs: float, windows: List[Tuple[float, float]]) -> float:
    for s, e in windows:
        if s <= t_abs <= e:
            return 1.0
    return 0.0


def simulate_forced(days_abs: np.ndarray, v0: float, k: float, r: float, alpha: float, windows: List[Tuple[float, float]]) -> np.ndarray:
    if len(days_abs) == 0:
        return np.array([], dtype=float)

    t0 = float(days_abs[0])
    t1 = float(days_abs[-1])
    if t1 == t0:
        return np.full_like(days_abs, fill_value=v0, dtype=float)

    def ode(t: float, y: np.ndarray) -> np.ndarray:
        v = float(max(y[0], 1e-6))
        it = is_treatment_on(t, windows)
        dv = r * v * (1.0 - v / max(k, 1.0)) - alpha * it * v
        return np.array([dv], dtype=float)

    sol = solve_ivp(
        ode,
        t_span=(t0, t1),
        y0=np.array([v0], dtype=float),
        t_eval=days_abs,
        method="RK45",
        max_step=7.0,
        rtol=1e-4,
        atol=1e-4,
    )
    if not sol.success or sol.y.shape[1] != len(days_abs):
        # fallback to constant if solver fails
        return np.full_like(days_abs, fill_value=v0, dtype=float)
    return sol.y[0]


def fit_forced_model(days_abs: np.ndarray, vols: np.ndarray, windows: List[Tuple[float, float]]) -> Tuple[float, float, float, bool]:
    v0 = float(vols[0])
    vmax = float(np.max(vols))
    k0 = max(vmax * 1.5, v0 * 1.1, 1.0)
    r0 = 0.01

    has_window = len(windows) > 0
    if not has_window:
        k_plain, r_plain, ok = fit_plain_logistic(days_abs, vols)
        return k_plain, r_plain, 0.0, ok

    # Optimize in log-space to enforce positivity.
    p0 = np.array([np.log(k0), np.log(r0), np.log(0.002)], dtype=float)
    lb = np.array([np.log(1.0), np.log(1e-6), np.log(1e-8)], dtype=float)
    ub = np.array([np.log(max(vmax * 50.0, 1e7)), np.log(0.2), np.log(0.2)], dtype=float)

    def residuals(p: np.ndarray) -> np.ndarray:
        k = float(np.exp(p[0]))
        r = float(np.exp(p[1]))
        alpha = float(np.exp(p[2]))
        pred = simulate_forced(days_abs, v0, k, r, alpha, windows)
        return pred - vols

    try:
        res = least_squares(residuals, p0, bounds=(lb, ub), max_nfev=1500)
        k = float(np.exp(res.x[0]))
        r = float(np.exp(res.x[1]))
        a = float(np.exp(res.x[2]))
        return k, r, a, bool(res.success)
    except Exception:
        return k0, r0, 0.0, False


def treatment_overlap(windows: List[Tuple[float, float]], d0: float, d1: float) -> bool:
    for s, e in windows:
        if max(s, d0) <= min(e, d1):
            return True
    return False


def summarize_rows(rows_df: pd.DataFrame, model_col_prefix: str) -> Dict[str, float]:
    y = rows_df["y_true"].values.astype(float)
    p = rows_df[f"{model_col_prefix}_pred"].values.astype(float)
    return compute_metrics(y, p)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["tumor_total_volume_mm3"] = pd.to_numeric(df["tumor_total_volume_mm3"], errors="coerce")
    df["day_from_diagnosis_imputed"] = pd.to_numeric(df["day_from_diagnosis_imputed"], errors="coerce")
    df = df.dropna(subset=["tumor_total_volume_mm3", "day_from_diagnosis_imputed"]).copy()

    patient_rows = []
    point_rows = []

    for pid, g in df.groupby("patient_id"):
        g = g.sort_values(["day_from_diagnosis_imputed", "timepoint"]).copy()
        days = g["day_from_diagnosis_imputed"].values.astype(float)
        vols = g["tumor_total_volume_mm3"].values.astype(float)

        # remove duplicate day rows by keeping first
        dedup = pd.DataFrame({"d": days, "v": vols}).drop_duplicates(subset=["d"], keep="first")
        days = dedup["d"].values.astype(float)
        vols = dedup["v"].values.astype(float)

        if len(days) < 4:
            continue
        if np.any(vols <= 0):
            continue

        n = len(days)
        split = max(3, int(np.ceil(0.7 * n)))
        if split >= n:
            split = n - 1

        train_days = days[:split]
        train_vols = vols[:split]
        test_days = days[split:]
        test_vols = vols[split:]
        if len(test_days) == 0:
            continue

        first_row = g.iloc[0]
        grade = grade_from_numeric(float(pd.to_numeric(first_row.get("grade_of_primary_brain_tumor", np.nan), errors="coerce")))
        wins = windows_from_row(first_row)
        has_overlap = treatment_overlap(wins, float(train_days[0]), float(test_days[-1])) if wins else False

        # Plain logistic
        k_plain, r_plain, ok_plain = fit_plain_logistic(train_days, train_vols)
        t_test_rel = test_days - train_days[0]
        plain_pred = logistic_with_v0(t_test_rel, k_plain, r_plain, float(train_vols[0]))

        # Forced logistic
        k_forced, r_forced, a_forced, ok_forced = fit_forced_model(train_days, train_vols, wins)
        forced_eval_days = np.concatenate(([train_days[0]], test_days))
        forced_full = simulate_forced(
            forced_eval_days,
            float(train_vols[0]),
            float(k_forced),
            float(r_forced),
            float(a_forced),
            wins,
        )
        forced_pred = forced_full[1:]

        m_plain = compute_metrics(test_vols, plain_pred)
        m_forced = compute_metrics(test_vols, forced_pred)

        patient_rows.append(
            {
                "patient_id": pid,
                "grade": grade,
                "n_total": n,
                "n_train": len(train_days),
                "n_test": len(test_days),
                "treatment_window_count": len(wins),
                "treatment_overlap_test": int(has_overlap),
                "plain_k": k_plain,
                "plain_r": r_plain,
                "plain_fit_ok": int(ok_plain),
                "forced_k": k_forced,
                "forced_r": r_forced,
                "forced_alpha": a_forced,
                "forced_fit_ok": int(ok_forced),
                "plain_mae_test": m_plain["mae"],
                "plain_rmse_test": m_plain["rmse"],
                "plain_r2_test": m_plain["r2"],
                "forced_mae_test": m_forced["mae"],
                "forced_rmse_test": m_forced["rmse"],
                "forced_r2_test": m_forced["r2"],
                "delta_rmse_forced_minus_plain": m_forced["rmse"] - m_plain["rmse"],
                "delta_mae_forced_minus_plain": m_forced["mae"] - m_plain["mae"],
            }
        )

        for dd, yt, yp0, yp1 in zip(test_days, test_vols, plain_pred, forced_pred):
            point_rows.append(
                {
                    "patient_id": pid,
                    "grade": grade,
                    "day": float(dd),
                    "y_true": float(yt),
                    "plain_pred": float(yp0),
                    "forced_pred": float(yp1),
                    "treatment_overlap_test": int(has_overlap),
                }
            )

    patient_df = pd.DataFrame(patient_rows)
    point_df = pd.DataFrame(point_rows)

    # Overall summaries
    overall_plain = summarize_rows(point_df, "plain") if not point_df.empty else {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
    overall_forced = summarize_rows(point_df, "forced") if not point_df.empty else {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}

    by_grade_rows = []
    for grade, gg in point_df.groupby("grade"):
        mp = summarize_rows(gg, "plain")
        mf = summarize_rows(gg, "forced")
        by_grade_rows.append(
            {
                "grade": grade,
                "plain_mae": mp["mae"],
                "plain_rmse": mp["rmse"],
                "plain_r2": mp["r2"],
                "forced_mae": mf["mae"],
                "forced_rmse": mf["rmse"],
                "forced_r2": mf["r2"],
                "delta_rmse_forced_minus_plain": mf["rmse"] - mp["rmse"],
                "n_obs": mp["n"],
            }
        )

    by_grade_df = pd.DataFrame(by_grade_rows)

    # Treatment-overlap subset (where forced term is actually active during test window)
    overlap_points = point_df[point_df["treatment_overlap_test"] == 1].copy()
    if not overlap_points.empty:
        overlap_plain = summarize_rows(overlap_points, "plain")
        overlap_forced = summarize_rows(overlap_points, "forced")
    else:
        overlap_plain = {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
        overlap_forced = {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}

    # Per-patient win counts.
    improved_rmse = int((patient_df["delta_rmse_forced_minus_plain"] < 0).sum()) if not patient_df.empty else 0
    worsened_rmse = int((patient_df["delta_rmse_forced_minus_plain"] > 0).sum()) if not patient_df.empty else 0

    summary = {
        "n_patients_evaluated": int(patient_df["patient_id"].nunique()) if not patient_df.empty else 0,
        "n_test_points": int(len(point_df)),
        "overall_plain": overall_plain,
        "overall_forced": overall_forced,
        "overall_delta_rmse_forced_minus_plain": float(overall_forced["rmse"] - overall_plain["rmse"])
        if np.isfinite(overall_forced["rmse"]) and np.isfinite(overall_plain["rmse"])
        else np.nan,
        "overlap_subset": {
            "n_points": int(len(overlap_points)),
            "plain": overlap_plain,
            "forced": overlap_forced,
            "delta_rmse_forced_minus_plain": float(overlap_forced["rmse"] - overlap_plain["rmse"])
            if np.isfinite(overlap_forced["rmse"]) and np.isfinite(overlap_plain["rmse"])
            else np.nan,
        },
        "patient_level": {
            "improved_rmse_count": improved_rmse,
            "worsened_rmse_count": worsened_rmse,
            "mean_delta_rmse_forced_minus_plain": float(patient_df["delta_rmse_forced_minus_plain"].mean())
            if not patient_df.empty
            else np.nan,
            "median_delta_rmse_forced_minus_plain": float(patient_df["delta_rmse_forced_minus_plain"].median())
            if not patient_df.empty
            else np.nan,
        },
        "by_grade": by_grade_df.to_dict(orient="records"),
    }

    out_patient = RESULTS_DIR / "phase1_treatment_forced_patient_metrics.csv"
    out_points = RESULTS_DIR / "phase1_treatment_forced_point_predictions.csv"
    out_grade = RESULTS_DIR / "phase1_treatment_forced_by_grade.csv"
    out_summary = RESULTS_DIR / "phase1_treatment_forced_summary.json"

    patient_df.to_csv(out_patient, index=False)
    point_df.to_csv(out_points, index=False)
    by_grade_df.to_csv(out_grade, index=False)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Treatment-forced logistic evaluation complete.")
    print(f"[OK] Patients evaluated: {summary['n_patients_evaluated']}")
    print(f"[OK] Test points: {summary['n_test_points']}")
    print(f"[OK] Wrote: {out_summary}")


if __name__ == "__main__":
    main()
