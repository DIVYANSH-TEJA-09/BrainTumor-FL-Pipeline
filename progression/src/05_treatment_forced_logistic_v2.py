"""
Treatment-forced logistic v2: separate chemo/radiation effects + carry-over.

Compares against plain logistic on per-patient temporal holdout.
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


def parse_windows(row: pd.Series) -> Dict[str, Tuple[float, float] | None]:
    def num(col: str) -> float:
        return float(pd.to_numeric(row.get(col, np.nan), errors="coerce"))

    out: Dict[str, Tuple[float, float] | None] = {"chemo": None, "rad": None}

    chemo_yes = num("initial_chemo_therapy_Yes")
    cs = num("number_of_days_from_diagnosis_to_initial_chemo_therapy_start_date")
    ce = num("number_of_days_from_diagnosis_to_initial_chemo_therapy_end_date")
    if np.isfinite(chemo_yes) and chemo_yes == 1.0 and np.isfinite(cs):
        if not np.isfinite(ce):
            ce = cs + 180.0
        if ce < cs:
            ce = cs
        out["chemo"] = (cs, ce)

    rad_yes = num("radiation_therapy_Yes")
    rs = num("number_of_days_from_diagnosis_to_radiation_therapy_start_date")
    re = num("number_of_days_from_diagnosis_to_radiation_therapy_end_date")
    if np.isfinite(rad_yes) and rad_yes == 1.0 and np.isfinite(rs):
        if not np.isfinite(re):
            re = rs + 42.0
        if re < rs:
            re = rs
        out["rad"] = (rs, re)

    return out


def effect_with_carryover(t_abs: float, window: Tuple[float, float] | None, tau: float) -> float:
    if window is None:
        return 0.0
    start, end = window
    if t_abs < start:
        return 0.0
    if start <= t_abs <= end:
        return 1.0
    # carry-over decay after treatment completion
    return float(np.exp(-(t_abs - end) / max(tau, 1e-3)))


def simulate_forced_v2(
    days_abs: np.ndarray,
    v0: float,
    k: float,
    r: float,
    alpha_chemo: float,
    alpha_rad: float,
    windows: Dict[str, Tuple[float, float] | None],
    tau_chemo: float = 60.0,
    tau_rad: float = 30.0,
) -> np.ndarray:
    if len(days_abs) == 0:
        return np.array([], dtype=float)

    t0 = float(days_abs[0])
    t1 = float(days_abs[-1])
    if t1 == t0:
        return np.full_like(days_abs, fill_value=v0, dtype=float)

    def ode(t: float, y: np.ndarray) -> np.ndarray:
        v = float(max(y[0], 1e-6))
        i_chemo = effect_with_carryover(t, windows.get("chemo"), tau_chemo)
        i_rad = effect_with_carryover(t, windows.get("rad"), tau_rad)
        kill = alpha_chemo * i_chemo + alpha_rad * i_rad
        dv = r * v * (1.0 - v / max(k, 1.0)) - kill * v
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
        return np.full_like(days_abs, fill_value=v0, dtype=float)
    return sol.y[0]


def fit_forced_v2(days_abs: np.ndarray, vols: np.ndarray, windows: Dict[str, Tuple[float, float] | None]) -> Tuple[float, float, float, float, bool]:
    v0 = float(vols[0])
    vmax = float(np.max(vols))
    k0 = max(vmax * 1.5, v0 * 1.1, 1.0)
    r0 = 0.01

    has_chemo = windows.get("chemo") is not None
    has_rad = windows.get("rad") is not None

    if not has_chemo and not has_rad:
        k_plain, r_plain, ok = fit_plain_logistic(days_abs, vols)
        return k_plain, r_plain, 0.0, 0.0, ok

    names = ["log_k", "log_r"]
    p0 = [np.log(k0), np.log(r0)]
    lb = [np.log(1.0), np.log(1e-6)]
    ub = [np.log(max(vmax * 50.0, 1e7)), np.log(0.2)]

    if has_chemo:
        names.append("log_ac")
        p0.append(np.log(0.002))
        lb.append(np.log(1e-8))
        ub.append(np.log(0.5))
    if has_rad:
        names.append("log_ar")
        p0.append(np.log(0.002))
        lb.append(np.log(1e-8))
        ub.append(np.log(0.5))

    p0 = np.array(p0, dtype=float)
    lb = np.array(lb, dtype=float)
    ub = np.array(ub, dtype=float)

    def unpack(p: np.ndarray) -> Tuple[float, float, float, float]:
        k = float(np.exp(p[0]))
        r = float(np.exp(p[1]))
        idx = 2
        ac = 0.0
        ar = 0.0
        if has_chemo:
            ac = float(np.exp(p[idx]))
            idx += 1
        if has_rad:
            ar = float(np.exp(p[idx]))
        return k, r, ac, ar

    def residuals(p: np.ndarray) -> np.ndarray:
        k, r, ac, ar = unpack(p)
        pred = simulate_forced_v2(days_abs, v0, k, r, ac, ar, windows)
        # stabilize heavy-tail volume errors with log residuals
        core = np.log1p(pred) - np.log1p(vols)
        # weak regularization on treatment effects
        reg = np.array([0.5 * ac, 0.5 * ar], dtype=float)
        return np.concatenate([core, reg])

    try:
        res = least_squares(residuals, p0, bounds=(lb, ub), max_nfev=3000)
        k, r, ac, ar = unpack(res.x)
        return k, r, ac, ar, bool(res.success)
    except Exception:
        return k0, r0, 0.0, 0.0, False


def treatment_overlap(windows: Dict[str, Tuple[float, float] | None], d0: float, d1: float) -> bool:
    for win in windows.values():
        if win is None:
            continue
        s, e = win
        if max(s, d0) <= min(e, d1):
            return True
    return False


def summarize_point_df(point_df: pd.DataFrame, pred_col: str) -> Dict[str, float]:
    return compute_metrics(point_df["y_true"].values.astype(float), point_df[pred_col].values.astype(float))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["tumor_total_volume_mm3"] = pd.to_numeric(df["tumor_total_volume_mm3"], errors="coerce")
    df["day_from_diagnosis_imputed"] = pd.to_numeric(df["day_from_diagnosis_imputed"], errors="coerce")
    df = df.dropna(subset=["tumor_total_volume_mm3", "day_from_diagnosis_imputed"]).copy()

    patient_rows: List[Dict[str, float | int | str]] = []
    point_rows: List[Dict[str, float | int | str]] = []

    for pid, g in df.groupby("patient_id"):
        g = g.sort_values(["day_from_diagnosis_imputed", "timepoint"]).copy()
        days = g["day_from_diagnosis_imputed"].values.astype(float)
        vols = g["tumor_total_volume_mm3"].values.astype(float)

        # remove duplicate day observations
        ded = pd.DataFrame({"d": days, "v": vols}).drop_duplicates(subset=["d"], keep="first")
        days = ded["d"].values.astype(float)
        vols = ded["v"].values.astype(float)

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
        grade_num = float(pd.to_numeric(first_row.get("grade_of_primary_brain_tumor", np.nan), errors="coerce"))
        grade = grade_from_numeric(grade_num)
        windows = parse_windows(first_row)
        overlap = treatment_overlap(windows, float(test_days[0]), float(test_days[-1]))

        # plain baseline
        kp, rp, okp = fit_plain_logistic(train_days, train_vols)
        plain_pred = logistic_with_v0(test_days - train_days[0], kp, rp, float(train_vols[0]))

        # forced v2
        kf, rf, ac, ar, okf = fit_forced_v2(train_days, train_vols, windows)
        eval_days = np.concatenate(([train_days[0]], test_days))
        forced_pred = simulate_forced_v2(eval_days, float(train_vols[0]), kf, rf, ac, ar, windows)[1:]

        mp = compute_metrics(test_vols, plain_pred)
        mf = compute_metrics(test_vols, forced_pred)

        patient_rows.append(
            {
                "patient_id": pid,
                "grade": grade,
                "n_total": n,
                "n_train": len(train_days),
                "n_test": len(test_days),
                "has_chemo_window": int(windows.get("chemo") is not None),
                "has_rad_window": int(windows.get("rad") is not None),
                "treatment_overlap_test": int(overlap),
                "plain_k": kp,
                "plain_r": rp,
                "plain_fit_ok": int(okp),
                "forced_k": kf,
                "forced_r": rf,
                "forced_alpha_chemo": ac,
                "forced_alpha_rad": ar,
                "forced_fit_ok": int(okf),
                "plain_mae_test": mp["mae"],
                "plain_rmse_test": mp["rmse"],
                "plain_r2_test": mp["r2"],
                "forced_mae_test": mf["mae"],
                "forced_rmse_test": mf["rmse"],
                "forced_r2_test": mf["r2"],
                "delta_rmse_forced_minus_plain": mf["rmse"] - mp["rmse"],
                "delta_mae_forced_minus_plain": mf["mae"] - mp["mae"],
            }
        )

        for dd, yt, y0, y1 in zip(test_days, test_vols, plain_pred, forced_pred):
            point_rows.append(
                {
                    "patient_id": pid,
                    "grade": grade,
                    "day": float(dd),
                    "y_true": float(yt),
                    "plain_pred": float(y0),
                    "forced_v2_pred": float(y1),
                    "treatment_overlap_test": int(overlap),
                }
            )

    patient_df = pd.DataFrame(patient_rows)
    point_df = pd.DataFrame(point_rows)

    if point_df.empty:
        raise RuntimeError("No valid patients for treatment-forced v2 evaluation.")

    overall_plain = summarize_point_df(point_df, "plain_pred")
    overall_forced = summarize_point_df(point_df, "forced_v2_pred")

    by_grade_rows = []
    for grade, gg in point_df.groupby("grade"):
        mp = summarize_point_df(gg, "plain_pred")
        mf = summarize_point_df(gg, "forced_v2_pred")
        by_grade_rows.append(
            {
                "grade": grade,
                "plain_mae": mp["mae"],
                "plain_rmse": mp["rmse"],
                "plain_r2": mp["r2"],
                "forced_v2_mae": mf["mae"],
                "forced_v2_rmse": mf["rmse"],
                "forced_v2_r2": mf["r2"],
                "delta_rmse_forced_minus_plain": mf["rmse"] - mp["rmse"],
                "n_obs": mp["n"],
            }
        )
    by_grade_df = pd.DataFrame(by_grade_rows)

    overlap = point_df[point_df["treatment_overlap_test"] == 1].copy()
    if not overlap.empty:
        ov_plain = summarize_point_df(overlap, "plain_pred")
        ov_forced = summarize_point_df(overlap, "forced_v2_pred")
        ov_delta = ov_forced["rmse"] - ov_plain["rmse"]
    else:
        ov_plain = {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
        ov_forced = {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n": 0}
        ov_delta = np.nan

    summary = {
        "n_patients_evaluated": int(patient_df["patient_id"].nunique()),
        "n_test_points": int(len(point_df)),
        "overall_plain": overall_plain,
        "overall_forced_v2": overall_forced,
        "overall_delta_rmse_forced_minus_plain": float(overall_forced["rmse"] - overall_plain["rmse"]),
        "patient_level": {
            "improved_rmse_count": int((patient_df["delta_rmse_forced_minus_plain"] < 0).sum()),
            "worsened_rmse_count": int((patient_df["delta_rmse_forced_minus_plain"] > 0).sum()),
            "mean_delta_rmse_forced_minus_plain": float(patient_df["delta_rmse_forced_minus_plain"].mean()),
            "median_delta_rmse_forced_minus_plain": float(patient_df["delta_rmse_forced_minus_plain"].median()),
        },
        "overlap_subset": {
            "n_points": int(len(overlap)),
            "plain": ov_plain,
            "forced_v2": ov_forced,
            "delta_rmse_forced_minus_plain": float(ov_delta) if np.isfinite(ov_delta) else np.nan,
        },
        "by_grade": by_grade_df.to_dict(orient="records"),
    }

    out_patient = RESULTS_DIR / "phase1_treatment_forced_v2_patient_metrics.csv"
    out_points = RESULTS_DIR / "phase1_treatment_forced_v2_point_predictions.csv"
    out_grade = RESULTS_DIR / "phase1_treatment_forced_v2_by_grade.csv"
    out_summary = RESULTS_DIR / "phase1_treatment_forced_v2_summary.json"

    patient_df.to_csv(out_patient, index=False)
    point_df.to_csv(out_points, index=False)
    by_grade_df.to_csv(out_grade, index=False)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Treatment-forced logistic v2 complete.")
    print(f"[OK] Patients evaluated: {summary['n_patients_evaluated']}")
    print(f"[OK] Test points: {summary['n_test_points']}")
    print(f"[OK] Wrote: {out_summary}")


if __name__ == "__main__":
    main()
