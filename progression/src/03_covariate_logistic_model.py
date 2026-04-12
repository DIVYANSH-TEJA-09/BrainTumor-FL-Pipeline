"""
Phase 1 (real-data extension): covariate-augmented logistic modeling.

What this script does:
1) Fit per-patient logistic curves on real longitudinal tumor volumes.
2) Build patient-level covariates from encoded clinical + baseline imaging features.
3) Compare two forecasting strategies with patient-level CV:
   - Grade-median logistic parameters (basic baseline)
   - Covariate-augmented logistic parameters (ElasticNet)
4) Export LGG/HGG metrics and covariate effect tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "longitudinal_modeling_dataset.csv"
RESULTS_DIR = BASE_DIR / "results"


def logistic_with_v0(t: np.ndarray, k: float, r: float, v0: float) -> np.ndarray:
    # k>0, r>0. If k<v0, trajectory can decline toward k.
    return k / (1.0 + ((k - v0) / v0) * np.exp(-r * t))


def safe_float(v: object, default: float = np.nan) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
        return default
    except Exception:
        return default


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "n_obs": int(len(y_true))}


def grade_from_numeric(g: float) -> str:
    if np.isnan(g):
        return "UNKNOWN"
    if g in (3.0, 4.0):
        return "HGG"
    if g == 2.0:
        return "LGG"
    return "UNKNOWN"


@dataclass
class PatientFit:
    patient_id: str
    grade: str
    n_points: int
    v0: float
    k_fit: float
    r_fit: float
    mae_fit: float
    rmse_fit: float
    r2_fit: float
    success: int


def fit_patient_logistic(t: np.ndarray, v: np.ndarray) -> Tuple[float, float, np.ndarray, bool]:
    v0 = float(v[0])
    vmax = float(np.max(v))
    vmin = float(np.min(v))

    # Initialize K so both growth and decay cases are feasible.
    if v[-1] >= v0:
        k0 = max(vmax * 1.5, v0 * 1.1)
    else:
        k0 = max(vmin * 0.8, 1.0)

    r0 = 0.01

    low = [1.0, 1e-6]
    high = [max(vmax * 50.0, 1e6), 0.2]

    try:
        popt, _ = curve_fit(
            lambda tt, kk, rr: logistic_with_v0(tt, kk, rr, v0),
            t,
            v,
            p0=[k0, r0],
            bounds=(low, high),
            maxfev=40000,
        )
        pred = logistic_with_v0(t, popt[0], popt[1], v0)
        return float(popt[0]), float(popt[1]), pred, True
    except Exception:
        # Fallback: near-static parameter guess
        k_fallback = max(v[-1], 1.0)
        r_fallback = 0.002
        pred = logistic_with_v0(t, k_fallback, r_fallback, v0)
        return float(k_fallback), float(r_fallback), pred, False


def build_patient_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    # Keep only rows with target and time.
    work = df.copy()
    work["tumor_total_volume_mm3"] = pd.to_numeric(work["tumor_total_volume_mm3"], errors="coerce")
    work["day_from_diagnosis_imputed"] = pd.to_numeric(work["day_from_diagnosis_imputed"], errors="coerce")
    work = work.dropna(subset=["tumor_total_volume_mm3", "day_from_diagnosis_imputed"])

    fits: List[PatientFit] = []
    traj: Dict[str, Dict[str, np.ndarray]] = {}

    for pid, g in work.groupby("patient_id"):
        g = g.sort_values(["day_from_diagnosis_imputed", "timepoint"])
        v = g["tumor_total_volume_mm3"].values.astype(float)
        day = g["day_from_diagnosis_imputed"].values.astype(float)

        if len(v) < 3:
            continue

        t = day - day[0]
        if np.allclose(t, 0):
            # Last fallback if all days collapsed.
            t = np.arange(len(v), dtype=float) * 30.0

        if np.any(v <= 0):
            continue

        grade_num = safe_float(g["grade_of_primary_brain_tumor"].iloc[0])
        grade = grade_from_numeric(grade_num)

        k_fit, r_fit, pred, ok = fit_patient_logistic(t, v)
        m = compute_metrics(v, pred)

        fits.append(
            PatientFit(
                patient_id=str(pid),
                grade=grade,
                n_points=int(len(v)),
                v0=float(v[0]),
                k_fit=float(k_fit),
                r_fit=float(r_fit),
                mae_fit=m["mae"],
                rmse_fit=m["rmse"],
                r2_fit=m["r2"],
                success=int(ok),
            )
        )

        traj[str(pid)] = {
            "t": t,
            "v": v,
            "v0": float(v[0]),
            "grade": np.array([grade]),
        }

    fit_df = pd.DataFrame([f.__dict__ for f in fits])

    # Build baseline per-patient covariates from first row.
    baseline = (
        work.sort_values(["patient_id", "day_from_diagnosis_imputed", "timepoint"]).groupby("patient_id").head(1)
    )

    # Choose clinically meaningful, mostly-available predictors.
    needed = [
        "patient_id",
        "age_at_diagnosis",
        "sex_at_birth_Male",
        "grade_of_primary_brain_tumor",
        "idh1_mutation",
        "idh1_mutation_is_unknown",
        "idh2_mutation",
        "idh2_mutation_is_unknown",
        "mgmt_methylation",
        "mgmt_methylation_is_unknown",
        "atrx_mutation",
        "atrx_mutation_is_unknown",
        "egfr_amplification",
        "egfr_amplification_is_unknown",
        "pten_mutation",
        "cdkn2a_b_deletion",
        "tp53_alteration",
        "previous_brain_tumor",
        "initial_chemo_therapy_Yes",
        "radiation_therapy_Yes",
        "enhancing_fraction",
        "edema_fraction",
        "necrotic_fraction",
    ]

    for col in needed:
        if col not in baseline.columns:
            baseline[col] = np.nan

    cov = baseline[needed].copy()

    # Derive cleaned binary features from coded biomarkers.
    cov["is_hgg"] = pd.to_numeric(cov["grade_of_primary_brain_tumor"], errors="coerce").isin([3.0, 4.0]).astype(float)
    cov["idh1_present"] = (pd.to_numeric(cov["idh1_mutation"], errors="coerce") == 1).astype(float)
    cov["idh2_present"] = (pd.to_numeric(cov["idh2_mutation"], errors="coerce") == 1).astype(float)
    cov["mgmt_present"] = (pd.to_numeric(cov["mgmt_methylation"], errors="coerce") == 1).astype(float)

    atrx_num = pd.to_numeric(cov["atrx_mutation"], errors="coerce")
    cov["atrx_present"] = atrx_num.isin([1.0, 2.0, 3.0]).astype(float)
    cov["egfr_present"] = (pd.to_numeric(cov["egfr_amplification"], errors="coerce") == 1).astype(float)

    cov["pten_present"] = (pd.to_numeric(cov["pten_mutation"], errors="coerce") == 1).astype(float)
    cov["cdkn2a_b_present"] = (pd.to_numeric(cov["cdkn2a_b_deletion"], errors="coerce") == 1).astype(float)
    cov["tp53_present"] = (pd.to_numeric(cov["tp53_alteration"], errors="coerce") == 1).astype(float)

    # Keep final model features.
    final_cols = [
        "patient_id",
        "age_at_diagnosis",
        "sex_at_birth_Male",
        "is_hgg",
        "idh1_present",
        "idh1_mutation_is_unknown",
        "idh2_present",
        "idh2_mutation_is_unknown",
        "mgmt_present",
        "mgmt_methylation_is_unknown",
        "atrx_present",
        "atrx_mutation_is_unknown",
        "egfr_present",
        "egfr_amplification_is_unknown",
        "pten_present",
        "cdkn2a_b_present",
        "tp53_present",
        "previous_brain_tumor",
        "initial_chemo_therapy_Yes",
        "radiation_therapy_Yes",
        "enhancing_fraction",
        "edema_fraction",
        "necrotic_fraction",
    ]

    cov = cov[final_cols].copy()

    patient_df = fit_df.merge(cov, on="patient_id", how="left")
    return patient_df, traj


def evaluate_predictions(patient_subset: pd.DataFrame, traj: Dict[str, Dict[str, np.ndarray]], pred_k: np.ndarray, pred_r: np.ndarray) -> Dict[str, object]:
    rows = []
    for i, (_, row) in enumerate(patient_subset.iterrows()):
        pid = row["patient_id"]
        if pid not in traj:
            continue
        tt = traj[pid]["t"]
        yy = traj[pid]["v"]
        v0 = traj[pid]["v0"]
        grade = str(traj[pid]["grade"][0])

        kk = float(np.clip(pred_k[i], 1.0, 1e7))
        rr = float(np.clip(pred_r[i], 1e-6, 0.2))
        yhat = logistic_with_v0(tt, kk, rr, v0)

        for y_true, y_pred in zip(yy, yhat):
            rows.append({"grade": grade, "y_true": float(y_true), "y_pred": float(y_pred)})

    out = pd.DataFrame(rows)
    if out.empty:
        return {"overall": {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "n_obs": 0}, "by_grade": {}}

    overall = compute_metrics(out["y_true"].values, out["y_pred"].values)
    by_grade = {}
    for gr, g in out.groupby("grade"):
        by_grade[gr] = compute_metrics(g["y_true"].values, g["y_pred"].values)

    return {"overall": overall, "by_grade": by_grade}


def fit_covariate_models(x_train: pd.DataFrame, yk_train: np.ndarray, yr_train: np.ndarray) -> Tuple[Pipeline, Pipeline]:
    # ElasticNet keeps this first pass sparse and interpretable.
    cv_folds = min(5, max(3, len(x_train) // 8))

    model_k = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "elasticnetcv",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9, 1.0],
                    alphas=np.logspace(-4, 1, 60),
                    cv=cv_folds,
                    random_state=42,
                    max_iter=20000,
                ),
            ),
        ]
    )

    model_r = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "elasticnetcv",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9, 1.0],
                    alphas=np.logspace(-4, 1, 60),
                    cv=cv_folds,
                    random_state=42,
                    max_iter=20000,
                ),
            ),
        ]
    )

    model_k.fit(x_train, yk_train)
    model_r.fit(x_train, yr_train)
    return model_k, model_r


def extract_effects(model: Pipeline, feature_names: List[str], target_name: str) -> pd.DataFrame:
    est = model.named_steps["elasticnetcv"]
    coef = est.coef_
    out = pd.DataFrame({"feature": feature_names, "coef_std": coef})
    out["effect_percent_per_1sd"] = (np.exp(out["coef_std"]) - 1.0) * 100.0
    out["target"] = target_name
    out["abs_coef"] = out["coef_std"].abs()
    out = out.sort_values("abs_coef", ascending=False)
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    patient_df, traj = build_patient_tables(df)

    # Keep patients with successful fit and known grade (LGG/HGG) for stratified summary.
    model_df = patient_df.copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df[model_df["success"] >= 0]

    # Features used for covariate-augmented parameter prediction.
    feature_cols = [
        c
        for c in model_df.columns
        if c
        not in {
            "patient_id",
            "grade",
            "n_points",
            "v0",
            "k_fit",
            "r_fit",
            "mae_fit",
            "rmse_fit",
            "r2_fit",
            "success",
        }
    ]

    # Targets on log-scale.
    yk = np.log(np.clip(model_df["k_fit"].values.astype(float), 1.0, 1e9))
    yr = np.log(np.clip(model_df["r_fit"].values.astype(float), 1e-6, 1.0))

    x = model_df[feature_cols]

    # Cross-validated comparison: grade-median vs covariate model.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(model_df), start=1):
        train_df = model_df.iloc[tr_idx].copy()
        test_df = model_df.iloc[te_idx].copy()

        # Baseline: grade-wise median parameters from training set.
        grade_med = train_df.groupby("grade")[["k_fit", "r_fit"]].median()
        glob_med_k = float(train_df["k_fit"].median())
        glob_med_r = float(train_df["r_fit"].median())

        base_k = []
        base_r = []
        for _, row in test_df.iterrows():
            gr = row["grade"]
            if gr in grade_med.index:
                base_k.append(float(grade_med.loc[gr, "k_fit"]))
                base_r.append(float(grade_med.loc[gr, "r_fit"]))
            else:
                base_k.append(glob_med_k)
                base_r.append(glob_med_r)

        base_eval = evaluate_predictions(test_df, traj, np.array(base_k), np.array(base_r))

        # Covariate model.
        model_k, model_r = fit_covariate_models(
            x.iloc[tr_idx],
            yk[tr_idx],
            yr[tr_idx],
        )

        pred_logk = model_k.predict(x.iloc[te_idx])
        pred_logr = model_r.predict(x.iloc[te_idx])
        pred_k = np.exp(np.clip(pred_logk, np.log(1.0), np.log(1e7)))
        pred_r = np.exp(np.clip(pred_logr, np.log(1e-6), np.log(0.2)))

        cov_eval = evaluate_predictions(test_df, traj, pred_k, pred_r)

        cv_rows.append(
            {
                "fold": fold,
                "model": "grade_median",
                **{f"overall_{k}": v for k, v in base_eval["overall"].items()},
            }
        )
        cv_rows.append(
            {
                "fold": fold,
                "model": "covariate_logistic",
                **{f"overall_{k}": v for k, v in cov_eval["overall"].items()},
            }
        )

        # Store grade-wise metrics per fold.
        for mname, ev in [("grade_median", base_eval), ("covariate_logistic", cov_eval)]:
            for gr, gm in ev["by_grade"].items():
                cv_rows.append(
                    {
                        "fold": fold,
                        "model": mname,
                        "grade": gr,
                        "mae": gm["mae"],
                        "rmse": gm["rmse"],
                        "r2": gm["r2"],
                        "n_obs": gm["n_obs"],
                        "type": "by_grade",
                    }
                )

    cv_df = pd.DataFrame(cv_rows)

    # Fit final covariate models on full patient set for effect interpretation.
    final_k, final_r = fit_covariate_models(x, yk, yr)
    effects_k = extract_effects(final_k, feature_cols, "log_k")
    effects_r = extract_effects(final_r, feature_cols, "log_r")

    # Grade-stratified fit quality from per-patient direct logistic fits.
    grade_summary = (
        model_df.groupby("grade")
        .agg(
            patients=("patient_id", "nunique"),
            mean_points=("n_points", "mean"),
            fit_r2_mean=("r2_fit", "mean"),
            fit_r2_std=("r2_fit", "std"),
            fit_mae_mean=("mae_fit", "mean"),
            fit_rmse_mean=("rmse_fit", "mean"),
            r_fit_mean=("r_fit", "mean"),
            k_fit_mean=("k_fit", "mean"),
        )
        .reset_index()
    )

    # Summarize CV results.
    cv_overall = (
        cv_df[cv_df["model"].isin(["grade_median", "covariate_logistic"]) & cv_df["overall_rmse"].notna()]
        .groupby("model")
        .agg(
            rmse_mean=("overall_rmse", "mean"),
            rmse_std=("overall_rmse", "std"),
            mae_mean=("overall_mae", "mean"),
            mae_std=("overall_mae", "std"),
            r2_mean=("overall_r2", "mean"),
            r2_std=("overall_r2", "std"),
            n_obs_mean=("overall_n_obs", "mean"),
        )
        .reset_index()
    )

    by_grade_cv = cv_df[cv_df.get("type", "") == "by_grade"].copy()
    if not by_grade_cv.empty:
        by_grade_summary = (
            by_grade_cv.groupby(["model", "grade"]) 
            .agg(
                rmse_mean=("rmse", "mean"),
                mae_mean=("mae", "mean"),
                r2_mean=("r2", "mean"),
                n_obs_mean=("n_obs", "mean"),
            )
            .reset_index()
        )
    else:
        by_grade_summary = pd.DataFrame(columns=["model", "grade", "rmse_mean", "mae_mean", "r2_mean", "n_obs_mean"])

    # Persist outputs.
    out_patient = RESULTS_DIR / "phase1_patient_logistic_parameters.csv"
    out_cv = RESULTS_DIR / "phase1_covariate_logistic_cv_metrics.csv"
    out_cv_overall = RESULTS_DIR / "phase1_covariate_logistic_cv_overall_summary.csv"
    out_cv_grade = RESULTS_DIR / "phase1_covariate_logistic_cv_grade_summary.csv"
    out_grade_fit = RESULTS_DIR / "phase1_logistic_grade_fit_summary.csv"
    out_eff_k = RESULTS_DIR / "phase1_covariate_effects_log_k.csv"
    out_eff_r = RESULTS_DIR / "phase1_covariate_effects_log_r.csv"
    out_json = RESULTS_DIR / "phase1_covariate_logistic_summary.json"

    model_df.to_csv(out_patient, index=False)
    cv_df.to_csv(out_cv, index=False)
    cv_overall.to_csv(out_cv_overall, index=False)
    by_grade_summary.to_csv(out_cv_grade, index=False)
    grade_summary.to_csv(out_grade_fit, index=False)
    effects_k.to_csv(out_eff_k, index=False)
    effects_r.to_csv(out_eff_r, index=False)

    summary = {
        "n_patients_modeled": int(model_df["patient_id"].nunique()),
        "n_rows_modeled": int(sum(len(traj[pid]["v"]) for pid in model_df["patient_id"] if pid in traj)),
        "grade_fit_summary": grade_summary.to_dict(orient="records"),
        "cv_overall_summary": cv_overall.to_dict(orient="records"),
        "cv_grade_summary": by_grade_summary.to_dict(orient="records"),
        "top_covariate_effects_log_r": effects_r.head(12)[
            ["feature", "coef_std", "effect_percent_per_1sd"]
        ].to_dict(orient="records"),
        "top_covariate_effects_log_k": effects_k.head(12)[
            ["feature", "coef_std", "effect_percent_per_1sd"]
        ].to_dict(orient="records"),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Covariate logistic modeling complete.")
    print(f"[OK] Patients modeled: {model_df['patient_id'].nunique()}")
    print(f"[OK] Wrote: {out_json}")


if __name__ == "__main__":
    main()
