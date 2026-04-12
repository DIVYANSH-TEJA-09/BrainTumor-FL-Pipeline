"""
Build numeric modeling tables from MU-Glioma-Post Excel sheets.

Outputs:
- data/processed/clinical_numeric_features.csv
- data/processed/longitudinal_volume_features.csv
- data/processed/longitudinal_modeling_dataset.csv
- data/processed/numeric_feature_build_report.json

This script converts mixed/categorical clinical metadata into machine-usable
numeric features and aligns segmentation-volume rows to patient/timepoint.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
CLINICAL_XLSX = BASE_DIR / "data" / "raw" / "mu_glioma_post" / "clinical" / "MU-Glioma-Post_ClinicalData-July2025.xlsx"
VOLUME_XLSX = BASE_DIR / "data" / "raw" / "mu_glioma_post" / "clinical" / "MU-Glioma-Post_Segmentation_Volumes.xlsx"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def sanitize_name(text: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(text).strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def parse_numeric_token(value: object) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s == "":
        return np.nan

    s = s.replace(",", "")
    s = re.sub(r"^q\s*", "", s, flags=re.IGNORECASE)

    if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        return float(s)

    m = re.match(r"^([-+]?\d+(?:\.\d+)?)(?:\s*[;/,].*)?$", s)
    if m:
        return float(m.group(1))

    return np.nan


def build_data_dictionary_map(dd_df: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    dd = dd_df.copy()
    dd["Data Collection Name"] = dd["Data Collection Name"].ffill()
    out: Dict[str, Dict[int, str]] = {}

    for col_name, grp in dd.groupby("Data Collection Name"):
        mapping: Dict[int, str] = {}
        for raw in grp["Data Descriptor /Metadata Name"].dropna().astype(str):
            m = re.match(r"^\s*(\d+)\s*-\s*(.+)$", raw.strip())
            if not m:
                continue
            code = int(m.group(1))
            desc = m.group(2).strip()
            mapping[code] = desc
        if mapping:
            out[str(col_name)] = mapping
    return out


def encode_clinical_numeric(clinical_df: pd.DataFrame, dd_map: Dict[str, Dict[int, str]]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = clinical_df.copy()
    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()

    # Normalize one known text binary
    if "Previous Brain Tumor" in df.columns:
        df["Previous Brain Tumor"] = (
            df["Previous Brain Tumor"].astype(str).str.strip().replace({"Yes": 1, "No": 0})
        )

    feature_df = pd.DataFrame({"patient_id": df["Patient_ID"]})

    numeric_cols: List[str] = []
    numeric_like_cols: List[str] = []
    mixed_cols: List[str] = []
    categorical_cols: List[str] = []

    object_token_flags: Dict[str, Dict[str, int]] = {}

    for col in df.columns:
        if col == "Patient_ID":
            continue

        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            new_col = sanitize_name(col)
            feature_df[new_col] = pd.to_numeric(s, errors="coerce")
            numeric_cols.append(col)
            continue

        raw = s.astype(str).str.strip()
        raw = raw.where(~s.isna(), np.nan)

        parsed = raw.map(parse_numeric_token)
        non_null = raw.notna().sum()
        numeric_rate = float(parsed.notna().sum() / non_null) if non_null else 0.0
        uniq = raw.dropna().nunique()

        if numeric_rate >= 0.95:
            new_col = sanitize_name(col)
            feature_df[new_col] = parsed
            numeric_like_cols.append(col)
            continue

        if 0.50 <= numeric_rate < 0.95:
            base = sanitize_name(col)
            feature_df[base] = parsed
            mixed_cols.append(col)

            tokens = raw.dropna().astype(str).str.strip()
            token_counts = {
                "is_ongoing": int(tokens.str.contains(r"ongoing|current", case=False, regex=True).sum()),
                "is_ltf": int(tokens.str.fullmatch(r"LTF", case=False).sum()),
                "is_h": int(tokens.str.fullmatch(r"H", case=False).sum()),
            }

            feature_df[f"{base}_is_ongoing"] = raw.astype(str).str.contains(
                r"ongoing|current", case=False, regex=True
            ).astype(float)
            feature_df.loc[s.isna(), f"{base}_is_ongoing"] = np.nan

            feature_df[f"{base}_is_ltf"] = raw.astype(str).str.fullmatch(r"LTF", case=False).astype(float)
            feature_df.loc[s.isna(), f"{base}_is_ltf"] = np.nan

            feature_df[f"{base}_is_h"] = raw.astype(str).str.fullmatch(r"H", case=False).astype(float)
            feature_df.loc[s.isna(), f"{base}_is_h"] = np.nan

            object_token_flags[col] = token_counts
            continue

        # Categorical: low cardinality one-hot
        if uniq <= 20:
            clean = raw.fillna("MISSING")
            one_hot = pd.get_dummies(clean, prefix=sanitize_name(col), dtype=float)
            feature_df = pd.concat([feature_df, one_hot], axis=1)
            categorical_cols.append(col)
        else:
            categorical_cols.append(col)

    # Add code-aware unknown flags using Data Dictionary (for coded clinical vars)
    for col, mapping in dd_map.items():
        if col not in df.columns:
            continue

        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue

        unknown_codes = [
            code
            for code, desc in mapping.items()
            if any(k in desc.lower() for k in ["unknown", "unable", "indeterminate"])
        ]
        if unknown_codes:
            feature_df[f"{sanitize_name(col)}_is_unknown"] = vals.isin(unknown_codes).astype(float)
            feature_df.loc[vals.isna(), f"{sanitize_name(col)}_is_unknown"] = np.nan

    # Missingness indicators for base numeric features
    for col in list(feature_df.columns):
        if col == "patient_id":
            continue
        if feature_df[col].dtype.kind in "biufc":
            miss_rate = feature_df[col].isna().mean()
            if miss_rate > 0.10:
                feature_df[f"{col}_is_missing"] = feature_df[col].isna().astype(float)

    report = {
        "numeric_cols": numeric_cols,
        "numeric_like_cols": numeric_like_cols,
        "mixed_cols": mixed_cols,
        "categorical_cols": categorical_cols,
        "token_flags": object_token_flags,
        "n_output_features": int(feature_df.shape[1] - 1),
        "n_patients": int(feature_df.shape[0]),
    }

    return feature_df, report


def parse_volume_patient_id(pid: str) -> Tuple[str, float]:
    s = str(pid).strip()
    m = re.match(r"^(PatientID_\d{4})(?:-Post-treatment_(\d+))?$", s)
    if not m:
        return s, np.nan
    base_pid = m.group(1)
    tp = float(m.group(2)) if m.group(2) else np.nan
    return base_pid, tp


def build_longitudinal_volume_table(volume_xlsx: Path, known_patients: pd.Index) -> Tuple[pd.DataFrame, Dict[str, object]]:
    label_map = {
        "Necrotic Tumor Core (Label1)": "label1_necrotic",
        "Tumor Infiltration and Edema": "label2_edema",
        "Enhancing Tumor Core (Label3)": "label3_enhancing",
        "Resection Cavity (Label4)": "label4_cavity",
    }

    xls = pd.ExcelFile(volume_xlsx)
    long_parts: List[pd.DataFrame] = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(volume_xlsx, sheet_name=sheet)
        if "Patient ID" not in df.columns or "Volume (mm^3)" not in df.columns:
            continue

        parsed = df["Patient ID"].astype(str).map(parse_volume_patient_id)
        df = df.copy()
        df["patient_id"] = [x[0] for x in parsed]
        df["timepoint"] = [x[1] for x in parsed]
        df = df[df["patient_id"].isin(set(known_patients))].copy()

        # For rows without explicit suffix, infer timepoint by sequence per patient within sheet
        df["_seq"] = df.groupby("patient_id").cumcount() + 1
        df.loc[df["timepoint"].isna(), "timepoint"] = df.loc[df["timepoint"].isna(), "_seq"]

        df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce")
        df = df[df["timepoint"].notna()].copy()
        df["timepoint"] = df["timepoint"].astype(int)

        tag = label_map.get(sheet, sanitize_name(sheet))
        df_out = pd.DataFrame(
            {
                "patient_id": df["patient_id"],
                "timepoint": df["timepoint"],
                f"{tag}_volume_mm3": pd.to_numeric(df["Volume (mm^3)"], errors="coerce"),
            }
        )
        long_parts.append(df_out)

    if not long_parts:
        return pd.DataFrame(columns=["patient_id", "timepoint"]), {"rows": 0}

    merged = long_parts[0]
    for part in long_parts[1:]:
        merged = merged.merge(part, on=["patient_id", "timepoint"], how="outer")

    merged = merged.sort_values(["patient_id", "timepoint"]).reset_index(drop=True)

    vol_cols = [c for c in merged.columns if c.endswith("_volume_mm3")]
    merged["tumor_total_volume_mm3"] = merged[vol_cols].sum(axis=1, min_count=1)

    if "label3_enhancing_volume_mm3" in merged.columns:
        merged["enhancing_fraction"] = (
            merged["label3_enhancing_volume_mm3"] / merged["tumor_total_volume_mm3"]
        )
    if "label2_edema_volume_mm3" in merged.columns:
        merged["edema_fraction"] = merged["label2_edema_volume_mm3"] / merged["tumor_total_volume_mm3"]
    if "label1_necrotic_volume_mm3" in merged.columns:
        merged["necrotic_fraction"] = merged["label1_necrotic_volume_mm3"] / merged["tumor_total_volume_mm3"]

    report = {
        "rows": int(len(merged)),
        "patients": int(merged["patient_id"].nunique()),
        "timepoints_min": int(merged.groupby("patient_id")["timepoint"].nunique().min())
        if len(merged)
        else 0,
        "timepoints_max": int(merged.groupby("patient_id")["timepoint"].nunique().max())
        if len(merged)
        else 0,
    }

    return merged, report


def build_mri_day_long(clinical_df: pd.DataFrame) -> pd.DataFrame:
    day_cols = {
        1: "Number of Days from Diagnosis to 1st MRI (Timepoint_1) ",
        2: "Number of Days from Diagnosis to 2nd MRI (Timepoint_2) ",
        3: "Number of Days from Diagnosis to 3rd MRI (Timepoint_3) ",
        4: "Number of Days from Diagnosis to 4th MRI (Timepoint_4) ",
        5: "Number of Days from Diagnosis to 5th MRI (Timepoint_5) ",
        6: "Number of Days from Diagnosis to 6th MRI (Timepoint_6) ",
    }

    out_rows = []
    for _, row in clinical_df.iterrows():
        pid = str(row["Patient_ID"]).strip()
        for tp, col in day_cols.items():
            if col not in clinical_df.columns:
                continue
            day = pd.to_numeric(row[col], errors="coerce")
            out_rows.append({"patient_id": pid, "timepoint": tp, "day_from_diagnosis": day})

    out = pd.DataFrame(out_rows)
    out = out.dropna(subset=["day_from_diagnosis"]).copy()
    return out


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    clinical_df = pd.read_excel(CLINICAL_XLSX, sheet_name="MU Glioma Post")
    dd_df = pd.read_excel(CLINICAL_XLSX, sheet_name="Data Dictionary")

    dd_map = build_data_dictionary_map(dd_df)
    clinical_num, clinical_report = encode_clinical_numeric(clinical_df, dd_map)

    volume_long, volume_report = build_longitudinal_volume_table(
        VOLUME_XLSX, clinical_num["patient_id"]
    )

    mri_days = build_mri_day_long(clinical_df)
    longitudinal = volume_long.merge(mri_days, on=["patient_id", "timepoint"], how="left")

    longitudinal["day_from_diagnosis_imputed"] = longitudinal["day_from_diagnosis"]
    need_impute = longitudinal["day_from_diagnosis_imputed"].isna()
    longitudinal.loc[need_impute, "day_from_diagnosis_imputed"] = (
        (longitudinal.loc[need_impute, "timepoint"] - 1) * 30
    )
    longitudinal["day_imputed_flag"] = need_impute.astype(float)

    modeling_df = longitudinal.merge(clinical_num, on="patient_id", how="left")

    out_clinical = PROCESSED_DIR / "clinical_numeric_features.csv"
    out_volume = PROCESSED_DIR / "longitudinal_volume_features.csv"
    out_modeling = PROCESSED_DIR / "longitudinal_modeling_dataset.csv"
    out_report = PROCESSED_DIR / "numeric_feature_build_report.json"

    clinical_num.to_csv(out_clinical, index=False)
    volume_long.to_csv(out_volume, index=False)
    modeling_df.to_csv(out_modeling, index=False)

    report = {
        "clinical_encoding": clinical_report,
        "volume_table": volume_report,
        "output_shapes": {
            "clinical_numeric_features": list(clinical_num.shape),
            "longitudinal_volume_features": list(volume_long.shape),
            "longitudinal_modeling_dataset": list(modeling_df.shape),
        },
        "notes": {
            "timepoint_alignment": "Rows with '-Post-treatment_n' use n directly; rows without suffix use per-patient sequence order.",
            "day_handling": "If MRI day is missing, day is imputed as (timepoint-1)*30 and flagged.",
        },
    }

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[OK] Wrote:")
    print(f"  - {out_clinical}")
    print(f"  - {out_volume}")
    print(f"  - {out_modeling}")
    print(f"  - {out_report}")
    print("[OK] Shapes:")
    print(f"  clinical_numeric_features: {clinical_num.shape}")
    print(f"  longitudinal_volume_features: {volume_long.shape}")
    print(f"  longitudinal_modeling_dataset: {modeling_df.shape}")


if __name__ == "__main__":
    main()
