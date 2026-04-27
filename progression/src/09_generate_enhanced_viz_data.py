"""
Phase 3: Generate Enhanced Streamlit Visualization Data
=======================================================

Creates a richer prediction index for the new Streamlit app.
Includes:
  - Real scan days (from clinical data, not hardcoded 30-day assumption)
  - Per-patient logistic model parameters (V0, K, r)
  - Residuals per timepoint
  - Overall model metrics
  - Grade-level aggregated stats
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path


def generate_enhanced_index(
    pred_csv: Path,
    patient_params_csv: Path,
    longitudinal_csv: Path,
    residuals_json: Path,
    metrics_json: Path,
    output_dir: Path,
) -> dict:
    """
    Build enhanced prediction index for the new Streamlit UI.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data sources
    pred_df = pd.read_csv(pred_csv)
    patient_params = pd.read_csv(patient_params_csv)
    longitudinal = pd.read_csv(longitudinal_csv)
    with open(residuals_json) as f:
        residuals_by_patient = json.load(f)
    with open(metrics_json) as f:
        eval_metrics = json.load(f)

    # Build real-day lookup: patient_id -> sorted list of (timepoint_order, day)
    day_lookup = {}
    for patient_id, group in longitudinal.groupby('patient_id'):
        group = group.sort_values('timepoint').reset_index(drop=True)
        days = group['day_from_diagnosis_imputed'].values.tolist()
        day_lookup[patient_id] = days

    # Build param lookup
    param_lookup = {}
    for _, row in patient_params.iterrows():
        param_lookup[row['patient_id']] = {
            'v0': float(row['v0']),
            'k_fit': float(row['k_fit']),
            'r_fit': float(row['r_fit']),
            'r2_fit': float(row['r2_fit']),
            'mae_fit': float(row['mae_fit']),
        }

    # Build index
    patient_groups = pred_df.groupby('patient_id')

    index = {
        'total_patients': len(patient_groups),
        'eval_metrics': eval_metrics,
        'patients': {},
    }

    for patient_id, group in patient_groups:
        grade = group.iloc[0]['grade']
        group = group.sort_values('timepoint_idx').reset_index(drop=True)

        # Get real days for this patient
        real_days = day_lookup.get(patient_id, [])

        # Get logistic parameters
        params = param_lookup.get(patient_id, {})

        # Get residuals
        resid_data = residuals_by_patient.get(patient_id, {})
        residuals = resid_data.get('residuals', [])

        timepoints = []
        for i, (_, row) in enumerate(group.iterrows()):
            tp = {
                'timepoint_idx': int(row['timepoint_idx']),
                'v_actual': float(row['v_actual']),
                'v_logistic': float(row['v_logistic']),
                'v_hybrid': float(row['v_hybrid']),
                'mae_baseline': float(row['mae_baseline']),
                'mae_hybrid': float(row['mae_hybrid']),
                'lstm_correction': float(row.get('lstm_correction', 0)),
                'day_from_diagnosis': float(real_days[i]) if i < len(real_days) else i * 30,
                'residual': float(residuals[i]) if i < len(residuals) else 0.0,
            }
            timepoints.append(tp)

        index['patients'][patient_id] = {
            'grade': grade,
            'n_timepoints': len(timepoints),
            'timepoints': timepoints,
            'mae_baseline_mean': float(group['mae_baseline'].mean()),
            'mae_hybrid_mean': float(group['mae_hybrid'].mean()),
            'logistic_params': params,
        }

    # Save
    index_file = output_dir / 'prediction_index.json'
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)

    csv_file = output_dir / 'all_predictions.csv'
    pred_df.to_csv(csv_file, index=False)

    n_hgg = sum(1 for p in index['patients'].values() if p['grade'] == 'HGG')
    n_lgg = sum(1 for p in index['patients'].values() if p['grade'] == 'LGG')
    print(f"[OK] Enhanced index: {len(index['patients'])} patients (HGG={n_hgg}, LGG={n_lgg})")
    print(f"     Saved: {index_file}")
    print(f"     Saved: {csv_file}")
    return index


if __name__ == '__main__':
    print("\n" + "="*70)
    print("GENERATE ENHANCED VISUALIZATION DATA")
    print("="*70 + "\n")

    base = Path(__file__).parent.parent

    index = generate_enhanced_index(
        pred_csv=base / 'results' / 'phase2_hybrid_predictions.csv',
        patient_params_csv=base / 'results' / 'phase1_patient_logistic_parameters.csv',
        longitudinal_csv=base / 'data' / 'processed' / 'longitudinal_modeling_dataset.csv',
        residuals_json=base / 'results' / 'phase2_residuals_by_patient.json',
        metrics_json=base / 'results' / 'phase2_evaluation_metrics.json',
        output_dir=base / 'streamlit_data',
    )

    print("\n[OK] Done. Ready for Streamlit app.\n")
