"""
Phase 2: Generate Streamlit Visualization Data
==============================================

Creates lightweight prediction data for Streamlit visualization.
Instead of loading all NIfTI masks, we store:
  - Volume predictions (logistic + hybrid) 
  - Metadata for each patient-timepoint
  
The Streamlit app will load the actual masks on-demand and scale them.
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np


def generate_visualization_index(pred_df: pd.DataFrame, output_dir: Path = Path('progression/streamlit_data')) -> dict:
    """
    Generate lightweight index for Streamlit visualization.
    
    Stores:
      - Per-patient, per-timepoint volume predictions
      - Metadata (grade, patient_id, timepoint_idx)
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by patient
    patient_groups = pred_df.groupby('patient_id')
    
    index = {
        'total_patients': len(patient_groups),
        'patients': {},
        'timestamp': str(pd.Timestamp.now()),
    }
    
    print(f"[Phase 2] Generating Streamlit visualization index...")
    print(f"  Total patients: {len(patient_groups)}")
    
    hgg_count = 0
    lgg_count = 0
    
    for patient_id, group in patient_groups:
        grade = group.iloc[0]['grade']
        
        # Sort by timepoint
        group = group.sort_values('timepoint_idx').reset_index(drop=True)
        
        timepoints = []
        for _, row in group.iterrows():
            timepoints.append({
                'timepoint_idx': int(row['timepoint_idx']),
                'v_actual': float(row['v_actual']),
                'v_logistic': float(row['v_logistic']),
                'v_hybrid': float(row['v_hybrid']),
                'mae_baseline': float(row['mae_baseline']),
                'mae_hybrid': float(row['mae_hybrid']),
            })
        
        index['patients'][patient_id] = {
            'grade': grade,
            'n_timepoints': len(timepoints),
            'timepoints': timepoints,
            'mae_baseline_mean': float(group['mae_baseline'].mean()),
            'mae_hybrid_mean': float(group['mae_hybrid'].mean()),
        }
        
        if grade == 'HGG':
            hgg_count += 1
        else:
            lgg_count += 1
    
    # Save index
    index_file = output_dir / 'prediction_index.json'
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"  [OK] HGG: {hgg_count} patients")
    print(f"       LGG: {lgg_count} patients")
    print(f"  Saved: {index_file}")
    
    # Save predictions CSV in streamlit_data for easy access
    csv_file = output_dir / 'all_predictions.csv'
    pred_df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file}")
    
    return index


if __name__ == '__main__':
    print("\n" + "="*70)
    print("PHASE 2: GENERATE STREAMLIT VISUALIZATION DATA")
    print("="*70 + "\n")
    
    # Load predictions
    print("[Step 1/2] Loading predictions...")
    pred_df = pd.read_csv('results/phase2_hybrid_predictions.csv')
    print(f"  Loaded {len(pred_df)} predictions")
    print()
    
    # Generate index
    print("[Step 2/2] Generating visualization index...")
    index = generate_visualization_index(pred_df)
    
    print("\n" + "="*70)
    print("VISUALIZATION DATA READY")
    print("="*70)
    print(f"\nFiles generated:")
    print(f"  progression/streamlit_data/prediction_index.json")
    print(f"  progression/streamlit_data/all_predictions.csv")
    print(f"\nNext: Create Streamlit page to visualize predictions")
    print("="*70 + "\n")
