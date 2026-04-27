"""
Phase 2: Convert Volume Predictions to 3D Spatial Masks
========================================================

Takes volume predictions (logistic baseline + LSTM hybrid) and converts them
to 3D spatial masks by scaling actual segmentation masks by volume ratio.

Then generates visualization data for Streamlit.
"""

import numpy as np
import nibabel as nib
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# MASK CONVERSION
# ============================================================================

class VolumeToMaskConverter:
    """Convert volume predictions to 3D spatial masks."""
    
    def __init__(self, data_dir='data/raw/mu_glioma_post', results_dir='results'):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
    
    def load_segmentation_mask(self, patient_id: str, timepoint: int) -> Optional[np.ndarray]:
        """
        Load actual segmentation mask for a patient at given timepoint.
        
        Expects: data_dir/MU-Glioma-Post/PatientID_XXXX/Timepoint_N/*_tumorMask.nii.gz
        """
        
        # Try to find the timepoint directory
        patient_dir = self.data_dir / 'MU-Glioma-Post' / patient_id
        
        if not patient_dir.exists():
            return None
        
        # List timepoint directories
        timepoint_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir() and 'Timepoint' in d.name])
        
        if timepoint >= len(timepoint_dirs):
            return None
        
        timepoint_dir = timepoint_dirs[timepoint]
        
        # Find tumorMask.nii.gz
        mask_files = list(timepoint_dir.glob('*tumorMask.nii.gz'))
        
        if not mask_files:
            return None
        
        try:
            mask_nifti = nib.load(mask_files[0])
            mask = mask_nifti.get_fdata()
            return mask
        except Exception as e:
            print(f"  Warning: Failed to load {mask_files[0]}: {e}")
            return None
    
    def volume_to_mask(
        self,
        actual_mask: np.ndarray,
        actual_volume: float,
        predicted_volume: float,
    ) -> np.ndarray:
        """
        Convert volume prediction to mask by scaling actual mask.
        
        Strategy: predicted_mask = actual_mask * (predicted_volume / actual_volume)
        
        This preserves spatial structure while scaling magnitude.
        """
        
        if actual_volume <= 0:
            return actual_mask.copy()
        
        scale_factor = predicted_volume / actual_volume
        predicted_mask = actual_mask * scale_factor
        
        return predicted_mask
    
    def generate_patient_predictions(
        self,
        patient_id: str,
        pred_df: pd.DataFrame,
    ) -> Dict:
        """
        Generate predicted masks for all timepoints of a patient.
        
        Returns:
            {
                'patient_id': str,
                'grade': str,
                'timepoints': [
                    {
                        't_idx': int,
                        'v_actual': float,
                        'v_logistic': float,
                        'v_hybrid': float,
                        'actual_mask': np.ndarray,
                        'logistic_mask': np.ndarray,
                        'hybrid_mask': np.ndarray,
                    },
                    ...
                ]
            }
        """
        
        # Get patient predictions
        patient_preds = pred_df[pred_df['patient_id'] == patient_id].sort_values('timepoint_idx')
        
        if len(patient_preds) == 0:
            return None
        
        grade = patient_preds.iloc[0]['grade']
        results = {
            'patient_id': patient_id,
            'grade': grade,
            'timepoints': [],
        }
        
        for _, row in patient_preds.iterrows():
            t_idx = int(row['timepoint_idx'])
            v_actual = float(row['v_actual'])
            v_logistic = float(row['v_logistic'])
            v_hybrid = float(row['v_hybrid'])
            
            # Load actual mask
            actual_mask = self.load_segmentation_mask(patient_id, t_idx)
            
            if actual_mask is None:
                continue
            
            # Generate predicted masks
            logistic_mask = self.volume_to_mask(actual_mask, v_actual, v_logistic)
            hybrid_mask = self.volume_to_mask(actual_mask, v_actual, v_hybrid)
            
            results['timepoints'].append({
                'timepoint_idx': t_idx,
                'v_actual': v_actual,
                'v_logistic': v_logistic,
                'v_hybrid': v_hybrid,
                'actual_mask': actual_mask.astype(np.float32),
                'logistic_mask': logistic_mask.astype(np.float32),
                'hybrid_mask': hybrid_mask.astype(np.float32),
            })
        
        return results if len(results['timepoints']) > 0 else None


# ============================================================================
# VISUALIZATION DATA GENERATION
# ============================================================================

def generate_visualization_data(
    pred_df: pd.DataFrame,
    converter: VolumeToMaskConverter,
    output_dir: Path = Path('progression/streamlit_data'),
) -> Dict:
    """
    Generate visualization data for all patients.
    
    Saves masks as NPZ files (more compact than NIfTI for this purpose).
    Returns index of available patients.
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index = {
        'hgg_patients': [],
        'lgg_patients': [],
        'timestamp': str(pd.Timestamp.now()),
    }
    
    unique_patients = pred_df['patient_id'].unique()
    
    print(f"[Phase 2] Generating visualization data for {len(unique_patients)} patients...")
    
    success_count = 0
    fail_count = 0
    
    for patient_id in unique_patients:
        try:
            # Generate masks
            patient_data = converter.generate_patient_predictions(patient_id, pred_df)
            
            if patient_data is None or len(patient_data['timepoints']) == 0:
                fail_count += 1
                continue
            
            # Save as NPZ
            output_file = output_dir / f"{patient_id}_masks.npz"
            
            # Prepare data for NPZ (convert arrays to storable format)
            npz_data = {
                'patient_id': patient_id,
                'grade': patient_data['grade'],
                'n_timepoints': len(patient_data['timepoints']),
            }
            
            for t_idx, tp in enumerate(patient_data['timepoints']):
                npz_data[f't{t_idx:02d}_actual_mask'] = tp['actual_mask']
                npz_data[f't{t_idx:02d}_logistic_mask'] = tp['logistic_mask']
                npz_data[f't{t_idx:02d}_hybrid_mask'] = tp['hybrid_mask']
                npz_data[f't{t_idx:02d}_v_actual'] = np.array([tp['v_actual']])
                npz_data[f't{t_idx:02d}_v_logistic'] = np.array([tp['v_logistic']])
                npz_data[f't{t_idx:02d}_v_hybrid'] = np.array([tp['v_hybrid']])
            
            np.savez_compressed(output_file, **npz_data)
            
            # Add to index
            grade = patient_data['grade']
            if grade == 'HGG':
                index['hgg_patients'].append({
                    'patient_id': patient_id,
                    'n_timepoints': len(patient_data['timepoints']),
                    'file': str(output_file.name),
                })
            elif grade == 'LGG':
                index['lgg_patients'].append({
                    'patient_id': patient_id,
                    'n_timepoints': len(patient_data['timepoints']),
                    'file': str(output_file.name),
                })
            
            success_count += 1
            
            if success_count % 20 == 0:
                print(f"  Processed {success_count} patients...")
        
        except Exception as e:
            fail_count += 1
            print(f"  Warning: {patient_id} failed: {e}")
    
    # Save index
    index_file = output_dir / 'index.json'
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"  [OK] Generated data for {success_count} patients")
    print(f"       Output: {output_dir}")
    print(f"       Failed: {fail_count} patients (missing masks)")
    print(f"       HGG: {len(index['hgg_patients'])}, LGG: {len(index['lgg_patients'])}")
    
    return index


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("PHASE 2: VOLUME PREDICTIONS -> 3D MASKS")
    print("="*70 + "\n")
    
    # Load predictions
    print("[Step 1/3] Loading predictions...")
    pred_df = pd.read_csv('results/phase2_hybrid_predictions.csv')
    print(f"  Loaded {len(pred_df)} predictions for {pred_df['patient_id'].nunique()} patients")
    print()
    
    # Initialize converter
    print("[Step 2/3] Initializing mask converter...")
    converter = VolumeToMaskConverter()
    print("  Ready to convert predictions to 3D masks")
    print()
    
    # Generate visualization data
    print("[Step 3/3] Generating visualization data...")
    index = generate_visualization_data(pred_df, converter)
    
    print("\n" + "="*70)
    print("VISUALIZATION DATA READY")
    print("="*70)
    print(f"\nNext step: Run Streamlit app to visualize 3D predictions")
    print(f"  cd progression")
    print(f"  streamlit run streamlit_3d_progression.py")
    print("\n" + "="*70 + "\n")
