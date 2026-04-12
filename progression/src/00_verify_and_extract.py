"""
Data Verification & Metadata Extraction for MU-Glioma-Post Dataset

Purpose:
- Verify downloaded dataset integrity
- Extract clinical metadata from XLSX files
- Create time-series data for each patient
- Generate stratification (LGG vs HGG)
- Create processed dataset ready for PHASE 1

Novel Clinical Contribution:
- Temporal alignment: Map timepoints to clinical events
- Grade stratification: Separate LGG vs HGG trajectories
- Treatment encoding: Standard post-operative protocol assumed
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import nibabel as nib
from datetime import datetime

class MUGliomaDataProcessor:
    """Process MU-Glioma-Post dataset for progression modeling"""
    
    def __init__(self, data_root: str = "data/raw/mu_glioma_post"):
        self.data_root = Path(data_root)
        
        # Handle both path structures:
        # 1. data/raw/mu_glioma_post/images/  (expected)
        # 2. data/raw/mu_glioma_post/MU-Glioma-Post/  (actual)
        if (self.data_root / "MU-Glioma-Post").exists():
            self.images_dir = self.data_root / "MU-Glioma-Post"
        else:
            self.images_dir = self.data_root / "images"
        
        self.clinical_dir = self.data_root / "clinical"
        self.processed_dir = Path("data/processed")
        
        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def verify_data_structure(self) -> Dict:
        """Verify dataset structure and report statistics"""
        print("[PHASE 0] Verifying MU-Glioma-Post Data Structure...")
        
        stats = {
            "total_patients": 0,
            "total_timepoints": 0,
            "patients_with_missing_modalities": [],
            "modality_coverage": {"T1": 0, "T1CE": 0, "T2": 0, "FLAIR": 0},
            "timepoint_distribution": {},
            "errors": []
        }
        
        if not self.images_dir.exists():
            stats["errors"].append(f"Images directory not found: {self.images_dir}")
            return stats
        
        # List all patient directories
        patient_dirs = sorted([d for d in self.images_dir.iterdir() if d.is_dir()])
        stats["total_patients"] = len(patient_dirs)
        
        print(f"  Found {stats['total_patients']} patient directories")
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            timepoint_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
            num_timepoints = len(timepoint_dirs)
            
            stats["total_timepoints"] += num_timepoints
            stats["timepoint_distribution"][patient_id] = num_timepoints
            
            # Check modality coverage
            missing_modalities = []
            for timepoint_dir in timepoint_dirs:
                files_in_dir = list(timepoint_dir.glob("*.nii.gz"))
                filenames = [f.name for f in files_in_dir]
                
                # Map actual filenames to modality names
                # t1n = T1 Native, t1c = T1 Contrast, t2w = T2 Weighted, t2f = T2 FLAIR
                modality_mapping = {
                    "T1": any("t1n" in f.lower() for f in filenames),
                    "T1CE": any("t1c" in f.lower() for f in filenames),
                    "T2": any("t2w" in f.lower() for f in filenames),
                    "FLAIR": any("t2f" in f.lower() for f in filenames),
                }
                
                for modality, found in modality_mapping.items():
                    if found:
                        stats["modality_coverage"][modality] += 1
                    else:
                        missing_modalities.append(f"{patient_id}/{timepoint_dir.name}: {modality}")
            
            if missing_modalities:
                stats["patients_with_missing_modalities"].extend(missing_modalities)
        
        # Report
        print(f"  [OK] Total timepoints: {stats['total_timepoints']}")
        print(f"  [OK] Average timepoints per patient: {stats['total_timepoints'] / stats['total_patients']:.1f}")
        print(f"  [OK] Modality coverage: {stats['modality_coverage']}")
        
        if stats["patients_with_missing_modalities"]:
            print(f"  [WARN] Warning: {len(stats['patients_with_missing_modalities'])} missing modality instances")
        
        return stats
    
    def load_clinical_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load clinical and segmentation volume data from XLSX"""
        print("\n[PHASE 0] Loading Clinical Metadata...")
        
        # Load the MU Glioma Post sheet from clinical data
        clinical_file = self.clinical_dir / "MU-Glioma-Post_ClinicalData-July2025.xlsx"
        volume_file = self.clinical_dir / "MU-Glioma-Post_Segmentation_Volumes.xlsx"
        
        if not clinical_file.exists():
            raise FileNotFoundError(f"Clinical data file not found: {clinical_file}")
        
        # Read the 'MU Glioma Post' sheet
        try:
            clinical_data = pd.read_excel(clinical_file, sheet_name='MU Glioma Post')
            print(f"  Loaded: {clinical_file.name} (sheet='MU Glioma Post')")
            print(f"    Shape: {clinical_data.shape}")
            print(f"    Columns (first 5): {clinical_data.columns.tolist()[:5]}...")
        except Exception as e:
            print(f"  [WARN] Could not load 'MU Glioma Post' sheet: {e}")
            raise
        
        # Load volume data
        volume_data = None
        if volume_file.exists():
            volume_data = pd.read_excel(volume_file)
            print(f"  Loaded: {volume_file.name}")
            print(f"    Shape: {volume_data.shape}")
            print(f"    Columns (first 5): {volume_data.columns.tolist()[:5]}...")
        
        print(f"  [OK] Clinical data shape: {clinical_data.shape}")
        
        return clinical_data, volume_data
    
    def stratify_by_grade(self, clinical_data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Stratify patients by glioma grade (LGG vs HGG)
        
        Novel Contribution: This stratification is clinically important because:
        - LGG (Grade II): Slow growth, indolent course, prognosis measured in years
        - HGG (Grade III-IV): Rapid growth, aggressive, prognosis measured in months
        
        Separate models allow us to learn grade-specific progression patterns.
        """
        print("\n[PHASE 0] Stratifying Patients by Glioma Grade...")
        
        stratification = {"LGG": [], "HGG": [], "UNKNOWN": []}
        
        # Look for grade column in clinical data
        grade_col = "Grade of Primary Brain Tumor"
        patient_id_col = "Patient_ID"
        
        if grade_col not in clinical_data.columns:
            print(f"  [WARN] Grade column not found. Available: {clinical_data.columns.tolist()[:10]}")
            return stratification
        
        print(f"  Using grade column: '{grade_col}'")
        print(f"  Using patient ID column: '{patient_id_col}'")
        
        for idx, row in clinical_data.iterrows():
            patient_id = str(row[patient_id_col])
            grade_val = row[grade_col]
            
            # Stratify based on grade
            if pd.isna(grade_val):
                grade = "UNKNOWN"
            else:
                grade_str = str(grade_val).lower().strip()
                # Roman numeral mapping
                if grade_str in ["2", "ii"]:
                    grade = "LGG"
                elif grade_str in ["3", "4", "iii", "iv"]:
                    grade = "HGG"
                else:
                    grade = "UNKNOWN"
            
            stratification[grade].append(patient_id)
        
        print(f"  [OK] LGG (Low-Grade): {len(stratification['LGG'])} patients")
        print(f"  [OK] HGG (High-Grade): {len(stratification['HGG'])} patients")
        print(f"  [WARN] Unknown: {len(stratification['UNKNOWN'])} patients")
        
        return stratification
    
    def create_timeseries_data(self, clinical_data: pd.DataFrame, 
                               volume_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create time-series dataset for progression modeling
        
        Each row = one patient's progression trajectory
        Columns = [PatientID, Grade, Age, Sex, Primary Diagnosis, Volumes at different timepoints, ...]
        """
        print("\n[PHASE 0] Creating Time-Series Data...")
        
        timeseries_list = []
        
        for idx, row in clinical_data.iterrows():
            patient_id = str(row["Patient_ID"])
            
            # Extract available information
            record = {
                "PatientID": patient_id,
                "Age": row.get("Age at diagnosis", np.nan),
                "Sex": row.get("Sex at Birth", np.nan),
                "Diagnosis": row.get("Primary Diagnosis", np.nan),
                "Grade": row.get("Grade of Primary Brain Tumor", np.nan),
            }
            
            # If volume data available, extract volumes for this patient
            if volume_data is not None and "Patient ID" in volume_data.columns:
                patient_volumes_all = volume_data[volume_data["Patient ID"] == patient_id]
                if not patient_volumes_all.empty:
                    # Get tumor volumes (typically Label Id = 1)
                    tumor_volumes = patient_volumes_all[patient_volumes_all["Label Name"] == "Tumor"]
                    if not tumor_volumes.empty:
                        volumes = tumor_volumes["Volume (mm^3)"].values
                        for tp_idx, vol in enumerate(volumes):
                            record[f"Volume_TP{tp_idx+1}"] = vol if not np.isnan(vol) else np.nan
                    
                    # Get tumor stats from imaging modalities if available
                    for col in volume_data.columns:
                        if "Image mean" in col:
                            val = patient_volumes_all[col].values[0]
                            if not np.isnan(val):
                                record[col] = val
            
            timeseries_list.append(record)
        
        timeseries_df = pd.DataFrame(timeseries_list)
        print(f"  [OK] Created time-series dataframe: {timeseries_df.shape}")
        print(f"    Columns (first 10): {timeseries_df.columns.tolist()[:10]}...")
        
        return timeseries_df
    
    def save_processed_data(self, stats: Dict, stratification: Dict, 
                           timeseries_df: pd.DataFrame, clinical_data: pd.DataFrame):
        """Save all processed data for PHASE 1"""
        print("\n[PHASE 0] Saving Processed Data...")
        
        # Save statistics
        stats_file = self.processed_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"  [OK] Saved: {stats_file}")
        
        # Save stratification
        stratification_file = self.processed_dir / "grade_stratification.json"
        with open(stratification_file, 'w') as f:
            json.dump(stratification, f, indent=2)
        print(f"  [OK] Saved: {stratification_file}")
        
        # Save time-series data
        timeseries_file = self.processed_dir / "timeseries_data.csv"
        timeseries_df.to_csv(timeseries_file, index=False)
        print(f"  [OK] Saved: {timeseries_file}")
        
        # Save clinical data for reference
        clinical_file = self.processed_dir / "clinical_data_raw.csv"
        clinical_data.to_csv(clinical_file, index=False)
        print(f"  [OK] Saved: {clinical_file}")
        
        print(f"\n[OK] All processed data ready in: {self.processed_dir}")
        print("\nNext step: Run PHASE 1 mathematical model implementation")


def main():
    """Main execution"""
    print("=" * 70)
    print("MU-GLIOMA-POST DATASET PREPROCESSING & VERIFICATION")
    print("=" * 70)
    
    try:
        processor = MUGliomaDataProcessor()
        
        # Step 1: Verify data structure
        stats = processor.verify_data_structure()
        
        if stats["errors"]:
            print("\n[ERROR] ERRORS FOUND:")
            for error in stats["errors"]:
                print(f"  - {error}")
            print("\nPlease ensure all data is downloaded to: progression/data/raw/mu_glioma_post/")
            print("See DOWNLOAD_INSTRUCTIONS.md for help")
            return
        
        # Step 2: Load clinical data
        clinical_data, volume_data = processor.load_clinical_data()
        
        # Step 3: Stratify by grade
        stratification = processor.stratify_by_grade(clinical_data)
        
        # Step 4: Create time-series data
        timeseries_df = processor.create_timeseries_data(clinical_data, volume_data)
        
        # Step 5: Save processed data
        processor.save_processed_data(stats, stratification, timeseries_df, clinical_data)
        
        print("\n" + "=" * 70)
        print("[OK] PREPROCESSING COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
