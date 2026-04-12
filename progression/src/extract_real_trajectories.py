"""
PHASE 1 (REVISED): Extract Real Tumor Volumes from Segmentation Masks

This script processes actual NIfTI segmentation masks (.nii.gz files) to extract
real tumor volumes for each patient timepoint. This enables fitting mathematical
models to ACTUAL tumor growth trajectories, not synthetic data.

Key difference from previous attempt:
- Before: Generated fake linear synthetic trajectories
- Now: Extract real tumor volumes from medical imaging segmentation masks
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# NIFTI FILE PROCESSING
# ============================================================================

class SegmentationProcessor:
    """Process NIfTI segmentation masks and extract tumor volumes."""
    
    def __init__(self, voxel_spacing_mm3=1.0):
        """
        Initialize processor.
        
        Args:
            voxel_spacing_mm3: Volume of each voxel in mm³ (depends on scan resolution)
                              Default 1.0 assumes unit voxel spacing
                              Will be auto-detected from NIfTI header if available
        """
        self.voxel_spacing = voxel_spacing_mm3
    
    def load_segmentation(self, nifti_path: Path) -> Optional[np.ndarray]:
        """Load NIfTI segmentation file and extract tumor mask."""
        try:
            img = nib.load(nifti_path)
            data = img.get_fdata()
            affine = img.affine
            
            # Calculate voxel spacing from affine matrix
            voxel_dims = np.abs(np.diag(affine[:3, :3]))
            voxel_volume = np.prod(voxel_dims)
            
            return data, voxel_volume
        except Exception as e:
            print(f"    [WARN] Failed to load {nifti_path.name}: {e}")
            return None, None
    
    def compute_tumor_volume(self, segmentation_data: np.ndarray, voxel_volume_mm3: float) -> float:
        """
        Compute tumor volume from segmentation mask.
        
        Args:
            segmentation_data: Binary or label mask from NIfTI file
            voxel_volume_mm3: Volume of each voxel in mm³
        
        Returns:
            Tumor volume in mm³
        """
        # Binarize if needed (any non-zero voxel = tumor)
        binary_mask = (segmentation_data > 0).astype(np.float32)
        
        # Count non-zero voxels
        tumor_voxels = np.sum(binary_mask)
        
        # Convert to volume
        tumor_volume_mm3 = tumor_voxels * voxel_volume_mm3
        
        return float(tumor_volume_mm3)


@dataclass
class PatientTrajectory:
    """Container for a patient's tumor progression trajectory."""
    patient_id: str
    grade: str
    timepoints: List[int]  # Timepoint indices (1, 2, 5, etc.)
    days_since_baseline: List[float]  # Days since first timepoint
    volumes_mm3: List[float]  # Tumor volume in mm³
    n_timepoints: int
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'patient_id': self.patient_id,
            'grade': self.grade,
            'n_timepoints': self.n_timepoints,
            'timepoints': self.timepoints,
            'days_since_baseline': self.days_since_baseline,
            'volumes_mm3': self.volumes_mm3,
            'volume_range': [min(self.volumes_mm3), max(self.volumes_mm3)] if self.volumes_mm3 else [0, 0]
        }
    
    def is_valid(self) -> bool:
        """Check if trajectory has sufficient data for modeling."""
        # Need at least 2 timepoints for fitting
        return self.n_timepoints >= 2


class RealTrajectoryExtractor:
    """Extract real tumor trajectories from segmentation masks."""
    
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processor = SegmentationProcessor()
        self.grade_file = self.processed_data_dir / "grade_stratification.json"
        
        # Load grade information
        with open(self.grade_file) as f:
            self.grades = json.load(f)
    
    def extract_patient_trajectory(self, patient_id: str) -> Optional[PatientTrajectory]:
        """Extract real tumor volumes for a patient across all timepoints."""
        patient_dir = self.raw_data_dir / "MU-Glioma-Post" / patient_id
        
        if not patient_dir.exists():
            return None
        
        # Find all timepoint folders
        timepoint_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir() and d.name.startswith('Timepoint_')])
        
        if len(timepoint_dirs) == 0:
            return None
        
        # Extract timepoint indices
        timepoint_indices = []
        volumes = []
        
        for tp_dir in timepoint_dirs:
            # Extract timepoint number (e.g., "Timepoint_1" → 1)
            tp_index = int(tp_dir.name.split('_')[1])
            timepoint_indices.append(tp_index)
            
            # Find tumor mask file
            mask_files = list(tp_dir.glob("*_tumorMask.nii.gz"))
            
            if not mask_files:
                # Try without .gz
                mask_files = list(tp_dir.glob("*_tumorMask.nii"))
            
            if not mask_files:
                print(f"    [WARN] No tumor mask found for {patient_id}/{tp_dir.name}")
                volumes.append(np.nan)
                continue
            
            # Load and process segmentation
            seg_data, voxel_vol = self.processor.load_segmentation(mask_files[0])
            
            if seg_data is None:
                volumes.append(np.nan)
                continue
            
            # Compute tumor volume
            volume = self.processor.compute_tumor_volume(seg_data, voxel_vol)
            volumes.append(volume)
        
        # Remove NaN volumes
        valid_indices = [(idx, vol) for idx, vol in zip(timepoint_indices, volumes) if not np.isnan(vol)]
        
        if len(valid_indices) < 2:
            return None  # Need at least 2 valid timepoints
        
        timepoint_indices, volumes = zip(*valid_indices)
        timepoint_indices = list(timepoint_indices)
        volumes = list(volumes)
        
        # Compute days since baseline (assuming ~30 days between consecutive clinical visits)
        # This is approximate - ideally would use actual scan dates from DICOM headers
        days_since_baseline = [(tp - 1) * 30 for tp in timepoint_indices]
        
        # Determine grade
        grade = None
        if patient_id in self.grades['LGG']:
            grade = 'LGG'
        elif patient_id in self.grades['HGG']:
            grade = 'HGG'
        else:
            grade = 'UNKNOWN'
        
        return PatientTrajectory(
            patient_id=patient_id,
            grade=grade,
            timepoints=timepoint_indices,
            days_since_baseline=days_since_baseline,
            volumes_mm3=volumes,
            n_timepoints=len(volumes)
        )
    
    def extract_all_trajectories(self, max_patients: Optional[int] = None) -> Dict[str, PatientTrajectory]:
        """Extract trajectories for all patients.
        
        Args:
            max_patients: Process only first N patients (for testing). None = all.
        """
        trajectories = {}
        
        # Get all patient IDs
        all_patients = self.grades['LGG'] + self.grades['HGG'] + self.grades['UNKNOWN']
        
        if max_patients:
            all_patients = all_patients[:max_patients]
            print(f"\n[TEST MODE] Processing first {max_patients} patients...")
        else:
            print(f"\nExtracting real tumor volumes from segmentation masks...")
            print(f"Processing {len(all_patients)} patients...")
        
        for i, patient_id in enumerate(all_patients):
            if (i + 1) % max(1, len(all_patients) // 10) == 0 or i == len(all_patients) - 1:
                print(f"  [{i + 1}/{len(all_patients)}] {patient_id}")
            
            trajectory = self.extract_patient_trajectory(patient_id)
            
            if trajectory and trajectory.is_valid():
                trajectories[patient_id] = trajectory
        
        print(f"[OK] Successfully extracted {len(trajectories)}/{len(all_patients)} valid trajectories")
        return trajectories


# ============================================================================
# PHASE 1 (REVISED): MAIN EXECUTION
# ============================================================================

def run_phase1_revised(raw_data_dir: Path, processed_data_dir: Path, output_dir: Path, test_mode: bool = False):
    """Execute Phase 1 (Revised): Real Tumor Volume Extraction and Analysis.
    
    Args:
        test_mode: If True, process only first 5 patients for testing
    """
    
    print("\n" + "="*80)
    print("PHASE 1 (REVISED): EXTRACT REAL TUMOR VOLUMES FROM SEGMENTATION MASKS")
    print("="*80)
    
    # Initialize extractor
    extractor = RealTrajectoryExtractor(raw_data_dir, processed_data_dir)
    
    # Extract all trajectories
    max_patients = 5 if test_mode else None
    trajectories = extractor.extract_all_trajectories(max_patients=max_patients)
    
    # Analyze trajectories by grade
    print(f"\n[2/4] Analyzing trajectories by grade...")
    
    grade_analysis = {'LGG': [], 'HGG': [], 'UNKNOWN': []}
    
    for traj in trajectories.values():
        grade_analysis[traj.grade].append(traj)
    
    # Print summary statistics
    for grade in ['LGG', 'HGG', 'UNKNOWN']:
        patients = grade_analysis[grade]
        if patients:
            n_patients = len(patients)
            timepoints = [p.n_timepoints for p in patients]
            avg_timepoints = np.mean(timepoints)
            max_volume = max([max(p.volumes_mm3) for p in patients])
            min_volume = min([min(p.volumes_mm3) for p in patients])
            
            print(f"\n  {grade}:")
            print(f"    Patients: {n_patients}")
            print(f"    Avg timepoints per patient: {avg_timepoints:.1f}")
            print(f"    Timepoint distribution: min={min(timepoints)}, max={max(timepoints)}")
            print(f"    Volume range: {min_volume:.1f} - {max_volume:.1f} mm³")
    
    # Save trajectories for Phase 1 (Mathematical Fitting)
    print(f"\n[3/4] Saving real trajectories...")
    
    trajectories_file = output_dir / "phase1_real_trajectories.json"
    trajectories_dict = {pid: traj.to_dict() for pid, traj in trajectories.items()}
    
    with open(trajectories_file, 'w') as f:
        json.dump(trajectories_dict, f, indent=2)
    
    print(f"  [OK] Saved {len(trajectories)} trajectories to {trajectories_file}")
    
    # Save summary report
    print(f"\n[4/4] Generating summary report...")
    
    report_file = output_dir / "phase1_trajectory_extraction_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 1 (REVISED): REAL TUMOR VOLUME EXTRACTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Patients Processed: {len(extractor.grades['LGG'] + extractor.grades['HGG'] + extractor.grades['UNKNOWN'])}\n")
        f.write(f"Valid Trajectories Extracted: {len(trajectories)}\n\n")
        
        for grade in ['LGG', 'HGG', 'UNKNOWN']:
            patients = grade_analysis[grade]
            if patients:
                f.write(f"\n{grade} STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Patients: {len(patients)}\n")
                
                timepoints = [p.n_timepoints for p in patients]
                volumes_all = [v for p in patients for v in p.volumes_mm3]
                
                f.write(f"Timepoints per patient:\n")
                f.write(f"  Mean: {np.mean(timepoints):.1f}\n")
                f.write(f"  Std: {np.std(timepoints):.1f}\n")
                f.write(f"  Range: {min(timepoints)} - {max(timepoints)}\n")
                
                f.write(f"\nTumor volumes (mm3):\n")
                f.write(f"  Mean: {np.mean(volumes_all):.1f}\n")
                f.write(f"  Std: {np.std(volumes_all):.1f}\n")
                f.write(f"  Range: {np.min(volumes_all):.1f} - {np.max(volumes_all):.1f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"  [OK] Saved report to {report_file}")
    
    print("\n" + "="*80)
    print("PHASE 1 (REVISED) COMPLETE [OK]")
    print("="*80)
    print("\nNext: Fit mathematical models to REAL tumor trajectories")
    print("      (Run 02_fit_real_models.py)")


if __name__ == "__main__":
    # Setup paths
    script_dir = Path(__file__).parent.parent
    raw_data_dir = script_dir / "data" / "raw" / "mu_glioma_post"
    processed_data_dir = script_dir / "data" / "processed"
    output_dir = script_dir / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Add Optional import at top of file
    # Run Phase 1 (Revised) - set max_patients=5 for quick test, None for all
    run_phase1_revised(raw_data_dir, processed_data_dir, output_dir, test_mode=False)
