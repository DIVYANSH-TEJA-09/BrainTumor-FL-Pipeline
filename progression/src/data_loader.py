"""
Progression Data Loader

This module provides the data loading interface for progression models.
It abstracts away the complexity of MU-Glioma-Post directory structure
and provides clean, normalized time-series data.

Classes:
- ProgressionDataLoader: Main interface for loading patient trajectories
- TimeseriesPatient: Individual patient's longitudinal data
- ProgressionDataset: PyTorch-compatible dataset for models
"""

import os
import json
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader


class TimeseriesPatient:
    """Represents a single patient's longitudinal progression data"""
    
    def __init__(self, patient_id: str, grade: str, volumes: np.ndarray, 
                 timepoints: np.ndarray, metadata: Dict = None):
        """
        Args:
            patient_id: Unique identifier
            grade: 'LGG' or 'HGG'
            volumes: Array of tumor volumes at each timepoint
            timepoints: Array of days post-surgery for each volume measurement
            metadata: Additional clinical information
        """
        self.patient_id = patient_id
        self.grade = grade
        self.volumes = volumes
        self.timepoints = timepoints
        self.metadata = metadata or {}
        
    def get_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (normalized_time, normalized_volume) for fitting"""
        # Normalize time to [0, 1] for model input
        if len(self.timepoints) > 0:
            t_norm = (self.timepoints - self.timepoints[0]) / max((self.timepoints[-1] - self.timepoints[0]), 1)
        else:
            t_norm = np.array([])
        
        # Normalize volume to [0, 1] for model input
        if len(self.volumes) > 0:
            v_min, v_max = self.volumes.min(), self.volumes.max()
            if v_max > v_min:
                v_norm = (self.volumes - v_min) / (v_max - v_min)
            else:
                v_norm = np.ones_like(self.volumes)
        else:
            v_norm = np.array([])
        
        return t_norm, v_norm
    
    def get_raw_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns raw (time_in_days, volume_in_mm3)"""
        return self.timepoints, self.volumes
    
    def __repr__(self) -> str:
        return f"TimeseriesPatient({self.patient_id}, grade={self.grade}, n_timepoints={len(self.volumes)})"


class ProgressionDataLoader:
    """Load and manage MU-Glioma-Post data for progression modeling"""
    
    def __init__(self, data_root: str = "data"):
        """
        Args:
            data_root: Path to data directory (containing 'raw' and 'processed' subdirs)
        """
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw" / "mu_glioma_post"
        self.processed_dir = self.data_root / "processed"
        self.images_dir = self.raw_dir / "images"
        
        # Load metadata
        self.stratification = self._load_stratification()
        self.clinical_data = self._load_clinical_data()
        self.timeseries_data = self._load_timeseries_data()
        
    def _load_stratification(self) -> Dict[str, List[str]]:
        """Load grade stratification"""
        strat_file = self.processed_dir / "grade_stratification.json"
        if not strat_file.exists():
            return {"LGG": [], "HGG": [], "UNKNOWN": []}
        
        with open(strat_file, 'r') as f:
            return json.load(f)
    
    def _load_clinical_data(self) -> pd.DataFrame:
        """Load raw clinical data"""
        clinical_file = self.processed_dir / "clinical_data_raw.csv"
        if not clinical_file.exists():
            return pd.DataFrame()
        
        return pd.read_csv(clinical_file)
    
    def _load_timeseries_data(self) -> pd.DataFrame:
        """Load preprocessed time-series data"""
        ts_file = self.processed_dir / "timeseries_data.csv"
        if not ts_file.exists():
            return pd.DataFrame()
        
        return pd.read_csv(ts_file)
    
    def get_patient(self, patient_id: str) -> Optional[TimeseriesPatient]:
        """
        Load a single patient's data
        
        Returns:
            TimeseriesPatient object or None if not found
        """
        # Find grade
        grade = None
        for g, patients in self.stratification.items():
            if patient_id in patients:
                grade = g
                break
        
        if grade is None:
            return None
        
        # Load volumes from timeseries data if available
        row = self.timeseries_data[self.timeseries_data['PatientID'] == patient_id]
        if row.empty:
            return None
        
        # Extract volume columns
        volume_cols = [col for col in row.columns if 'volume' in col.lower()]
        volumes = row[volume_cols].values.flatten()
        volumes = volumes[~np.isnan(volumes)]  # Remove NaN
        
        # Create timepoints (assume evenly spaced if not provided)
        timepoints = np.arange(len(volumes)) * 90  # Assume 3-month intervals
        
        # Get metadata
        metadata = {
            'Age': row['Age'].values[0] if 'Age' in row.columns else np.nan,
            'Sex': row['Sex'].values[0] if 'Sex' in row.columns else np.nan,
        }
        
        return TimeseriesPatient(patient_id, grade, volumes, timepoints, metadata)
    
    def get_patients_by_grade(self, grade: str) -> List[TimeseriesPatient]:
        """
        Load all patients of a specific grade
        
        Args:
            grade: 'LGG' or 'HGG'
        
        Returns:
            List of TimeseriesPatient objects
        """
        patients = []
        for patient_id in self.stratification.get(grade, []):
            patient = self.get_patient(patient_id)
            if patient is not None:
                patients.append(patient)
        
        return patients
    
    def get_all_patients(self) -> List[TimeseriesPatient]:
        """Load all patients from all grades"""
        all_patients = []
        for grade in ['LGG', 'HGG', 'UNKNOWN']:
            all_patients.extend(self.get_patients_by_grade(grade))
        
        return all_patients
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_patients': len(self.stratification['LGG']) + len(self.stratification['HGG']),
            'lgg_patients': len(self.stratification['LGG']),
            'hgg_patients': len(self.stratification['HGG']),
            'unknown_patients': len(self.stratification['UNKNOWN']),
        }
        
        # Add volume statistics
        all_patients = self.get_all_patients()
        if all_patients:
            volumes = np.concatenate([p.volumes for p in all_patients if len(p.volumes) > 0])
            stats['volume_min'] = float(volumes.min())
            stats['volume_max'] = float(volumes.max())
            stats['volume_mean'] = float(volumes.mean())
            stats['volume_std'] = float(volumes.std())
        
        return stats


class ProgressionDataset(Dataset):
    """PyTorch Dataset for progression model training"""
    
    def __init__(self, patients: List[TimeseriesPatient], min_timepoints: int = 2):
        """
        Args:
            patients: List of TimeseriesPatient objects
            min_timepoints: Minimum number of observations per patient
        """
        self.patients = [p for p in patients if len(p.volumes) >= min_timepoints]
        
    def __len__(self) -> int:
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a patient's data as tensors
        
        Returns dict with:
            - 'patient_id': Patient ID
            - 'grade': LGG or HGG
            - 't': Normalized time points [T]
            - 'v': Normalized volumes [T]
            - 't_raw': Raw time in days [T]
            - 'v_raw': Raw volumes in mm³ [T]
            - 'metadata': Clinical metadata
        """
        patient = self.patients[idx]
        t_norm, v_norm = patient.get_trajectory()
        t_raw, v_raw = patient.get_raw_trajectory()
        
        return {
            'patient_id': patient.patient_id,
            'grade': patient.grade,
            't': torch.FloatTensor(t_norm),
            'v': torch.FloatTensor(v_norm),
            't_raw': torch.FloatTensor(t_raw),
            'v_raw': torch.FloatTensor(v_raw),
            'metadata': patient.metadata,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader
    
    Handles variable-length sequences by padding
    """
    # Find max length in batch
    max_len = max(len(item['t']) for item in batch)
    
    # Pad sequences
    t_padded = []
    v_padded = []
    masks = []
    
    for item in batch:
        t = item['t']
        v = item['v']
        
        # Pad with zeros
        t_pad = torch.nn.functional.pad(t, (0, max_len - len(t)), value=float('nan'))
        v_pad = torch.nn.functional.pad(v, (0, max_len - len(v)), value=float('nan'))
        mask = torch.cat([torch.ones(len(t)), torch.zeros(max_len - len(t))])
        
        t_padded.append(t_pad)
        v_padded.append(v_pad)
        masks.append(mask)
    
    return {
        'patient_ids': [item['patient_id'] for item in batch],
        'grades': [item['grade'] for item in batch],
        't': torch.stack(t_padded),
        'v': torch.stack(v_padded),
        'mask': torch.stack(masks),
        'metadata': [item['metadata'] for item in batch],
    }


def create_dataloaders(data_root: str = "data", 
                       batch_size: int = 16,
                       grade: Optional[str] = None) -> Tuple[DataLoader, Dict]:
    """
    Create PyTorch DataLoaders for progression modeling
    
    Args:
        data_root: Root data directory
        batch_size: Batch size for loader
        grade: Specific grade to load ('LGG', 'HGG', or None for all)
    
    Returns:
        (dataloader, dataset_stats)
    """
    loader = ProgressionDataLoader(data_root)
    
    if grade:
        patients = loader.get_patients_by_grade(grade)
    else:
        patients = loader.get_all_patients()
    
    dataset = ProgressionDataset(patients)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    stats = loader.get_statistics()
    
    return dataloader, stats


if __name__ == "__main__":
    # Example usage
    print("Loading MU-Glioma-Post data...")
    loader = ProgressionDataLoader()
    
    print("\nDataset statistics:")
    stats = loader.get_statistics()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    print("\nCreating PyTorch DataLoaders...")
    train_loader, stats = create_dataloaders(grade='HGG', batch_size=8)
    print(f"Loaded {len(train_loader)} batches for HGG patients")
    
    # Show first batch
    batch = next(iter(train_loader))
    print(f"\nFirst batch shape:")
    print(f"  Time: {batch['t'].shape}")
    print(f"  Volume: {batch['v'].shape}")
    print(f"  Mask: {batch['mask'].shape}")
