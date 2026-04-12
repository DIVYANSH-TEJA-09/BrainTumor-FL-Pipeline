"""
Phase 2: Hybrid LSTM Residual Model
====================================

Strategy:
  1. Load logistic baseline predictions for all 111 patients
  2. Compute residuals: residual(t) = V_actual(t) - V_pred_logistic(t)
  3. Prepare sequential data for LSTM: each patient's residual trajectory
  4. Build grade-stratified LSTM models (separate LGG and HGG)
  5. Train on 70% early timepoints, evaluate on 30% late timepoints
  6. Final prediction: V_final = V_pred_logistic + LSTM_correction

Key insight:
  - Logistic baseline captures ~60% of variance (R²=0.617 HGG, 0.756 LGG at per-patient level)
  - LSTM learns non-linear residual patterns: where baseline fails
  - Expected improvement: +5-10% R² (typical hybrid literature)
  - Maintains interpretability: logistic trend + LSTM noise correction
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 1. DATA LOADING AND RESIDUAL COMPUTATION
# ============================================================================

class ResidualDataPreparation:
    """Compute residuals from logistic baseline for all patients."""
    
    def __init__(self, data_dir='data/processed', results_dir='results'):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
    def load_patient_data(self) -> pd.DataFrame:
        """Load patient logistic parameters and fit results."""
        df = pd.read_csv(self.results_dir / 'phase1_patient_logistic_parameters.csv')
        return df
    
    def load_trajectories(self) -> Dict:
        """Load actual patient trajectories from longitudinal dataset."""
        # Load the longitudinal modeling dataset
        df = pd.read_csv(self.data_dir / 'longitudinal_modeling_dataset.csv')
        
        trajs = {}
        
        # Group by patient to reconstruct trajectories
        for patient_id, group in df.groupby('patient_id'):
            # Sort by timepoint index (ascending order gives chronological)
            group = group.sort_values('timepoint').reset_index(drop=True)
            
            volumes = group['tumor_total_volume_mm3'].values.astype(float)
            times = group['day_from_diagnosis_imputed'].values.astype(float)
            
            # Remove any NaN entries
            valid_idx = ~(np.isnan(volumes) | np.isnan(times))
            volumes = volumes[valid_idx]
            times = times[valid_idx]
            
            if len(volumes) >= 3:  # Only keep trajectories with 3+ points
                trajs[patient_id] = {
                    'volumes': volumes.tolist(),
                    'times': times.tolist(),
                }
        
        return trajs
    
    def logistic_function(self, t: np.ndarray, v0: float, k: float, r: float) -> np.ndarray:
        """Evaluate logistic model at times t."""
        return k / (1 + ((k - v0) / v0) * np.exp(-r * t))
    
    def compute_residuals(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Compute residuals for each patient-timepoint pair.
        
        Returns:
            residual_df: DataFrame with columns [patient_id, grade, t_index, t_days, v_actual, v_pred, residual, mae, rmse]
            residuals_by_patient: Dict[patient_id] -> {grade, residuals_list, times_list, n_points}
        """
        
        patient_params = self.load_patient_data()
        trajectories = self.load_trajectories()
        
        residuals_list = []
        residuals_by_patient = {}
        
        print("[Phase 2] Computing residuals for all patients...")
        
        for _, row in patient_params.iterrows():
            patient_id = row['patient_id']
            grade = row['grade']
            v0 = row['v0']
            k = row['k_fit']
            r = row['r_fit']
            
            # Get actual trajectory
            if patient_id not in trajectories:
                print(f"  Warning: {patient_id} not in trajectories, skipping")
                continue
            
            traj = trajectories[patient_id]
            volumes = np.array(traj['volumes'])
            times = np.array(traj['times'])
            
            # Compute predictions
            v_pred = self.logistic_function(times, v0, k, r)
            residuals = volumes - v_pred
            
            # Store per-patient summary
            residuals_by_patient[patient_id] = {
                'grade': grade,
                'residuals': residuals.tolist(),
                'times': times.tolist(),
                'volumes': volumes.tolist(),
                'v_pred': v_pred.tolist(),
                'n_points': len(times),
                'mae_residual': np.abs(residuals).mean(),
                'rmse_residual': np.sqrt((residuals ** 2).mean()),
            }
            
            # Store point-level data
            for t_idx, (t, v_actual, v_p, residual) in enumerate(zip(times, volumes, v_pred, residuals)):
                residuals_list.append({
                    'patient_id': patient_id,
                    'grade': grade,
                    't_index': t_idx,
                    't_days': t,
                    'v_actual': v_actual,
                    'v_pred': v_p,
                    'residual': residual,
                })
        
        residual_df = pd.DataFrame(residuals_list)
        
        print(f"  [OK] Computed residuals for {len(residuals_by_patient)} patients")
        print(f"  [OK] Total points: {len(residual_df)}")
        print(f"  HGG: {(residual_df['grade'] == 'HGG').sum()} points")
        print(f"  LGG: {(residual_df['grade'] == 'LGG').sum()} points")
        
        return residual_df, residuals_by_patient
    
    def save_residuals(self, residual_df: pd.DataFrame, residuals_by_patient: Dict):
        """Save residual data for Phase 2."""
        residual_df.to_csv(self.results_dir / 'phase2_residuals_pointwise.csv', index=False)
        
        with open(self.results_dir / 'phase2_residuals_by_patient.json', 'w') as f:
            json.dump(residuals_by_patient, f, indent=2)
        
        print(f"  [OK] Saved to phase2_residuals_pointwise.csv")
        print(f"  [OK] Saved to phase2_residuals_by_patient.json")


# ============================================================================
# 2. LSTM DATASET
# ============================================================================

class ResidualSequenceDataset(Dataset):
    """
    PyTorch Dataset for LSTM training on residual sequences.
    
    Each sample:
      - Input: sequence of residuals (look-back window)
      - Output: next residual to predict
      - Grade: for stratified training
      - Temporal split: marked as train or test
    """
    
    def __init__(
        self,
        residuals_by_patient: Dict,
        grade: str,
        lookback: int = 3,
        test_split: float = 0.3,
        split_type: str = 'train',
    ):
        """
        Args:
            residuals_by_patient: Dict from ResidualDataPreparation
            grade: 'HGG' or 'LGG'
            lookback: how many previous residuals to use as input
            test_split: fraction of late timepoints to use for testing
            split_type: 'train' or 'test'
        """
        
        self.grade = grade
        self.lookback = lookback
        self.samples = []
        self.scalers = {}  # Per-patient residual scaler
        
        print(f"[Phase 2] Preparing {grade} dataset (split={split_type})...")
        
        patient_count = 0
        sample_count = 0
        
        for patient_id, traj_data in residuals_by_patient.items():
            if traj_data['grade'] != grade:
                continue
            
            patient_count += 1
            residuals = np.array(traj_data['residuals'], dtype=np.float32)
            n_points = len(residuals)
            
            if n_points < lookback + 1:
                continue  # Not enough points for this lookback
            
            # Temporal split: 70% early (train), 30% late (test)
            split_idx = int(n_points * (1 - test_split))
            
            if split_type == 'train':
                max_idx = split_idx
            else:  # test
                min_idx = split_idx
                max_idx = n_points
            
            # Create sequences
            for t in range(lookback, max_idx if split_type == 'train' else n_points):
                if split_type == 'test' and t < split_idx + lookback:
                    continue
                
                # Input: residuals at t-lookback, ..., t-1
                x_seq = residuals[t - lookback:t]
                # Output: residual at t
                y = residuals[t]
                
                self.samples.append({
                    'patient_id': patient_id,
                    'x_seq': x_seq,
                    'y': y,
                    'n_points': n_points,
                    't_idx': t,
                })
                
                sample_count += 1
        
        print(f"  [OK] {grade}: {patient_count} patients, {sample_count} samples")
        
        # Normalize sequences
        self._normalize()
    
    def _normalize(self):
        """Normalize residual sequences to zero mean, unit variance."""
        if len(self.samples) == 0:
            return
        
        all_residuals = np.concatenate([s['x_seq'] for s in self.samples])
        self.mean = np.mean(all_residuals)
        self.std = np.std(all_residuals) + 1e-6  # Avoid division by zero
        
        for sample in self.samples:
            sample['x_seq'] = (sample['x_seq'] - self.mean) / self.std
            sample['y'] = (sample['y'] - self.mean) / self.std
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        x = torch.from_numpy(sample['x_seq']).float()  # Shape: (lookback,)
        y = torch.tensor(sample['y'], dtype=torch.float32)  # Scalar
        return x, y


# ============================================================================
# 3. LSTM ARCHITECTURE
# ============================================================================

class ResidualLSTM(nn.Module):
    """
    LSTM model to predict next residual from previous residuals.
    
    Architecture:
      - Input: sequence of residuals (lookback length)
      - LSTM: capture temporal dependencies in residuals
      - Fully connected layers: output prediction
      - Output: predicted next residual
    """
    
    def __init__(
        self,
        lookback: int = 3,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention layer (optional): weight different timesteps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, lookback)
        
        Returns:
            output: Tensor of shape (batch_size, 1) - predicted residual
        """
        
        # Reshape for LSTM: (batch_size, lookback, 1)
        x = x.unsqueeze(-1)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, lookback, hidden)
        
        # Attention: weight the LSTM outputs
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, lookback, hidden)
        
        # Take the final hidden state (could also use attention)
        final_hidden = attn_out[:, -1, :]  # (batch, hidden)
        
        # Fully connected layers
        output = self.fc(final_hidden)  # (batch, 1)
        
        return output.squeeze(-1)  # (batch,)


# ============================================================================
# 4. MAIN ORCHESTRATION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("PHASE 2: LSTM RESIDUAL ENHANCEMENT")
    print("="*70 + "\n")
    
    # Step 1: Compute residuals
    print("[Step 1/4] Computing residuals from logistic baseline...")
    prep = ResidualDataPreparation()
    residual_df, residuals_by_patient = prep.compute_residuals()
    prep.save_residuals(residual_df, residuals_by_patient)
    print()
    
    # Step 2: Create datasets
    print("[Step 2/4] Creating LSTM datasets...")
    
    # HGG dataset
    hgg_train_ds = ResidualSequenceDataset(
        residuals_by_patient, 'HGG', lookback=3, split_type='train'
    )
    hgg_test_ds = ResidualSequenceDataset(
        residuals_by_patient, 'HGG', lookback=3, split_type='test'
    )
    
    # LGG dataset
    lgg_train_ds = ResidualSequenceDataset(
        residuals_by_patient, 'LGG', lookback=3, split_type='train'
    )
    lgg_test_ds = ResidualSequenceDataset(
        residuals_by_patient, 'LGG', lookback=3, split_type='test'
    )
    
    print()
    print("[Step 3/4] LSTM Model Architecture:")
    model = ResidualLSTM(lookback=3, hidden_size=32, num_layers=1, dropout=0.2)
    print(model)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("\n[Step 4/4] Infrastructure ready for training")
    print(f"  HGG: {len(hgg_train_ds)} train, {len(hgg_test_ds)} test")
    print(f"  LGG: {len(lgg_train_ds)} train, {len(lgg_test_ds)} test")
    print("\n[OK] Phase 2 infrastructure complete")
    print("  Next: 07_hybrid_lstm_training.py for training and evaluation")
