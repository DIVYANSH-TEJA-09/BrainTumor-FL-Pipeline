"""
Phase 2: LSTM Training & Evaluation
===================================

Trains grade-stratified LSTM models on residuals from logistic baseline.
Evaluates hybrid model (logistic + LSTM) vs baseline.
Generates predictions for all patients.

Timeline: ~10 minutes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import time

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure_lstm import (
    ResidualDataPreparation,
    ResidualSequenceDataset,
    ResidualLSTM,
)


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

class LSTMTrainer:
    """Train LSTM model on residual sequences."""
    
    def __init__(
        self,
        model: ResidualLSTM,
        train_dataset: ResidualSequenceDataset,
        test_dataset: ResidualSequenceDataset,
        grade: str,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.grade = grade
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.MSELoss()
        
        self.history = {'train_loss': [], 'test_loss': []}
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(x_batch)
        
        return total_loss / len(self.train_dataset)
    
    def evaluate(self, test_loader) -> float:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_pred = self.model(x_batch)
                loss = self.criterion(y_pred, y_batch)
                
                total_loss += loss.item() * len(x_batch)
        
        return total_loss / len(self.test_dataset) if len(self.test_dataset) > 0 else 0.0
    
    def train(self, epochs: int = 100, batch_size: int = 16, patience: int = 15) -> dict:
        """
        Train model with early stopping.
        
        Args:
            epochs: max number of epochs
            batch_size: batch size
            patience: early stopping patience
        
        Returns:
            training history
        """
        
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        
        best_test_loss = float('inf')
        patience_counter = 0
        
        print(f"  Training {self.grade} LSTM for max {epochs} epochs...")
        print(f"    Device: {self.device}")
        print(f"    Train: {len(self.train_dataset)} samples, Test: {len(self.test_dataset)} samples")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            test_loss = self.evaluate(test_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            
            self.scheduler.step(test_loss)
            
            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        elapsed = time.time() - start_time
        print(f"  [OK] Training complete ({elapsed:.1f}s)")
        print(f"    Best test loss: {best_test_loss:.6f}")
        
        return self.history


# ============================================================================
# HYBRID PREDICTION
# ============================================================================

def generate_hybrid_predictions(
    residuals_by_patient: dict,
    patient_params: pd.DataFrame,
    hgg_model: ResidualLSTM,
    lgg_model: ResidualLSTM,
    hgg_scaler_stats: dict,
    lgg_scaler_stats: dict,
) -> pd.DataFrame:
    """
    Generate hybrid predictions (logistic + LSTM) for all patients.
    
    Returns:
        DataFrame with columns: patient_id, grade, timepoint_idx, 
                               logistic_pred, lstm_correction, hybrid_pred,
                               v_actual, mae_baseline, mae_hybrid
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hgg_model.to(device).eval()
    lgg_model.to(device).eval()
    
    predictions = []
    
    print("[Phase 2] Generating hybrid predictions...")
    
    with torch.no_grad():
        for patient_id, traj_data in residuals_by_patient.items():
            grade = traj_data['grade']
            residuals = np.array(traj_data['residuals'], dtype=np.float32)
            volumes = np.array(traj_data['volumes'], dtype=np.float32)
            v_pred = np.array(traj_data['v_pred'], dtype=np.float32)
            
            # Select model and scaler
            model = hgg_model if grade == 'HGG' else lgg_model
            scaler_stats = hgg_scaler_stats if grade == 'HGG' else lgg_scaler_stats
            
            # For each timepoint, predict residual correction
            for t_idx in range(len(residuals)):
                # LSTM needs lookback window (3 previous residuals)
                if t_idx < 3:
                    # Not enough history, use logistic only
                    lstm_correction = 0.0
                else:
                    # Get last 3 residuals
                    lookback = residuals[t_idx-3:t_idx]
                    # Normalize
                    lookback_norm = (lookback - scaler_stats['mean']) / scaler_stats['std']
                    # Predict
                    x = torch.from_numpy(lookback_norm).float().to(device)
                    lstm_out_norm = model(x.unsqueeze(0)).item()
                    # Denormalize
                    lstm_correction = lstm_out_norm * scaler_stats['std'] + scaler_stats['mean']
                
                # Hybrid prediction
                hybrid_pred = v_pred[t_idx] + lstm_correction
                
                predictions.append({
                    'patient_id': patient_id,
                    'grade': grade,
                    'timepoint_idx': t_idx,
                    'v_actual': volumes[t_idx],
                    'v_logistic': v_pred[t_idx],
                    'lstm_correction': lstm_correction,
                    'v_hybrid': hybrid_pred,
                    'mae_baseline': abs(v_pred[t_idx] - volumes[t_idx]),
                    'mae_hybrid': abs(hybrid_pred - volumes[t_idx]),
                })
    
    pred_df = pd.DataFrame(predictions)
    print(f"  [OK] Generated {len(pred_df)} predictions")
    print(f"    HGG: {(pred_df['grade']=='HGG').sum()} points")
    print(f"    LGG: {(pred_df['grade']=='LGG').sum()} points")
    
    return pred_df


def evaluate_predictions(pred_df: pd.DataFrame) -> dict:
    """Compute metrics comparing baseline vs hybrid."""
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {
        'overall': {},
        'by_grade': {}
    }
    
    # Overall
    mae_baseline = mean_absolute_error(pred_df['v_actual'], pred_df['v_logistic'])
    mae_hybrid = mean_absolute_error(pred_df['v_actual'], pred_df['v_hybrid'])
    rmse_baseline = np.sqrt(mean_squared_error(pred_df['v_actual'], pred_df['v_logistic']))
    rmse_hybrid = np.sqrt(mean_squared_error(pred_df['v_actual'], pred_df['v_hybrid']))
    r2_baseline = r2_score(pred_df['v_actual'], pred_df['v_logistic'])
    r2_hybrid = r2_score(pred_df['v_actual'], pred_df['v_hybrid'])
    
    results['overall'] = {
        'mae_baseline': float(mae_baseline),
        'mae_hybrid': float(mae_hybrid),
        'mae_improvement_pct': float(100 * (mae_baseline - mae_hybrid) / mae_baseline),
        'rmse_baseline': float(rmse_baseline),
        'rmse_hybrid': float(rmse_hybrid),
        'rmse_improvement_pct': float(100 * (rmse_baseline - rmse_hybrid) / rmse_baseline),
        'r2_baseline': float(r2_baseline),
        'r2_hybrid': float(r2_hybrid),
        'r2_improvement': float(r2_hybrid - r2_baseline),
        'n_points': int(len(pred_df)),
    }
    
    # By grade
    for grade in ['HGG', 'LGG']:
        subset = pred_df[pred_df['grade'] == grade]
        if len(subset) > 0:
            mae_base = mean_absolute_error(subset['v_actual'], subset['v_logistic'])
            mae_hyb = mean_absolute_error(subset['v_actual'], subset['v_hybrid'])
            rmse_base = np.sqrt(mean_squared_error(subset['v_actual'], subset['v_logistic']))
            rmse_hyb = np.sqrt(mean_squared_error(subset['v_actual'], subset['v_hybrid']))
            r2_base = r2_score(subset['v_actual'], subset['v_logistic'])
            r2_hyb = r2_score(subset['v_actual'], subset['v_hybrid'])
            
            results['by_grade'][grade] = {
                'mae_baseline': float(mae_base),
                'mae_hybrid': float(mae_hyb),
                'mae_improvement_pct': float(100 * (mae_base - mae_hyb) / mae_base) if mae_base > 0 else 0,
                'rmse_baseline': float(rmse_base),
                'rmse_hybrid': float(rmse_hyb),
                'rmse_improvement_pct': float(100 * (rmse_base - rmse_hyb) / rmse_base) if rmse_base > 0 else 0,
                'r2_baseline': float(r2_base),
                'r2_hybrid': float(r2_hyb),
                'r2_improvement': float(r2_hyb - r2_base),
                'n_points': int(len(subset)),
            }
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("PHASE 2: LSTM TRAINING & EVALUATION")
    print("="*70 + "\n")
    
    results_dir = Path('results')
    
    # Step 1: Load infrastructure & data
    print("[Step 1/5] Loading data...")
    prep = ResidualDataPreparation()
    residual_df, residuals_by_patient = prep.compute_residuals()
    patient_params = pd.read_csv(results_dir / 'phase1_patient_logistic_parameters.csv')
    print()
    
    # Step 2: Create datasets
    print("[Step 2/5] Preparing datasets...")
    hgg_train_ds = ResidualSequenceDataset(residuals_by_patient, 'HGG', lookback=3, split_type='train')
    hgg_test_ds = ResidualSequenceDataset(residuals_by_patient, 'HGG', lookback=3, split_type='test')
    lgg_train_ds = ResidualSequenceDataset(residuals_by_patient, 'LGG', lookback=3, split_type='train')
    lgg_test_ds = ResidualSequenceDataset(residuals_by_patient, 'LGG', lookback=3, split_type='test')
    print()
    
    # Step 3: Train HGG model
    print("[Step 3/5] Training HGG LSTM model...")
    hgg_model = ResidualLSTM(lookback=3, hidden_size=32, num_layers=1, dropout=0.2)
    hgg_trainer = LSTMTrainer(hgg_model, hgg_train_ds, hgg_test_ds, 'HGG')
    hgg_history = hgg_trainer.train(epochs=100, batch_size=16, patience=15)
    print()
    
    # Step 4: Train LGG model
    print("[Step 4/5] Training LGG LSTM model...")
    lgg_model = ResidualLSTM(lookback=3, hidden_size=32, num_layers=1, dropout=0.2)
    lgg_trainer = LSTMTrainer(lgg_model, lgg_train_ds, lgg_test_ds, 'LGG')
    lgg_history = lgg_trainer.train(epochs=100, batch_size=16, patience=15)
    print()
    
    # Step 5: Generate predictions & evaluate
    print("[Step 5/5] Generating hybrid predictions...")
    pred_df = generate_hybrid_predictions(
        residuals_by_patient,
        patient_params,
        hgg_model,
        lgg_model,
        hgg_train_ds.__dict__,  # Scaler stats
        lgg_train_ds.__dict__,
    )
    
    # Evaluate
    metrics = evaluate_predictions(pred_df)
    print()
    
    # Save results
    print("[Saving] Results...")
    pred_df.to_csv(results_dir / 'phase2_hybrid_predictions.csv', index=False)
    
    with open(results_dir / 'phase2_training_history.json', 'w') as f:
        json.dump({
            'hgg': hgg_history,
            'lgg': lgg_history,
        }, f, indent=2)
    
    with open(results_dir / 'phase2_evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save models
    torch.save(hgg_model.state_dict(), results_dir / 'phase2_hgg_lstm_model.pth')
    torch.save(lgg_model.state_dict(), results_dir / 'phase2_lgg_lstm_model.pth')
    
    print(f"  [OK] phase2_hybrid_predictions.csv")
    print(f"  [OK] phase2_training_history.json")
    print(f"  [OK] phase2_evaluation_metrics.json")
    print(f"  [OK] phase2_hgg_lstm_model.pth")
    print(f"  [OK] phase2_lgg_lstm_model.pth")
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 RESULTS SUMMARY")
    print("="*70)
    print(f"\nOverall Performance:")
    print(f"  Baseline MAE:       {metrics['overall']['mae_baseline']:.1f} mm³")
    print(f"  Hybrid MAE:         {metrics['overall']['mae_hybrid']:.1f} mm³")
    print(f"  Improvement:        {metrics['overall']['mae_improvement_pct']:.2f}%")
    print(f"\n  Baseline RMSE:      {metrics['overall']['rmse_baseline']:.1f} mm³")
    print(f"  Hybrid RMSE:        {metrics['overall']['rmse_hybrid']:.1f} mm³")
    print(f"\n  Baseline R²:        {metrics['overall']['r2_baseline']:.4f}")
    print(f"  Hybrid R²:          {metrics['overall']['r2_hybrid']:.4f}")
    print(f"  R² Improvement:     {metrics['overall']['r2_improvement']:+.4f}")
    
    print(f"\nBy Grade:")
    for grade in ['HGG', 'LGG']:
        if grade in metrics['by_grade']:
            g_metrics = metrics['by_grade'][grade]
            print(f"\n  {grade} ({g_metrics['n_points']} points):")
            print(f"    MAE baseline:   {g_metrics['mae_baseline']:.1f} mm³")
            print(f"    MAE hybrid:     {g_metrics['mae_hybrid']:.1f} mm³")
            print(f"    Improvement:    {g_metrics['mae_improvement_pct']:.2f}%")
            print(f"    R² baseline:    {g_metrics['r2_baseline']:.4f}")
            print(f"    R² hybrid:      {g_metrics['r2_hybrid']:.4f}")
    
    print("\n" + "="*70 + "\n")
