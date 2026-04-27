#!/usr/bin/env python
"""Quick test to verify growth metrics calculations"""
import json
from pathlib import Path

pred_index_file = Path('streamlit_data/prediction_index.json')
if pred_index_file.exists():
    with open(pred_index_file) as f:
        data = json.load(f)
    
    patient_id = list(data['patients'].keys())[0]
    patient = data['patients'][patient_id]
    
    print(f"Patient: {patient_id}, Grade: {patient['grade']}, Timepoints: {patient['n_timepoints']}")
    
    if len(patient['timepoints']) >= 2:
        tp1 = patient['timepoints'][0]
        tp2 = patient['timepoints'][1]
        
        print(f"TP1 - Actual: {tp1['v_actual']:.0f}, Logistic: {tp1['v_logistic']:.0f}, Hybrid: {tp1['v_hybrid']:.0f}")
        print(f"TP2 - Actual: {tp2['v_actual']:.0f}, Logistic: {tp2['v_logistic']:.0f}, Hybrid: {tp2['v_hybrid']:.0f}")
        
        growth_actual = tp2['v_actual'] - tp1['v_actual']
        growth_pct = (growth_actual / tp1['v_actual'] * 100) if tp1['v_actual'] > 0 else 0
        
        print(f"Growth (Actual): {growth_actual:+.0f} mm³ ({growth_pct:+.1f}%)")
        
        # Check baseline prediction
        growth_baseline = tp2['v_logistic'] - tp1['v_actual']
        growth_baseline_pct = (growth_baseline / tp1['v_actual'] * 100) if tp1['v_actual'] > 0 else 0
        print(f"Growth (Baseline predicted): {growth_baseline:+.0f} mm³ ({growth_baseline_pct:+.1f}%)")
        
        # Check hybrid prediction
        growth_hybrid = tp2['v_hybrid'] - tp1['v_actual']
        growth_hybrid_pct = (growth_hybrid / tp1['v_actual'] * 100) if tp1['v_actual'] > 0 else 0
        print(f"Growth (Hybrid predicted): {growth_hybrid:+.0f} mm³ ({growth_hybrid_pct:+.1f}%)")
        
        print("\nData structure is valid for growth visualization")
else:
    print('Prediction index file not found')
