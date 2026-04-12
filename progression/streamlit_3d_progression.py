"""
3D Progression Prediction Visualization
========================================

Streamlit page showing:
  - Actual tumor segmentation (blue)
  - Logistic baseline prediction (red)
  - LSTM hybrid prediction (green)
  
All overlaid on brain MRI in 3D interactive view.

Adapted from: segmentation/streamlit_app/pages/2_3D_Visualization.py
"""

import streamlit as st
import os
import sys
import json
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from pathlib import Path
from skimage.measure import marching_cubes

st.set_page_config(page_title="3D Tumor Growth Prediction", layout="wide")

# ============================================================================
# DATA LOADING
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "mu_glioma_post"
RESULTS_DIR = Path(__file__).parent.parent / "streamlit_data"
PRED_INDEX_FILE = RESULTS_DIR / "prediction_index.json"

@st.cache_data
def load_prediction_index():
    """Load patient prediction index."""
    if PRED_INDEX_FILE.exists():
        with open(PRED_INDEX_FILE) as f:
            return json.load(f)
    return None

@st.cache_data
def load_nifti(path):
    """Load NIfTI file."""
    if not os.path.exists(path):
        return None
    try:
        return nib.load(path).get_fdata()
    except:
        return None

def find_patient_data_path(patient_id: str):
    """Find patient data directory."""
    patient_dir = DATA_DIR / "MU-Glioma-Post" / patient_id
    if patient_dir.exists():
        return patient_dir
    return None

def get_tumor_mask_path(patient_dir, timepoint_idx: int):
    """Get path to tumor mask for timepoint."""
    timepoint_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir() and 'Timepoint' in d.name])
    if timepoint_idx < len(timepoint_dirs):
        timepoint_dir = timepoint_dirs[timepoint_idx]
        masks = list(timepoint_dir.glob('*tumorMask.nii.gz'))
        if masks:
            return masks[0]
    return None

def get_brain_image_path(patient_dir, timepoint_idx: int):
    """Get path to T1CE brain image for timepoint."""
    timepoint_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir() and 'Timepoint' in d.name])
    if timepoint_idx < len(timepoint_dirs):
        timepoint_dir = timepoint_dirs[timepoint_idx]
        # Try T1CE first, then others
        for pattern in ['*_t1c.nii.gz', '*_T1c.nii.gz', '*_t1ce.nii.gz', '*brain*.nii.gz']:
            images = list(timepoint_dir.glob(pattern))
            if images:
                return images[0]
    return None

# ============================================================================
# 3D MESH GENERATION
# ============================================================================

def extract_mesh(volume, level=0.5, step_size=2):
    """Extract 3D mesh from volume using marching cubes."""
    vol = volume[::step_size, ::step_size, ::step_size]
    if vol.sum() == 0:
        return None
    try:
        verts, faces, _, _ = marching_cubes(vol, level=level)
        verts = verts * step_size
        return verts, faces
    except:
        return None

def make_mesh_trace(volume, color, name, opacity, step_size=2, scene="scene"):
    """Create Plotly mesh trace from volume."""
    result = extract_mesh(volume, level=0.5, step_size=step_size)
    if result is None:
        return None
    
    verts, faces = result
    x, y, z = verts.T
    i, j, k = faces.T
    
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity,
        name=name, showlegend=True,
        flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.2, roughness=0.6),
        lightposition=dict(x=100, y=200, z=300),
        scene=scene,
    )

def scale_mask_by_volume(actual_mask, actual_volume, predicted_volume):
    """Scale mask by volume ratio."""
    if actual_volume <= 0 or predicted_volume <= 0:
        return actual_mask.copy()
    scale = predicted_volume / actual_volume
    return actual_mask * scale

# ============================================================================
# UI
# ============================================================================

st.title("3D Tumor Growth Prediction")
st.markdown("**Actual** (blue) vs **Baseline** (red) vs **LSTM Hybrid** (green) predictions")

# Load prediction index
pred_index = load_prediction_index()
if pred_index is None:
    st.error("Prediction data not found. Please run Phase 2 training first.")
    st.stop()

# Sidebar controls
st.sidebar.title("Controls")

# Patient selection
all_patients = sorted(pred_index['patients'].keys())
hgg_patients = [p for p in all_patients if pred_index['patients'][p]['grade'] == 'HGG']
lgg_patients = [p for p in all_patients if pred_index['patients'][p]['grade'] == 'LGG']

grade_choice = st.sidebar.radio("Grade", ["All", "HGG", "LGG"], index=0)
if grade_choice == "HGG":
    available_patients = hgg_patients
elif grade_choice == "LGG":
    available_patients = lgg_patients
else:
    available_patients = all_patients

patient_id = st.sidebar.selectbox("Patient ID", available_patients)
patient_data = pred_index['patients'][patient_id]
grade = patient_data['grade']
n_timepoints = patient_data['n_timepoints']

st.sidebar.markdown(f"**Grade:** {grade}")
st.sidebar.markdown(f"**Timepoints:** {n_timepoints}")
st.sidebar.markdown(f"**Baseline MAE:** {patient_data['mae_baseline_mean']:.0f} mm³")
st.sidebar.markdown(f"**Hybrid MAE:** {patient_data['mae_hybrid_mean']:.0f} mm³")

# Timepoint selection
timepoint_idx = st.sidebar.slider("Timepoint", 0, n_timepoints - 1, 0)
timepoint_data = patient_data['timepoints'][timepoint_idx]

st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")
show_actual = st.sidebar.checkbox("Show Actual Tumor", value=True)
show_baseline = st.sidebar.checkbox("Show Baseline Prediction", value=True)
show_hybrid = st.sidebar.checkbox("Show LSTM Hybrid Prediction", value=True)
tumor_opacity = st.sidebar.slider("Opacity", 0.2, 1.0, 0.7)
step_size = st.sidebar.select_slider("Mesh Quality", options=[1, 2, 3, 4], value=2)

# ============================================================================
# 3D VISUALIZATION
# ============================================================================

col1, col2 = st.columns([3, 1])

with col1:
    # Try to load patient data
    patient_dir = find_patient_data_path(patient_id)
    
    if patient_dir is None:
        st.error(f"Patient data not found: {patient_id}")
    else:
        # Load actual mask
        mask_path = get_tumor_mask_path(patient_dir, timepoint_idx)
        
        if mask_path is None:
            st.error(f"Tumor mask not found for timepoint {timepoint_idx}")
        else:
            actual_mask = load_nifti(mask_path)
            
            if actual_mask is None:
                st.error("Failed to load tumor mask")
            else:
                # Generate predicted masks
                v_actual = timepoint_data['v_actual']
                v_logistic = timepoint_data['v_logistic']
                v_hybrid = timepoint_data['v_hybrid']
                
                logistic_mask = scale_mask_by_volume(actual_mask, v_actual, v_logistic)
                hybrid_mask = scale_mask_by_volume(actual_mask, v_actual, v_hybrid)
                
                # Create 3D visualization
                fig = go.Figure()
                
                # Add meshes
                traces = []
                if show_actual:
                    trace = make_mesh_trace(actual_mask, '#3498db', 'Actual Tumor', tumor_opacity, step_size)
                    if trace:
                        traces.append(trace)
                
                if show_baseline:
                    trace = make_mesh_trace(logistic_mask, '#e74c3c', 'Baseline (Logistic)', tumor_opacity, step_size)
                    if trace:
                        traces.append(trace)
                
                if show_hybrid:
                    trace = make_mesh_trace(hybrid_mask, '#2ecc71', 'LSTM Hybrid', tumor_opacity, step_size)
                    if trace:
                        traces.append(trace)
                
                for trace in traces:
                    fig.add_trace(trace)
                
                # Update layout
                fig.update_layout(
                    title=f"{patient_id} - Timepoint {timepoint_idx}",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        aspectmode="data",
                    ),
                    width=1000,
                    height=800,
                    showlegend=True,
                    hovermode="closest",
                )
                
                st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Predictions")
    st.metric("Actual Volume", f"{v_actual:.0f} mm³")
    st.metric("Baseline Pred", f"{v_logistic:.0f} mm³")
    st.metric("Hybrid Pred", f"{v_hybrid:.0f} mm³")
    
    st.subheader("Errors")
    mae_baseline = abs(v_actual - v_logistic)
    mae_hybrid = abs(v_actual - v_hybrid)
    
    st.metric("Baseline MAE", f"{mae_baseline:.0f} mm³")
    st.metric("Hybrid MAE", f"{mae_hybrid:.0f} mm³")
    
    improvement = (mae_baseline - mae_hybrid) / mae_baseline * 100 if mae_baseline > 0 else 0
    st.metric("Improvement", f"{improvement:+.1f}%")

# ============================================================================
# TRAJECTORY OVER TIME
# ============================================================================

st.markdown("---")
st.subheader("Volume Trajectory")

import pandas as pd

trajectory_data = {
    'Timepoint': [t['timepoint_idx'] for t in patient_data['timepoints']],
    'Actual': [t['v_actual'] for t in patient_data['timepoints']],
    'Baseline': [t['v_logistic'] for t in patient_data['timepoints']],
    'LSTM Hybrid': [t['v_hybrid'] for t in patient_data['timepoints']],
}

traj_df = pd.DataFrame(trajectory_data)

fig_traj = go.Figure()
fig_traj.add_trace(go.Scatter(
    x=traj_df['Timepoint'], y=traj_df['Actual'],
    mode='lines+markers', name='Actual', line=dict(color='#3498db', width=3),
    marker=dict(size=8)
))
fig_traj.add_trace(go.Scatter(
    x=traj_df['Timepoint'], y=traj_df['Baseline'],
    mode='lines+markers', name='Baseline', line=dict(color='#e74c3c', width=2, dash='dash'),
    marker=dict(size=6)
))
fig_traj.add_trace(go.Scatter(
    x=traj_df['Timepoint'], y=traj_df['LSTM Hybrid'],
    mode='lines+markers', name='LSTM Hybrid', line=dict(color='#2ecc71', width=2, dash='dash'),
    marker=dict(size=6)
))

fig_traj.update_layout(
    title="Volume Over Time",
    xaxis_title="Timepoint",
    yaxis_title="Volume (mm³)",
    hovermode="x unified",
    height=400,
)

st.plotly_chart(fig_traj, use_container_width=True)

# Summary statistics
st.subheader("Summary Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Baseline Mean MAE",
        f"{patient_data['mae_baseline_mean']:.0f} mm³"
    )

with col2:
    st.metric(
        "Hybrid Mean MAE",
        f"{patient_data['mae_hybrid_mean']:.0f} mm³"
    )

with col3:
    improvement_pct = (patient_data['mae_baseline_mean'] - patient_data['mae_hybrid_mean']) / patient_data['mae_baseline_mean'] * 100
    st.metric(
        "Mean Improvement",
        f"{improvement_pct:+.1f}%"
    )
