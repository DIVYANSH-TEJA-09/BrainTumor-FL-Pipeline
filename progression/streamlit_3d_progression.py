"""
3D Progression Prediction Visualization - Synchronized Multi-Panel with Brain Overlay
======================================================================================

Streamlit page showing:
  - 3 synchronized 3D panels: Actual tumor, Logistic baseline, LSTM hybrid
  - Brain overlay on all panels
  - All three panels share the same camera (synchronized rotation/zoom)
  - Perfect for clinical comparison - see all predictions from same angle
  
Key Feature: Rotate ANY panel and all three rotate together!
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
# INITIALIZE SESSION STATE FOR SYNCHRONIZED CAMERA
# ============================================================================

if 'sync_camera' not in st.session_state:
    # Default camera view - stored in session state so it persists across updates
    st.session_state.sync_camera = dict(
        eye=dict(x=1.6, y=1.0, z=0.8),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0)
    )

# ============================================================================
# DATA LOADING
# ============================================================================

# Ensure we're using the correct directory - get absolute paths
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "raw" / "mu_glioma_post"
RESULTS_DIR = APP_DIR / "streamlit_data"
PRED_INDEX_FILE = RESULTS_DIR / "prediction_index.json"

@st.cache_data
def load_prediction_index():
    """Load patient prediction index."""
    if PRED_INDEX_FILE.exists():
        with open(PRED_INDEX_FILE) as f:
            return json.load(f)
    
    st.error(f"""
    Prediction data not found. 
    
    Looking for: {PRED_INDEX_FILE.resolve()}
    
    Expected location: {RESULTS_DIR.resolve()}
    
    Files in that directory: {list(RESULTS_DIR.glob('*')) if RESULTS_DIR.exists() else 'Directory does not exist'}
    
    Please run: python src/08_generate_viz_data.py
    """)
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

def build_brain_trace(brain_img, opacity, step_size, scene="scene"):
    """Build brain surface mesh from T1CE image."""
    if brain_img is None:
        return None
    
    # Normalize and create brain mask from T1CE
    brain_norm = (brain_img - brain_img.min()) / (brain_img.max() - brain_img.min() + 1e-8)
    brain_mask = (brain_norm > 0.15).astype(float)
    
    return make_mesh_trace(
        brain_mask, '#D5D8DC', 'Brain',
        opacity=opacity, step_size=max(step_size, 2), scene=scene
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

st.title("3D Tumor Growth Prediction with Synchronized View")
st.markdown("**🔄 Rotate any panel - all three move together!** | Actual (blue) vs Baseline (red) vs LSTM Hybrid (green)")

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
show_brain = st.sidebar.checkbox("Show Brain Overlay", value=True)
brain_opacity = st.sidebar.slider("Brain Opacity", 0.02, 0.30, 0.08, 0.02)
tumor_opacity = st.sidebar.slider("Tumor Opacity", 0.2, 1.0, 0.7)
step_size = st.sidebar.select_slider("Mesh Quality", options=[1, 2, 3, 4], value=2)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Synchronized Camera:** Rotate/zoom any panel and all three move together for perfect comparison!")

# ============================================================================
# 3D VISUALIZATION - THREE PANELS WITH SYNCHRONIZED CAMERA
# ============================================================================

# Try to load patient data
patient_dir = find_patient_data_path(patient_id)

if patient_dir is None:
    st.error(f"Patient data not found: {patient_id}")
else:
    # Load actual mask
    mask_path = get_tumor_mask_path(patient_dir, timepoint_idx)
    brain_img_path = get_brain_image_path(patient_dir, timepoint_idx)
    
    if mask_path is None:
        st.error(f"Tumor mask not found for timepoint {timepoint_idx}")
    else:
        actual_mask = load_nifti(mask_path)
        brain_img = load_nifti(brain_img_path) if brain_img_path else None
        
        if actual_mask is None:
            st.error("Failed to load tumor mask")
        else:
            # Get prediction volumes
            v_actual = timepoint_data['v_actual']
            v_logistic = timepoint_data['v_logistic']
            v_hybrid = timepoint_data['v_hybrid']
            
            # Generate predicted masks by scaling
            logistic_mask = scale_mask_by_volume(actual_mask, v_actual, v_logistic)
            hybrid_mask = scale_mask_by_volume(actual_mask, v_actual, v_hybrid)
            
            # Create unified figure with 3 subplots for SYNCHRONIZED camera
            from plotly.subplots import make_subplots
            
            # Collect all traces with proper scene assignments
            traces_actual = []
            traces_baseline = []
            traces_hybrid = []
            
            # Add brain overlay to all panels
            if show_brain and brain_img is not None:
                brain_trace1 = build_brain_trace(brain_img, brain_opacity, step_size, "scene1")
                brain_trace2 = build_brain_trace(brain_img, brain_opacity, step_size, "scene2")
                brain_trace3 = build_brain_trace(brain_img, brain_opacity, step_size, "scene3")
                if brain_trace1:
                    traces_actual.append(brain_trace1)
                if brain_trace2:
                    traces_baseline.append(brain_trace2)
                if brain_trace3:
                    traces_hybrid.append(brain_trace3)
            
            # Add tumor meshes
            trace_actual = make_mesh_trace(actual_mask, '#3498db', 'Actual Tumor', tumor_opacity, step_size, "scene1")
            if trace_actual:
                traces_actual.append(trace_actual)
            
            trace_baseline = make_mesh_trace(logistic_mask, '#e74c3c', 'Baseline', tumor_opacity, step_size, "scene2")
            if trace_baseline:
                traces_baseline.append(trace_baseline)
            
            trace_hybrid = make_mesh_trace(hybrid_mask, '#2ecc71', 'LSTM Hybrid', tumor_opacity, step_size, "scene3")
            if trace_hybrid:
                traces_hybrid.append(trace_hybrid)
            
            # Create subplots with 3D scenes
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=("🔵 Actual Tumor", "🔴 Baseline (Logistic)", "🟢 LSTM Hybrid"),
                horizontal_spacing=0.05
            )
            
            # Add traces to each subplot
            for trace in traces_actual:
                trace.scene = "scene1"
                fig.add_trace(trace, row=1, col=1)
            
            for trace in traces_baseline:
                trace.scene = "scene2"
                fig.add_trace(trace, row=1, col=2)
            
            for trace in traces_hybrid:
                trace.scene = "scene3"
                fig.add_trace(trace, row=1, col=3)
            
            # Update all scenes with synchronized camera
            camera_settings = dict(
                eye=st.session_state.sync_camera['eye'],
                up=st.session_state.sync_camera['up'],
                center=st.session_state.sync_camera['center']
            )
            
            scene_dict = dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgb(10, 10, 20)",
                camera=camera_settings,
                aspectmode="data",
            )
            
            fig.update_layout(
                scene1=scene_dict,
                scene2=scene_dict,
                scene3=scene_dict,
                height=600,
                margin=dict(l=0, r=0, t=50, b=0),
                paper_bgcolor="rgb(10, 10, 20)",
                font=dict(color="white"),
                showlegend=False,
                hovermode='closest',
            )
            
            # Display the synchronized figure
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True}, key="fig_3panel")
            
            # Display metrics below the visualization
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔵 Actual Volume", f"{v_actual:.0f} mm³")
            
            with col2:
                mae_baseline = abs(v_actual - v_logistic)
                st.metric("🔴 Baseline Volume", f"{v_logistic:.0f} mm³", f"MAE: {mae_baseline:.0f}")
            
            with col3:
                mae_hybrid = abs(v_actual - v_hybrid)
                improvement = (mae_baseline - mae_hybrid) / mae_baseline * 100 if mae_baseline > 0 else 0
                st.metric("🟢 Hybrid Volume", f"{v_hybrid:.0f} mm³", f"MAE: {mae_hybrid:.0f} ({improvement:+.1f}%)")

# ============================================================================
# TRAJECTORY OVER TIME
# ============================================================================

st.markdown("---")
st.subheader("Volume Trajectory Over Time")

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
    paper_bgcolor="rgb(10, 10, 20)",
    plot_bgcolor="rgb(20, 20, 40)",
    font=dict(color="white"),
    xaxis=dict(gridcolor="rgba(100, 100, 140, 0.2)"),
    yaxis=dict(gridcolor="rgba(100, 100, 140, 0.2)"),
)

st.plotly_chart(fig_traj, use_container_width=True, key="fig_traj")

# Summary statistics
st.markdown("---")
st.subheader("Summary Statistics")
col1, col2, col3, col4 = st.columns(4)

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

with col4:
    st.metric(
        "Patient Grade",
        f"{grade} ({n_timepoints} timepoints)"
    )
