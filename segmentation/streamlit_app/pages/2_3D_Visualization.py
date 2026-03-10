"""
3D Interactive Brain Tumor Visualization
=========================================
Renders the brain + tumor regions as interactive 3D surfaces.
Supports side-by-side Ground Truth vs AI Prediction comparison.
Runs live inference if prediction doesn't exist yet.
"""

import streamlit as st
import os
import sys
import glob
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes

st.set_page_config(page_title="3D Tumor Visualization", layout="wide")

# ─── paths & inference ───────────────────────────────────────────────────
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
from inference import ensure_prediction, get_all_patients, DEMO_DIR


def load_nifti(path):
    if not os.path.exists(path):
        return None
    return nib.load(path).get_fdata()



def extract_mesh(volume, level=0.5, step_size=2):
    vol = volume[::step_size, ::step_size, ::step_size]
    if vol.sum() == 0:
        return None
    try:
        verts, faces, _, _ = marching_cubes(vol, level=level)
        verts = verts * step_size
        return verts, faces
    except Exception:
        return None


def make_mesh_trace(volume, color, name, opacity, step_size, level=0.5,
                    flatshading=True, scene="scene"):
    result = extract_mesh(volume, level=level, step_size=step_size)
    if result is None:
        return None
    verts, faces = result
    x, y, z = verts.T
    i, j, k = faces.T
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity,
        name=name, showlegend=True,
        flatshading=flatshading,
        lighting=dict(ambient=0.6, diffuse=0.7, specular=0.2, roughness=0.6),
        lightposition=dict(x=100, y=200, z=300),
        scene=scene,
    )


# ─── colors ──────────────────────────────────────────────────────────────
PRED_COLORS = {
    "Whole Tumor (WT)": "#2ECC71",   # emerald green
    "Tumor Core (TC)":  "#E74C3C",   # vivid red
    "Enhancing Tumor (ET)": "#F1C40F",  # bright gold
}
GT_COLORS = {
    "Whole Tumor (WT)": "#1ABC9C",   # turquoise
    "Tumor Core (TC)":  "#9B59B6",   # amethyst
    "Enhancing Tumor (ET)": "#3498DB",  # ocean blue
}
PRED_CHANNELS = {"Whole Tumor (WT)": 1, "Tumor Core (TC)": 0, "Enhancing Tumor (ET)": 2}
BRAIN_COLOR = "#D5D8DC"

# ─── sidebar ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ 3D Controls")

samples = get_all_patients()
if not samples:
    st.error("No processed prediction volumes found.")
    st.stop()

selected_id = st.sidebar.selectbox("🧑‍⚕️ Patient", samples)

st.sidebar.markdown("---")
view_mode = st.sidebar.radio(
    "View Mode",
    ["Prediction Only", "Ground Truth Only", "Side-by-Side Comparison"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Brain Surface")
show_brain = st.sidebar.checkbox("Show Brain", value=True)
brain_opacity = st.sidebar.slider("Brain Opacity", 0.02, 0.30, 0.08, 0.02)

st.sidebar.subheader("🎯 Tumor")
region_choice = st.sidebar.multiselect(
    "Regions",
    ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"],
    default=["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"],
)
tumor_opacity = st.sidebar.slider("Tumor Opacity", 0.20, 1.0, 0.70, 0.05)

st.sidebar.markdown("---")
step_size = st.sidebar.select_slider("Mesh Quality", options=[1, 2, 3, 4], value=2)

# ─── load data (run inference if needed) ─────────────────────────────────
ensure_prediction(selected_id)

img_data = load_nifti(os.path.join(DEMO_DIR, f"{selected_id}_image.nii.gz"))
pred_data = load_nifti(os.path.join(DEMO_DIR, f"{selected_id}_pred.nii.gz"))
lbl_data = load_nifti(os.path.join(DEMO_DIR, f"{selected_id}_label.nii.gz"))

# ─── title ───────────────────────────────────────────────────────────────
st.title("🌐 3D Brain Tumor Visualization")
st.markdown(f"**Patient:** `{selected_id}` · **Drag** to rotate · **Scroll** to zoom")

# ─── color legend ────────────────────────────────────────────────────────
legend_cols = st.columns(6)
for idx, (region, color) in enumerate(PRED_COLORS.items()):
    with legend_cols[idx]:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:14px;height:14px;background:{color};'
            f'border-radius:3px;"></div><span style="font-size:13px;">Pred: {region}</span></div>',
            unsafe_allow_html=True,
        )
if view_mode in ["Ground Truth Only", "Side-by-Side Comparison"]:
    for idx, (region, color) in enumerate(GT_COLORS.items()):
        with legend_cols[idx + 3]:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<div style="width:14px;height:14px;background:{color};'
                f'border-radius:3px;"></div><span style="font-size:13px;">GT: {region}</span></div>',
                unsafe_allow_html=True,
            )


# ─── helper: build brain trace ───────────────────────────────────────────
def build_brain_trace(scene_name="scene"):
    if img_data is None:
        return None
    flair = img_data[:, :, :, 3]
    flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    brain_mask = (flair_norm > 0.15).astype(float)
    return make_mesh_trace(
        brain_mask, BRAIN_COLOR, "Brain",
        opacity=brain_opacity,
        step_size=max(step_size, 2),
        flatshading=False, scene=scene_name,
    )


def build_pred_traces(scene_name="scene"):
    traces = []
    if pred_data is None:
        return traces
    for region in region_choice:
        ch = PRED_CHANNELS[region]
        vol = pred_data[:, :, :, ch]
        t = make_mesh_trace(vol, PRED_COLORS[region], f"Pred: {region}",
                            tumor_opacity, step_size, scene=scene_name)
        if t:
            traces.append(t)
    return traces


def build_gt_traces(scene_name="scene"):
    traces = []
    if lbl_data is None:
        return traces
    gt_masks = {
        "Whole Tumor (WT)": (lbl_data > 0).astype(float),
        "Tumor Core (TC)": ((lbl_data == 1) | (lbl_data == 4)).astype(float),
        "Enhancing Tumor (ET)": (lbl_data == 4).astype(float),
    }
    for region in region_choice:
        vol = gt_masks[region]
        t = make_mesh_trace(vol, GT_COLORS[region], f"GT: {region}",
                            tumor_opacity, step_size, scene=scene_name)
        if t:
            traces.append(t)
    return traces


SCENE_LAYOUT = dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False),
    bgcolor="rgb(10, 10, 20)",
    aspectmode="data",
    camera=dict(eye=dict(x=1.6, y=1.0, z=0.8), up=dict(x=0, y=0, z=1)),
)

# ─── render ──────────────────────────────────────────────────────────────

if view_mode == "Side-by-Side Comparison":
    # Two 3D plots: GT on the left, Prediction on the right
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🟢 Ground Truth")
        gt_traces = []
        if show_brain:
            bt = build_brain_trace("scene")
            if bt:
                gt_traces.append(bt)
        gt_traces.extend(build_gt_traces("scene"))

        if gt_traces:
            fig_gt = go.Figure(data=gt_traces)
            fig_gt.update_layout(
                scene=SCENE_LAYOUT,
                margin=dict(l=0, r=0, t=0, b=0),
                height=600,
                paper_bgcolor="rgb(10, 10, 20)",
                legend=dict(font=dict(color="white", size=11),
                            bgcolor="rgba(20,20,40,0.8)", x=0.01, y=0.99),
            )
            st.plotly_chart(fig_gt, width="stretch")
        else:
            st.info("No ground truth data available for this patient.")

    with col_right:
        st.markdown("### 🔴 AI Prediction")
        pred_traces = []
        if show_brain:
            bt = build_brain_trace("scene")
            if bt:
                pred_traces.append(bt)
        pred_traces.extend(build_pred_traces("scene"))

        if pred_traces:
            fig_pred = go.Figure(data=pred_traces)
            fig_pred.update_layout(
                scene=SCENE_LAYOUT,
                margin=dict(l=0, r=0, t=0, b=0),
                height=600,
                paper_bgcolor="rgb(10, 10, 20)",
                legend=dict(font=dict(color="white", size=11),
                            bgcolor="rgba(20,20,40,0.8)", x=0.01, y=0.99),
            )
            st.plotly_chart(fig_pred, width="stretch")
        else:
            st.warning("No prediction data available.")

else:
    # Single 3D view
    all_traces = []
    if show_brain:
        bt = build_brain_trace("scene")
        if bt:
            all_traces.append(bt)

    if view_mode == "Prediction Only":
        all_traces.extend(build_pred_traces("scene"))
    elif view_mode == "Ground Truth Only":
        all_traces.extend(build_gt_traces("scene"))

    if not all_traces:
        st.warning("Nothing to render. Check that data exists and regions are selected.")
        st.stop()

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        scene=SCENE_LAYOUT,
        margin=dict(l=0, r=0, t=0, b=0),
        height=750,
        paper_bgcolor="rgb(10, 10, 20)",
        legend=dict(font=dict(color="white", size=13),
                    bgcolor="rgba(20,20,40,0.85)",
                    bordercolor="rgba(100,100,140,0.5)", borderwidth=1,
                    x=0.01, y=0.99),
    )
    st.plotly_chart(fig, width="stretch")

# ─── volume stats ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Tumor Volume Statistics")

if pred_data is not None:
    cols = st.columns(3)
    for idx, (region, ch) in enumerate(PRED_CHANNELS.items()):
        vol = pred_data[:, :, :, ch]
        voxel_count = int(vol.sum())
        volume_cc = voxel_count / 1000.0
        color = PRED_COLORS[region]
        with cols[idx]:
            st.markdown(
                f'<div style="background:rgba(30,30,50,0.8);padding:14px;'
                f'border-radius:10px;border-left:4px solid {color};">'
                f'<div style="color:{color};font-size:13px;font-weight:600;">{region}</div>'
                f'<div style="color:white;font-size:26px;font-weight:700;">{volume_cc:.1f} cm³</div>'
                f'<div style="color:#888;font-size:11px;">{voxel_count:,} voxels</div></div>',
                unsafe_allow_html=True,
            )

# ─── dice ────────────────────────────────────────────────────────────────
if lbl_data is not None and pred_data is not None:
    st.markdown("---")
    st.subheader("🔬 Dice Scores")
    cols = st.columns(3)
    gt_masks_dice = {
        "Whole Tumor (WT)": (lbl_data > 0).astype(float),
        "Tumor Core (TC)": ((lbl_data == 1) | (lbl_data == 4)).astype(float),
        "Enhancing Tumor (ET)": (lbl_data == 4).astype(float),
    }
    for idx, (region, ch) in enumerate(PRED_CHANNELS.items()):
        p = pred_data[:, :, :, ch]
        g = gt_masks_dice[region]
        dice = (2.0 * (p * g).sum()) / (p.sum() + g.sum() + 1e-8)
        color = PRED_COLORS[region]
        grade = "Excellent" if dice > 0.8 else "Good" if dice > 0.6 else "Fair"
        with cols[idx]:
            st.markdown(
                f'<div style="background:rgba(30,30,50,0.8);padding:14px;'
                f'border-radius:10px;border-left:4px solid {color};">'
                f'<div style="color:{color};font-size:13px;">{region}</div>'
                f'<div style="color:white;font-size:30px;font-weight:700;">{dice:.4f}</div>'
                f'<div style="color:#888;">{grade}</div></div>',
                unsafe_allow_html=True,
            )
