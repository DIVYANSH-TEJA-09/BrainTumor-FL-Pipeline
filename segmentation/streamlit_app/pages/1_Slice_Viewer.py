"""
Slice-by-Slice Segmentation Viewer
====================================
View MRI slices with ground truth and AI prediction overlay.
Supports all 4 modalities and 3 tumor sub-regions.
"""

import streamlit as st
import os
import sys
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(page_title="Slice Viewer", layout="wide")

# ─── paths & inference ───────────────────────────────────────────────────
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
from inference import ensure_prediction, get_all_patients, DEMO_DIR


def load_nifti(path):
    if not os.path.exists(path):
        return None
    return nib.load(path).get_fdata()


# ─── sidebar ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Slice Viewer Controls")

samples = get_all_patients()
if not samples:
    st.error("No demo data found. Please ensure patient volumes exist in demo_data/")
    st.stop()

selected_id = st.sidebar.selectbox("🧑‍⚕️ Patient", samples)

modality = st.sidebar.selectbox(
    "MRI Modality",
    ["FLAIR", "T1", "T1ce", "T2"],
    index=0,
)
MOD_MAP = {"T1": 0, "T1ce": 1, "T2": 2, "FLAIR": 3}

overlay = st.sidebar.radio(
    "Overlay",
    ["AI Prediction", "Ground Truth", "Both (Side-by-Side)", "None"],
    index=0,
)

overlay_alpha = st.sidebar.slider("Overlay Opacity", 0.1, 0.9, 0.5, 0.05)

# ─── load data (run inference if needed) ─────────────────────────────────
ensure_prediction(selected_id)

img_path = os.path.join(DEMO_DIR, f"{selected_id}_image.nii.gz")
pred_path = os.path.join(DEMO_DIR, f"{selected_id}_pred.nii.gz")
lbl_path = os.path.join(DEMO_DIR, f"{selected_id}_label.nii.gz")

img_data = load_nifti(img_path)    # (D, H, W, 4)
pred_data = load_nifti(pred_path)  # (D, H, W, 3) channels: 0=TC, 1=WT, 2=ET
lbl_data = load_nifti(lbl_path)    # (D, H, W) labels: 1=NCR, 2=ED, 4=ET

if img_data is None:
    st.error("Failed to load MRI volume.")
    st.stop()

depth = img_data.shape[0]

# ─── title ───────────────────────────────────────────────────────────────
st.title("🔬 Slice-by-Slice Segmentation Viewer")
st.markdown(f"**Patient:** `{selected_id}` · **Modality:** {modality} · "
            f"**Volume:** {img_data.shape[0]}×{img_data.shape[1]}×{img_data.shape[2]}")

slice_idx = st.slider("Z-Axis Slice", 0, depth - 1, depth // 2)

# ─── color maps ──────────────────────────────────────────────────────────
# Tumor overlay: WT=green, TC=red, ET=yellow (matching 3D view)
TUMOR_COLORS = np.array([
    [0, 0, 0, 0],            # background (transparent)
    [0.18, 0.80, 0.44, 1],   # WT - green
    [0.91, 0.30, 0.24, 1],   # TC - red
    [0.95, 0.77, 0.06, 1],   # ET - gold
])


def make_overlay_from_pred(pred_slice):
    """Convert (H, W, 3) prediction channels to (H, W, 4) RGBA overlay."""
    h, w = pred_slice.shape[:2]
    overlay_img = np.zeros((h, w, 4))
    # Order: WT first (background), then TC, then ET on top
    wt = pred_slice[:, :, 1] > 0.5
    tc = pred_slice[:, :, 0] > 0.5
    et = pred_slice[:, :, 2] > 0.5
    overlay_img[wt] = TUMOR_COLORS[1]
    overlay_img[tc] = TUMOR_COLORS[2]
    overlay_img[et] = TUMOR_COLORS[3]
    return overlay_img


def make_overlay_from_gt(lbl_slice):
    """Convert integer label slice to (H, W, 4) RGBA overlay."""
    h, w = lbl_slice.shape
    overlay_img = np.zeros((h, w, 4))
    wt = lbl_slice > 0
    tc = (lbl_slice == 1) | (lbl_slice == 4)
    et = lbl_slice == 4
    overlay_img[wt] = TUMOR_COLORS[1]
    overlay_img[tc] = TUMOR_COLORS[2]
    overlay_img[et] = TUMOR_COLORS[3]
    return overlay_img


def render_slice(mri_slice, overlay_img, title, alpha):
    """Render an MRI slice with optional overlay."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mri_slice, cmap="gray", origin="lower")
    if overlay_img is not None:
        ax.imshow(overlay_img, alpha=alpha, origin="lower")
    ax.set_title(title, fontsize=14, color="white", fontweight="bold")
    ax.axis("off")
    fig.patch.set_facecolor("#0a0a14")
    return fig


# ─── render ──────────────────────────────────────────────────────────────
mri_slice = img_data[slice_idx, :, :, MOD_MAP[modality]]

if overlay == "None":
    fig = render_slice(mri_slice, None, f"{modality} — Slice {slice_idx}", 0)
    st.pyplot(fig)

elif overlay == "AI Prediction":
    if pred_data is not None:
        ov = make_overlay_from_pred(pred_data[slice_idx])
        fig = render_slice(mri_slice, ov, f"AI Prediction — Slice {slice_idx}",
                           overlay_alpha)
        st.pyplot(fig)
    else:
        st.warning("Prediction not available for this patient.")

elif overlay == "Ground Truth":
    if lbl_data is not None:
        ov = make_overlay_from_gt(lbl_data[slice_idx])
        fig = render_slice(mri_slice, ov, f"Ground Truth — Slice {slice_idx}",
                           overlay_alpha)
        st.pyplot(fig)
    else:
        st.warning("Ground truth not available.")

elif overlay == "Both (Side-by-Side)":
    col1, col2 = st.columns(2)
    with col1:
        if lbl_data is not None:
            ov = make_overlay_from_gt(lbl_data[slice_idx])
            fig = render_slice(mri_slice, ov, "Ground Truth", overlay_alpha)
            st.pyplot(fig)
        else:
            st.info("No ground truth available.")
    with col2:
        if pred_data is not None:
            ov = make_overlay_from_pred(pred_data[slice_idx])
            fig = render_slice(mri_slice, ov, "AI Prediction", overlay_alpha)
            st.pyplot(fig)
        else:
            st.info("No prediction available.")

# ─── all modalities row ──────────────────────────────────────────────────
with st.expander("📋 All 4 Modalities (this slice)", expanded=False):
    cols = st.columns(4)
    mod_names = ["T1", "T1ce", "T2", "FLAIR"]
    for i, col in enumerate(cols):
        with col:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img_data[slice_idx, :, :, i], cmap="gray", origin="lower")
            ax.set_title(mod_names[i], fontsize=11, color="white")
            ax.axis("off")
            fig.patch.set_facecolor("#0a0a14")
            st.pyplot(fig)

# ─── color legend ────────────────────────────────────────────────────────
st.markdown("---")
legend_cols = st.columns(3)
labels = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
colors = ["#2ECC71", "#E74C3C", "#F1C40F"]
for i, col in enumerate(legend_cols):
    with col:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="width:16px;height:16px;background:{colors[i]};'
            f'border-radius:3px;"></div>'
            f'<span>{labels[i]}</span></div>',
            unsafe_allow_html=True,
        )
