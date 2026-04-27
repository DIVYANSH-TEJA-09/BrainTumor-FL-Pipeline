
"""
Tumor Progression Forecasting — Reviewer Dashboard
====================================================

A Streamlit application for tumor growth prediction visualization.
Designed to demonstrate and validate three forecasting approaches:

  1. Mathematical Model (Logistic Growth) — interpretable, clinically motivated
  2. LSTM Hybrid (Residual Correction) — deep learning enhancement over baseline
  3. 3D U-Net Spatial Prediction — predicts WHERE growth will occur

Core workflow:
  - Select a patient and an input scan (timepoint)
  - App projects what the next scan should look like
  - Ground truth of the next scan is shown for verification
  - Side-by-side comparison of math vs LSTM vs spatial predictions
  - Full method explanation for reviewer clarity

Author: FL-QPSO Brain Tumor Management System — Module 3
"""

import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─────────────────────────── Page Config ───────────────────────────

st.set_page_config(
    page_title="Tumor Progression Forecasting",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Light Professional CSS ────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #f7f8fc; }

    /* Title */
    .main-title {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0; letter-spacing: -0.5px;
    }
    .subtitle { color: #6b7280; font-size: 0.95rem; margin-top: 0; margin-bottom: 1.5rem; }

    /* Cards */
    .glass-card {
        background: #ffffff; border: 1px solid #e5e7eb;
        border-radius: 14px; padding: 1.1rem 1.2rem;
        margin-bottom: 0.8rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        transition: box-shadow 0.25s ease;
    }
    .glass-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
    .card-label {
        font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 1.1px; color: #9ca3af; font-weight: 600; margin-bottom: 0.35rem;
    }
    .card-value { font-size: 1.5rem; font-weight: 700; color: #111827; }
    .card-delta { font-size: 0.82rem; font-weight: 500; margin-top: 0.15rem; }
    .delta-positive { color: #059669; }
    .delta-negative { color: #dc2626; }
    .delta-neutral  { color: #6b7280; }

    /* Method explanation cards */
    .method-card {
        background: #ffffff; border: 1px solid #e5e7eb;
        border-radius: 14px; padding: 1.2rem; margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .method-title { font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; }
    .method-formula {
        font-family: 'Courier New', monospace; background: #f3f4f6;
        border-radius: 8px; padding: 0.8rem; color: #1e40af;
        font-size: 0.84rem; margin: 0.5rem 0; overflow-x: auto;
        border: 1px solid #e5e7eb;
    }
    .param-grid {
        display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
        gap: 0.5rem; margin-top: 0.5rem;
    }
    .param-item {
        background: #f9fafb; border-radius: 8px; padding: 0.45rem 0.6rem;
        border: 1px solid #f3f4f6;
    }
    .param-name { color: #9ca3af; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.8px; }
    .param-val  { color: #111827; font-size: 1rem; font-weight: 600; }

    /* Section headers */
    .section-header {
        font-size: 1.15rem; font-weight: 600; color: #374151;
        border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem;
        margin: 1.8rem 0 1rem;
    }

    /* Info/Warning banners */
    .info-banner {
        background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
        padding: 0.8rem 1rem; margin: 0.5rem 0; color: #1e40af;
        font-size: 0.85rem;
    }
    .warn-banner {
        background: #fef3c7; border: 1px solid #fcd34d; border-radius: 10px;
        padding: 0.8rem 1rem; margin: 0.5rem 0; color: #92400e;
        font-size: 0.85rem;
    }

    /* Panel header colors */
    .panel-actual  { color: #2563eb; }
    .panel-math    { color: #d97706; }
    .panel-lstm    { color: #059669; }
    .panel-truth   { color: #7c3aed; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff; border-right: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Paths ─────────────────────────────────

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "raw" / "mu_glioma_post"
RESULTS_DIR = APP_DIR / "streamlit_data"
PRED_INDEX_FILE = RESULTS_DIR / "prediction_index.json"
SPATIAL_MODEL_PATH = APP_DIR / "results" / "spatial_unet_best.pth"
SPATIAL_EVAL_PATH = APP_DIR / "results" / "spatial_eval.json"
SPATIAL_CROP = (96, 96, 64)

# ─────────────────────────── Data Loading ──────────────────────────

@st.cache_data
def load_prediction_index():
    if PRED_INDEX_FILE.exists():
        with open(PRED_INDEX_FILE) as f:
            return json.load(f)
    return None

@st.cache_data
def load_nifti(path):
    if not HAS_NIBABEL or not os.path.exists(path):
        return None
    try:
        return nib.load(path).get_fdata()
    except Exception:
        return None

def find_patient_dir(patient_id: str):
    d = DATA_DIR / "MU-Glioma-Post" / patient_id
    return d if d.exists() else None

def get_mask_path(patient_dir, tp_idx: int):
    dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir() and 'Timepoint' in d.name])
    if tp_idx < len(dirs):
        masks = list(dirs[tp_idx].glob('*tumorMask.nii.gz'))
        if masks:
            return masks[0]
    return None

def get_brain_path(patient_dir, tp_idx: int):
    dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir() and 'Timepoint' in d.name])
    if tp_idx < len(dirs):
        for pat in ['*_t1c.nii.gz', '*_T1c.nii.gz', '*_t1ce.nii.gz', '*brain*.nii.gz']:
            imgs = list(dirs[tp_idx].glob(pat))
            if imgs:
                return imgs[0]
    return None

# ─────────────────────────── 3D Mesh Helpers ───────────────────────

def extract_mesh(vol, level=0.5, step=2):
    if not HAS_SKIMAGE:
        return None
    v = vol[::step, ::step, ::step]
    if v.sum() == 0:
        return None
    try:
        verts, faces, _, _ = marching_cubes(v, level=level)
        return verts * step, faces
    except Exception:
        return None

def make_mesh_trace(vol, color, name, opacity, step=2):
    r = extract_mesh(vol, step=step)
    if r is None:
        return None
    verts, faces = r
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, flatshading=True,
        lighting=dict(ambient=0.65, diffuse=0.7, specular=0.2, roughness=0.5),
        lightposition=dict(x=100, y=200, z=300),
    )

def build_brain_trace(brain_img, opacity, step):
    if brain_img is None:
        return None
    norm = (brain_img - brain_img.min()) / (brain_img.max() - brain_img.min() + 1e-8)
    mask = (norm > 0.15).astype(float)
    return make_mesh_trace(mask, '#c4c4cc', 'Brain', opacity, max(step, 3))

def scale_mask(mask, v_actual, v_pred):
    if v_actual <= 0 or v_pred <= 0:
        return mask.copy()
    return mask * (v_pred / v_actual)

def make_3d_fig(traces, height=440):
    fig = go.Figure(data=[t for t in traces if t is not None])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="#f0f0f5", aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=0.8, z=0.6), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig

# ─────────────────────────── Logistic function ─────────────────────

def logistic_curve(t, v0, k, r):
    with np.errstate(over='ignore'):
        denom = 1 + ((k - v0) / v0) * np.exp(-r * t)
        denom = np.clip(denom, 1e-10, None)
    return k / denom


# ─────────────────────────── Spatial U-Net ─────────────────────────

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.block(x)


class SpatialUNet3D(nn.Module):
    def __init__(self, base=16, dropout=0.15):
        super().__init__()
        self.enc1 = ConvBlock3D(1, base)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.enc4 = ConvBlock3D(base * 4, base * 8, dropout)
        self.pool = nn.MaxPool3d(2)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock3D(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)
        self.out_conv = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        orig = x.shape[2:]
        pad = []
        for s in reversed(orig):
            d = (8 - s % 8) % 8
            pad.extend([0, d])
        if any(p > 0 for p in pad):
            x = F.pad(x, pad)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self._cat(self.up3(e4), e3); d3 = self.dec3(d3)
        d2 = self._cat(self.up2(d3), e2); d2 = self.dec2(d2)
        d1 = self._cat(self.up1(d2), e1); d1 = self.dec1(d1)
        out = torch.sigmoid(self.out_conv(d1))
        return out[:, :, :orig[0], :orig[1], :orig[2]]

    def _cat(self, up, skip):
        diff = [s - u for s, u in zip(skip.shape[2:], up.shape[2:])]
        if any(d != 0 for d in diff):
            up = F.pad(up, [0, diff[2], 0, diff[1], 0, diff[0]])
        return torch.cat([up, skip], dim=1)


@st.cache_resource
def load_spatial_model():
    """Load the trained 3D U-Net for spatial prediction."""
    if not HAS_TORCH:
        st.sidebar.warning("PyTorch not installed — spatial prediction unavailable.")
        return None
    if not SPATIAL_MODEL_PATH.exists():
        st.sidebar.warning(f"Model not found at: {SPATIAL_MODEL_PATH}")
        return None
    try:
        ckpt = torch.load(SPATIAL_MODEL_PATH, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        model = SpatialUNet3D(base=cfg.get('base_filters', 16), dropout=cfg.get('dropout', 0.15))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.sidebar.error(f"Spatial model load error: {e}")
        return None


def crop_around_tumor(mask, crop_size=SPATIAL_CROP):
    """Crop a full-size mask to the tumor region for U-Net input."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        center = np.array(mask.shape) // 2
    else:
        center = coords.mean(axis=0).astype(int)
    d, h, w = crop_size
    starts = [center[0] - d // 2, center[1] - h // 2, center[2] - w // 2]
    ends = [s + sz for s, sz in zip(starts, crop_size)]
    pad_before = [max(0, -s) for s in starts]
    pad_after = [max(0, e - mask.shape[i]) for i, e in enumerate(ends)]
    starts = [max(0, s) for s in starts]
    ends = [min(mask.shape[i], e) for i, e in enumerate(ends)]
    cropped = mask[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    if any(p > 0 for p in pad_before + pad_after):
        cropped = np.pad(cropped, [(pad_before[i], pad_after[i]) for i in range(3)],
                         mode='constant', constant_values=0)
    return cropped, center, starts, ends, pad_before, pad_after


def uncrop_prediction(pred_crop, full_shape, starts, ends, pad_before, pad_after):
    """Place the cropped prediction back into full volume coordinates."""
    # Remove padding from prediction
    crop_size = pred_crop.shape
    slices = [
        slice(pad_before[i], crop_size[i] - pad_after[i] if pad_after[i] > 0 else crop_size[i])
        for i in range(3)
    ]
    unpadded = pred_crop[slices[0], slices[1], slices[2]]
    full = np.zeros(full_shape, dtype=np.float32)
    full[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]] = unpadded
    return full


@torch.no_grad()
def predict_spatial(model, mask_input):
    """Run spatial prediction: input mask -> predicted next mask."""
    if model is None:
        return None
    binary = (mask_input > 0).astype(np.float32)
    cropped, center, starts, ends, pb, pa = crop_around_tumor(binary)
    x = torch.from_numpy(cropped[np.newaxis, np.newaxis].astype(np.float32))
    pred = model(x)
    pred_crop = (pred[0, 0].numpy() > 0.5).astype(np.float32)
    return uncrop_prediction(pred_crop, mask_input.shape, starts, ends, pb, pa)


def load_spatial_eval():
    if SPATIAL_EVAL_PATH.exists():
        with open(SPATIAL_EVAL_PATH) as f:
            return json.load(f)
    return None


# ═══════════════════════════ MAIN APP ══════════════════════════════

pred_index = load_prediction_index()
if pred_index is None:
    st.error("Prediction data not found. Run: `python src/09_generate_enhanced_viz_data.py`")
    st.stop()

patients = pred_index['patients']
eval_metrics = pred_index.get('eval_metrics', {})

# ─── Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="main-title" style="font-size:1.3rem;">🧠 Controls</p>', unsafe_allow_html=True)

    all_pids = sorted(patients.keys())
    hgg_pids = [p for p in all_pids if patients[p]['grade'] == 'HGG']
    lgg_pids = [p for p in all_pids if patients[p]['grade'] == 'LGG']

    grade_filter = st.radio("Tumor Grade", ["All", "HGG", "LGG"], horizontal=True)
    if grade_filter == "HGG":
        pool = hgg_pids
    elif grade_filter == "LGG":
        pool = lgg_pids
    else:
        pool = all_pids

    pid = st.selectbox("Patient", pool, index=0)
    pdata = patients[pid]
    grade = pdata['grade']
    n_tp = pdata['n_timepoints']

    st.markdown("---")
    st.markdown(f"**Grade:** `{grade}` &nbsp; **Timepoints:** `{n_tp}`")

    # Input timepoint (predict FROM this scan)
    max_from = n_tp - 2
    if max_from < 0:
        st.warning("Only 1 timepoint — cannot predict growth.")
        st.stop()

    from_idx = st.slider("Input Scan (predict FROM)", 0, max_from, 0)
    to_idx = from_idx + 1

    tp_from = pdata['timepoints'][from_idx]
    tp_to = pdata['timepoints'][to_idx]

    day_from = tp_from.get('day_from_diagnosis', from_idx * 30)
    day_to = tp_to.get('day_from_diagnosis', to_idx * 30)
    day_delta = abs(day_to - day_from)

    st.markdown(f"**Predicting:** Day `{day_from:.0f}` → Day `{day_to:.0f}` (`{day_delta:.0f}` days)")

    # LSTM status indicator
    lstm_active = to_idx >= 3
    if lstm_active:
        st.success("🧠 LSTM active — has 3+ prior residuals")
    else:
        st.warning(f"⚠️ LSTM inactive — needs 3 prior timepoints, only has {to_idx}")

    st.markdown("---")
    st.markdown("**3D Options**")
    show_brain = st.checkbox("Brain overlay", value=True)
    show_spatial = st.checkbox("Spatial prediction (U-Net)", value=True)
    brain_op = st.slider("Brain opacity", 0.02, 0.20, 0.06, 0.02)
    tumor_op = st.slider("Tumor opacity", 0.3, 1.0, 0.75)
    mesh_step = st.select_slider("Mesh quality", [1, 2, 3, 4], 2)

# ─── Title ─────────────────────────────────────────────────────────

st.markdown('<h1 class="main-title">Tumor Progression Forecasting</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Given one MRI scan, predict the next — then validate against ground truth. Volume prediction (Math + LSTM) and spatial growth prediction (3D U-Net), compared side by side.</p>', unsafe_allow_html=True)

# Load spatial model
spatial_model = load_spatial_model() if show_spatial else None
spatial_eval = load_spatial_eval()

# ─── Data for this prediction ─────────────────────────────────────

v_from = tp_from['v_actual']
v_to_actual = tp_to['v_actual']
v_to_math = tp_to['v_logistic']
v_to_lstm = tp_to['v_hybrid']
mae_math = abs(v_to_actual - v_to_math)
mae_lstm = abs(v_to_actual - v_to_lstm)
improvement = ((mae_math - mae_lstm) / mae_math * 100) if mae_math > 0 else 0.0
growth_pct = ((v_to_actual - v_from) / v_from * 100) if v_from > 0 else 0.0

# LSTM transparency
lstm_correction = tp_to.get('lstm_correction', 0)
is_tied = abs(v_to_math - v_to_lstm) < 1.0  # effectively same

# ─── LSTM Explanation Banner ──────────────────────────────────────

if is_tied:
    st.markdown(f"""
    <div class="warn-banner">
        ⚠️ <strong>Math = LSTM for this timepoint.</strong> The LSTM requires 3 prior residuals
        (lookback window) before it can compute a correction. Timepoint {to_idx} only has
        {to_idx} prior points, so the LSTM correction is <strong>0 mm³</strong>.
        The LSTM starts contributing at timepoint index ≥ 3.
        {f'Try moving the slider to a later timepoint to see LSTM in action.' if n_tp > 4 else
         f'This patient has only {n_tp} scans — LSTM cannot contribute.'}
    </div>
    """, unsafe_allow_html=True)

# ─── Quick Stats Row ──────────────────────────────────────────────

cols = st.columns(5)
stat_data = [
    ("Input Volume", f"{v_from:,.0f} mm³", f"Day {day_from:.0f}", "neutral"),
    ("Actual Growth", f"{growth_pct:+.1f}%", f"{v_to_actual - v_from:+,.0f} mm³",
     "positive" if growth_pct > 0 else "negative"),
    ("Math Prediction", f"{v_to_math:,.0f} mm³", f"Error: {mae_math:,.0f}", "neutral"),
    ("LSTM Prediction", f"{v_to_lstm:,.0f} mm³",
     f"Error: {mae_lstm:,.0f}" + (" (= Math)" if is_tied else ""), "neutral"),
    ("LSTM Advantage",
     f"{improvement:+.1f}%" if not is_tied else "N/A",
     "LSTM inactive" if is_tied else ("lower error" if improvement > 0 else "higher error"),
     "neutral" if is_tied else ("positive" if improvement > 0 else "negative")),
]

for col, (label, value, delta, cls) in zip(cols, stat_data):
    col.markdown(f"""
    <div class="glass-card">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        <div class="card-delta delta-{cls}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── 3D Prediction Panels ─────────────────────────────────────────

st.markdown('<div class="section-header">🔬 Scan-to-Scan Prediction</div>', unsafe_allow_html=True)
st.markdown(f"**Input:** Scan at day {day_from:.0f} → **Predict:** Scan at day {day_to:.0f} ({day_delta:.0f} days later)")

# Spatial prediction info
if spatial_model is not None:
    sp_dice = spatial_eval.get('best_dice', 0) if spatial_eval else 0
    st.markdown(f"""
    <div class="info-banner">
        🧠 <strong>Spatial U-Net active</strong> — predicts WHERE growth occurs (Dice: {sp_dice:.3f}).
        Blue = stable tumor, Red = predicted growth, Green = predicted regression.
    </div>
    """, unsafe_allow_html=True)
elif show_spatial:
    st.markdown('<div class="warn-banner">Spatial U-Net model not found. Run training first.</div>', unsafe_allow_html=True)

patient_dir = find_patient_dir(pid)
can_render_3d = patient_dir is not None and HAS_NIBABEL and HAS_SKIMAGE

if can_render_3d:
    mask_from_path = get_mask_path(patient_dir, from_idx)
    mask_to_path = get_mask_path(patient_dir, to_idx)
    brain_path = get_brain_path(patient_dir, from_idx)

    mask_from = load_nifti(str(mask_from_path)) if mask_from_path else None
    mask_to = load_nifti(str(mask_to_path)) if mask_to_path else None
    brain_img = load_nifti(str(brain_path)) if brain_path else None

    if mask_from is not None:
        # Spatial prediction
        mask_spatial = None
        if spatial_model is not None:
            mask_spatial = predict_spatial(spatial_model, mask_from)

        has_spatial = mask_spatial is not None

        # 3-panel layout: Input -> Spatial Prediction -> Ground Truth
        c1, c2, c3 = st.columns(3)

        # Panel 1: Input Scan
        with c1:
            st.markdown('<p class="card-label panel-actual">📌 Input Scan (Current)</p>', unsafe_allow_html=True)
            traces = []
            if show_brain and brain_img is not None:
                traces.append(build_brain_trace(brain_img, brain_op, mesh_step))
            traces.append(make_mesh_trace(mask_from, '#3b82f6', 'Input', tumor_op, mesh_step))
            st.plotly_chart(make_3d_fig(traces, height=500), use_container_width=True, key="fig_input")
            st.markdown(f"""<div class="glass-card">
                <div class="card-label">Volume</div>
                <div class="card-value">{v_from:,.0f} mm\u00b3</div>
                <div class="card-delta delta-neutral">Day {day_from:.0f}</div>
            </div>""", unsafe_allow_html=True)

        # Panel 2: Spatial U-Net Prediction
        with c2:
            if has_spatial:
                st.markdown('<p class="card-label" style="color:#e11d48;">Spatial Prediction (3D U-Net)</p>', unsafe_allow_html=True)
                traces = []
                if show_brain and brain_img is not None:
                    traces.append(build_brain_trace(brain_img, brain_op, mesh_step))

                input_bin = (mask_from > 0).astype(float)
                pred_bin = (mask_spatial > 0.5).astype(float)
                stable = input_bin * pred_bin
                new_growth = np.clip(pred_bin - input_bin, 0, 1)
                regression = np.clip(input_bin - pred_bin, 0, 1)

                t_stable = make_mesh_trace(stable, '#3b82f6', 'Stable', tumor_op * 0.7, mesh_step)
                t_growth = make_mesh_trace(new_growth, '#ef4444', 'Growth', tumor_op, mesh_step)
                t_regress = make_mesh_trace(regression, '#22c55e', 'Regression', tumor_op * 0.6, mesh_step)
                traces.extend([t for t in [t_stable, t_growth, t_regress] if t is not None])

                st.plotly_chart(make_3d_fig(traces, height=500), use_container_width=True, key="fig_spatial")

                sp_dice = 0.0
                if mask_to is not None:
                    gt_bin = (mask_to > 0).astype(float)
                    inter = (pred_bin * gt_bin).sum()
                    total = pred_bin.sum() + gt_bin.sum()
                    sp_dice = 2 * inter / total if total > 0 else 1.0

                n_growth_vox = int(new_growth.sum())
                n_regress_vox = int(regression.sum())
                st.markdown(f"""<div class="glass-card" style="border-left: 4px solid #e11d48;">
                    <div class="card-label">Predicted Shape (Dice vs Ground Truth)</div>
                    <div class="card-value" style="color:#e11d48;">{sp_dice:.3f}</div>
                    <div class="card-delta delta-neutral">
                        <span style="color:#ef4444;">&#9632; +{n_growth_vox:,} growth</span>
                        <span style="color:#3b82f6;">&#9632; stable</span>
                        <span style="color:#22c55e;">&#9632; -{n_regress_vox:,} regr.</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                mask_math_scaled = scale_mask(mask_from, v_from, v_to_math)
                st.markdown('<p class="card-label" style="color:#d97706;">Volume-Scaled Prediction</p>', unsafe_allow_html=True)
                traces = []
                if show_brain and brain_img is not None:
                    traces.append(build_brain_trace(brain_img, brain_op, mesh_step))
                traces.append(make_mesh_trace(mask_math_scaled, '#d97706', 'Math', tumor_op, mesh_step))
                st.plotly_chart(make_3d_fig(traces, height=500), use_container_width=True, key="fig_fallback")
                st.markdown(f"""<div class="glass-card">
                    <div class="card-label">Predicted Volume (Math)</div>
                    <div class="card-value">{v_to_math:,.0f} mm\u00b3</div>
                    <div class="card-delta delta-neutral">Spatial model unavailable</div>
                </div>""", unsafe_allow_html=True)

        # Panel 3: Ground Truth
        with c3:
            st.markdown('<p class="card-label panel-truth">Ground Truth (Actual Next Scan)</p>', unsafe_allow_html=True)
            traces = []
            if show_brain and brain_img is not None:
                traces.append(build_brain_trace(brain_img, brain_op, mesh_step))
            if mask_to is not None:
                traces.append(make_mesh_trace(mask_to, '#7c3aed', 'Ground Truth', tumor_op, mesh_step))
            st.plotly_chart(make_3d_fig(traces, height=500), use_container_width=True, key="fig_truth")
            gp_cls = "positive" if growth_pct > 0 else "negative"
            st.markdown(f"""<div class="glass-card">
                <div class="card-label">Actual Volume</div>
                <div class="card-value">{v_to_actual:,.0f} mm\u00b3</div>
                <div class="card-delta delta-{gp_cls}">{growth_pct:+.1f}% from input</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.warning("Could not load tumor mask for the selected timepoint.")
else:
    # Fallback: volume bars when 3D unavailable
    st.info("3D rendering unavailable (NIfTI data or dependencies not found). Showing volume comparison.")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name='Input', x=['Volume'], y=[v_from], marker_color='#3b82f6'))
    fig_bar.add_trace(go.Bar(name='Math', x=['Volume'], y=[v_to_math], marker_color='#d97706'))
    fig_bar.add_trace(go.Bar(name='LSTM', x=['Volume'], y=[v_to_lstm], marker_color='#059669'))
    fig_bar.add_trace(go.Bar(name='Actual', x=['Volume'], y=[v_to_actual], marker_color='#7c3aed'))
    fig_bar.update_layout(
        barmode='group', height=350,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#f7f8fc',
        font=dict(color='#374151'),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════ METHOD EXPLANATION ═════════════════════

st.markdown('<div class="section-header">📖 Method Explanation (For Reviewer)</div>', unsafe_allow_html=True)

params = pdata.get('logistic_params', {})
v0_p = params.get('v0', 0)
k_p = params.get('k_fit', 0)
r_p = params.get('r_fit', 0)
r2_p = params.get('r2_fit', 0)

col_math, col_lstm, col_unet = st.columns(3)

with col_math:
    st.markdown(f"""
    <div class="method-card">
        <div class="method-title panel-math">📐 Mathematical Model: Logistic Growth</div>
        <p style="color:#6b7280; font-size:0.85rem;">
            A classical population dynamics model fitted per-patient. Captures bounded tumor growth with
            an asymptotic carrying capacity. <strong>Interpretable</strong> — parameters have clinical meaning.
        </p>
        <div class="method-formula">V(t) = K / (1 + ((K - V0) / V0) x e<sup>-r*t</sup>)</div>
        <div class="param-grid">
            <div class="param-item"><div class="param-name">V0 (Initial)</div><div class="param-val">{v0_p:,.0f}</div></div>
            <div class="param-item"><div class="param-name">K (Capacity)</div><div class="param-val">{k_p:,.0f}</div></div>
            <div class="param-item"><div class="param-name">r (Rate/day)</div><div class="param-val">{r_p:.4f}</div></div>
            <div class="param-item"><div class="param-name">R2 (Fit)</div><div class="param-val">{r2_p:.3f}</div></div>
        </div>
        <p style="color:#9ca3af; font-size:0.78rem; margin-top:0.8rem;">
            Predicts HOW MUCH &nbsp;|&nbsp; Interpretable &nbsp;|&nbsp; Cannot predict WHERE
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_lstm:
    corr_display = f"{lstm_correction:+,.0f} mm3" if not is_tied else "0 mm3 (inactive)"
    corr_color = "#059669" if not is_tied else "#9ca3af"
    st.markdown(f"""
    <div class="method-card">
        <div class="method-title panel-lstm">🧠 LSTM Hybrid: Residual Correction</div>
        <p style="color:#6b7280; font-size:0.85rem;">
            A neural network trained on <em>residuals</em> (errors) of the logistic model.
            Learns temporal patterns in where the math model fails, then applies a correction.
            <strong>Grade-stratified</strong>: separate models for HGG and LGG.
        </p>
        <div class="method-formula">V_hybrid(t) = V_logistic(t) + LSTM_correction(t)<br>
LSTM: Input(3 residuals) -> LSTM(32) -> Attention(4-head) -> FC(64->32->1)</div>
        <div class="param-grid">
            <div class="param-item"><div class="param-name">Correction</div><div class="param-val" style="color:{corr_color};">{corr_display}</div></div>
            <div class="param-item"><div class="param-name">Architecture</div><div class="param-val">Attn-LSTM</div></div>
            <div class="param-item"><div class="param-name">Parameters</div><div class="param-val">12,929</div></div>
            <div class="param-item"><div class="param-name">Lookback</div><div class="param-val">3 steps</div></div>
        </div>
        <p style="color:#9ca3af; font-size:0.78rem; margin-top:0.8rem;">
            Predicts HOW MUCH (refined) &nbsp;|&nbsp; Needs 4+ timepoints
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_unet:
    sp_eval = load_spatial_eval()
    sp_dice_val = sp_eval.get('best_dice', 0) if sp_eval else 0
    sp_epochs = sp_eval.get('best_epoch', 0) if sp_eval else 0
    sp_params = sp_eval.get('n_params', 0) if sp_eval else 0
    st.markdown(f"""
    <div class="method-card">
        <div class="method-title" style="color:#e11d48;">🔮 3D U-Net: Spatial Growth Prediction</div>
        <p style="color:#6b7280; font-size:0.85rem;">
            A 3D convolutional neural network trained on <strong>391 consecutive mask pairs</strong> from 155 patients.
            Given the current tumor shape, predicts WHERE growth will occur in the next scan.
        </p>
        <div class="method-formula">Mask(T+1) = UNet3D(Mask(T))<br>
Encoder(4-level) -> Bottleneck -> Decoder(skip connections) -> Sigmoid</div>
        <div class="param-grid">
            <div class="param-item"><div class="param-name">Best Dice</div><div class="param-val" style="color:#e11d48;">{sp_dice_val:.3f}</div></div>
            <div class="param-item"><div class="param-name">Architecture</div><div class="param-val">3D U-Net</div></div>
            <div class="param-item"><div class="param-name">Parameters</div><div class="param-val">{sp_params:,}</div></div>
            <div class="param-item"><div class="param-name">Best Epoch</div><div class="param-val">{sp_epochs}</div></div>
        </div>
        <p style="color:#9ca3af; font-size:0.78rem; margin-top:0.8rem;">
            Predicts WHERE &nbsp;|&nbsp; Shape-aware &nbsp;|&nbsp; Works from single scan
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════ GROWTH CURVE FIT ══════════════════════

st.markdown('<div class="section-header">📈 Logistic Growth Curve Fit</div>', unsafe_allow_html=True)

all_days = [tp.get('day_from_diagnosis', i * 30) for i, tp in enumerate(pdata['timepoints'])]
all_actual = [tp['v_actual'] for tp in pdata['timepoints']]
all_logistic = [tp['v_logistic'] for tp in pdata['timepoints']]
all_hybrid = [tp['v_hybrid'] for tp in pdata['timepoints']]
all_residuals = [tp.get('residual', 0) for tp in pdata['timepoints']]

# Smooth logistic curve
if k_p > 0 and r_p > 0:
    t_min = min(all_days)
    t_max = max(all_days) * 1.2
    t_smooth = np.linspace(max(0, t_min - 30), t_max, 200)
    v_smooth = logistic_curve(t_smooth, v0_p, k_p, r_p)
else:
    t_smooth = np.array(all_days)
    v_smooth = np.array(all_logistic)

fig_curve = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Volume Trajectory & Fitted Curve", "Residuals (Actual − Predicted)"),
    horizontal_spacing=0.08,
)

# Left: trajectory + fitted curve
fig_curve.add_trace(go.Scatter(
    x=t_smooth.tolist(), y=v_smooth.tolist(), mode='lines', name='Logistic Fit',
    line=dict(color='#d97706', width=2, dash='dot'),
), row=1, col=1)

fig_curve.add_trace(go.Scatter(
    x=all_days, y=all_actual, mode='lines+markers', name='Actual',
    line=dict(color='#3b82f6', width=3), marker=dict(size=9, symbol='circle'),
), row=1, col=1)

fig_curve.add_trace(go.Scatter(
    x=all_days, y=all_logistic, mode='markers', name='Math Predicted',
    marker=dict(size=8, symbol='diamond', color='#d97706', line=dict(width=1, color='#f59e0b')),
), row=1, col=1)

fig_curve.add_trace(go.Scatter(
    x=all_days, y=all_hybrid, mode='markers', name='LSTM Hybrid',
    marker=dict(size=8, symbol='star', color='#059669', line=dict(width=1, color='#34d399')),
), row=1, col=1)

# Highlight current prediction pair
fig_curve.add_trace(go.Scatter(
    x=[day_from], y=[v_from], mode='markers', name='Input Scan',
    marker=dict(size=14, color='#3b82f6', symbol='circle-open', line=dict(width=3)),
    showlegend=False,
), row=1, col=1)

fig_curve.add_trace(go.Scatter(
    x=[day_to], y=[v_to_actual], mode='markers', name='Target',
    marker=dict(size=14, color='#7c3aed', symbol='star-open', line=dict(width=3)),
    showlegend=False,
), row=1, col=1)

fig_curve.add_annotation(
    x=day_to, y=v_to_actual, ax=day_from, ay=v_from,
    xref="x", yref="y", axref="x", ayref="y",
    showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2,
    arrowcolor="#7c3aed", opacity=0.5,
    row=1, col=1,
)

# Right: residuals
bar_colors = ['#dc2626' if r < 0 else '#059669' for r in all_residuals]
fig_curve.add_trace(go.Bar(
    x=all_days, y=all_residuals, name='Residual',
    marker=dict(color=bar_colors, opacity=0.8),
    showlegend=False,
), row=1, col=2)
fig_curve.add_hline(y=0, line_dash="dash", line_color="#9ca3af", row=1, col=2)

fig_curve.update_layout(
    height=400,
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff',
    font=dict(color='#374151', family='Inter'),
    legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#e5e7eb', font=dict(size=11)),
    margin=dict(l=50, r=30, t=40, b=50),
)

for i in [1, 2]:
    fig_curve.update_xaxes(title_text="Days from Diagnosis", row=1, col=i, gridcolor='#f3f4f6', zeroline=False)
fig_curve.update_yaxes(title_text="Volume (mm³)", row=1, col=1, gridcolor='#f3f4f6')
fig_curve.update_yaxes(title_text="Residual (mm³)", row=1, col=2, gridcolor='#f3f4f6')

st.plotly_chart(fig_curve, use_container_width=True, key="fig_curve")


# ═══════════════════════════ ERROR COMPARISON ══════════════════════

st.markdown('<div class="section-header">📊 Per-Timepoint Error Comparison</div>', unsafe_allow_html=True)

comp_data = []
for i, tp in enumerate(pdata['timepoints']):
    tp_tied = abs(tp['v_logistic'] - tp['v_hybrid']) < 1.0
    comp_data.append({
        'TP': i,
        'Day': tp.get('day_from_diagnosis', i * 30),
        'Actual (mm³)': tp['v_actual'],
        'Math Pred': tp['v_logistic'],
        'LSTM Pred': tp['v_hybrid'],
        'Math Error': tp['mae_baseline'],
        'LSTM Error': tp['mae_hybrid'],
        'LSTM Active': '✅' if not tp_tied else '❌ (< 3 history)',
        'Winner': '🧠 LSTM' if tp['mae_hybrid'] < tp['mae_baseline'] - 1 else '📐 Math' if tp['mae_baseline'] < tp['mae_hybrid'] - 1 else '🤝 Tied',
    })

comp_df = pd.DataFrame(comp_data)
col_table, col_bar = st.columns([3, 2])

with col_table:
    st.dataframe(
        comp_df.style.format({
            'Actual (mm³)': '{:,.0f}', 'Math Pred': '{:,.0f}', 'LSTM Pred': '{:,.0f}',
            'Math Error': '{:,.0f}', 'LSTM Error': '{:,.0f}', 'Day': '{:.0f}',
        }),
        use_container_width=True,
        height=min(400, 80 + 35 * len(comp_df)),
    )

with col_bar:
    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(
        x=[f"T{d['TP']}" for d in comp_data],
        y=[d['Math Error'] for d in comp_data],
        name='Math Error', marker_color='#d97706', opacity=0.85,
    ))
    fig_err.add_trace(go.Bar(
        x=[f"T{d['TP']}" for d in comp_data],
        y=[d['LSTM Error'] for d in comp_data],
        name='LSTM Error', marker_color='#059669', opacity=0.85,
    ))
    fig_err.update_layout(
        barmode='group', height=min(400, 80 + 35 * len(comp_df)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff',
        font=dict(color='#374151'), margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', font=dict(size=10)),
        xaxis=dict(gridcolor='#f3f4f6'),
        yaxis=dict(title='Error (mm³)', gridcolor='#f3f4f6'),
    )
    st.plotly_chart(fig_err, use_container_width=True, key="fig_err")


# ═══════════════════════════ OVERALL MODEL STATS ═══════════════════

st.markdown('<div class="section-header">🏆 Overall Model Performance</div>', unsafe_allow_html=True)

hgg_m = eval_metrics.get('by_grade', {}).get('HGG', {})
lgg_m = eval_metrics.get('by_grade', {}).get('LGG', {})
overall_m = eval_metrics.get('overall', {})

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #059669;">
        <div class="card-label">HGG (High-Grade) — {hgg_m.get('n_points', 0)} observations</div>
        <div class="card-value" style="font-size:1.3rem; color:#059669;">MAE ↓ {hgg_m.get('mae_improvement_pct', 0):.1f}%</div>
        <div class="card-delta delta-positive">
            Baseline: {hgg_m.get('mae_baseline', 0):,.0f} → Hybrid: {hgg_m.get('mae_hybrid', 0):,.0f} mm³<br>
            R² improved: {hgg_m.get('r2_baseline', 0):.3f} → {hgg_m.get('r2_hybrid', 0):.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #d97706;">
        <div class="card-label">LGG (Low-Grade) — {lgg_m.get('n_points', 0)} observations</div>
        <div class="card-value" style="font-size:1.3rem; color:#d97706;">MAE {lgg_m.get('mae_improvement_pct', 0):+.1f}%</div>
        <div class="card-delta delta-neutral">
            Baseline: {lgg_m.get('mae_baseline', 0):,.0f} → Hybrid: {lgg_m.get('mae_hybrid', 0):,.0f} mm³<br>
            LSTM marginal on slow-growing tumors (expected)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #3b82f6;">
        <div class="card-label">Overall — {overall_m.get('n_points', 0)} observations</div>
        <div class="card-value" style="font-size:1.3rem; color:#3b82f6;">MAE ↓ {overall_m.get('mae_improvement_pct', 0):.1f}%</div>
        <div class="card-delta delta-neutral">
            111 patients (89 HGG, 22 LGG) &nbsp;|&nbsp; Temporal cross-validation<br>
            LSTM adds value on HGG; neutral on LGG
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Patient Summary ──────────────────────────────────────────────

st.markdown('<div class="section-header">📋 Patient Summary</div>', unsafe_allow_html=True)

p_mae_base = pdata['mae_baseline_mean']
p_mae_hyb = pdata['mae_hybrid_mean']
p_imp = ((p_mae_base - p_mae_hyb) / p_mae_base * 100) if p_mae_base > 0 else 0

cs1, cs2, cs3, cs4 = st.columns(4)
cs1.markdown(f"""<div class="glass-card">
    <div class="card-label">Patient</div>
    <div class="card-value" style="font-size:1.1rem;">{pid}</div>
    <div class="card-delta delta-neutral">{grade} • {n_tp} scans</div>
</div>""", unsafe_allow_html=True)
cs2.markdown(f"""<div class="glass-card">
    <div class="card-label">Mean Math Error</div>
    <div class="card-value">{p_mae_base:,.0f}</div>
    <div class="card-delta delta-neutral">mm³ across all timepoints</div>
</div>""", unsafe_allow_html=True)
cs3.markdown(f"""<div class="glass-card">
    <div class="card-label">Mean LSTM Error</div>
    <div class="card-value">{p_mae_hyb:,.0f}</div>
    <div class="card-delta delta-neutral">mm³ across all timepoints</div>
</div>""", unsafe_allow_html=True)
delta_cls = 'positive' if p_imp > 0 else 'negative' if p_imp < 0 else 'neutral'
winner = 'LSTM wins' if p_imp > 0 else 'Math wins' if p_imp < 0 else 'Equal'
cs4.markdown(f"""<div class="glass-card">
    <div class="card-label">LSTM Improvement</div>
    <div class="card-value">{p_imp:+.1f}%</div>
    <div class="card-delta delta-{delta_cls}">{winner}</div>
</div>""", unsafe_allow_html=True)


# ─── Footer ────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.78rem; padding:0.8rem 0;">
    Module 3: Tumor Progression Forecasting &nbsp;|&nbsp; FL-QPSO Brain Tumor Management System &nbsp;|&nbsp;
    MU-Glioma-Post Dataset (TCIA) &nbsp;|&nbsp; 111 patients, 654 predictions
</div>
""", unsafe_allow_html=True)
