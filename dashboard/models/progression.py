"""
Progression Inference
======================
Tumor growth prediction using:
  1. Mathematical Model (Logistic Growth) — per-patient fitted curve
  2. Spatial U-Net (3D) — predicts WHERE growth occurs

For single-scan uploads, uses grade-median defaults for logistic params.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from config import (
    SPATIAL_UNET_PATH, SPATIAL_EVAL_PATH, SPATIAL_CROP,
    LOGISTIC_DEFAULTS, PREDICTION_INDEX_PATH,
)

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ═══════════════════════════ LOGISTIC GROWTH MODEL ═══════════════════

def logistic_curve(t, v0, k, r):
    """Logistic growth: V(t) = K / (1 + ((K - V0) / V0) * exp(-r*t))"""
    with np.errstate(over='ignore'):
        denom = 1 + ((k - v0) / v0) * np.exp(-r * t)
        denom = np.clip(denom, 1e-10, None)
    return k / denom


def predict_logistic_growth(current_volume, grade="HGG", days_forward=None):
    """
    Project tumor growth using logistic model with grade-median defaults.

    Parameters
    ----------
    current_volume : float
        Current tumor volume in mm³.
    grade : str
        Tumor grade: "HGG" or "LGG".
    days_forward : list[int] or None
        Days to project forward. Defaults to [30, 60, 90, 180, 365].

    Returns
    -------
    dict with keys:
        - projections: list of {day, volume, growth_pct}
        - params: {v0, k, r, grade}
        - curve: {days, volumes} for smooth plotting
    """
    if days_forward is None:
        days_forward = [30, 60, 90, 180, 365]

    params = LOGISTIC_DEFAULTS.get(grade, LOGISTIC_DEFAULTS["HGG"])
    v0 = current_volume
    k = params["k"]
    r = params["r"]

    projections = []
    for d in days_forward:
        v = logistic_curve(d, v0, k, r)
        growth_pct = ((v - v0) / v0 * 100) if v0 > 0 else 0.0
        projections.append({
            "day": d,
            "volume": float(v),
            "growth_pct": float(growth_pct),
        })

    # Smooth curve for plotting
    t_smooth = np.linspace(0, max(days_forward) * 1.1, 200)
    v_smooth = logistic_curve(t_smooth, v0, k, r)

    return {
        "projections": projections,
        "params": {
            "v0": float(v0),
            "k": float(k),
            "r": float(r),
            "grade": grade,
        },
        "curve": {
            "days": t_smooth.tolist(),
            "volumes": v_smooth.tolist(),
        },
    }


# ═══════════════════════════ SPATIAL U-NET ════════════════════════════

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


# ─── Spatial Model Cache ─────────────────────────────────────────────
_spatial_model = None


def load_spatial_model():
    """Load the trained 3D U-Net for spatial prediction. Cached."""
    global _spatial_model

    if _spatial_model is not None:
        return _spatial_model

    if not SPATIAL_UNET_PATH.exists():
        return None

    try:
        ckpt = torch.load(str(SPATIAL_UNET_PATH), map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        model = SpatialUNet3D(
            base=cfg.get('base_filters', 16),
            dropout=cfg.get('dropout', 0.15),
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        _spatial_model = model
        return model
    except Exception as e:
        print(f"[WARN] Spatial model load error: {e}")
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
def predict_spatial(mask_input, image_data=None):
    """
    Run spatial prediction: input mask → predicted next mask.

    Parameters
    ----------
    mask_input : np.ndarray (D, H, W)
        Current tumor segmentation mask.
    image_data : np.ndarray or None
        Optional 4-channel MRI volume for brain context mesh.

    Returns
    -------
    dict with keys:
        - predicted_mask: np.ndarray — predicted next-timepoint mask
        - stable: np.ndarray — voxels present in both
        - growth: np.ndarray — new voxels (growth)
        - regression: np.ndarray — lost voxels (regression)
        - stats: growth/regression voxel counts
        - mesh_data: 3D mesh data for visualization
    """
    model = load_spatial_model()
    if model is None:
        return None

    binary = (mask_input > 0).astype(np.float32)
    cropped, center, starts, ends, pb, pa = crop_around_tumor(binary)
    x = torch.from_numpy(cropped[np.newaxis, np.newaxis].astype(np.float32))
    pred = model(x)
    pred_crop = (pred[0, 0].numpy() > 0.5).astype(np.float32)
    predicted_mask = uncrop_prediction(pred_crop, mask_input.shape, starts, ends, pb, pa)

    # Compute regions
    input_bin = (binary > 0).astype(float)
    pred_bin = (predicted_mask > 0.5).astype(float)
    stable = input_bin * pred_bin
    growth = np.clip(pred_bin - input_bin, 0, 1)
    regression = np.clip(input_bin - pred_bin, 0, 1)

    stats = {
        "growth_voxels": int(growth.sum()),
        "regression_voxels": int(regression.sum()),
        "stable_voxels": int(stable.sum()),
        "input_volume": float(input_bin.sum()),
        "predicted_volume": float(pred_bin.sum()),
        "volume_change_pct": float(
            (pred_bin.sum() - input_bin.sum()) / input_bin.sum() * 100
        ) if input_bin.sum() > 0 else 0.0,
    }

    # Mesh data for 3D viz
    mesh_data = None
    if HAS_SKIMAGE:
        mesh_data = _build_spatial_meshes(stable, growth, regression, image_data=image_data)

    return {
        "predicted_mask": predicted_mask,
        "stable": stable,
        "growth": growth,
        "regression": regression,
        "stats": stats,
        "mesh_data": mesh_data,
    }


def _build_spatial_meshes(stable, growth, regression, image_data=None, step=2):
    """Build 3D mesh data for the spatial prediction visualization."""
    meshes = {}
    parts = [
        ("stable", stable, "#3b82f6", 0.6),
        ("growth", growth, "#ef4444", 0.9),
        ("regression", regression, "#22c55e", 0.6),
    ]

    for name, vol, color, opacity in parts:
        v = vol[::step, ::step, ::step]
        if v.sum() == 0:
            continue
        try:
            verts, faces, _, _ = marching_cubes(v, level=0.5)
            verts = verts * step
            meshes[name] = {
                "vertices": verts.tolist(),
                "faces": faces.tolist(),
                "color": color,
                "opacity": opacity,
            }
        except Exception:
            pass

    # Add a faint outer envelope to provide anatomical context.
    try:
        envelope = np.clip(stable + growth + regression, 0, 1)
        if envelope.sum() > 0:
            v = envelope[::step, ::step, ::step]
            verts, faces, _, _ = marching_cubes(v, level=0.5)
            verts = verts * step
            meshes["envelope"] = {
                "vertices": verts.tolist(),
                "faces": faces.tolist(),
                "color": "#94A3B8",
                "opacity": 0.10,
            }
    except Exception:
        pass

    # Optional brain context mesh from MRI image volume.
    if image_data is not None:
        try:
            if image_data.ndim == 4 and image_data.shape[3] > 0:
                tissue = np.mean(np.abs(image_data), axis=3)
            else:
                tissue = np.abs(image_data)
            tissue_mask = (tissue > 1e-6).astype(np.uint8)
            v = tissue_mask[::step, ::step, ::step]
            if v.sum() > 0:
                verts, faces, _, _ = marching_cubes(v, level=0.5)
                verts = verts * step
                meshes["brain"] = {
                    "vertices": verts.tolist(),
                    "faces": faces.tolist(),
                    "color": "#94A3B8",
                    "opacity": 0.10,
                }
        except Exception:
            pass

    return meshes


def get_spatial_eval_metrics():
    """Load spatial model evaluation metrics."""
    if SPATIAL_EVAL_PATH.exists():
        with open(SPATIAL_EVAL_PATH) as f:
            return json.load(f)
    return None


def run_full_progression(seg_mask, grade="HGG", image_data=None):
    """
    Run complete progression analysis for a segmented tumor.

    Parameters
    ----------
    seg_mask : np.ndarray
        3D tumor segmentation mask (usually WT channel).
    grade : str
        Tumor grade.

    Returns
    -------
    dict with keys:
        - logistic: logistic growth projections
        - spatial: spatial U-Net predictions (or None)
        - eval_metrics: spatial model eval metrics
    """
    # Volume from mask
    current_volume = float((seg_mask > 0).sum())

    # Logistic growth
    logistic_result = predict_logistic_growth(current_volume, grade)

    # Spatial prediction
    spatial_result = predict_spatial(seg_mask, image_data=image_data)

    # Eval metrics
    eval_metrics = get_spatial_eval_metrics()

    return {
        "logistic": logistic_result,
        "spatial": spatial_result,
        "eval_metrics": eval_metrics,
        "current_volume": current_volume,
        "grade": grade,
    }
