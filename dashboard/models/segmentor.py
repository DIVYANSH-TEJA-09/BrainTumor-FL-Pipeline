"""
Segmentation Inference
=======================
Loads the 3D Attention U-Net (MONAI) and runs BraTS-style
brain tumor segmentation on 4-modality NIfTI input.

Output: 3-channel binary mask (TC, WT, ET) + volume statistics.
"""

import os
import numpy as np
import torch
from pathlib import Path

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import AttentionUnet
    from monai.transforms import (
        Compose, LoadImaged, NormalizeIntensityd, Orientationd,
        Spacingd, EnsureChannelFirstd, EnsureTyped,
    )
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from config import SEGMENTATION_MODEL_PATH, SEG_TUMOR_REGIONS, SEG_REGION_COLORS


# ─── MONAI Transforms (must match training) ─────────────────────────
if HAS_MONAI:
    INFERENCE_TRANSFORMS = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear",)),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])


# ─── Model Cache ────────────────────────────────────────────────────
_seg_model = None
_seg_device = None


def load_seg_model():
    """Load the 3D Attention U-Net model. Cached after first load."""
    global _seg_model, _seg_device

    if _seg_model is not None:
        return _seg_model, _seg_device

    if not HAS_MONAI:
        raise ImportError("MONAI is required for segmentation. Install: pip install monai")

    if SEGMENTATION_MODEL_PATH is None or not SEGMENTATION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Segmentation model not found. Checked paths — ensure "
            f"best_metric_model.pth or best_metric_model_refined.pth exists "
            f"in segmentation/ or segmentation/streamlit_app/"
        )

    _seg_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _seg_model = AttentionUnet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(_seg_device)

    _seg_model.load_state_dict(
        torch.load(str(SEGMENTATION_MODEL_PATH), map_location=_seg_device)
    )
    _seg_model.eval()
    return _seg_model, _seg_device


def segment_patient(t1_path, t1ce_path, t2_path, flair_path, progress_cb=None):
    """
    Run 3D segmentation on a patient's 4 modalities.

    Parameters
    ----------
    t1_path, t1ce_path, t2_path, flair_path : str or Path
        Paths to NIfTI files for each modality.
    progress_cb : callable or None
        Optional callback(percent, message) for progress updates.

    Returns
    -------
    dict with keys:
        - pred_mask: np.ndarray (D, H, W, 3) — binary predictions
        - uncertainty_map: np.ndarray (D, H, W) — normalized voxel uncertainty [0,1]
        - image_data: np.ndarray (D, H, W, 4) — preprocessed input
        - volumes: dict — TC, WT, ET volumes in mm³
        - total_voxels: int
        - mesh_data: dict — 3D mesh data for visualization (if skimage available)
        - slices: list — representative axial slices for preview
    """
    if not HAS_MONAI:
        raise ImportError("MONAI is required for segmentation")
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required for segmentation")

    if progress_cb:
        progress_cb(5, "Loading model...")

    model, device = load_seg_model()

    # Build MONAI data dict
    data_dict = {
        "image": [str(t1_path), str(t1ce_path), str(t2_path), str(flair_path)],
    }

    if progress_cb:
        progress_cb(15, "Preprocessing with MONAI transforms...")

    sample_data = INFERENCE_TRANSFORMS(data_dict)

    if progress_cb:
        progress_cb(35, "Running 3D U-Net inference (sliding window)...")

    inputs = sample_data["image"].unsqueeze(0).to(device)  # (1, 4, D, H, W)

    with torch.no_grad():
        logits = sliding_window_inference(inputs, (96, 96, 96), 4, model)
        probs = logits.sigmoid()
        outputs = (probs > 0.5).float()

    if progress_cb:
        progress_cb(80, "Processing results...")

    # Extract numpy arrays
    img_np = inputs[0].cpu().numpy().transpose(1, 2, 3, 0)   # (D,H,W,4)
    probs_np = probs[0].cpu().numpy().transpose(1, 2, 3, 0)  # (D,H,W,3)
    pred_np = outputs[0].cpu().numpy().transpose(1, 2, 3, 0)  # (D,H,W,3)

    # Voxel-wise uncertainty from channel-wise predictive entropy
    eps = 1e-6
    entropy = -(
        probs_np * np.log(np.clip(probs_np, eps, 1.0)) +
        (1.0 - probs_np) * np.log(np.clip(1.0 - probs_np, eps, 1.0))
    )
    uncertainty_map = np.mean(entropy, axis=3) / np.log(2.0)
    uncertainty_map = np.clip(uncertainty_map, 0.0, 1.0).astype(np.float32)

    high_uncertainty_threshold = float(max(0.6, np.percentile(uncertainty_map, 92)))
    high_uncertainty_ratio = float((uncertainty_map >= high_uncertainty_threshold).sum() / max(1, uncertainty_map.size))
    uncertainty_mean = float(uncertainty_map.mean())
    uncertainty_p95 = float(np.percentile(uncertainty_map, 95))

    if high_uncertainty_ratio >= 0.08 or uncertainty_mean >= 0.45:
        uncertainty_level = "high"
    elif high_uncertainty_ratio >= 0.03 or uncertainty_mean >= 0.30:
        uncertainty_level = "medium"
    else:
        uncertainty_level = "low"

    uncertainty_summary = {
        "level": uncertainty_level,
        "mean": uncertainty_mean,
        "p95": uncertainty_p95,
        "high_uncertainty_threshold": high_uncertainty_threshold,
        "high_uncertainty_ratio": high_uncertainty_ratio,
        "review_recommended": uncertainty_level != "low",
    }

    # Volume statistics
    voxel_vol = 1.0  # isotropic 1mm spacing
    volumes = {
        "TC": float(pred_np[:, :, :, 0].sum() * voxel_vol),
        "WT": float(pred_np[:, :, :, 1].sum() * voxel_vol),
        "ET": float(pred_np[:, :, :, 2].sum() * voxel_vol),
    }
    total_voxels = int(np.prod(pred_np.shape[:3]))

    # Representative slices (5 evenly-spaced through tumor region)
    slices_data = _extract_preview_slices(img_np, pred_np, uncertainty_map=uncertainty_map)

    # 3D mesh data
    mesh_data = None
    if HAS_SKIMAGE:
        mesh_data = _extract_mesh_data(pred_np, img_np, uncertainty_map=uncertainty_map)

    if progress_cb:
        progress_cb(100, "Segmentation complete!")

    return {
        "pred_mask": pred_np,
        "uncertainty_map": uncertainty_map,
        "uncertainty_summary": uncertainty_summary,
        "image_data": img_np,
        "volumes": volumes,
        "total_voxels": total_voxels,
        "mesh_data": mesh_data,
        "slices": slices_data,
        "region_names": SEG_TUMOR_REGIONS,
        "region_colors": SEG_REGION_COLORS,
    }


def _extract_preview_slices(img_np, pred_np, uncertainty_map=None, n_slices=7):
    """Extract representative axial slices through the tumor region."""
    # Find slices with tumor
    wt_channel = pred_np[:, :, :, 1]  # Whole Tumor
    tumor_slices = np.where(wt_channel.sum(axis=(1, 2)) > 0)[0]

    if len(tumor_slices) == 0:
        # Fallback: use middle slices
        total_d = img_np.shape[0]
        indices = np.linspace(total_d * 0.3, total_d * 0.7, n_slices, dtype=int)
    else:
        indices = np.linspace(tumor_slices[0], tumor_slices[-1], n_slices, dtype=int)

    slices = []
    for idx in indices:
        idx = min(idx, img_np.shape[0] - 1)
        # FLAIR modality (channel 3) as background
        flair_slice = img_np[idx, :, :, 3]
        # Normalize for display
        flair_min, flair_max = flair_slice.min(), flair_slice.max()
        if flair_max > flair_min:
            flair_norm = ((flair_slice - flair_min) / (flair_max - flair_min) * 255).astype(np.uint8)
        else:
            flair_norm = np.zeros_like(flair_slice, dtype=np.uint8)

        # Prediction overlay
        tc = pred_np[idx, :, :, 0]
        wt = pred_np[idx, :, :, 1]
        et = pred_np[idx, :, :, 2]
        unc = uncertainty_map[idx] if uncertainty_map is not None else np.zeros_like(tc, dtype=np.float32)
        unc_u8 = np.clip(unc * 255.0, 0, 255).astype(np.uint8)

        slices.append({
            "index": int(idx),
            "flair": flair_norm.tolist(),
            "tc": tc.astype(np.uint8).tolist(),
            "wt": wt.astype(np.uint8).tolist(),
            "et": et.astype(np.uint8).tolist(),
            "uncertainty": unc_u8.tolist(),
        })

    return slices


def _extract_mesh_data(pred_np, img_np=None, uncertainty_map=None, step=2):
    """Extract 3D mesh vertices and faces for Plotly rendering."""
    meshes = {}

    labels = [("TC", 0, "#E74C3C"), ("WT", 1, "#F59E0B"), ("ET", 2, "#F97316")]

    for name, ch, color in labels:
        vol = pred_np[:, :, :, ch]
        v = vol[::step, ::step, ::step]
        if v.sum() == 0:
            meshes[name] = None
            continue
        try:
            verts, faces, _, _ = marching_cubes(v, level=0.5)
            verts = verts * step  # Scale back
            meshes[name] = {
                "vertices": verts.tolist(),
                "faces": faces.tolist(),
                "color": color,
                "name": name,
            }
        except Exception:
            meshes[name] = None

    # Optional brain context mesh for anatomical reference.
    if img_np is not None:
        try:
            # Use non-zero signal footprint as a robust brain/tissue mask.
            tissue = np.mean(np.abs(img_np), axis=3)
            tissue_mask = (tissue > 1e-6).astype(np.uint8)
            v = tissue_mask[::step, ::step, ::step]
            if v.sum() > 0:
                verts, faces, _, _ = marching_cubes(v, level=0.5)
                verts = verts * step
                meshes["BRAIN"] = {
                    "vertices": verts.tolist(),
                    "faces": faces.tolist(),
                    "color": "#94A3B8",
                    "name": "Brain",
                    "opacity": 0.12,
                }
        except Exception:
            meshes["BRAIN"] = None

    # Optional uncertainty shell mesh for interpretability.
    if uncertainty_map is not None:
        try:
            threshold = float(max(0.6, np.percentile(uncertainty_map, 92)))
            unc_mask = (uncertainty_map >= threshold).astype(np.uint8)
            v = unc_mask[::step, ::step, ::step]
            if v.sum() > 0:
                verts, faces, _, _ = marching_cubes(v, level=0.5)
                verts = verts * step
                meshes["UNC"] = {
                    "vertices": verts.tolist(),
                    "faces": faces.tolist(),
                    "color": "#C084FC",
                    "name": "Uncertainty",
                    "opacity": 0.22,
                }
        except Exception:
            meshes["UNC"] = None

    return meshes
