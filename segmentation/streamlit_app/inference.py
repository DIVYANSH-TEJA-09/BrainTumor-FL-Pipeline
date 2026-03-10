"""
Shared inference module for 3D brain tumor segmentation.
Loads the AttentionUnet model and runs sliding_window_inference
on patients that don't have pre-computed predictions.
"""

import os
import shutil
import numpy as np
import nibabel as nib
import streamlit as st
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
)


# ─── paths ───────────────────────────────────────────────────────────────
SEG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "segmentation"))
DEMO_DIR = os.path.join(SEG_DIR, "demo_data")
# streamlit_app/ is inside segmentation/, so go up one level to reach segmentation/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Model checkpoint — prefer refined model (better calibration)
# 1. Check streamlit_app/ (where refined model lives)
# 2. Check segmentation/ (parent dir, where base model lives)
_THIS_DIR = os.path.dirname(__file__)
_candidates = [
    os.path.join(_THIS_DIR, "best_metric_model_refined.pth"),          # streamlit_app/
    os.path.join(PROJECT_ROOT, "best_metric_model_refined.pth"),       # segmentation/
    os.path.join(_THIS_DIR, "best_metric_model.pth"),                  # streamlit_app/
    os.path.join(PROJECT_ROOT, "best_metric_model.pth"),               # segmentation/
]
CKPT_PATH = None
for _c in _candidates:
    if os.path.exists(_c):
        CKPT_PATH = _c
        break

# MONAI transforms — must match training exactly
INFERENCE_TRANSFORMS = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
])

# Transforms for image-only (no label available)
INFERENCE_TRANSFORMS_IMG_ONLY = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    EnsureTyped(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear",)),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
])


@st.cache_resource
def load_seg_model():
    """Load the 3D Attention U-Net model (cached across sessions)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)

    if os.path.exists(CKPT_PATH):
        try:
            model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
            model.eval()
            return model, device
        except Exception as e:
            st.error(f"Failed to load model weights: {e}")
            return None, None
    else:
        st.error(f"Model checkpoint not found at {CKPT_PATH}")
        return None, None


def ensure_prediction(patient_id):
    """
    Ensure that the prediction volume exists for a patient.
    If _pred.nii.gz already exists, returns True immediately.
    Otherwise, runs live inference using the exact same MONAI
    transforms as the training pipeline.
    """
    pred_path = os.path.join(DEMO_DIR, f"{patient_id}_pred.nii.gz")
    img_path = os.path.join(DEMO_DIR, f"{patient_id}_image.nii.gz")
    lbl_path = os.path.join(DEMO_DIR, f"{patient_id}_label.nii.gz")

    # Already have prediction — skip
    if os.path.exists(pred_path) and os.path.exists(img_path):
        return True

    # Check if raw MRI modalities exist in patient subfolder
    p_dir = os.path.join(DEMO_DIR, patient_id)
    if not os.path.isdir(p_dir):
        return False

    # Build file paths (same order as extract_demo_data.py: t1, t1ce, t2, flair)
    mod_paths = {
        "t1":    os.path.join(p_dir, f"{patient_id}_t1.nii.gz"),
        "t1ce":  os.path.join(p_dir, f"{patient_id}_t1ce.nii.gz"),
        "t2":    os.path.join(p_dir, f"{patient_id}_t2.nii.gz"),
        "flair": os.path.join(p_dir, f"{patient_id}_flair.nii.gz"),
    }
    seg_path = os.path.join(p_dir, f"{patient_id}_seg.nii.gz")

    for m, mp in mod_paths.items():
        if not os.path.exists(mp):
            st.warning(f"Missing modality: {m} at {mp}")
            return False

    # ─── Run live inference ──────────────────────────────────────────
    st.info(f"🧠 **Running AI Inference** on `{patient_id}`... This may take 30-60 seconds.")
    progress = st.progress(0)
    status = st.empty()

    try:
        # Build MONAI data dict (image is a list of 4 modality paths)
        has_label = os.path.exists(seg_path)
        data_dict = {
            "image": [mod_paths["t1"], mod_paths["t1ce"], mod_paths["t2"], mod_paths["flair"]],
        }
        if has_label:
            data_dict["label"] = seg_path

        # Apply MONAI transforms (Orientation, Spacing, Normalize — matching training)
        status.text("Loading & preprocessing with MONAI transforms...")
        if has_label:
            sample_data = INFERENCE_TRANSFORMS(data_dict)
        else:
            sample_data = INFERENCE_TRANSFORMS_IMG_ONLY(data_dict)
        progress.progress(30)

        # Run model inference
        status.text("Running 3D U-Net inference (sliding window)...")
        model, device = load_seg_model()
        if model is None:
            return False

        inputs = sample_data["image"].unsqueeze(0).to(device)  # (1, 4, D, H, W)
        with torch.no_grad():
            outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model)
            outputs = (outputs.sigmoid() > 0.5).float()
        progress.progress(80)

        # Save processed image volume (D, H, W, 4)
        status.text("Saving results...")
        img_np = inputs[0].cpu().numpy().transpose(1, 2, 3, 0)
        nib.save(nib.Nifti1Image(img_np, affine=np.eye(4)), img_path)

        # Save prediction (D, H, W, 3)
        pred_np = outputs[0].cpu().numpy().transpose(1, 2, 3, 0)
        nib.save(nib.Nifti1Image(pred_np, affine=np.eye(4)), pred_path)

        # Save ground truth label (D, H, W)
        if has_label:
            lbl_np = sample_data["label"][0].cpu().numpy()
            nib.save(nib.Nifti1Image(lbl_np.astype(np.float32), affine=np.eye(4)), lbl_path)
        elif not os.path.exists(lbl_path):
            empty = np.zeros(pred_np.shape[:3])
            nib.save(nib.Nifti1Image(empty.astype(np.float32), affine=np.eye(4)), lbl_path)

        progress.progress(100)
        status.text("✅ Inference complete!")
        return True

    except Exception as e:
        st.error(f"Inference failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


def get_all_patients():
    """
    Return all patient IDs that have either pre-computed predictions
    OR raw MRI data (can be inferred on-demand).
    """
    patients = set()

    # Patients with pre-computed predictions
    import glob
    for f in glob.glob(os.path.join(DEMO_DIR, "*_pred.nii.gz")):
        pid = os.path.basename(f).replace("_pred.nii.gz", "")
        patients.add(pid)

    # Patients with raw MRI data (subfolder with modality files)
    if os.path.isdir(DEMO_DIR):
        for d in os.listdir(DEMO_DIR):
            full = os.path.join(DEMO_DIR, d)
            if os.path.isdir(full) and d.startswith("BraTS"):
                # Check it has at least the flair file
                if os.path.exists(os.path.join(full, f"{d}_flair.nii.gz")):
                    patients.add(d)

    return sorted(patients)
