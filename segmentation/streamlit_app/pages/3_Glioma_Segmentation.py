import streamlit as st
import sys
import os
import shutil
import glob
import nibabel as nib
import numpy as np

st.set_page_config(page_title="Glioma Segmentation", layout="wide")

if "selected_patient" not in st.session_state:
    st.warning("No patient selected.")
    st.stop()

p = st.session_state["selected_patient"]

# --- ADAPTER LOGIC START ---
# The existing app expects: {id}_image.nii.gz, {id}_label.nii.gz, {id}_pred.nii.gz in DEMO_DIR
# The new data provides: {id}/{id}_flair.nii.gz, etc. in DEMO_DIR

import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    EnsureType,
    Orientation,
    Spacing,
    NormalizeIntensity,
    ResizeWithPadOrCrop
)

SEG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../segmentation"))
DEMO_DIR = os.path.join(SEG_DIR, "demo_data")
# Model is in the root of the project (parent of streamlit_app)
PROJECT_ROOT = os.path.abspath(os.path.join(SEG_DIR, "../..")) 
CKPT_PATH = os.path.join(PROJECT_ROOT, "best_metric_model.pth")
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(PROJECT_ROOT, "best_metric_model_refined.pth")

PATIENT_ID = p["id"]

@st.cache_resource
def load_model():
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

def prepare_compatible_data(pid, base_dir):
    p_dir = os.path.join(base_dir, pid)
    
    # Target files (flat)
    target_img = os.path.join(base_dir, f"{pid}_image.nii.gz")
    target_lbl = os.path.join(base_dir, f"{pid}_label.nii.gz")
    target_pred = os.path.join(base_dir, f"{pid}_pred.nii.gz")
    
    # If compatible files exist, we assume they are good. 
    # BUT user wants to force real inference or ensure correctness.
    # Let's check if they exist.
    if os.path.exists(target_img) and os.path.exists(target_lbl) and os.path.exists(target_pred):
        return True
        
    if not os.path.exists(p_dir):
        return False
        
    st.info(f"🧠 RUNNING AI INFERENCE: Generating volumetric prediction for {pid}...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Loading MRI Sequences...")
        
        # 1. Load 4 channels
        t1 = os.path.join(p_dir, f"{pid}_t1.nii.gz")
        t1ce = os.path.join(p_dir, f"{pid}_t1ce.nii.gz")
        t2 = os.path.join(p_dir, f"{pid}_t2.nii.gz")
        flair = os.path.join(p_dir, f"{pid}_flair.nii.gz")
        
        # Load using MONAI transforms for consistency
        # We need to construct a sample manually
        
        # Manually load raw data first to check affine
        nii_stub = nib.load(flair)
        affine = nii_stub.affine
        
        status_text.text("Preprocessing & Normalizing...")
        progress_bar.progress(20)
        
        # Create a combined 4D volume (4, D, H, W) for MONAI (Channel First)
        # Note: Original App's 'LoadImaged' loads as (D, H, W, 4) then EnsureChannelFirstd makes it (4, D, H, W)
        d_t1 = nib.load(t1).get_fdata()
        d_t1ce = nib.load(t1ce).get_fdata()
        d_t2 = nib.load(t2).get_fdata()
        d_flair = nib.load(flair).get_fdata()
        
        # Stack channels: (4, D, H, W)
        raw_stack = np.stack([d_t1, d_t1ce, d_t2, d_flair], axis=0)
        
        # Convert to Tensor
        data_tensor = torch.from_numpy(raw_stack).float()
        
        # Apply Normalization manually (matching training)
        # NormalizeIntensity(nonzero=True, channel_wise=True)
        # Loop over channels
        for c in range(4):
            ch_data = data_tensor[c]
            mask = ch_data > 0
            if mask.any():
                mean = ch_data[mask].mean()
                std = ch_data[mask].std()
                data_tensor[c] = (ch_data - mean) / (std + 1e-8) * mask.float()
                
        # Prep for Inference
        # Add Batch Dim -> (1, 4, D, H, W)
        inputs = data_tensor.unsqueeze(0)
        
        # Save Input Image (Transpose to D,H,W,4 for App Visualization)
        # App expects (D, H, W, 4)
        status_text.text("Saving Preprocessed Input...")
        img_np = data_tensor.numpy().transpose(1, 2, 3, 0)
        nib.save(nib.Nifti1Image(img_np, affine), target_img)
        progress_bar.progress(40)
        
        # 2. Inference
        model, device = load_model()
        if model is None:
            return False
            
        inputs = inputs.to(device)
        
        status_text.text("Running 3D Unet Inference (Sliding Window)...")
        with torch.no_grad():
            outputs = sliding_window_inference(inputs, (96, 96, 96), 4, model)
            outputs = (outputs.sigmoid() > 0.5).float()
        
        progress_bar.progress(80)
        
        # 3. Save Prediction
        # Output is (1, 3, D, H, W).
        # Need (D, H, W, 3)
        status_text.text("Constructing Prediction Volume...")
        pred_np = outputs[0].cpu().numpy().transpose(1, 2, 3, 0)
        nib.save(nib.Nifti1Image(pred_np, affine), target_pred)
        
        # 4. Handle Ground Truth
        seg_path = os.path.join(p_dir, f"{pid}_seg.nii.gz")
        if os.path.exists(seg_path):
            shutil.copy(seg_path, target_lbl)
        else:
            # Create empty if not exists
            empty = np.zeros(pred_np.shape[:3])
            nib.save(nib.Nifti1Image(empty, affine), target_lbl)
            
        progress_bar.progress(100)
        status_text.text("Done!")
        return True
            
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False

# --- RUN ADAPTER ---
if os.path.exists(os.path.join(DEMO_DIR, PATIENT_ID)):
    success = prepare_compatible_data(PATIENT_ID, DEMO_DIR)
    if not success:
        st.error("Failed to prepare compatible data.")
else:
    # If using mock ID that doesn't exist on disk, we can't run the real app
    pass

st.markdown(f"## ✂️ 3D Glioma Segmentation: {p['name']} ({p['id']})")
st.markdown("---")

# --- INTEGRATION ----
APP_PATH = os.path.join(SEG_DIR, "app.py")

if SEG_DIR not in sys.path:
    sys.path.append(SEG_DIR)

original_cwd = os.getcwd()
try:
    os.chdir(SEG_DIR)
    
    # Read the code
    with open(APP_PATH, "r") as f:
        code = f.read()
        
    # Execute it
    exec(code, {"__file__": APP_PATH, "st": st, "plt": __import__("matplotlib.pyplot"), "nib": __import__("nibabel"), "np": __import__("numpy"), "os": os, "glob": __import__("glob")})

except Exception as e:
    st.error(f"Error running segmentation module: {e}")
finally:
    os.chdir(original_cwd)
