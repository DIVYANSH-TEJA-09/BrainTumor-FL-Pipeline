
import streamlit as st
import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
DEMO_DIR = "demo_data"
# st.set_page_config(page_title="Brain Tumor Segmentation AI", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    h1 {
        color: #4facfe;
    }
    h2 {
        color: #00f2fe;
    }
    .stSlider {
        color: #4facfe;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
def load_nifti(path):
    if not os.path.exists(path):
        return None
    img = nib.load(path)
    return img.get_fdata()

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

def get_demo_samples():
    files = sorted(glob.glob(os.path.join(DEMO_DIR, "*_image.nii.gz")))
    ids = [os.path.basename(f).replace("_image.nii.gz", "") for f in files]
    return ids

# --- APP LOGIC ---
st.title("🧠 3D Brain Tumor Segmentation Visualizer")

samples = get_demo_samples()
if not samples:
    st.error(f"No demo data found in `{DEMO_DIR}`. Please run data extraction first.")
    st.stop()

# Sidebar
st.sidebar.title("Configuration")
selected_id = st.sidebar.selectbox("Select Patient Sample", samples)
view_mode = st.sidebar.radio("View Mode", ["3D Interactive Scanner", "Reviewer Deep Dive"])

# Load Data
img_path = os.path.join(DEMO_DIR, f"{selected_id}_image.nii.gz")
lbl_path = os.path.join(DEMO_DIR, f"{selected_id}_label.nii.gz")
pred_path = os.path.join(DEMO_DIR, f"{selected_id}_pred.nii.gz")

img_data = load_nifti(img_path) # (D, H, W, 4)
lbl_data = load_nifti(lbl_path) # (D, H, W)
pred_data = load_nifti(pred_path) # (D, H, W, 3)

if img_data is None:
    st.error("Failed to load image data.")
    st.stop()

# Slice Slider
depth = img_data.shape[0]
slice_idx = st.slider("Slice Index (Z-Axis)", 0, depth-1, depth//2)

if view_mode == "3D Interactive Scanner":
    st.subheader(f"Patient ID: {selected_id}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### MRI Scan (FLAIR)")
        # FLAIR is usually channel 3, but let's check input. 
        # Our extraction saved (D,H,W,4). 0=T1, 1=T1ce, 2=T2, 3=FLAIR.
        flair = img_data[slice_idx, :, :, 3]
        
        fig, ax = plt.subplots()
        ax.imshow(flair, cmap="gray", origin="lower")
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.markdown("### AI Prediction (Whole Tumor)")
        if pred_data is not None:
            # Pred is (D,H,W,3). 
            # MONAI Output: 0=TC, 1=WT, 2=ET. We want WT.
            wt_pred = pred_data[slice_idx, :, :, 1]
            
            fig, ax = plt.subplots()
            ax.imshow(flair, cmap="gray", origin="lower")
            ax.imshow(wt_pred, cmap="Reds", alpha=0.5, origin="lower")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("Prediction not available.")

elif view_mode == "Reviewer Deep Dive":
    st.subheader("Deep Dive Analysis")
    
    # ROW 1: INPUTS
    st.markdown("#### Input Modalities")
    c1, c2, c3, c4 = st.columns(4)
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]
    
    for i, col in enumerate([c1, c2, c3, c4]):
        with col:
            st.caption(modality_names[i])
            sl = img_data[slice_idx, :, :, i]
            fig, ax = plt.subplots()
            ax.imshow(sl, cmap="gray", origin="lower")
            ax.axis("off")
            st.pyplot(fig)

    # ROW 2: GROUND TRUTH
    if lbl_data is not None:
        st.markdown("#### Ground Truth")
        c1, c2, c3, c4 = st.columns(4)
        
        # Labels: 0=WT, 1=TC, 2=ET in our "Fixed" dataset? 
        # Wait, the validation extraction saved `lbl_np` which was the result of `LoadImaged`.
        # So it's the original mask with integer labels: 1 (NCR), 2 (ED), 4 (ET).
        # We need to map these to WT/TC/ET for display if we are showing components.
        # WT = 1 + 2 + 4
        # TC = 1 + 4
        # ET = 4
        
        mask = lbl_data[slice_idx, :, :]
        
        # WT
        wt_mask = (mask > 0).astype(float)
        with c1:
            st.caption("Whole Tumor")
            fig, ax = plt.subplots()
            ax.imshow(wt_mask, cmap="Greens", origin="lower")
            ax.axis("off")
            st.pyplot(fig)
            
        # TC
        tc_mask = ((mask == 1) | (mask == 4)).astype(float)
        with c2:
            st.caption("Tumor Core")
            fig, ax = plt.subplots()
            ax.imshow(tc_mask, cmap="Greens", origin="lower")
            ax.axis("off")
            st.pyplot(fig)
            
        # ET
        et_mask = (mask == 4).astype(float)
        with c3:
            st.caption("Enhancing Tumor")
            fig, ax = plt.subplots()
            ax.imshow(et_mask, cmap="Greens", origin="lower")
            ax.axis("off")
            st.pyplot(fig)
            
        # Overlay
        with c4:
            st.caption("GT Overlay (WT)")
            flair = img_data[slice_idx, :, :, 3]
            fig, ax = plt.subplots()
            ax.imshow(flair, cmap="gray", origin="lower")
            ax.imshow(wt_mask, cmap="Greens", alpha=0.5, origin="lower")
            ax.axis("off")
            st.pyplot(fig)

    # ROW 3: PREDICTIONS
    if pred_data is not None:
        st.markdown("#### AI Predictions")
        c1, c2, c3, c4 = st.columns(4)
        
        # Preds are CHANNELS: 
        # MONAI Default: 0=TC, 1=WT, 2=ET
        # We want to display: WT, TC, ET
        
        # Display WT (Channel 1)
        with c1:
            st.caption("Whole Tumor")
            p = pred_data[slice_idx, :, :, 1]
            fig, ax = plt.subplots()
            ax.imshow(p, cmap="Reds", origin="lower")
            ax.axis("off")
            st.pyplot(fig)

        # Display TC (Channel 0)
        with c2:
            st.caption("Tumor Core")
            p = pred_data[slice_idx, :, :, 0]
            fig, ax = plt.subplots()
            ax.imshow(p, cmap="Reds", origin="lower")
            ax.axis("off")
            st.pyplot(fig)

        # Display ET (Channel 2)
        with c3:
            st.caption("Enhancing Tumor")
            p = pred_data[slice_idx, :, :, 2]
            fig, ax = plt.subplots()
            ax.imshow(p, cmap="Reds", origin="lower")
            ax.axis("off")
            st.pyplot(fig)
        
        with c4:
            st.caption("Pred Overlay (WT)")
            flair = img_data[slice_idx, :, :, 3]
            wt_p = pred_data[slice_idx, :, :, 1] # WT is Ch 1
            fig, ax = plt.subplots()
            ax.imshow(flair, cmap="gray", origin="lower")
            ax.imshow(wt_p, cmap="Reds", alpha=0.5, origin="lower")
            ax.axis("off")
            st.pyplot(fig)
