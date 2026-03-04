import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Patient Detail", layout="wide")

if "selected_patient" not in st.session_state:
    st.warning("No patient selected.")
    st.stop()

p = st.session_state["selected_patient"]

# --- HEADER ---
st.markdown(f"# 👤 Patient Detail: {p['name']}")
st.markdown(f"**ID**: `{p['id']}` | **Age**: {p['age']} | **Gender**: {p['gender']}")

col_left, col_right = st.columns([1, 1])

# --- LEFT: SCANS ---
with col_left:
    st.markdown("### 🖼️ MRI Scan Preview")
    
    # Check for real data (Adapter Logic for Visualization)
    # New Data Structure: demo_data/{id}/{id}_flair.nii.gz etc.
    DEMO_ROOT = os.path.join(os.path.dirname(__file__), "../segmentation/demo_data")
    patient_dir = os.path.join(DEMO_ROOT, p['id'])
    
    found_real_data = False
    
    if os.path.exists(patient_dir) and os.path.isdir(patient_dir):
        # Try to load FLAIR/T1/T2/T1ce
        # Filenames: {id}_flair.nii.gz
        flair_path = os.path.join(patient_dir, f"{p['id']}_flair.nii.gz")
        t1_path = os.path.join(patient_dir, f"{p['id']}_t1.nii.gz")
        t1ce_path = os.path.join(patient_dir, f"{p['id']}_t1ce.nii.gz")
        t2_path = os.path.join(patient_dir, f"{p['id']}_t2.nii.gz")
        
        paths = [t1_path, t1ce_path, t2_path, flair_path]
        names = ["T1", "T1ce", "T2", "FLAIR"]
        
        if os.path.exists(flair_path):
            found_real_data = True
            st.success("DICOM/NIfTI Volume Loaded Successfully")
            
            # Load Data
            # Just show middle slice of each
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            cols = [c1, c2, c3, c4]
            
            for i, path in enumerate(paths):
                with cols[i]:
                    st.caption(f"**{names[i]}**")
                    if os.path.exists(path):
                        try:
                            data = nib.load(path).get_fdata()
                            mid = data.shape[0] // 2
                            sl = data[mid, :, :]
                            fig, ax = plt.subplots()
                            ax.imshow(sl, cmap="gray", origin="lower")
                            ax.axis("off")
                            st.pyplot(fig)
                        except:
                            st.error("Error")
                    else:
                        st.info("Not available")
                        
    if not found_real_data:
        st.warning("No real MRI data found for this patient (Mock Entry).")
        st.image("https://prod-images-static.radiopaedia.org/images/53381669/59b1e9c2f6d6288647008107955562_gallery.jpeg", width=300)

# --- RIGHT: CLASSIFICATION & ACTION ---
with col_right:
    st.markdown("### 🧠 AI Classification Result")
    
    if "Glioma" in p["status"]:
        st.markdown("""
        <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #d32f2f;">
            <h2 style="color: #d32f2f; margin:0;">Detected: High-Grade Glioma</h2>
            <p>Confidence: <strong>94.2%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Recommended Action")
        st.write("Volumetric segmentation is recommended for surgical planning.")
        
        st.markdown("---")
        if found_real_data:
            if st.button("🚀 Run 3D Segmentation Model", type="primary"):
                st.switch_page("pages/3_Glioma_Segmentation.py")
        elif "Glioma" in p["status"]:
            st.button("🚀 Run 3D Segmentation Model", disabled=True, help="Requires real NIfTI data")
            
    else:
        color = "#e3f2fd" if "Pituitary" in p["status"] else "#fff3e0"
        border = "#1976d2" if "Pituitary" in p["status"] else "#fbc02d"
        name = "Pituitary Tumor" if "Pituitary" in p["status"] else "Meningioma"
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 20px; border-radius: 10px; border-left: 5px solid {border};">
            <h2 style="color: black; margin:0;">Detected: {name}</h2>
            <p>Confidence: <strong>89.5%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("ℹ️ Segmentation module is currently optimized for GLIOMA only. This case does not meet inclusion criteria.")

st.markdown("---")
st.markdown("#### Clinical Notes")
st.text(p["notes"])
