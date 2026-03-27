"""
📤 Upload & Demo Data
======================
Upload your own MRI modalities (T1, T1ce, T2, FLAIR)
or use built-in demo patients for quick testing.
"""

import streamlit as st
import os
import sys
import shutil
import tempfile
import nibabel as nib
import numpy as np

st.set_page_config(page_title="Upload & Demo Data", layout="wide")

# ─── paths & inference ───────────────────────────────────────────────────
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
from inference import (
    ensure_prediction,
    get_all_patients,
    get_demo_candidates,
    prepare_demo_patient,
    prepare_uploaded_patient,
    DEMO_DIR,
)

# ─── CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }
    .main .block-container { padding-top: 1.5rem; }
    .upload-card {
        background: rgba(30, 30, 50, 0.8);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(100, 126, 234, 0.3);
        margin-bottom: 16px;
    }
    .upload-card:hover {
        border-color: rgba(100, 126, 234, 0.6);
        box-shadow: 0 4px 20px rgba(100, 126, 234, 0.15);
    }
    .demo-card {
        background: linear-gradient(135deg, rgba(30, 30, 60, 0.9), rgba(40, 40, 70, 0.9));
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(46, 204, 113, 0.3);
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .demo-card:hover {
        border-color: rgba(46, 204, 113, 0.6);
        box-shadow: 0 4px 20px rgba(46, 204, 113, 0.15);
        transform: translateY(-2px);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-ready { background: rgba(46, 204, 113, 0.2); color: #2ECC71; }
    .badge-pending { background: rgba(241, 196, 15, 0.2); color: #F1C40F; }
    .badge-missing { background: rgba(231, 76, 60, 0.2); color: #E74C3C; }
    .hero-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .section-title {
        font-size: 1.3rem; font-weight: 700;
        color: #667eea;
        margin-bottom: 8px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover { transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ─── title ───────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">📤 Upload & Demo Data</p>',
            unsafe_allow_html=True)
st.markdown("**Upload your own MRI scans or use built-in demo patients for quick testing.**")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: DEMO DATA    |    TAB 2: MANUAL UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
tab_demo, tab_upload = st.tabs(["🧪 Demo Patients", "📁 Upload Your Own MRI"])

# ─── TAB 1: Demo Data ────────────────────────────────────────────────────
with tab_demo:
    st.markdown("""
    <div style="background:rgba(30,30,50,0.6);padding:16px;border-radius:10px;
                border-left:4px solid #2ECC71;margin-bottom:20px;">
        <b>🧪 Demo Patients</b> — Pre-loaded BraTS 2021 brain MRI scans with
        ground truth segmentations. Select a patient and click
        <b>"Prepare & Run Inference"</b> to generate AI predictions instantly.
    </div>
    """, unsafe_allow_html=True)

    # Get demo candidates (raw BraTS folders) and already-prepared patients
    candidates = get_demo_candidates()
    ready_patients = get_all_patients()

    if not candidates and not ready_patients:
        st.warning("No demo data found in `demo_data/` subfolders. "
                   "Please add BraTS patient folders or use the Upload tab.")
    else:
        # Show ready patients
        if ready_patients:
            st.markdown("#### ✅ Ready Patients")
            st.caption("These patients have pre-computed predictions and can be viewed immediately.")

            cols = st.columns(3)
            for idx, pid in enumerate(ready_patients):
                with cols[idx % 3]:
                    # Check what data exists
                    has_pred = os.path.exists(os.path.join(DEMO_DIR, f"{pid}_pred.nii.gz"))
                    has_img = os.path.exists(os.path.join(DEMO_DIR, f"{pid}_image.nii.gz"))
                    has_lbl = os.path.exists(os.path.join(DEMO_DIR, f"{pid}_label.nii.gz"))

                    pred_badge = '<span class="status-badge badge-ready">✓ Prediction</span>' if has_pred else '<span class="status-badge badge-missing">✗ No Pred</span>'
                    lbl_badge = '<span class="status-badge badge-ready">✓ Ground Truth</span>' if has_lbl else '<span class="status-badge badge-pending">○ No GT</span>'

                    st.markdown(f"""
                    <div class="demo-card">
                        <div style="font-size:16px;font-weight:700;color:#E8E8E8;margin-bottom:6px;">
                            🧠 {pid}
                        </div>
                        <div style="margin-bottom:8px;">
                            {pred_badge} {lbl_badge}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("🔬 Slice Viewer", key=f"sv_{pid}"):
                            st.session_state["selected_patient"] = pid
                            st.switch_page("pages/1_Slice_Viewer.py")
                    with c2:
                        if st.button("🌐 3D View", key=f"3d_{pid}"):
                            st.session_state["selected_patient"] = pid
                            st.switch_page("pages/2_3D_Visualization.py")

        # Show candidates that need inference
        pending = [c for c in candidates if c not in ready_patients]
        if pending:
            st.markdown("---")
            st.markdown("#### 🔄 Available for Processing")
            st.caption("These patients have raw MRI data but need inference to generate predictions.")

            cols = st.columns(3)
            for idx, pid in enumerate(pending):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="demo-card" style="border-color:rgba(241,196,15,0.3);">
                        <div style="font-size:16px;font-weight:700;color:#E8E8E8;margin-bottom:6px;">
                            🧠 {pid}
                        </div>
                        <span class="status-badge badge-pending">⏳ Needs Processing</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("🚀 Prepare & Run Inference", key=f"prep_{pid}"):
                        with st.spinner(f"Running inference on {pid}..."):
                            success = prepare_demo_patient(pid)
                            if success:
                                st.success(f"✅ {pid} is ready!")
                                st.rerun()
                            else:
                                st.error(f"Failed to process {pid}.")


# ─── TAB 2: Manual Upload ────────────────────────────────────────────────
with tab_upload:
    st.markdown("""
    <div style="background:rgba(30,30,50,0.6);padding:16px;border-radius:10px;
                border-left:4px solid #667eea;margin-bottom:20px;">
        <b>📁 Upload Your Own MRI Scans</b> — Upload 4 NIfTI files
        (T1, T1ce, T2, FLAIR) to run brain tumor segmentation.
        Optionally include a ground truth segmentation mask for comparison.
    </div>
    """, unsafe_allow_html=True)

    # Patient ID input
    custom_id = st.text_input(
        "🏷️ Patient / Case ID",
        value="Custom_Patient_01",
        help="A unique identifier for this upload. Will be used to name the output files.",
    )

    st.markdown("---")
    st.markdown('<p class="section-title">📂 Upload MRI Modalities</p>',
                unsafe_allow_html=True)
    st.caption("Upload NIfTI files (.nii or .nii.gz) for each modality. All 4 modalities are **required**.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="upload-card">
            <div style="font-size:15px;font-weight:600;color:#667eea;margin-bottom:4px;">
                🔵 T1-weighted
            </div>
            <div style="color:#888;font-size:12px;">Pre-contrast structural scan</div>
        </div>
        """, unsafe_allow_html=True)
        t1_file = st.file_uploader("T1", type=["nii", "nii.gz", "gz"], key="t1_upload",
                                    label_visibility="collapsed")

        st.markdown("""
        <div class="upload-card">
            <div style="font-size:15px;font-weight:600;color:#9B59B6;margin-bottom:4px;">
                🟣 T1ce (Contrast-Enhanced)
            </div>
            <div style="color:#888;font-size:12px;">Post-gadolinium contrast scan</div>
        </div>
        """, unsafe_allow_html=True)
        t1ce_file = st.file_uploader("T1ce", type=["nii", "nii.gz", "gz"], key="t1ce_upload",
                                      label_visibility="collapsed")

    with col2:
        st.markdown("""
        <div class="upload-card">
            <div style="font-size:15px;font-weight:600;color:#3498DB;margin-bottom:4px;">
                🔷 T2-weighted
            </div>
            <div style="color:#888;font-size:12px;">Highlights edema and fluid</div>
        </div>
        """, unsafe_allow_html=True)
        t2_file = st.file_uploader("T2", type=["nii", "nii.gz", "gz"], key="t2_upload",
                                    label_visibility="collapsed")

        st.markdown("""
        <div class="upload-card">
            <div style="font-size:15px;font-weight:600;color:#2ECC71;margin-bottom:4px;">
                🟢 FLAIR
            </div>
            <div style="color:#888;font-size:12px;">Fluid-attenuated inversion recovery</div>
        </div>
        """, unsafe_allow_html=True)
        flair_file = st.file_uploader("FLAIR", type=["nii", "nii.gz", "gz"], key="flair_upload",
                                       label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p class="section-title">📂 Optional: Ground Truth Segmentation</p>',
                unsafe_allow_html=True)
    st.caption("Upload a segmentation mask (labels: 1=NCR, 2=ED, 4=ET) for Dice score comparison.")
    seg_file = st.file_uploader("Segmentation Mask", type=["nii", "nii.gz", "gz"],
                                 key="seg_upload", label_visibility="collapsed")

    # ─── Upload status summary ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Upload Summary")

    uploads = {
        "T1": t1_file,
        "T1ce": t1ce_file,
        "T2": t2_file,
        "FLAIR": flair_file,
    }
    status_cols = st.columns(5)
    all_required = True
    for i, (name, f) in enumerate(uploads.items()):
        with status_cols[i]:
            if f is not None:
                st.markdown(f'<span class="status-badge badge-ready">✓ {name}</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="status-badge badge-missing">✗ {name}</span>',
                            unsafe_allow_html=True)
                all_required = False
    with status_cols[4]:
        if seg_file is not None:
            st.markdown('<span class="status-badge badge-ready">✓ Seg (optional)</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-pending">○ Seg (optional)</span>',
                        unsafe_allow_html=True)

    # ─── process button ──────────────────────────────────────────────────
    st.markdown("")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        run_disabled = not all_required
        if run_disabled:
            st.warning("⚠️ Please upload all 4 required modalities (T1, T1ce, T2, FLAIR) before proceeding.")

        run_clicked = st.button(
            "🚀 Run Segmentation Inference",
            disabled=run_disabled,
            use_container_width=True,
        )

    if run_clicked and all_required:
        with st.spinner("Processing uploaded files and running inference..."):
            success = prepare_uploaded_patient(
                patient_id=custom_id,
                t1_file=t1_file,
                t1ce_file=t1ce_file,
                t2_file=t2_file,
                flair_file=flair_file,
                seg_file=seg_file,
            )

        if success:
            st.success(f"✅ Inference complete for **{custom_id}**!")
            st.balloons()

            st.markdown("---")
            st.markdown("#### 🎯 View Results")
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                if st.button("🔬 Open Slice Viewer", key="goto_slice", use_container_width=True):
                    st.session_state["selected_patient"] = custom_id
                    st.switch_page("pages/1_Slice_Viewer.py")
            with res_c2:
                if st.button("🌐 Open 3D Viewer", key="goto_3d", use_container_width=True):
                    st.session_state["selected_patient"] = custom_id
                    st.switch_page("pages/2_3D_Visualization.py")
        else:
            st.error("❌ Inference failed. Please check the error messages above.")

# ─── footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#555;font-size:12px;padding:10px;">
    🧠 NeuroAI Brain Tumor Segmentation · 3D Attention U-Net · BraTS 2021 · MONAI Framework
</div>
""", unsafe_allow_html=True)
