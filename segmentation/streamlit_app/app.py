"""
🧠 NeuroAI Brain Tumor Segmentation
=====================================
Streamlit app with multiple modes:
  1. Upload & Demo Data — Load demo patients or upload your own MRI
  2. Slice-by-Slice Segmentation Viewer
  3. 3D Interactive Visualization
"""

import streamlit as st

st.set_page_config(
    page_title="NeuroAI Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }
    .main .block-container { padding-top: 1.5rem; }
    .stButton>button {
        width: 100%; border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: 600; border: none;
        padding: 12px; font-size: 16px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
    .hero-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .feature-card {
        background: rgba(30, 30, 50, 0.8);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(100, 126, 234, 0.2);
        height: 100%;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: rgba(100, 126, 234, 0.5);
        box-shadow: 0 8px 30px rgba(100, 126, 234, 0.15);
        transform: translateY(-3px);
    }
    .feature-icon { font-size: 2.2rem; margin-bottom: 12px; }
    .feature-title { font-size: 1.2rem; font-weight: 700; color: #E8E8E8; margin-bottom: 8px; }
    .feature-desc { color: #999; font-size: 0.9rem; line-height: 1.5; }
    .quick-start-box {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.1), rgba(39, 174, 96, 0.05));
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Landing Page ────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🧠 NeuroAI Brain Tumor Segmentation</p>',
            unsafe_allow_html=True)
st.markdown("**3D Attention U-Net · BraTS 2021 · MONAI Framework**")
st.markdown("---")

# ─── Quick Start ─────────────────────────────────────────────────────────
st.markdown("""
<div class="quick-start-box">
    <div style="font-size:1.1rem;font-weight:700;color:#2ECC71;margin-bottom:8px;">
        🚀 Quick Start
    </div>
    <div style="color:#CCC;font-size:0.95rem;">
        <b>New here?</b> Head to <b>Upload & Demo Data</b> to try the built-in demo patients
        or upload your own MRI scans (T1, T1ce, T2, FLAIR).
        The AI will run 3D brain tumor segmentation in real-time!
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ─── Feature Cards ───────────────────────────────────────────────────────
col0, col1, col2 = st.columns(3)

with col0:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📤</div>
        <div class="feature-title">Upload & Demo Data</div>
        <div class="feature-desc">
            Load pre-built BraTS 2021 demo patients or upload your own
            MRI modalities (T1, T1ce, T2, FLAIR). Run AI inference
            instantly and compare with ground truth.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    if st.button("Open Upload & Demo →", key="upload"):
        st.switch_page("pages/0_Upload_Data.py")

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <div class="feature-title">Slice-by-Slice Viewer</div>
        <div class="feature-desc">
            Scroll through MRI slices with segmentation overlay.
            View all 4 modalities (T1, T1ce, T2, FLAIR) alongside
            ground truth and AI predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    if st.button("Open Slice Viewer →", key="slice"):
        st.switch_page("pages/1_Slice_Viewer.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🌐</div>
        <div class="feature-title">3D Visualization</div>
        <div class="feature-desc">
            Interactive 3D rendering of brain and tumor regions.
            Rotate, zoom, compare AI prediction vs ground truth
            side-by-side with Dice scores.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    if st.button("Open 3D Viewer →", key="3d"):
        st.switch_page("pages/2_3D_Visualization.py")

st.markdown("---")

# ─── Quick stats ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Architecture", "3D Attn U-Net")
c2.metric("Mean Dice", "0.76")
c3.metric("Tumor Core", "0.85")
c4.metric("Dataset", "BraTS 2021")

# ─── How it works ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔄 How It Works")

steps = [
    ("1️⃣", "**Upload / Select Data**", "Upload your own MRI scans or choose from built-in BraTS demo patients."),
    ("2️⃣", "**AI Inference**", "The 3D Attention U-Net processes all 4 modalities using sliding window inference."),
    ("3️⃣", "**Visualize Results**", "Explore segmentation results in 2D slice view or interactive 3D rendering."),
    ("4️⃣", "**Compare & Evaluate**", "Compare AI predictions with ground truth. View Dice scores and volume statistics."),
]

step_cols = st.columns(4)
for i, (icon, title, desc) in enumerate(steps):
    with step_cols[i]:
        st.markdown(f"""
        <div style="text-align:center;padding:12px;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-size:0.95rem;font-weight:600;color:#E8E8E8;margin:8px 0 4px;">{title}</div>
            <div style="color:#888;font-size:0.85rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#555;font-size:12px;padding:10px;">
    🧠 NeuroAI Brain Tumor Segmentation · Built with Streamlit, MONAI & PyTorch
</div>
""", unsafe_allow_html=True)
