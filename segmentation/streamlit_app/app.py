"""
🧠 NeuroAI Brain Tumor Segmentation
=====================================
Streamlit app with 2 modes:
  1. Slice-by-Slice Segmentation Viewer
  2. 3D Interactive Visualization
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
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
    .hero-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Landing Page ────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🧠 NeuroAI Brain Tumor Segmentation</p>',
            unsafe_allow_html=True)
st.markdown("**3D Attention U-Net · BraTS 2021 · MONAI Framework**")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔬 Slice-by-Slice Viewer
    Scroll through MRI slices with segmentation overlay.
    View all 4 modalities (T1, T1ce, T2, FLAIR) alongside
    ground truth and AI predictions.
    """)
    if st.button("Open Slice Viewer →", key="slice"):
        st.switch_page("pages/1_Slice_Viewer.py")

with col2:
    st.markdown("""
    ### 🌐 3D Visualization
    Interactive 3D rendering of the brain and tumor regions.
    Rotate, zoom, compare prediction vs ground truth side-by-side.
    """)
    if st.button("Open 3D Viewer →", key="3d"):
        st.switch_page("pages/2_3D_Visualization.py")

st.markdown("---")

# Quick stats
c1, c2, c3, c4 = st.columns(4)
c1.metric("Architecture", "3D Attn U-Net")
c2.metric("Mean Dice", "0.76")
c3.metric("Tumor Core", "0.85")
c4.metric("Dataset", "BraTS 2021")
