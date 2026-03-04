"""
This application extends an existing working 3D glioma segmentation Streamlit app.
The segmentation module is preserved without modification.
All additional components (classification, longitudinal analysis, reports)
are mock implementations for demonstration and review purposes.
"""

import streamlit as st
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NeuroAI Clinical Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS STYLING ---
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- APP NAVIGATION ---
st.title("🧠 NeuroAI Brain Tumor Management System")

# Check for mock login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Clinician Login")
        st.info("Demo Access: Press Login")
        if st.button("Login"):
            st.session_state.logged_in = True
            st.rerun()
else:
    st.success(f"Verified Clinician Access. System Ready.")
    st.switch_page("pages/1_Dashboard.py")
