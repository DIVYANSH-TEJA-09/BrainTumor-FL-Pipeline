import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Longitudinal Analysis", layout="wide")

st.markdown("# 📈 Longitudinal Analysis - Prototype")
st.markdown("*Tracking tumor volume and treatment response over time.*")
st.info("🚧 This feature is currently under development. Data shown is simulated.")

if "selected_patient" in st.session_state:
    p = st.session_state["selected_patient"]
    st.subheader(f"Patient: {p['name']} ({p['id']})")
else:
    st.warning("No patient selected.")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Tumor Volume Trend")
    # Fake data
    data = pd.DataFrame({
        "Date": ["2023-01", "2023-03", "2023-06", "2023-09", "2023-12"],
        "Volume (cm3)": [45, 48, 52, 30, 25]
    })
    
    st.line_chart(data.set_index("Date"))
    st.caption("Note: Drop in volume correlates with surgical intervention in August 2023.")

with col2:
    st.markdown("### Key Metrics")
    st.metric("Current Volume", "25 cm³", "-16%")
    st.metric("Growth Rate", "-2.1 cm³/mo", "Regression")
    st.metric("RANO Criteria", "Partial Response", "Stable")

st.markdown("---")
st.markdown("### 🧬 Biomarker Tracking (Future)")
st.text("Placeholders for genomic markers (IDH1, MGMT) tracking.")
