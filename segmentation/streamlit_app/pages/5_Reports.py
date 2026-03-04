import streamlit as st
import datetime

st.set_page_config(page_title="Reports", layout="wide")

st.markdown("# 📑 Clinical Reporting")

if "selected_patient" not in st.session_state:
    st.warning("No patient selected.")
    st.stop()

p = st.session_state["selected_patient"]

# Mock Report Generation
report_date = datetime.date.today()

st.sidebar.header("Report Settings")
report_type = st.sidebar.selectbox("Report Type", ["Radiology Standard", "Surgical Planning", "Oncology Summary"])

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"### Generated Report: {p['id']}")
    
    report_content = f"""
    NEURO-ONCOLOGY TUMOR BOARD REPORT
    --------------------------------------------------
    Patient ID: {p['id']}
    Name: {p['name']}
    Date: {report_date}
    Type: {report_type}
    
    CLINICAL HISTORY:
    {p['notes']}
    
    IMAGING FINDINGS:
    MRI Brain demonstrates a mass lesion consistent with {p['status']}.
    Multi-modal sequences (T1, T1ce, T2, FLAIR) acquired.
    
    AI ANALYSIS:
    - Classification: {p['status']}
    - Confidence: High
    - Segmentation Status: {'Completed' if 'Glioma' in p['status'] else 'Not Indicated'}
    
    RECOMMENDATIONS:
    Please refer to the 3D Segmentation module for volumetric analysis.
    Correlate with histopathology.
    --------------------------------------------------
    Electronically Signed by NeuroAI System
    """
    
    st.text_area("Report Preview", report_content, height=400)

with col2:
    st.markdown("### Actions")
    st.button("🖨️ Print Report")
    st.button("📥 Export to EMR")
    st.button("✉️ Email to Referrer")
    
    st.markdown("---")
    st.download_button("Download PDF", report_content, file_name=f"Report_{p['id']}.txt")
