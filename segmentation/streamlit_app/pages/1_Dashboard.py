import streamlit as st
import json
import os

st.set_page_config(page_title="Dashboard", layout="wide")

st.markdown("# 🏥 Clinician Dashboard")

# Load mock data
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/mock_patients.json")
try:
    with open(DATA_PATH, "r") as f:
        patients = json.load(f)
except FileNotFoundError:
    patients = []

# --- METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", "142", "+4 this week")
col2.metric("Pending Reviews", str(len(patients)), "Urgent")
col3.metric("Glioma Cases", "2", "Segmented")

st.markdown("---")
st.markdown("### 📋 Patient List")

# Display as a table with items
col1, col2, col3, col4, col5 = st.columns([1.5, 2, 1.5, 3, 2])
col1.markdown("**Patient ID**")
col2.markdown("**Scan Date**")
col3.markdown("**Tumor Type**")
col4.markdown("**Status**")
col5.markdown("**Action**")

for p in patients:
    with st.container():
        c1, c2, c3, c4, c5 = st.columns([1.5, 2, 1.5, 3, 2])
        c1.text(p["id"])
        c2.text(p["scan_date"])
        
        tumor_type = ""
        if "Glioma" in p["status"]:
            tumor_type = "Glioma"
            status_badge = "🟢 Segmented"
        elif "Meningioma" in p["status"]:
            tumor_type = "Meningioma"
            status_badge = "🔴 Seg. Not Available"
        elif "Pituitary" in p["status"]:
            tumor_type = "Pituitary"
            status_badge = "🔴 Seg. Not Available"
            
        c3.text(tumor_type)
        c4.markdown(f"`{status_badge}`")
        
        if c5.button("Open Detail", key=p["id"]):
            st.session_state["selected_patient"] = p
            st.switch_page("pages/2_Patient_Detail.py")
        
        st.markdown("---")
