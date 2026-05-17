You are working on `BrainTumor-FL-Pipeline`, a clinical brain tumor AI dashboard built with Streamlit (dashboard/app.py). The ML backend has three modules: FL-QPSO classification (ResNet-18, 3 classes: Glioma / Meningioma / Pituitary), 3D Attention U-Net segmentation, and LSTM tumor progression forecasting.

Your job is to diagnose and fix the existing dashboard, then rebuild its UI to be clinician-ready. Work through the following in order.

---

## PHASE 1 — DIAGNOSE & FIX CRITICAL BUGS

### Bug 1: Hardcoded classification label
Search the entire dashboard/ and federated_learning/ directories for any place where the tumor type is hardcoded as "glioma" regardless of model output. The model (ResNet-18) was trained on 3 classes. Fix the inference pipeline so:
- The model's actual softmax output is used to determine the predicted class
- Class index → label mapping is: {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}
- The predicted label, confidence score, and per-class probabilities are all surfaced to the UI

### Bug 2: MRI scan vs MRI image format mismatch
The model was trained on 2D MRI image slices (JPG/PNG from Kaggle dataset). The app currently accepts MRI scan files (NIfTI .nii / DICOM .dcm). This causes a preprocessing mismatch.

Fix the input pipeline:
- Accept both 2D images (JPG/PNG) AND 3D scans (NIfTI/DICOM) as input
- For 3D scans: extract the middle axial slice (or let the user pick a slice index), convert to a 224x224 grayscale → 3-channel tensor, apply the same normalization used during training (ImageNet mean/std)
- For 2D images: apply the same preprocessing directly
- Add clear UI labels telling the user which format they're uploading

### Bug 3: Laggy / frozen inference
Profile where the latency comes from. Common culprits:
- Model being reloaded from disk on every inference call — fix by caching the model with @st.cache_resource
- Segmentation module (3D U-Net) running synchronously on CPU — add a loading spinner and run heavy inference in a thread or async where possible
- Large NIfTI files being fully loaded into memory — only load the slice needed for classification

---

## PHASE 2 — REBUILD THE UI (Light, Clean, Clinical)

Redesign dashboard/app.py with the following specification. Use Streamlit's native components + custom CSS injected via st.markdown.

### Visual Theme
- Background: #FFFFFF (pure white)
- Primary accent: #3B82F6 (light blue) for buttons and highlights
- Secondary: #F0F7FF for card backgrounds
- Text: #1E293B (near-black)
- Font: Inter or system-sans-serif
- No dark backgrounds anywhere
- Sidebar: light gray (#F8FAFC)

### Layout & Navigation
Use a sidebar with these pages:
1. Patient Upload
2. Classification Report
3. Segmentation Viewer
4. Progression Forecast
5. Multi-Patient View (stub — show "Coming Soon" placeholder for now)

### Page 1 — Patient Upload
- Clean upload widget: accepts JPG/PNG (MRI image slice) or NII/DCM (MRI scan)
- Patient metadata form: Patient ID, Age, Sex, Scan Date, Clinical Notes (optional)
- Single "Analyze" button — nothing runs until this is clicked
- Show a preview of the uploaded image/slice after upload

### Page 2 — Classification Report
Only shown after "Analyze" is clicked.

Show:
- Predicted tumor type (large, bold — Glioma / Meningioma / Pituitary)
- Confidence percentage with a color-coded bar (green ≥ 85%, yellow 65–84%, red < 65%)
- Per-class probability breakdown as a horizontal bar chart
- A collapsible "Clinical Context" section — expand on click — that shows:
  - **If Glioma:** "High-grade gliomas are aggressive. Immediate neurosurgery referral, MRI contrast enhancement, molecular profiling (IDH1, MGMT) recommended. RANO criteria apply for follow-up."
  - **If Meningioma:** "Usually benign. Watchful waiting or surgical resection depending on size and symptoms. Annual MRI follow-up standard."
  - **If Pituitary:** "Evaluate for hormonal dysfunction (prolactin, GH, ACTH). Ophthalmology referral if visual field defects. Transsphenoidal surgery if indicated."
- A "Download PDF Report" button that generates a simple PDF with patient metadata + classification results

### Page 3 — Segmentation Viewer
Only loads when user clicks "Run Segmentation" button (lazy — not automatic).

Show:
- Axial/Coronal/Sagittal slice selector (slider)
- Overlay toggle: show mask on/off
- Color-coded regions: Whole Tumor (yellow), Tumor Core (red), Enhancing Tumor (hot pink)
- Dice scores displayed beneath: WT / TC / ET
- Download segmentation mask button

### Page 4 — Progression Forecast
Only loads when user clicks "Run Progression Forecast" button.

Show:
- Input: series of historical volume measurements (user can add rows: date + volume in cm³)
- Output: line chart showing 6-month projection using the LSTM model
- RANO status badge: CR / PR / SD / PD with color coding
- Risk alert if projected growth > 25% in 3 months

### General UX Rules
- Every heavy computation must be behind a button click — nothing runs on page load
- Every section that has detail should be in an st.expander (collapsed by default)
- Add a st.spinner with a descriptive message for all model inference calls
- Use st.columns for side-by-side layouts wherever possible
- Mobile-friendly: don't use fixed pixel widths

---

## PHASE 3 — EXTENDABILITY HOOKS (implement stubs, don't build yet)

Add clearly marked TODO stubs for:
1. Multi-patient session: a selectbox to switch between loaded patient records stored in st.session_state
2. AI chatbot panel: a chat interface (st.chat_input / st.chat_message) that will later query a RAG system about the patient's scan
3. Federated node selector: a sidebar toggle to switch which hospital node's model weights to use for inference

---

## CONSTRAINTS
- Do not break the existing ML model loading or training code — only touch dashboard/ and the inference preprocessing pipeline
- All changes must be backward compatible with the existing model checkpoints
- Add docstrings to every new function
- Use type hints throughout
- Keep all new dependencies in requirements.txt

Start by reading dashboard/app.py and dashboard/config.py in full, then proceed with Phase 1 bug diagnosis.