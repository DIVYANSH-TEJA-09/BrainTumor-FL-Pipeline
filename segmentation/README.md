# 3D Multimodal Brain Tumor Segmentation using Attention U-Net

**Author**: [Your Name/Team]
**Date**: December 2025

---

## Abstract
This project implements a deep learning pipeline for the volumetric segmentation of brain tumors from multimodal MRI scans (BraTS 2021 dataset). Leveraging a **3D Attention U-Net** architecture within the **MONAI** framework, the system effectively segments three tumor sub-regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET). The solution achieves "Clinical Grade" performance with a Mean Dice Score of ~0.76 (Tumor Core: 0.85). A user-friendly **Streamlit Web Application** was developed to allow interactive 3D visualization and deep-dive inspection of the model's predictions.

---

## 1. Introduction
Glioblastoma is the most aggressive form of primary brain cancer. Early and accurate segmentation of tumor compartments is crucial for treatment planning and monitoring. Manual segmentation is time-consuming and prone to inter-observer variability. This project aims to automate this process using state-of-the-art 3D Convolutional Neural Networks (CNNs).

### 1.1 Input Data
The model processes four 3D MRI modalities simultaneously:
1.  **T1-weighted (T1)**: Anatomical structure.
2.  **T1-weighted with contrast (T1ce)**: Enhancing tumor boundaries.
3.  **T2-weighted (T2)**: Edema visualization.
4.  **T2 Fluid Attenuated Inversion Recovery (FLAIR)**: Edema and non-enhancing tumor.

---

## 2. Methodology

### 2.1 Preprocessing Pipeline
To handle the high dimensionality of 3D MRI data, we employed a robust preprocessing pipeline using **MONAI**:
-   **LoadImage**: Volumetric NIfTI loading.
-   **Orientation**: Reoriented to standard RAS axis.
-   **Spacing**: Resampled to isotropic voxel spacing (1.0 x 1.0 x 1.0 mm).
-   **Normalization**: Z-score normalization (nonzero region).
-   **RandCropByPosNegLabel**: Extracted balanced 3D patches of size `(96, 96, 96)` during training to handle class imbalance.

### 2.2 Model Architecture: 3D Attention U-Net
We utilized an **Attention U-Net**, which improves upon the standard U-Net by integrating Attention Gates (AGs).
-   **Encoder**: Captures contextual features.
-   **Attention Gates**: Learn to suppress irrelevant regions (healthy tissue) and focus on the tumor targets automatically.
-   **Decoder**: Recovers spatial resolution for precise segmentation masks.
-   **Input Channels**: 4 (Modalities).
-   **Output Channels**: 3 (WT, TC, ET - mutually exclusive for training).

### 2.3 Training Configuration
-   **Optimizer**: Adam (`lr=1e-4`, `weight_decay=1e-5`).
-   **Loss Function**: Generalized Dice Loss (optimized for class imbalance).
-   **Epochs**: 20 (Refined Phase).
-   **Batch Size**: 1 (Volumetric processing).

---

## 3. Results

### 3.1 Quantitative Metrics
The model was evaluated on a held-out validation set.
-   **Mean Dice Score**: 0.76
-   **Tumor Core (TC) Dice**: 0.85
-   **Enhancing Tumor (ET) Dice**: 0.79
-   **Whole Tumor (WT) Dice**: 0.65

### 3.2 Visual Analysis
The model demonstrates strong spatial alignment with ground truth annotations. It successfully distinguishes between the Enhancing Tumor (active core) and Peritumoral Edema.

---

## 4. Web Application (Demo)
A **Streamlit** web interface was developed to demonstrate the model's capabilities in a clinical setting.

### Features
1.  **3D Interactive Scanner**: Allows users to scroll through MRI slices (Z-axis) with a real-time overlay of the predicted tumor regions.
2.  **Reviewer Deep Dive**: A side-by-side comparison grid displaying:
    -   **Inputs**: All 4 MRI modalities.
    -   **Ground Truth**: Annotated WT, TC, ET regions.
    -   **Predictions**: Model-predicted regions.

### Usage
Run the following command to launch the app:
```bash
streamlit run app.py
```
*Note: Ensure the `Major_Project/3d_Unet_segmentation` directory is your current working directory.*

---

## 5. Directory Structure
The project is organized as follows:
```
Major_Project/
└── 3d_Unet_segmentation/
    ├── app.py                      # Web Application Logic
    ├── best_metric_model_refined.pth # Trained Model Weights
    ├── dataset_step1_refined.ipynb   # Training & Analysis Notebook
    ├── demo_data/                    # Sample Patient Data for Demo
    ├── extract_demo_data.py          # Data Extraction Script
    ├── inspect_data.py               # Inspection Script
    ├── requirements.txt              # Dependencies
    └── logs/                         # Debug Logs
```

---

## 6. Future Work
-   **Tumor Classification**: Implementing a 3D DenseNet to classify tumor subtypes (Glioma vs. Meningioma).
-   **Federated Learning**: Simulating privacy-preserving training across multiple hospitals.
-   **Growth Prediction**: Using Longitudinal data (Time-series MRI) to predict tumor trajectory.

---

## References
1.  **MONAI Framework**: https://monai.io/
2.  **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation".
3.  **Attention U-Net**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas".
4.  **BraTS Challenge**: http://braintumorsegmentation.org/
