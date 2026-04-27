"""
Dashboard Configuration
========================
Centralized paths, constants, and model configs for the NeuroAI Dashboard.
"""

import os
from pathlib import Path

# ─── Directory Layout ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent

# Upload storage (temporary)
UPLOAD_DIR = DASHBOARD_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ─── Classification ─────────────────────────────────────────────────
# ResNet-18 (Transfer Learning) trained with QPSO-FL
# 3 classes: Glioma (0), Meningioma (1), Pituitary (2)
# Input: 224×224 RGB, ImageNet-normalized
CLASSIFICATION_DIR = PROJECT_ROOT / "federated_learning"
CLASSIFICATION_RESULTS = CLASSIFICATION_DIR / "results" / "results_transfer_learning" / "Natural Setup"

CLASSIFICATION_MODELS = {
    "QPSO-FL": CLASSIFICATION_RESULTS / "models" / "qpso_best.pth",
}

NUM_CLASSES = 4
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]
CLASS_ICONS = ["R", "B", "G", "P"]
IMG_SIZE = 224  # ResNet-18 input size

# ─── Segmentation ────────────────────────────────────────────────────
SEGMENTATION_DIR = PROJECT_ROOT / "segmentation"
SEGMENTATION_APP_DIR = SEGMENTATION_DIR / "streamlit_app"

# Prefer refined model
_seg_candidates = [
    SEGMENTATION_APP_DIR / "best_metric_model_refined.pth",
    SEGMENTATION_DIR / "best_metric_model_refined.pth",
    SEGMENTATION_APP_DIR / "best_metric_model.pth",
    SEGMENTATION_DIR / "best_metric_model.pth",
]
SEGMENTATION_MODEL_PATH = None
for _c in _seg_candidates:
    if _c.exists():
        SEGMENTATION_MODEL_PATH = _c
        break

SEG_TUMOR_REGIONS = ["Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"]
SEG_REGION_COLORS = ["#E74C3C", "#3498DB", "#F39C12"]

# ─── Progression ─────────────────────────────────────────────────────
PROGRESSION_DIR = PROJECT_ROOT / "progression"
PROGRESSION_RESULTS = PROGRESSION_DIR / "results"

LSTM_MODELS = {
    "HGG": PROGRESSION_RESULTS / "phase2_hgg_lstm_model.pth",
    "LGG": PROGRESSION_RESULTS / "phase2_lgg_lstm_model.pth",
}
SPATIAL_UNET_PATH = PROGRESSION_RESULTS / "spatial_unet_best.pth"
SPATIAL_EVAL_PATH = PROGRESSION_RESULTS / "spatial_eval.json"
SPATIAL_CROP = (96, 96, 64)

PREDICTION_INDEX_PATH = PROGRESSION_DIR / "streamlit_data" / "prediction_index.json"

# Grade-median logistic parameters (from phase1 fitting)
LOGISTIC_DEFAULTS = {
    "HGG": {"v0": 100000, "k": 1900000, "r": 0.073},
    "LGG": {"v0": 50000, "k": 2500000, "r": 0.062},
}

# ─── Flask ───────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "neuroai-dashboard-dev-key")
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB upload limit
