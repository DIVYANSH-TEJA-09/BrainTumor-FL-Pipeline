"""
🧠 Brain Tumor Classification — Federated Learning Demo
=========================================================
Demonstrates the FL-trained SimpleCNN models (FedAvg, FedProx, QPSO).
Users can upload images or use sample test images from the dataset.
Shows predicted class with confidence bars and compares all 3 models.
"""

import streamlit as st
import os
import sys
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠", layout="wide")

# ─── paths ───────────────────────────────────────────────────────────────
FL_ROOT = os.path.abspath(os.path.dirname(__file__))
RESULTS = os.path.join(FL_ROOT, "results", "results_layer_by_layer_QPSO")
# Use Setup 1 models by default (best performing)
MODELS_DIR = os.path.join(RESULTS, "Setup_1", "models")

NUM_CLASSES = 4
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]
CLASS_ICONS = ["🔴", "🔵", "🟢", "🟣"]
IMG_SIZE = 112

# ─── model ───────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(NUM_CLASSES).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, device


def predict(model, device, image):
    """Run inference on a PIL Image. Returns (class_idx, probabilities)."""
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)), probs


def render_prediction_card(title, color_accent, pred_idx, probs, image):
    """Render a styled prediction result."""
    confidence = probs[pred_idx] * 100
    st.markdown(
        f'<div style="background:rgba(20,20,35,0.9);padding:20px;border-radius:12px;'
        f'border-top:4px solid {color_accent};margin-bottom:16px;">'
        f'<h3 style="color:{color_accent};margin:0 0 8px 0;">{title}</h3>'
        f'<div style="color:white;font-size:28px;font-weight:800;margin:4px 0;">'
        f'{CLASS_ICONS[pred_idx]} {CLASS_NAMES[pred_idx]}</div>'
        f'<div style="color:#aaa;font-size:14px;">Confidence: {confidence:.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Probability bar chart
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=CLASS_NAMES,
        orientation='h',
        marker_color=CLASS_COLORS,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='auto',
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.5)",
        xaxis=dict(range=[0, 100], title="Probability (%)", color="#aaa",
                   gridcolor="rgba(100,100,140,0.2)"),
        yaxis=dict(color="white"),
        font=dict(color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    .hero-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ─── sidebar ─────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Classification Controls")

# Setup selector
setup = st.sidebar.radio("Experiment Setup", ["Setup 1 (Natural)", "Setup 2 (Label Skew)"])
setup_dir = "Setup_1" if "1" in setup else "Setup_2"
models_dir = os.path.join(RESULTS, setup_dir, "models")

# Check available models
model_files = {}
for name, fname in [("FedAvg", "fedavg_best.pth"),
                     ("FedProx", "fedprox_best.pth"),
                     ("QPSO-FL", "qpso_best.pth")]:
    path = os.path.join(models_dir, fname)
    if os.path.exists(path):
        model_files[name] = path

if not model_files:
    st.error(f"No model weights found in `{models_dir}`. Please ensure .pth files are present.")
    st.stop()

selected_models = st.sidebar.multiselect(
    "Compare Models",
    list(model_files.keys()),
    default=list(model_files.keys()),
)

st.sidebar.markdown("---")
input_mode = st.sidebar.radio("Image Source", ["Upload Image", "Sample from Dataset"])

# ─── title ───────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🧠 Brain Tumor Classification</p>',
            unsafe_allow_html=True)
st.markdown("**Federated Learning · SimpleCNN (~120K params) · "
            "FedAvg vs FedProx vs QPSO-FL**")
st.markdown("---")

# ─── image input ─────────────────────────────────────────────────────────
image = None

if input_mode == "Upload Image":
    uploaded = st.file_uploader(
        "Upload a brain MRI image (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

elif input_mode == "Sample from Dataset":
    # Try to find sample images from the results plots (confusion matrices show sample data)
    # Or look for actual dataset images
    sample_dirs = [
        os.path.join(FL_ROOT, "data"),
        os.path.join(FL_ROOT, "sample_images"),
    ]

    # Search for any sample images
    sample_images = []
    for d in sample_dirs:
        if os.path.isdir(d):
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                sample_images.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))

    if sample_images:
        # Group by class if possible
        selected = st.selectbox("Select a sample image", sample_images,
                                format_func=lambda x: os.path.basename(x))
        image = Image.open(selected).convert("RGB")
    else:
        st.info(
            "💡 No sample images found locally. You can either:\n"
            "1. **Upload an image** using the sidebar option\n"
            "2. **Add sample images** to `federated_learning/sample_images/` "
            "(one subfolder per class: glioma/, meningioma/, notumor/, pituitary/)"
        )

# ─── inference ───────────────────────────────────────────────────────────
if image is not None:
    # Show the input image
    st.subheader("📷 Input Image")
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(image, caption="Input MRI", use_container_width=True)
    with col_info:
        w, h = image.size
        st.markdown(f"**Resolution:** {w}×{h} → resized to {IMG_SIZE}×{IMG_SIZE}")
        st.markdown(f"**Models:** {', '.join(selected_models)}")
        st.markdown(f"**Setup:** {setup}")

    st.markdown("---")
    st.subheader("🔬 Classification Results")

    if not selected_models:
        st.warning("Select at least one model from the sidebar.")
    else:
        # Run all selected models
        cols = st.columns(len(selected_models))
        model_colors = {"FedAvg": "#1f77b4", "FedProx": "#ff7f0e", "QPSO-FL": "#2ca02c"}

        results = {}
        for idx, name in enumerate(selected_models):
            with cols[idx]:
                model, device = load_model(model_files[name])
                pred_idx, probs = predict(model, device, image)
                results[name] = (pred_idx, probs)
                render_prediction_card(
                    name,
                    model_colors.get(name, "#666"),
                    pred_idx, probs, image,
                )

        # Consensus section
        if len(results) > 1:
            st.markdown("---")
            st.subheader("🤝 Model Consensus")

            predictions = [CLASS_NAMES[r[0]] for r in results.values()]
            unanimous = len(set(predictions)) == 1

            if unanimous:
                st.success(
                    f"✅ **All {len(results)} models agree:** "
                    f"{CLASS_ICONS[list(results.values())[0][0]]} "
                    f"**{predictions[0]}**"
                )
            else:
                # Majority vote
                from collections import Counter
                votes = Counter(predictions)
                winner, count = votes.most_common(1)[0]
                winner_idx = CLASS_NAMES.index(winner)
                st.warning(
                    f"⚠️ **Models disagree.** Majority vote ({count}/{len(results)}): "
                    f"{CLASS_ICONS[winner_idx]} **{winner}**"
                )

                # Show disagreement details
                for name, (pred_idx, probs) in results.items():
                    emoji = "✅" if CLASS_NAMES[pred_idx] == winner else "❌"
                    st.markdown(
                        f"  {emoji} **{name}:** {CLASS_NAMES[pred_idx]} "
                        f"({probs[pred_idx]*100:.1f}%)"
                    )

        # Average confidence across models
        if len(results) > 1:
            avg_probs = np.mean([r[1] for r in results.values()], axis=0)
            ensemble_pred = int(np.argmax(avg_probs))

            st.markdown("---")
            st.subheader("📊 Ensemble Average (All Models)")
            render_prediction_card(
                "Ensemble Average",
                "#E91E63",
                ensemble_pred, avg_probs, image,
            )

else:
    # Welcome state
    st.markdown("""
    ### How it works
    1. **Choose a setup** — Natural heterogeneity (Setup 1) or Label Skew (Setup 2)
    2. **Select models** — Compare FedAvg, FedProx, and QPSO-FL side by side
    3. **Upload or select an image** — Any brain MRI (axial slice)
    4. **See results** — Class prediction with confidence bars for each model

    The models were trained using **federated learning** across 3 simulated hospitals,
    each with different data distributions. The QPSO-FL model uses our novel
    **Layer-by-Layer QPSO aggregation** for fairer global model performance.
    """)

    # Show model info cards
    st.markdown("---")
    info_cols = st.columns(3)
    model_info = [
        ("FedAvg", "#1f77b4", "Weighted average of client updates. Standard baseline."),
        ("FedProx", "#ff7f0e", "Adds proximal regularization (μ=0.01) to prevent client drift."),
        ("QPSO-FL", "#2ca02c", "Layer-by-layer quantum PSO with validation-loss fitness. Our contribution."),
    ]
    for col, (name, color, desc) in zip(info_cols, model_info):
        with col:
            st.markdown(
                f'<div style="background:rgba(20,20,35,0.9);padding:20px;border-radius:12px;'
                f'border-top:3px solid {color};">'
                f'<h4 style="color:{color};margin:0 0 8px 0;">{name}</h4>'
                f'<p style="color:#aaa;font-size:13px;margin:0;">{desc}</p></div>',
                unsafe_allow_html=True,
            )
