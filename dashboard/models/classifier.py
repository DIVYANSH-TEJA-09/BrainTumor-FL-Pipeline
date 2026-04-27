"""
Classification Inference
=========================
Loads the FL-trained ResNet-18 (QPSO-FL) model for 3-class brain tumor
classification: Glioma (0), Meningioma (1), Pituitary (2).

Model: BrainTumorResNet — ResNet-18 with replaced FC (512 → 3).
Training: Federated QPSO, transfer learning, Natural Setup.
Input: 224×224 RGB, ImageNet-normalized.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import base64
from io import BytesIO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSIFICATION_MODELS, CLASS_NAMES, CLASS_COLORS, CLASS_ICONS


# ─── Model Architecture (from federated_learning/src/model.py) ──────
class BrainTumorResNet(nn.Module):
    """
    ResNet-18 with final FC replaced: 512 → num_classes.
    Must match the architecture used in FL training exactly.
    """
    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        # During inference we don't need pretrained ImageNet weights —
        # we load our own FL-trained weights.
        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ─── Constants ──────────────────────────────────────────────────────
# The actual checkpoint FC layer has shape [4, 512] → 4 classes were used
NUM_CLASSES = 4
CLASS_NAMES_3 = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS_3 = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]
CLASS_ICONS_3 = ["R", "B", "G", "P"]
IMG_SIZE = 224  # ResNet training size


# ─── Transform (must match training TEST_TRANSFORM exactly) ─────────
# Training data was: float32 [0, 1] numpy → PIL Image → ToTensor → ImageNet Norm
# ToTensor converts PIL [0,255] uint8 → [0,1] float tensor
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Model Cache ────────────────────────────────────────────────────
_loaded_models = {}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name: str):
    """Load a classification model by name. Cached after first load."""
    if name in _loaded_models:
        return _loaded_models[name]

    path = CLASSIFICATION_MODELS.get(name)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Model '{name}' not found at {path}")

    device = get_device()
    model = BrainTumorResNet(num_classes=NUM_CLASSES).to(device)
    state = torch.load(str(path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    _loaded_models[name] = (model, device)
    return model, device


def load_all_models():
    """Pre-load all available classification models."""
    loaded = {}
    for name, path in CLASSIFICATION_MODELS.items():
        if path.exists():
            try:
                load_model(name)
                loaded[name] = True
            except Exception as e:
                loaded[name] = str(e)
        else:
            loaded[name] = f"Missing: {path}"
    return loaded


def predict(model, device, image: Image.Image):
    """
    Run inference on a PIL Image.
    Returns (class_index, probabilities_array).
    """
    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)), probs


def _pil_to_b64(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _compute_uncertainty(prob_arr: np.ndarray):
    """Compute uncertainty metrics from class probabilities."""
    p = np.asarray(prob_arr, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()

    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(len(p))) if len(p) > 1 else 1.0
    entropy_norm = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    order = np.sort(p)[::-1]
    margin = float(order[0] - order[1]) if len(order) > 1 else 1.0

    if entropy_norm >= 0.65 or margin < 0.15:
        level = "high"
    elif entropy_norm >= 0.45 or margin < 0.30:
        level = "medium"
    else:
        level = "low"

    return {
        "entropy": float(entropy),
        "entropy_normalized": float(entropy_norm),
        "margin_top1_top2": float(margin),
        "level": level,
        "review_recommended": level != "low",
    }


def _generate_gradcam(model, device, image: Image.Image, target_class_idx=None):
    """Generate a Grad-CAM style attention map for ResNet-18 layer4."""
    image_rgb = image.convert("RGB")
    resized = image_rgb.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    tensor = TRANSFORM(image_rgb).unsqueeze(0).to(device)

    activations = []
    gradients = []

    target_layer = model.model.layer4[-1]

    def f_hook(_, __, output):
        activations.append(output.detach())

    def b_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    h1 = target_layer.register_forward_hook(f_hook)
    if hasattr(target_layer, "register_full_backward_hook"):
        h2 = target_layer.register_full_backward_hook(b_hook)
    else:
        h2 = target_layer.register_backward_hook(b_hook)

    try:
        logits = model(tensor)
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]
        model.zero_grad(set_to_none=True)
        score.backward()

        if not activations or not gradients:
            return None

        act = activations[0][0]   # (C, H, W)
        grad = gradients[0][0]    # (C, H, W)
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0))
        cam_np = cam.cpu().numpy()

        cmin, cmax = float(cam_np.min()), float(cam_np.max())
        if cmax > cmin:
            cam_np = (cam_np - cmin) / (cmax - cmin)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        heat_l = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L")
        heat_l = heat_l.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        heat = np.asarray(heat_l, dtype=np.float32) / 255.0

        orig = np.asarray(resized, dtype=np.float32) / 255.0
        heat_rgb = np.zeros_like(orig)
        heat_rgb[:, :, 0] = np.clip(heat * 1.2, 0.0, 1.0)
        heat_rgb[:, :, 1] = np.clip((heat - 0.35) / 0.65, 0.0, 1.0) * 0.9

        blend = np.clip(orig * 0.62 + heat_rgb * 0.38, 0.0, 1.0)
        heat_img = Image.fromarray((heat_rgb * 255).astype(np.uint8), mode="RGB")
        blend_img = Image.fromarray((blend * 255).astype(np.uint8), mode="RGB")

        return {
            "method": "gradcam_layer4",
            "heatmap_b64": _pil_to_b64(heat_img),
            "blend_b64": _pil_to_b64(blend_img),
            "target_class_idx": int(target_class_idx),
        }
    except Exception:
        return None
    finally:
        h1.remove()
        h2.remove()


def classify_image(image: Image.Image, model_names=None, include_explainability=False):
    """
    Classify a single image with the QPSO-FL model (or multiple models).

    Parameters
    ----------
    image : PIL.Image — Input MRI slice (RGB, any size).
    model_names : list[str] or None — Defaults to ["QPSO-FL"].

    Returns
    -------
    dict with keys:
        - results: {model_name: {class_name, class_idx, confidence, probabilities}}
        - consensus: {class_name, class_idx, unanimous, vote_count}
        - ensemble: {class_name, class_idx, confidence, probabilities}
        - is_glioma: bool
    """
    if model_names is None:
        # Default: only use QPSO-FL (user's request)
        model_names = ["QPSO-FL"]

    image = image.convert("RGB")
    results = {}

    for name in model_names:
        try:
            model, device = load_model(name)
            pred_idx, probs = predict(model, device, image)
            results[name] = {
                "class_name": CLASS_NAMES_3[pred_idx],
                "class_idx": pred_idx,
                "confidence": float(probs[pred_idx]),
                "probabilities": {
                    CLASS_NAMES_3[i]: float(probs[i]) for i in range(len(probs))
                },
                "color": CLASS_COLORS_3[pred_idx],
                "icon": CLASS_ICONS_3[pred_idx],
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    # Consensus
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    uncertainty = None
    explainability = None

    if valid_results:
        predictions = [v["class_name"] for v in valid_results.values()]
        from collections import Counter
        votes = Counter(predictions)
        winner, count = votes.most_common(1)[0]
        winner_idx = CLASS_NAMES_3.index(winner)

        # Ensemble average
        all_probs = np.mean(
            [list(v["probabilities"].values()) for v in valid_results.values()],
            axis=0,
        )
        ens_idx = int(np.argmax(all_probs))

        consensus = {
            "class_name": winner,
            "class_idx": winner_idx,
            "unanimous": len(set(predictions)) == 1,
            "vote_count": count,
            "total_models": len(valid_results),
            "color": CLASS_COLORS_3[winner_idx],
            "icon": CLASS_ICONS_3[winner_idx],
        }
        ensemble = {
            "class_name": CLASS_NAMES_3[ens_idx],
            "class_idx": ens_idx,
            "confidence": float(all_probs[ens_idx]),
            "probabilities": {
                CLASS_NAMES_3[i]: float(all_probs[i]) for i in range(len(all_probs))
            },
        }

        uncertainty = _compute_uncertainty(all_probs)

        if include_explainability:
            primary_name = "QPSO-FL" if "QPSO-FL" in valid_results else next(iter(valid_results.keys()))
            model, device = _loaded_models[primary_name]
            explainability = _generate_gradcam(
                model,
                device,
                image,
                target_class_idx=consensus["class_idx"],
            )
    else:
        consensus = {"error": "No models produced valid results"}
        ensemble = {"error": "No models produced valid results"}
        uncertainty = {"error": "No models produced valid results"}
        explainability = {"error": "No models produced valid results"}

    return {
        "results": results,
        "consensus": consensus,
        "ensemble": ensemble,
        "uncertainty": uncertainty,
        "explainability": explainability,
        "is_glioma": consensus.get("class_name") == "Glioma",
    }
