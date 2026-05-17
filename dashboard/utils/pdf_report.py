"""
Clinical PDF Report Generator
==============================
Builds a clean, demo-ready report with:
- Patient metadata and scan summary
- Classification results with per-class explainability
- Segmentation overlays across modalities (when available)
- Progression analysis with RANO status and forecasts
"""
from __future__ import annotations

import base64
import os
import tempfile
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False


# --- Color constants (R, G, B) ---
BLUE = (59, 130, 246)
NAVY = (15, 23, 42)
SLATE = (100, 116, 139)
LIGHT = (240, 247, 255)
WHITE = (255, 255, 255)
GREEN = (34, 197, 94)
RED = (239, 68, 68)
AMBER = (245, 158, 11)
PURPLE = (139, 92, 246)
ORANGE = (249, 115, 22)

CLASS_COLORS = {
    "Glioma": RED,
    "Meningioma": BLUE,
    "No Tumor": GREEN,
    "Pituitary": PURPLE,
}

SEG_COLORS = {
    "WT": (255, 213, 0),
    "TC": (255, 0, 0),
    "ET": (249, 115, 22),
}

MODALITY_LABELS = {
    "t1": "T1",
    "t1ce": "T1ce",
    "t2": "T2",
    "flair": "FLAIR",
}


def _b64_to_tmpfile(b64: str, suffix: str = ".png") -> str:
    data = base64.b64decode(b64)
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _pil_to_tmpfile(img: Image.Image, suffix: str = ".png") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        img.save(f, format="PNG")
    return path


def _conf_color(conf: float) -> tuple[int, int, int]:
    if conf >= 0.85:
        return GREEN
    if conf >= 0.65:
        return AMBER
    return RED


def _rano(growth_pct: float) -> tuple[str, str, tuple[int, int, int]]:
    if growth_pct <= -100:
        return "CR", "Complete Response", GREEN
    if growth_pct <= -25:
        return "PR", "Partial Response", BLUE
    if growth_pct <= 25:
        return "SD", "Stable Disease", AMBER
    return "PD", "Progressive Disease", RED


def _draw_probability_bars(
    probabilities: dict[str, float],
    width: int = 720,
    height: int = 260,
) -> Image.Image:
    """Create a chart image for class probabilities."""
    img = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(img)

    classes = list(probabilities.keys())
    values = [float(probabilities[c]) for c in classes]

    margin_x = 140
    bar_height = 26
    gap = 12
    top = 30
    max_bar = width - margin_x - 40

    draw.text((20, 8), "Per-Class Probability", fill=NAVY)

    count = min(len(classes), 6)
    for idx, (cls, val) in enumerate(zip(classes[:count], values[:count])):
        y = top + idx * (bar_height + gap)
        color = CLASS_COLORS.get(cls, BLUE)
        draw.text((20, y + 4), cls, fill=SLATE)
        draw.rectangle([margin_x, y + 4, margin_x + max_bar, y + bar_height], fill=(226, 232, 240))
        fill_w = max(2, int(max_bar * val))
        draw.rectangle([margin_x, y + 4, margin_x + fill_w, y + bar_height], fill=color)
        pct = f"{val * 100:.1f}%"
        draw.text((margin_x + max_bar + 8, y + 4), pct, fill=NAVY)

    return img


def _make_overlay(
    base_img: Image.Image,
    mask_wt: np.ndarray,
    mask_tc: np.ndarray,
    mask_et: np.ndarray,
) -> Image.Image:
    """Overlay segmentation masks on a base image."""
    base = np.asarray(base_img.convert("RGB"), dtype=np.float32)
    out = base.copy()

    def blend(mask: np.ndarray, color: tuple[int, int, int], alpha: float):
        if mask is None:
            return
        mask_bool = mask > 0
        if not np.any(mask_bool):
            return
        color_arr = np.array(color, dtype=np.float32)
        out[mask_bool] = out[mask_bool] * (1.0 - alpha) + color_arr * alpha

    blend(mask_wt, SEG_COLORS["WT"], 0.35)
    blend(mask_tc, SEG_COLORS["TC"], 0.55)
    blend(mask_et, SEG_COLORS["ET"], 0.7)

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to uint8 for reporting."""
    arr = np.asarray(slice_2d, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    flat = arr[np.isfinite(arr)]
    if flat.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    p1, p99 = np.percentile(flat, [1, 99])
    if p99 <= p1:
        return np.zeros(arr.shape, dtype=np.uint8)
    clipped = np.clip(arr, p1, p99)
    norm = (clipped - p1) / (p99 - p1)
    return (norm * 255).astype(np.uint8)


def slice_to_image(slice_2d: np.ndarray) -> Image.Image:
    """Convert a 2D array to a grayscale RGB PIL image."""
    img = Image.fromarray(_normalize_slice(slice_2d), mode="L")
    return img.convert("RGB")


def _draw_growth_curve(curve: dict[str, list], width: int = 720, height: int = 260) -> Image.Image:
    """Render a simple growth curve plot using PIL."""
    img = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(img)

    days = curve.get("days") or []
    vols = curve.get("volumes") or []
    if not days or not vols:
        return img

    padding = 40
    x0, y0 = padding, height - padding
    x1, y1 = width - padding, padding

    draw.line((x0, y0, x1, y0), fill=(226, 232, 240), width=2)
    draw.line((x0, y0, x0, y1), fill=(226, 232, 240), width=2)
    draw.text((x0, 10), "Projected Growth", fill=NAVY)

    dmin, dmax = float(min(days)), float(max(days))
    vmin, vmax = float(min(vols)), float(max(vols))
    if dmax == dmin:
        dmax += 1.0
    if vmax == vmin:
        vmax += 1.0

    points = []
    for d, v in zip(days, vols):
        x = x0 + (float(d) - dmin) / (dmax - dmin) * (x1 - x0)
        y = y0 - (float(v) - vmin) / (vmax - vmin) * (y0 - y1)
        points.append((x, y))

    if len(points) > 1:
        draw.line(points, fill=BLUE, width=3)

    draw.text((x0, height - padding + 8), "Days", fill=SLATE)
    draw.text((5, y1), "Volume", fill=SLATE)
    return img


def _prepare_tile(
    img: Image.Image,
    size: int = 320,
    bg: tuple[int, int, int] = (235, 238, 245),
) -> Image.Image:
    """Resize and letterbox an image into a square tile."""
    tile = Image.new("RGB", (size, size), bg)
    source = img.convert("RGB")
    source.thumbnail((size, size), Image.LANCZOS)
    x = (size - source.width) // 2
    y = (size - source.height) // 2
    tile.paste(source, (x, y))
    return tile


def _render_image_grid(
    pdf: "ClinicalReport",
    items: list[tuple[str, Image.Image]],
    tmp_files: list[str],
    cols: int = 2,
    tile_w: float = 78.0,
    tile_h: float = 78.0,
    label_h: float = 5.0,
    h_gap: float = 10.0,
    v_gap: float = 10.0,
) -> None:
    """Render labeled image tiles in a grid and advance cursor."""
    if not items:
        return
    start_x = 10.0
    start_y = pdf.get_y()
    for idx, (label, img) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = start_x + col * (tile_w + h_gap)
        y = start_y + row * (tile_h + label_h + v_gap)
        pdf.set_xy(x, y)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*SLATE)
        pdf.cell(tile_w, label_h, label, align="C")
        tile = _prepare_tile(img)
        p = _pil_to_tmpfile(tile)
        tmp_files.append(p)
        pdf.image(p, x=x, y=y + label_h, w=tile_w, h=tile_h)
    rows = (len(items) + cols - 1) // cols
    pdf.set_y(start_y + rows * (tile_h + label_h + v_gap))


def build_segmentation_overlays(
    modalities: dict[str, Image.Image],
    masks: dict[str, np.ndarray],
) -> dict[str, Image.Image]:
    """
    Build overlay images for each modality.

    Parameters
    ----------
    modalities : dict
        Mapping of modality key -> PIL.Image base slice.
    masks : dict
        Mapping of mask names (WT/TC/ET) -> 2D arrays.

    Returns
    -------
    dict
        Mapping of label -> overlay image.
    """
    overlays: dict[str, Image.Image] = {}
    wt = masks.get("WT")
    tc = masks.get("TC")
    et = masks.get("ET")

    for mod, img in modalities.items():
        label = MODALITY_LABELS.get(mod, mod.upper())
        overlays[label] = _make_overlay(img, wt, tc, et)

    return overlays


class ClinicalReport(FPDF):
    """Custom FPDF layout helpers."""

    def header(self):
        self.set_fill_color(*BLUE)
        self.rect(0, 0, 210, 7, "F")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*SLATE)
        self.cell(0, 5, f"NeuroAI Clinical Report | Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*NAVY)
        self.cell(0, 7, title, ln=True)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def kv_row(self, label: str, value: str):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*SLATE)
        self.cell(40, 6, label, align="R")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*NAVY)
        self.cell(0, 6, f"   {value}", ln=True)

    def insight(self, text: str):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*NAVY)
        self.set_x(12)
        line = f"- {str(text)}"
        self.multi_cell(0, 5, line)


def generate_clinical_pdf(
    patient_meta: dict[str, Any],
    classification: dict | None = None,
    segmentation: dict | None = None,
    progression: dict | None = None,
    report_assets: dict[str, Any] | None = None,
) -> bytes:
    if not HAS_FPDF:
        raise ImportError("fpdf2 required: pip install fpdf2")

    report_assets = report_assets or {}
    tmp_files: list[str] = []

    pdf = ClinicalReport()
    pdf.alias_nb_pages()

    # === COVER ===
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 14, "NeuroAI", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*SLATE)
    pdf.cell(0, 6, "Clinical Decision Support Report", ln=True, align="C")
    pdf.ln(4)

    pdf.set_fill_color(*LIGHT)
    y0 = pdf.get_y()
    pdf.rect(15, y0, 180, 38, "F")
    pdf.set_xy(15, y0 + 3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*NAVY)
    pdf.cell(180, 6, "   Patient Information", ln=True)
    pdf.set_x(15)
    pdf.kv_row("Patient ID:", str(patient_meta.get("patient_id", "N/A")))
    pdf.set_x(15)
    pdf.kv_row("Age / Sex:", f"{patient_meta.get('age', 'N/A')} / {patient_meta.get('sex', 'N/A')}")
    pdf.set_x(15)
    pdf.kv_row("Scan Date:", str(patient_meta.get("scan_date", "N/A")))
    notes = str(patient_meta.get("notes", "")).strip()
    if notes:
        pdf.set_x(15)
        pdf.kv_row("Notes:", notes[:120])
    pdf.set_y(y0 + 42)

    if classification:
        cs = classification.get("consensus", {})
        ens = classification.get("ensemble", {})
        cn = cs.get("class_name", "Unknown")
        conf = float(ens.get("confidence", 0))
        cc = CLASS_COLORS.get(cn, BLUE)

        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*SLATE)
        pdf.cell(0, 6, "DIAGNOSIS SUMMARY", ln=True, align="C")
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(*cc)
        pdf.cell(0, 12, cn, ln=True, align="C")
        level = "HIGH" if conf >= 0.85 else "MODERATE" if conf >= 0.65 else "LOW"
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*_conf_color(conf))
        pdf.cell(0, 6, f"Confidence: {conf:.1%} ({level})", ln=True, align="C")
        pdf.ln(1)
        pdf.set_fill_color(226, 232, 240)
        x_bar = 50
        pdf.rect(x_bar, pdf.get_y(), 110, 6, "F")
        pdf.set_fill_color(*_conf_color(conf))
        pdf.rect(x_bar, pdf.get_y(), max(1, conf * 110), 6, "F")
        pdf.ln(10)

    # === CLASSIFICATION ===
    if classification:
        pdf.add_page()
        pdf.section_title("Classification Analysis")

        scan_summary = report_assets.get("scan_summary") or {}
        if scan_summary:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*NAVY)
            pdf.cell(0, 6, "Scan Summary", ln=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*SLATE)
            pdf.multi_cell(0, 5, scan_summary.get("summary", ""))
            quality = scan_summary.get("quality")
            if quality:
                pdf.kv_row("Quality:", quality)
            warnings = scan_summary.get("warnings") or []
            if warnings:
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*SLATE)
                pdf.cell(0, 5, "Warnings:", ln=True)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(*NAVY)
                for w in warnings:
                    pdf.insight(w)
            pdf.ln(2)

        slice_b64 = classification.get("slice_image_b64")
        if slice_b64:
            try:
                p = _b64_to_tmpfile(slice_b64)
                tmp_files.append(p)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(*SLATE)
                pdf.cell(0, 5, "INPUT MRI SLICE", ln=True)
                pdf.image(p, x=10, w=50)
                pdf.ln(3)
            except Exception:
                pass

        ens = classification.get("ensemble", {})
        probs = ens.get("probabilities", {})
        if probs:
            chart = _draw_probability_bars(probs)
            p = _pil_to_tmpfile(chart)
            tmp_files.append(p)
            pdf.image(p, x=10, w=180)
            pdf.ln(5)

        # Explainability bundle
        exp_bundle = classification.get("explainability_bundle", {})
        if exp_bundle:
            pdf.section_title("Explainability by Class (Grad-CAM)")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*SLATE)
            pdf.cell(0, 5, "Heatmaps show the regions driving each class decision.", ln=True)
            pdf.ln(2)
            items = []
            for cls, payload in list(exp_bundle.items())[:4]:
                img_b64 = payload.get("blend_b64") or payload.get("heatmap_b64")
                if not img_b64:
                    continue
                try:
                    img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
                    items.append((cls, img))
                except Exception:
                    continue
            _render_image_grid(pdf, items, tmp_files, cols=2)
            pdf.ln(2)

        unc = classification.get("uncertainty", {})
        if unc.get("level"):
            pdf.section_title("Risk & Quality Assessment")
            pdf.kv_row("Uncertainty:", f"{unc['level'].upper()} (entropy={float(unc.get('entropy_normalized', 0)):.3f})")
            if unc.get("review_recommended"):
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*RED)
                pdf.cell(0, 5, "    [!] Manual review recommended", ln=True)
            qc = classification.get("quality_check", {})
            if qc.get("status"):
                pdf.kv_row("Data Quality:", qc["status"].upper())

        ctx = classification.get("clinical_context") or ""
        if ctx:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*GREEN)
            pdf.cell(0, 6, "Clinical Context", ln=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*NAVY)
            pdf.multi_cell(0, 5, ctx)

        insights = classification.get("insights", [])
        if insights:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*NAVY)
            pdf.cell(0, 6, "Clinical Insights", ln=True)
            for ins in insights[:8]:
                pdf.insight(ins)

    # === SEGMENTATION ===
    if segmentation and not segmentation.get("error"):
        pdf.add_page()
        pdf.section_title("Segmentation Analysis")

        volumes = segmentation.get("volumes", {})
        total_vox = segmentation.get("total_voxels", 1)
        regions = [
            ("Whole Tumor (WT)", volumes.get("WT", 0), AMBER),
            ("Tumor Core (TC)", volumes.get("TC", 0), RED),
            ("Enhancing Tumor (ET)", volumes.get("ET", 0), ORANGE),
        ]
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(*LIGHT)
        pdf.cell(70, 7, "Region", border=1, fill=True)
        pdf.cell(40, 7, "Volume (mm^3)", border=1, fill=True, align="C")
        pdf.cell(40, 7, "% of Brain", border=1, fill=True, align="C")
        pdf.cell(30, 7, "", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for name, vol, color in regions:
            pdf.set_fill_color(*color)
            pdf.rect(pdf.get_x(), pdf.get_y() + 1, 3, 5, "F")
            pdf.cell(4, 7, "")
            pdf.cell(66, 7, name, border=1)
            pdf.cell(40, 7, f"{vol:,.0f}", border=1, align="C")
            pct = (vol / total_vox * 100) if total_vox else 0
            pdf.cell(40, 7, f"{pct:.2f}%", border=1, align="C")
            pdf.cell(30, 7, "", ln=True)
        pdf.ln(4)

        modality_slices = report_assets.get("modality_slices", {})
        if modality_slices:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*NAVY)
            pdf.cell(0, 6, "Modality Slice Gallery", ln=True)
            pdf.ln(2)
            items = [(label, img) for label, img in list(modality_slices.items())[:4]]
            _render_image_grid(pdf, items, tmp_files, cols=2)
            pdf.ln(2)

        overlays = report_assets.get("segmentation_overlays", {})
        if overlays:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*NAVY)
            pdf.cell(0, 6, "Overlay Samples (Axial)", ln=True)
            pdf.ln(2)
            items = [(label, img) for label, img in list(overlays.items())[:4]]
            _render_image_grid(pdf, items, tmp_files, cols=2)
            pdf.ln(2)

        unc_seg = segmentation.get("uncertainty_summary", {})
        if unc_seg.get("level"):
            pdf.section_title("Segmentation Uncertainty")
            pdf.kv_row("Level:", unc_seg["level"].upper())
            pdf.kv_row("Mean:", f"{float(unc_seg.get('mean', 0)):.4f}")
            pdf.kv_row("P95:", f"{float(unc_seg.get('p95', 0)):.4f}")
            ratio = float(unc_seg.get("high_uncertainty_ratio", 0)) * 100
            pdf.kv_row("High-unc voxels:", f"{ratio:.1f}%")
            if unc_seg.get("review_recommended"):
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*RED)
                pdf.cell(0, 5, "    [!] Manual review recommended", ln=True)

    # === PROGRESSION ===
    if progression and not progression.get("error"):
        pdf.add_page()
        pdf.section_title("Progression Forecast")
        pdf.kv_row("Current Volume:", f"{progression.get('current_volume', 0):,.0f} mm^3")
        pdf.kv_row("Grade:", progression.get("grade", "HGG"))
        pdf.ln(2)

        proj = progression.get("logistic", {}).get("projections", [])
        if proj:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(*LIGHT)
            pdf.cell(40, 7, "Timepoint", border=1, fill=True)
            pdf.cell(45, 7, "Projected Volume", border=1, fill=True, align="C")
            pdf.cell(35, 7, "Growth", border=1, fill=True, align="C")
            pdf.cell(45, 7, "RANO Status", border=1, fill=True, align="C")
            pdf.ln()
            pdf.set_font("Helvetica", "", 9)
            for p in proj:
                day = p.get("day", 0)
                label = f"{day} days" + (f" ({day//30} mo)" if day >= 30 else "")
                code, desc, color = _rano(p.get("growth_pct", 0))
                pdf.cell(40, 7, label, border=1)
                pdf.cell(45, 7, f"{p.get('volume', 0):,.0f} mm^3", border=1, align="C")
                g = p.get("growth_pct", 0)
                pdf.set_text_color(*(RED if g > 0 else GREEN))
                pdf.cell(35, 7, f"{'+' if g >= 0 else ''}{g:.1f}%", border=1, align="C")
                pdf.set_text_color(*NAVY)
                pdf.set_fill_color(*color)
                pdf.rect(pdf.get_x() + 2, pdf.get_y() + 1.5, 4, 4, "F")
                pdf.cell(45, 7, f"  {code} - {desc}", border=1)
                pdf.ln()
            pdf.ln(3)

        p90 = next((p for p in proj if p.get("day") == 90), None)
        if p90 and p90.get("growth_pct", 0) > 25:
            pdf.set_fill_color(254, 242, 242)
            pdf.rect(10, pdf.get_y(), 190, 12, "F")
            pdf.set_draw_color(*RED)
            pdf.set_line_width(0.8)
            pdf.line(10, pdf.get_y(), 10, pdf.get_y() + 12)
            pdf.set_x(14)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*RED)
            pdf.cell(0, 12, f"  ALERT: Projected growth of {p90['growth_pct']:.1f}% in 3 months exceeds RANO PD threshold (25%)", ln=True)
            pdf.ln(2)

        curve = ((progression.get("logistic") or {}).get("curve")) or {}
        if curve:
            chart = _draw_growth_curve(curve)
            p = _pil_to_tmpfile(chart)
            tmp_files.append(p)
            pdf.image(p, x=10, w=180)
            pdf.ln(6)

        stats = ((progression.get("spatial") or {}).get("stats")) or {}
        if stats:
            pdf.section_title("Spatial Growth Prediction (3D U-Net)")
            pdf.kv_row("Stable voxels:", f"{stats.get('stable_voxels', 0):,}")
            pdf.kv_row("Growth voxels:", f"{stats.get('growth_voxels', 0):,}")
            pdf.kv_row("Regression voxels:", f"{stats.get('regression_voxels', 0):,}")

        insights = progression.get("insights", [])
        if insights:
            pdf.section_title("Progression Insights")
            for ins in insights[:8]:
                pdf.insight(ins)

    # === FINAL PAGE ===
    pdf.add_page()
    pdf.section_title("Methodology & Disclaimer")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*SLATE)
    pdf.multi_cell(
        0,
        5,
        "This report is AI-assisted clinical decision support and not a standalone diagnosis. "
        "All results must be reviewed by qualified clinicians. "
        "Classification: ResNet-18 federated QPSO (4 classes). "
        "Segmentation: 3D Attention U-Net with uncertainty estimates. "
        "Progression: Logistic growth model with spatial prediction.\n\n"
        f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    result = bytes(pdf.output())

    for f in tmp_files:
        try:
            os.unlink(f)
        except OSError:
            pass

    return result
