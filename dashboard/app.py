"""
NeuroAI Dashboard — FastAPI Backend
=====================================
Professional brain tumor analysis pipeline:
Upload → Classify → Segment (Glioma) → Progression (Glioma)

Run:
    uvicorn app:app --reload --port 5000
"""

import os
import sys
import json
import uuid
import base64
import asyncio
import traceback
import textwrap
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure dashboard dir is on path for config imports
DASHBOARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DASHBOARD_DIR))

from config import UPLOAD_DIR, CLASSIFICATION_MODELS
from utils.file_handler import (
    create_session_dir, extract_zip, analyze_uploaded_files,
    cleanup_session,
)
from utils.preprocessing import assess_nifti_quality


# ─── Lifespan: pre-load models ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load classification models at startup."""
    print("[NeuroAI] Dashboard starting...")
    try:
        from models.classifier import load_all_models
        status = load_all_models()
        for name, s in status.items():
            icon = "[OK]" if s is True else "[FAIL]"
            print(f"  {icon} Classification/{name}: {s}")
    except Exception as e:
        print(f"  [WARN] Classification models: {e}")

    print("  [INFO] Segmentation & Progression models load on first use")
    print("[NeuroAI] Dashboard ready at http://localhost:5000")
    yield
    print("[NeuroAI] Shutting down...")


# ─── FastAPI App ────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroAI Brain Tumor Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR / "static")), name="static")


def _build_classification_insights(result: dict):
    """Create lightweight clinical-style insights for classification output."""
    consensus = result.get("consensus", {})
    cls = consensus.get("class_name")
    uncertainty = result.get("uncertainty", {})
    level = uncertainty.get("level")
    quality = result.get("quality_check") or result.get("quality_overview") or {}
    q_status = str(quality.get("status", "")).lower()

    insights = []
    if cls == "Glioma":
        insights.append("Glioma pathway active: proceed with segmentation and progression modules.")
    elif cls:
        insights.append("Non-glioma classification: downstream glioma-only modules may be skipped.")

    if level == "high":
        insights.append("High uncertainty: manual review is recommended before clinical interpretation.")
    elif level == "medium":
        insights.append("Medium uncertainty: correlate with radiologist assessment and multi-sequence evidence.")
    elif level == "low":
        insights.append("Low uncertainty: prediction confidence is consistent across top probabilities.")

    if q_status == "high_risk":
        insights.append("Input quality appears high-risk; verify modality integrity before relying on prediction.")
    elif q_status == "review":
        insights.append("Input quality flags detected; correlate with radiology review.")

    return insights


def _build_progression_insights(result: dict):
    """Create explainability insights for progression output."""
    insights = []
    current_vol = float(result.get("current_volume", 0.0))
    logistic = result.get("logistic", {})
    projections = logistic.get("projections", [])

    if current_vol > 0:
        if current_vol > 150000:
            insights.append("Current tumor burden is high; prioritize close follow-up intervals.")
        elif current_vol > 60000:
            insights.append("Current tumor burden is moderate; monitor trend progression closely.")
        else:
            insights.append("Current tumor burden is relatively low; trend stability is key.")

    proj_180 = next((p for p in projections if int(p.get("day", -1)) == 180), None)
    if proj_180 is not None:
        g = float(proj_180.get("growth_pct", 0.0))
        if g >= 15:
            insights.append("6-month logistic projection indicates notable volumetric increase.")
        elif g <= -5:
            insights.append("6-month logistic projection indicates potential regression trajectory.")
        else:
            insights.append("6-month logistic projection indicates relatively stable volume trend.")

    spatial = result.get("spatial")
    if spatial and spatial.get("stats"):
        st = spatial["stats"]
        gv = int(st.get("growth_voxels", 0))
        rv = int(st.get("regression_voxels", 0))
        if gv > rv * 1.3:
            insights.append("Spatial model suggests growth-dominant morphology.")
        elif rv > gv * 1.3:
            insights.append("Spatial model suggests regression-dominant morphology.")
        else:
            insights.append("Spatial model suggests mixed or balanced change pattern.")

    return insights


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compact_payload_for_storage(payload):
    def _compact(obj):
        if isinstance(obj, dict):
            compacted = {}
            for k, v in obj.items():
                if isinstance(k, str) and k.endswith("_b64"):
                    continue
                compacted[k] = _compact(v)
            return compacted
        if isinstance(obj, list):
            return [_compact(v) for v in obj]
        return obj

    return _compact(payload)


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _compute_uncertainty_from_probabilities(probabilities: dict):
    probs = np.asarray(list(probabilities.values()), dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()

    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(len(probs))) if len(probs) > 1 else 1.0
    entropy_norm = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    order = np.sort(probs)[::-1]
    margin = float(order[0] - order[1]) if len(order) > 1 else 1.0

    if entropy_norm >= 0.65 or margin < 0.15:
        level = "high"
    elif entropy_norm >= 0.45 or margin < 0.30:
        level = "medium"
    else:
        level = "low"

    return {
        "entropy": entropy,
        "entropy_normalized": entropy_norm,
        "margin_top1_top2": margin,
        "level": level,
        "review_recommended": level != "low",
    }


def _assess_image_quality(image: Image.Image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    mean_intensity = float(gray.mean()) if gray.size else 0.0
    contrast_std = float(gray.std()) if gray.size else 0.0

    warnings = []
    if contrast_std < 20:
        warnings.append("Low contrast image")
    if mean_intensity < 15:
        warnings.append("Image is very dark")
    if mean_intensity > 240:
        warnings.append("Image is very bright")

    status = "ok"
    if len(warnings) >= 2:
        status = "high_risk"
    elif warnings:
        status = "review"

    return {
        "shape": [int(v) for v in gray.shape],
        "mean_intensity": mean_intensity,
        "contrast_std": contrast_std,
        "status": status,
        "warnings": warnings,
    }


def _summarize_quality_checks(quality_checks: dict):
    if not quality_checks:
        return {
            "status": "review",
            "warnings": ["No quality checks were available"],
            "modalities_checked": 0,
        }

    statuses = [str(v.get("status", "review")) for v in quality_checks.values()]
    if "high_risk" in statuses:
        status = "high_risk"
    elif "review" in statuses:
        status = "review"
    else:
        status = "ok"

    warnings = []
    for mod, payload in quality_checks.items():
        for w in payload.get("warnings", []):
            warnings.append(f"{mod.upper()}: {w}")

    return {
        "status": status,
        "warnings": warnings,
        "modalities_checked": len(quality_checks),
    }


def _build_report_payload(session_id: str, session_dir: Path):
    upload = _load_json(session_dir / "upload_analysis.json") or {}
    classification = _load_json(session_dir / "classification_result.json") or {}
    segmentation = _load_json(session_dir / "segmentation_result.json") or {}
    progression = _load_json(session_dir / "progression_result.json") or {}

    return {
        "session_id": session_id,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "upload": upload,
        "classification": classification,
        "segmentation": segmentation,
        "progression": progression,
    }


def _render_report_lines(payload: dict):
    upload = payload.get("upload", {})
    cls = payload.get("classification", {})
    seg = payload.get("segmentation", {})
    prog = payload.get("progression", {})

    lines = [
        "NeuroAI Clinical Decision Support Report",
        f"Generated: {payload.get('generated_at_utc', 'N/A')}",
        f"Session ID: {payload.get('session_id', 'N/A')}",
        "",
        "Data Intake",
        f"Detected modalities: {', '.join(sorted((upload.get('modalities') or {}).keys())) or 'N/A'}",
    ]

    q_overview = upload.get("quality_overview", {})
    if q_overview:
        lines.append(f"Quality status: {str(q_overview.get('status', 'N/A')).upper()}")
        for w in q_overview.get("warnings", [])[:4]:
            lines.append(f"- {w}")

    lines.extend(["", "Classification"])
    consensus = cls.get("consensus", {})
    if consensus:
        lines.append(f"Predicted class: {consensus.get('class_name', 'N/A')}")

    uncertainty = cls.get("uncertainty", {})
    if uncertainty and "level" in uncertainty:
        lines.append(
            f"Uncertainty: {uncertainty.get('level', 'N/A')} "
            f"(entropy_norm={float(uncertainty.get('entropy_normalized', 0.0)):.3f}, "
            f"margin={float(uncertainty.get('margin_top1_top2', 0.0)):.3f})"
        )
    q_check = cls.get("quality_check", {})
    if q_check:
        lines.append(f"Input quality: {str(q_check.get('status', 'N/A')).upper()}")
        for w in q_check.get("warnings", [])[:3]:
            lines.append(f"- {w}")

    for insight in cls.get("insights", [])[:3]:
        lines.append(f"- Insight: {insight}")

    if seg:
        lines.extend(["", "Segmentation"])
        vols = seg.get("volumes", {})
        lines.append(f"TC volume: {float(vols.get('TC', 0.0)):.0f} mm^3")
        lines.append(f"WT volume: {float(vols.get('WT', 0.0)):.0f} mm^3")
        lines.append(f"ET volume: {float(vols.get('ET', 0.0)):.0f} mm^3")

        unc = seg.get("uncertainty_summary", {})
        if unc:
            lines.append(
                f"Segmentation uncertainty: {unc.get('level', 'N/A')} "
                f"(high-uncertainty ratio={float(unc.get('high_uncertainty_ratio', 0.0)):.3f})"
            )

    if prog:
        lines.extend(["", "Progression Forecast"])
        lines.append(f"Current volume: {float(prog.get('current_volume', 0.0)):.0f} mm^3")
        lines.append(f"Grade assumption: {prog.get('grade', 'N/A')}")

        projections = (((prog.get("logistic") or {}).get("projections")) or [])
        for p in projections:
            day = int(p.get("day", 0))
            if day in (30, 90, 180, 365):
                lines.append(
                    f"Day {day}: {float(p.get('volume', 0.0)):.0f} mm^3 "
                    f"({float(p.get('growth_pct', 0.0)):+.1f}%)"
                )

        for insight in prog.get("insights", [])[:4]:
            lines.append(f"- Insight: {insight}")

    lines.extend([
        "",
        "Disclaimer",
        "This report is AI-assisted decision support and not a standalone diagnosis.",
    ])

    wrapped = []
    for ln in lines:
        if not ln:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(ln, width=95) or [""])
    return wrapped


def _pdf_escape(text: str):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_minimal_pdf(lines: list[str]):
    max_lines = 48
    page_lines = lines[:max_lines]

    stream_lines = [
        "BT",
        "/F1 10 Tf",
        "50 760 Td",
        "14 TL",
    ]
    for ln in page_lines:
        stream_lines.append(f"({_pdf_escape(ln)}) Tj")
        stream_lines.append("T*")
    stream_lines.append("ET")
    stream = "\n".join(stream_lines).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> "
            b"/Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
    ]

    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"

    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    pdf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n".encode("ascii")

    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    ).encode("ascii")
    return pdf


# ─── Serve SPA ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(DASHBOARD_DIR / "templates" / "index.html"))


# ─── Upload Endpoint ────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
):
    """Handle file uploads (ZIP or individual NIfTI files)."""
    session_id, session_dir = create_session_dir()

    max_upload_bytes = 2 * 1024 * 1024 * 1024
    max_filename_len = 255

    def _safe_upload_target(name: str) -> Path:
        upload_name = Path(name).name
        if not upload_name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        if len(upload_name) > max_filename_len:
            raise HTTPException(status_code=400, detail=f"Filename too long: {upload_name}")

        target_path = (session_dir / upload_name).resolve()
        base_path = session_dir.resolve()
        if target_path != base_path and base_path not in target_path.parents:
            raise HTTPException(status_code=400, detail=f"Unsafe filename: {name}")
        return target_path

    try:
        saved_paths = []
        is_zip = False
        total_uploaded = 0

        for f in files:
            target = _safe_upload_target(f.filename or "")
            with target.open("wb") as dst:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    total_uploaded += len(chunk)
                    if total_uploaded > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Upload exceeds 2GB limit")
                    dst.write(chunk)
            saved_paths.append(target)
            await f.close()

            if (f.filename or "").lower().endswith('.zip'):
                is_zip = True

        # If ZIP, extract it
        if is_zip and len(saved_paths) == 1:
            result = extract_zip(str(saved_paths[0]), session_dir)
        else:
            result = analyze_uploaded_files([str(p) for p in saved_paths])

        quality_checks = {}
        for mod, path in (result.get("modalities") or {}).items():
            if mod == "seg":
                continue
            try:
                quality_checks[mod] = assess_nifti_quality(path)
            except Exception as q_err:
                quality_checks[mod] = {
                    "status": "review",
                    "warnings": [f"Quality check unavailable: {q_err}"],
                }

        if quality_checks:
            result["quality_checks"] = quality_checks
            result["quality_overview"] = _summarize_quality_checks(quality_checks)

        result["session_id"] = session_id
        _write_json(
            session_dir / "upload_analysis.json",
            _compact_payload_for_storage(result),
        )
        return JSONResponse(result)

    except Exception as e:
        cleanup_session(session_id)
        raise HTTPException(status_code=400, detail=str(e))


# ─── Classification Endpoint ────────────────────────────────────────
@app.post("/api/classify")
async def classify(session_id: str = Form(...)):
    """Run classification on the uploaded MRI data."""
    session_dir = UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        from models.classifier import (
            classify_image,
            CLASS_NAMES_3,
            CLASS_COLORS_3,
            CLASS_ICONS_3,
        )
        from utils.file_handler import analyze_uploaded_files
        from utils.preprocessing import (
            extract_classification_slice,
            extract_classification_slices_for_voting,
        )

        # Re-analyze to get modality paths
        all_files = list(session_dir.rglob("*"))
        analysis = analyze_uploaded_files([str(f) for f in all_files if f.is_file()])
        modalities = analysis.get("modalities", {})

        quality_checks = {}
        for mod, path in modalities.items():
            if mod == "seg":
                continue
            try:
                quality_checks[mod] = assess_nifti_quality(path)
            except Exception as q_err:
                quality_checks[mod] = {
                    "status": "review",
                    "warnings": [f"Quality check unavailable: {q_err}"],
                }

        quality_overview = _summarize_quality_checks(quality_checks)

        def finalize_response(payload: dict):
            payload["quality_checks"] = quality_checks

            if quality_checks:
                payload["quality_overview"] = quality_overview
                src_mod = payload.get("source_modality")
                if src_mod and src_mod in quality_checks and "quality_check" not in payload:
                    payload["quality_check"] = quality_checks[src_mod]
            else:
                source_quality = payload.get("quality_check") or {
                    "status": "review",
                    "warnings": ["No modality-level quality checks were available"],
                }
                payload["quality_overview"] = {
                    "status": source_quality.get("status", "review"),
                    "warnings": source_quality.get("warnings", []),
                    "modalities_checked": 0,
                }

            if not payload.get("uncertainty"):
                ensemble_probs = ((payload.get("ensemble") or {}).get("probabilities")) or None
                if ensemble_probs:
                    payload["uncertainty"] = _compute_uncertainty_from_probabilities(ensemble_probs)
            elif "error" in payload.get("uncertainty", {}):
                ensemble_probs = ((payload.get("ensemble") or {}).get("probabilities")) or None
                if ensemble_probs:
                    payload["uncertainty"] = _compute_uncertainty_from_probabilities(ensemble_probs)

            payload["insights"] = _build_classification_insights(payload)
            _write_json(
                session_dir / "classification_result.json",
                _compact_payload_for_storage(payload),
            )
            return JSONResponse(payload)

        # BraTS 2020/2021 protocol is glioma-only by definition.
        # If a full BraTS-style multi-modal study is detected, bypass
        # the general tumor-type classifier to avoid out-of-domain errors.
        has_all_brats_modalities = all(m in modalities for m in ("t1", "t1ce", "t2", "flair"))
        is_brats_named = any("brats" in f.name.lower() for f in all_files if f.is_file())
        is_brats_protocol = has_all_brats_modalities and (is_brats_named or "seg" in modalities)

        if is_brats_protocol:
            glioma_idx = CLASS_NAMES_3.index("Glioma")
            other_prob = 0.01 / max(1, len(CLASS_NAMES_3) - 1)
            prob_values = [other_prob for _ in CLASS_NAMES_3]
            prob_values[glioma_idx] = 0.99
            probabilities = {
                CLASS_NAMES_3[i]: float(prob_values[i])
                for i in range(len(CLASS_NAMES_3))
            }

            response = {
                "results": {
                    "BraTS-Protocol": {
                        "class_name": "Glioma",
                        "class_idx": glioma_idx,
                        "confidence": float(prob_values[glioma_idx]),
                        "probabilities": probabilities,
                        "color": CLASS_COLORS_3[glioma_idx],
                        "icon": CLASS_ICONS_3[glioma_idx],
                    }
                },
                "consensus": {
                    "class_name": "Glioma",
                    "class_idx": glioma_idx,
                    "unanimous": True,
                    "vote_count": 1,
                    "total_models": 1,
                    "color": CLASS_COLORS_3[glioma_idx],
                    "icon": CLASS_ICONS_3[glioma_idx],
                },
                "ensemble": {
                    "class_name": "Glioma",
                    "class_idx": glioma_idx,
                    "confidence": float(prob_values[glioma_idx]),
                    "probabilities": probabilities,
                },
                "is_glioma": True,
                "source": "brats_protocol",
                "brats_override": True,
                "note": "BraTS datasets are glioma-only; tumor-type classification was bypassed.",
                "uncertainty": _compute_uncertainty_from_probabilities(probabilities),
                "explainability": None,
            }

            try:
                preview_img, preview_idx, total_slices = extract_classification_slice(
                    modalities["flair"],
                    slice_strategy="tumor_region",
                )
                buf = BytesIO()
                preview_img.save(buf, format="PNG")
                response["source_modality"] = "flair"
                response["slice_index"] = preview_idx
                response["total_slices"] = total_slices
                response["slice_image_b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                pass

            return finalize_response(response)

        # Extract a 2D slice for classification
        # Priority: FLAIR > T1ce > T2 > T1
        slice_source = None
        for mod in ["flair", "t1ce", "t2", "t1"]:
            if mod in modalities:
                slice_source = modalities[mod]
                break

        if slice_source is None:
            # Look for any image files (JPG/PNG)
            img_files = [f for f in all_files
                         if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            if img_files:
                img = Image.open(str(img_files[0])).convert("RGB")
                result = classify_image(img, include_explainability=True)
                result["source"] = "uploaded_image"
                result["source_file"] = str(img_files[0].name)
                result["quality_check"] = _assess_image_quality(img)
                return finalize_response(result)
            raise HTTPException(
                status_code=400,
                detail="No valid MRI files found for classification"
            )

        # Multi-slice voting: extract 5 high-variance slices and vote
        voted_slices, total_slices = extract_classification_slices_for_voting(
            slice_source, n_slices=5
        )

        if voted_slices:
            # Classify each slice and vote
            from collections import Counter
            per_slice = []

            for img, s_idx in voted_slices:
                res = classify_image(img, include_explainability=False)
                cls_name = (res.get("consensus") or {}).get("class_name")
                conf = float((res.get("ensemble") or {}).get("confidence", 0.0))
                per_slice.append({
                    "image": img,
                    "slice_idx": s_idx,
                    "result": res,
                    "class_name": cls_name,
                    "confidence": conf,
                })

            valid_predictions = [p for p in per_slice if p.get("class_name")]
            if not valid_predictions:
                raise HTTPException(status_code=500, detail="No valid slice prediction produced")

            # Final vote: majority across slices
            votes = Counter(p["class_name"] for p in valid_predictions)
            winner, count = votes.most_common(1)[0]

            winner_candidates = [p for p in valid_predictions if p["class_name"] == winner]
            selected = max(
                winner_candidates if winner_candidates else valid_predictions,
                key=lambda p: p["confidence"],
            )

            # If vote winner differs from best-confidence single slice,
            # use the vote winner result but find a slice that predicted it
            result = classify_image(selected["image"], include_explainability=True)
            result["slice_voting"] = {
                "num_slices": len(voted_slices),
                "votes": dict(votes),
                "winner": winner,
                "unanimous": len(set(p["class_name"] for p in valid_predictions)) == 1,
            }

            # Override consensus if vote disagrees
            if winner != result["consensus"]["class_name"]:
                winner_idx = CLASS_NAMES_3.index(winner)
                result["consensus"]["class_name"] = winner
                result["consensus"]["class_idx"] = winner_idx
                result["consensus"]["color"] = CLASS_COLORS_3[winner_idx]
                result["consensus"]["icon"] = CLASS_ICONS_3[winner_idx]
                result["is_glioma"] = winner == "Glioma"

            result["source"] = "nifti_multi_slice_vote"
            result["source_modality"] = [k for k, v in modalities.items() if v == slice_source][0]
            result["slice_index"] = selected["slice_idx"]
            result["total_slices"] = total_slices
            result["quality_check"] = quality_checks.get(result["source_modality"])

            # Encode the best slice image as base64
            buf = BytesIO()
            selected["image"].save(buf, format="PNG")
            result["slice_image_b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")

            return finalize_response(result)

        # Fallback: single slice
        img, slice_idx, total_slices = extract_classification_slice(
            slice_source, slice_strategy="tumor_region"
        )

        result = classify_image(img, include_explainability=True)
        result["source"] = "nifti_slice"
        result["source_modality"] = [k for k, v in modalities.items() if v == slice_source][0]
        result["slice_index"] = slice_idx
        result["total_slices"] = total_slices
        result["quality_check"] = quality_checks.get(result["source_modality"])

        buf = BytesIO()
        img.save(buf, format="PNG")
        result["slice_image_b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        return finalize_response(result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Segmentation Endpoint ──────────────────────────────────────────
@app.post("/api/segment")
async def segment(session_id: str = Form(...)):
    """Run 3D segmentation on uploaded MRI modalities (Glioma only)."""
    session_dir = UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        from models.segmentor import segment_patient
        from utils.file_handler import analyze_uploaded_files

        all_files = list(session_dir.rglob("*"))
        analysis = analyze_uploaded_files([str(f) for f in all_files if f.is_file()])
        modalities = analysis.get("modalities", {})

        required = ["t1", "t1ce", "t2", "flair"]
        missing = [m for m in required if m not in modalities]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required modalities: {missing}"
            )

        # Run segmentation (this is CPU/GPU heavy — runs synchronously)
        result = await asyncio.to_thread(
            segment_patient,
            modalities["t1"], modalities["t1ce"],
            modalities["t2"], modalities["flair"],
        )

        # Remove large numpy arrays from response (keep summary data)
        # Store masks in session for progression step
        mask_path = session_dir / "seg_prediction.npz"
        np.savez_compressed(
            str(mask_path),
            pred_mask=result["pred_mask"],
            image_data=result["image_data"],
        )

        response = {
            "volumes": result["volumes"],
            "total_voxels": result["total_voxels"],
            "region_names": result["region_names"],
            "region_colors": result["region_colors"],
            "slices": result["slices"],
            "mesh_data": result["mesh_data"],
            "uncertainty_summary": result.get("uncertainty_summary"),
            "mask_saved": True,
        }

        seg_report_payload = {
            "volumes": response["volumes"],
            "total_voxels": response["total_voxels"],
            "region_names": response["region_names"],
            "region_colors": response["region_colors"],
            "uncertainty_summary": response.get("uncertainty_summary"),
        }
        _write_json(
            session_dir / "segmentation_result.json",
            _compact_payload_for_storage(seg_report_payload),
        )

        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Progression Endpoint ───────────────────────────────────────────
@app.post("/api/progression")
async def progression(
    session_id: str = Form(...),
    grade: str = Form("HGG"),
):
    """Run progression analysis on segmented tumor."""
    session_dir = UPLOAD_DIR / session_id
    mask_path = session_dir / "seg_prediction.npz"

    if not mask_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Segmentation must be run first"
        )

    try:
        from models.progression import run_full_progression

        data = np.load(str(mask_path))
        pred_mask = data["pred_mask"]
        image_data = data["image_data"] if "image_data" in data.files else None

        # Use Whole Tumor channel (index 1) for progression
        wt_mask = pred_mask[:, :, :, 1]

        result = await asyncio.to_thread(
            run_full_progression, wt_mask, grade, image_data
        )

        # Serialize for JSON (remove numpy arrays)
        response = {
            "current_volume": result["current_volume"],
            "grade": result["grade"],
            "logistic": result["logistic"],
            "eval_metrics": result.get("eval_metrics"),
        }

        # Spatial results
        if result["spatial"] is not None:
            sp = result["spatial"]
            response["spatial"] = {
                "stats": sp["stats"],
                "mesh_data": sp["mesh_data"],
            }
        else:
            response["spatial"] = None

        response["insights"] = _build_progression_insights(result)

        params = ((response.get("logistic") or {}).get("params")) or {}
        explainability = {
            "drivers": {
                "initial_volume_mm3": float(params.get("v0", response.get("current_volume", 0.0))),
                "growth_rate_r_per_day": float(params.get("r", 0.0)),
                "carrying_capacity_mm3": float(params.get("k", 0.0)),
            },
            "grade_context": "High-Grade Glioma" if response.get("grade") == "HGG" else "Low-Grade Glioma",
        }

        if response.get("spatial") and response["spatial"].get("stats"):
            st = response["spatial"]["stats"]
            explainability["spatial_balance"] = {
                "growth_voxels": int(st.get("growth_voxels", 0)),
                "stable_voxels": int(st.get("stable_voxels", 0)),
                "regression_voxels": int(st.get("regression_voxels", 0)),
                "volume_change_pct": float(st.get("volume_change_pct", 0.0)),
            }

        response["explainability"] = explainability

        prog_report_payload = {
            "current_volume": response["current_volume"],
            "grade": response["grade"],
            "logistic": {
                "params": (response.get("logistic") or {}).get("params"),
                "projections": (response.get("logistic") or {}).get("projections"),
            },
            "spatial": {
                "stats": ((response.get("spatial") or {}).get("stats")),
            } if response.get("spatial") else None,
            "insights": response.get("insights"),
            "explainability": response.get("explainability"),
        }
        _write_json(
            session_dir / "progression_result.json",
            _compact_payload_for_storage(prog_report_payload),
        )

        return JSONResponse(response)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Report Endpoint ────────────────────────────────────────────────
@app.get("/api/report/{session_id}")
async def download_report(session_id: str):
    """Generate and download a concise clinical-style PDF report."""
    session_dir = UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    classification_json = session_dir / "classification_result.json"
    if not classification_json.exists():
        raise HTTPException(
            status_code=400,
            detail="Classification must be completed before report export",
        )

    payload = _build_report_payload(session_id, session_dir)
    lines = _render_report_lines(payload)
    pdf_bytes = _build_minimal_pdf(lines)

    report_json_path = session_dir / "report_payload.json"
    _write_json(report_json_path, _compact_payload_for_storage(payload))

    headers = {
        "Content-Disposition": f'attachment; filename="neuroai_report_{session_id}.pdf"'
    }
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers=headers,
    )


# ─── Status / Health ────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Health check with model availability info."""
    models_available = {}
    for name, path in CLASSIFICATION_MODELS.items():
        models_available[f"classification/{name}"] = path.exists()

    from config import SEGMENTATION_MODEL_PATH, SPATIAL_UNET_PATH
    models_available["segmentation"] = (
        SEGMENTATION_MODEL_PATH is not None and SEGMENTATION_MODEL_PATH.exists()
    )
    models_available["spatial_unet"] = SPATIAL_UNET_PATH.exists()

    return {
        "status": "ok",
        "models": models_available,
    }


# ─── Cleanup ────────────────────────────────────────────────────────
@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up a session's uploaded files."""
    cleanup_session(session_id)
    return {"status": "cleaned"}


# ─── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
