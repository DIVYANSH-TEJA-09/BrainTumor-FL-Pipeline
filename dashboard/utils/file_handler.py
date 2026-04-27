"""
File Handler
==============
ZIP extraction, file validation, and modality auto-detection
for uploaded brain MRI data.
"""

import os
import re
import zipfile
import shutil
import uuid
from pathlib import Path

from config import UPLOAD_DIR


# Modality detection patterns
MODALITY_PATTERNS = {
    "t1ce": [
        r"t1ce", r"t1c[^a-z]", r"t1_ce", r"t1gd", r"t1_contrast",
        r"[_\-]0001(?:\.nii(?:\.gz)?)?$",
    ],
    "t1": [
        r"(?<![a-z])t1(?!c)", r"t1[_\-]?w(?:eighted)?",
        r"[_\-]0000(?:\.nii(?:\.gz)?)?$",
    ],
    "t2": [
        r"(?<![a-z])t2(?![a-z])", r"t2[_\-]?w(?:eighted)?",
        r"[_\-]0002(?:\.nii(?:\.gz)?)?$",
    ],
    "flair": [
        r"flair", r"fl(?:uid)?[_\-]?att", r"t2f",
        r"[_\-]0003(?:\.nii(?:\.gz)?)?$",
    ],
    "seg": [r"seg", r"mask", r"label", r"tumor[_\-]?mask"],
}


def create_session_dir():
    """Create a unique upload session directory."""
    session_id = str(uuid.uuid4())[:8]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_id, session_dir


def extract_zip(zip_path, session_dir):
    """
    Extract a ZIP file into the session directory.

    Returns
    -------
    dict with keys:
        - files: list of extracted file paths
        - modalities: dict mapping modality name → file path
        - has_all_required: bool — whether T1, T1ce, T2, FLAIR are all present
    """
    extracted_files = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        base_dir = session_dir.resolve()
        for info in zf.infolist():
            if info.is_dir():
                continue

            # Block symlink entries and path traversal.
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"Unsafe symlink entry in ZIP: {info.filename}")

            member_path = Path(info.filename)
            if member_path.is_absolute() or member_path.drive:
                raise ValueError(f"Unsafe absolute path in ZIP: {info.filename}")

            target_path = (base_dir / member_path).resolve()
            if target_path != base_dir and base_dir not in target_path.parents:
                raise ValueError(f"Unsafe ZIP path traversal attempt: {info.filename}")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, 'r') as src, target_path.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted_files.append(target_path)

    return _analyze_files(extracted_files)


def analyze_uploaded_files(file_paths):
    """
    Analyze a list of uploaded file paths.

    Returns
    -------
    dict with same structure as extract_zip output.
    """
    return _analyze_files([Path(f) for f in file_paths])


def _analyze_files(file_paths):
    """Detect modalities from a list of file paths."""
    # Filter to NIfTI files
    nifti_files = [f for f in file_paths if _is_nifti(f)]
    all_files = [f for f in file_paths if f.is_file()]

    modalities = {}
    unmatched = []

    for fpath in nifti_files:
        fname_lower = fpath.name.lower()
        matched = False

        # Order matters: check t1ce before t1
        for mod_name, patterns in MODALITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, fname_lower):
                    if mod_name not in modalities:
                        modalities[mod_name] = str(fpath)
                        matched = True
                        break
            if matched:
                break

        if not matched:
            unmatched.append(str(fpath))

    required = {"t1", "t1ce", "t2", "flair"}
    has_all_required = required.issubset(set(modalities.keys()))

    return {
        "files": [str(f) for f in all_files],
        "nifti_files": [str(f) for f in nifti_files],
        "modalities": modalities,
        "unmatched": unmatched,
        "has_all_required": has_all_required,
        "has_seg": "seg" in modalities,
        "missing": list(required - set(modalities.keys())),
    }


def _is_nifti(path):
    """Check if a file is a NIfTI file."""
    name = str(path).lower()
    return name.endswith('.nii.gz') or name.endswith('.nii')


def save_uploaded_file(file_storage, session_dir, filename=None):
    """Save an uploaded file to the session directory."""
    if filename is None:
        filename = file_storage.filename
    target = session_dir / filename
    file_storage.save(str(target))
    return target


def cleanup_session(session_id):
    """Remove a session's upload directory."""
    session_dir = UPLOAD_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
