"""
Preprocessing Utilities
========================
Extract 2D classification slices from 3D NIfTI volumes,
and prepare data for the MONAI segmentation pipeline.

Key insight: The classification model was trained on 2D MRI images
with specific appearance (CLAHE-like contrast, tight brain crops).
Raw NIfTI slices look different, so we match the training distribution
with percentile-based windowing and brain-region cropping.
"""

import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from typing import Tuple

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def _normalize_brain_slice(slice_2d):
    """
    Normalize a 2D brain MRI slice to match the appearance of
    typical classification training data.

    Uses percentile windowing (rather than min-max) to handle
    the bright skull and dark background in raw NIfTI slices.
    Then applies CLAHE-like contrast enhancement.
    """
    # Remove zero background for percentile calculation
    nonzero = slice_2d[slice_2d > 0]
    if len(nonzero) == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    # Percentile windowing: clip to 1st-99th percentile
    p1, p99 = np.percentile(nonzero, [1, 99])
    clipped = np.clip(slice_2d, p1, p99)

    # Normalize to 0-255
    s_min, s_max = clipped.min(), clipped.max()
    if s_max > s_min:
        normalized = ((clipped - s_min) / (s_max - s_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(slice_2d, dtype=np.uint8)

    return normalized


def _crop_brain_region(img_array):
    """
    Crop to the bounding box of the brain region (non-zero),
    with a small margin. This makes the image look more like
    the training data which was tightly cropped around the brain.
    """
    # Find non-zero bounding box
    nonzero_rows = np.any(img_array > 10, axis=1)
    nonzero_cols = np.any(img_array > 10, axis=0)

    if not nonzero_rows.any() or not nonzero_cols.any():
        return img_array

    rmin, rmax = np.where(nonzero_rows)[0][[0, -1]]
    cmin, cmax = np.where(nonzero_cols)[0][[0, -1]]

    # Add margin (5%)
    h, w = img_array.shape
    margin_r = max(1, int((rmax - rmin) * 0.05))
    margin_c = max(1, int((cmax - cmin) * 0.05))

    rmin = max(0, rmin - margin_r)
    rmax = min(h, rmax + margin_r)
    cmin = max(0, cmin - margin_c)
    cmax = min(w, cmax + margin_c)

    return img_array[rmin:rmax, cmin:cmax]


def extract_classification_slice(nifti_path, modality="flair", slice_strategy="tumor_region"):
    """
    Extract a 2D axial slice from a 3D NIfTI volume for classification.

    Uses improved preprocessing:
    - Percentile windowing (not min-max) for better contrast
    - Brain region cropping to match training data appearance
    - Selects slices from the tumor region (high-variance central slices)

    Parameters
    ----------
    nifti_path : str or Path
        Path to a NIfTI file.
    modality : str
        Which modality this is (for logging).
    slice_strategy : str
        "tumor_region" — pick the slice with highest variance in central brain.
        "middle" — use the middle axial slice.
        "multi_vote" — extract 5 slices, return the best one for classification.

    Returns
    -------
    PIL.Image — RGB image suitable for classification model input.
    int — slice index used.
    int — total number of slices.
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required: pip install nibabel")

    nii = nib.load(str(nifti_path))
    data = nii.get_fdata()

    if len(data.shape) == 4:
        data = data[:, :, :, 0]

    n_slices = data.shape[2]

    if slice_strategy == "tumor_region":
        # Find slices in the central region with highest tissue variance
        # (tumor regions have heterogeneous signal → high variance)
        central_start = int(n_slices * 0.3)
        central_end = int(n_slices * 0.7)

        best_idx = n_slices // 2
        best_score = -1

        for i in range(central_start, central_end):
            s = data[:, :, i]
            nonzero = s[s > 0]
            if len(nonzero) < 100:
                continue
            # Score: variance of nonzero intensities (tumor = high variance)
            score = float(np.var(nonzero))
            if score > best_score:
                best_score = score
                best_idx = i

        slice_idx = best_idx

    elif slice_strategy == "multi_vote":
        # Return multiple slices for voting (handled by caller)
        slice_idx = n_slices // 2
    else:
        slice_idx = n_slices // 2

    slice_2d = data[:, :, slice_idx]

    # Apply improved normalization
    normalized = _normalize_brain_slice(slice_2d)

    # Crop to brain region
    cropped = _crop_brain_region(normalized)

    # Convert to RGB PIL image
    img = Image.fromarray(cropped, mode='L').convert('RGB')

    return img, slice_idx, n_slices


def extract_classification_slices_for_voting(nifti_path, n_slices=5):
    """
    Extract multiple slices for multi-slice voting classification.
    Returns list of (PIL.Image, slice_idx).
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required")

    data = nib.load(str(nifti_path)).get_fdata()
    if len(data.shape) == 4:
        data = data[:, :, :, 0]

    total = data.shape[2]

    # Focus on central 40% of slices (where tumor is most likely)
    start = int(total * 0.3)
    end = int(total * 0.7)

    # Score all central slices by variance
    scores = []
    for i in range(start, end):
        s = data[:, :, i]
        nonzero = s[s > 0]
        if len(nonzero) < 100:
            scores.append((i, 0))
        else:
            scores.append((i, float(np.var(nonzero))))

    # Pick top n_slices by variance
    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [s[0] for s in scores[:n_slices]]
    top_indices.sort()

    results = []
    for idx in top_indices:
        slice_2d = data[:, :, idx]
        normalized = _normalize_brain_slice(slice_2d)
        cropped = _crop_brain_region(normalized)
        img = Image.fromarray(cropped, mode='L').convert('RGB')
        results.append((img, idx))

    return results, total


def extract_axial_slice(nifti_path: str | Path, slice_index: int | None = None) -> tuple[np.ndarray, int, int]:
    """
    Extract a single axial slice from a NIfTI volume without loading the full data.

    Parameters
    ----------
    nifti_path : str or Path
        Path to a NIfTI file.
    slice_index : int or None
        Optional axial index. Defaults to middle slice if out of range or None.

    Returns
    -------
    np.ndarray
        2D slice array.
    int
        Slice index used.
    int
        Total number of slices.
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required")

    nii = nib.load(str(nifti_path))
    data = nii.dataobj

    if data.ndim == 4:
        data = data[:, :, :, 0]

    n_slices = int(data.shape[2])
    if slice_index is None or slice_index < 0 or slice_index >= n_slices:
        slice_index = n_slices // 2

    slice_2d = np.asanyarray(data[:, :, slice_index])
    return slice_2d, int(slice_index), n_slices


def render_report_slice(nifti_path: str | Path, slice_index: int | None = None) -> tuple[Image.Image, int, int]:
    """
    Render a normalized axial slice for reporting from a NIfTI file.

    Parameters
    ----------
    nifti_path : str or Path
        Path to NIfTI file.
    slice_index : int or None
        Optional axial slice index. Defaults to middle slice.

    Returns
    -------
    PIL.Image
        RGB image suitable for reporting.
    int
        Slice index used.
    int
        Total number of slices.
    """
    slice_2d, idx, total = extract_axial_slice(nifti_path, slice_index=slice_index)
    normalized = _normalize_brain_slice(slice_2d)
    cropped = _crop_brain_region(normalized)
    img = Image.fromarray(cropped, mode="L").convert("RGB")
    return img, idx, total


def extract_multi_slices(nifti_path, n_slices=5):
    """
    Extract multiple representative axial slices for thumbnail preview.
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required")

    data = nib.load(str(nifti_path)).get_fdata()
    if len(data.shape) == 4:
        data = data[:, :, :, 0]

    total = data.shape[2]
    indices = np.linspace(total * 0.25, total * 0.75, n_slices, dtype=int)

    images = []
    for idx in indices:
        idx = min(idx, total - 1)
        slice_2d = data[:, :, idx]
        normalized = _normalize_brain_slice(slice_2d)
        images.append(Image.fromarray(normalized, mode='L'))

    return images


def compute_volume_from_mask(mask_data, voxel_dims=(1.0, 1.0, 1.0)):
    """
    Compute tumor volume in mm³ from a binary mask.

    Parameters
    ----------
    mask_data : np.ndarray
        Binary segmentation mask.
    voxel_dims : tuple
        Voxel dimensions in mm.

    Returns
    -------
    float — volume in mm³
    """
    voxel_volume = float(np.prod(voxel_dims))
    tumor_voxels = (mask_data > 0).sum()
    return float(tumor_voxels * voxel_volume)


def assess_nifti_quality(nifti_path):
    """Basic scan quality and OOD checks for uploaded NIfTI volumes."""
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required")

    nii = nib.load(str(nifti_path))
    data = nii.get_fdata()
    if data.ndim == 4:
        data = data[:, :, :, 0]

    shape = tuple(int(v) for v in data.shape)
    nonzero = data[data > 0]
    nonzero_ratio = float((data > 0).sum() / max(1, data.size))

    if len(nonzero) > 0:
        p1, p50, p99 = np.percentile(nonzero, [1, 50, 99])
        dynamic_range = float(p99 - p1)
    else:
        p1 = p50 = p99 = 0.0
        dynamic_range = 0.0

    warnings = []
    if min(shape) < 64:
        warnings.append("Very low spatial dimension detected")
    if nonzero_ratio < 0.03:
        warnings.append("Low brain tissue occupancy (possible wrong scan/mask)")
    if dynamic_range < 20:
        warnings.append("Low signal dynamic range")

    status = "ok"
    if len(warnings) >= 2:
        status = "high_risk"
    elif warnings:
        status = "review"

    return {
        "shape": shape,
        "nonzero_ratio": nonzero_ratio,
        "intensity_percentiles": {
            "p1": float(p1),
            "p50": float(p50),
            "p99": float(p99),
        },
        "dynamic_range": dynamic_range,
        "status": status,
        "warnings": warnings,
    }
