"""
Step 1: Prepare Spatial Prediction Dataset
==========================================

Extracts consecutive mask pairs from MU-Glioma-Post, crops them around the
tumor region (saves GPU memory), and packages everything into a single
compressed file for upload to Kaggle.

Usage:
    python src/10_prepare_spatial_data.py

Output:
    results/spatial_pairs.npz   (~50-200 MB, upload this to Kaggle)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict
import json
import sys

# ─────────────────────────── Config ────────────────────────────────

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "mu_glioma_post" / "MU-Glioma-Post"
OUT_DIR = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)

CROP_SIZE = (96, 96, 64)   # D, H, W crop around tumor center
PAD_VALUE = 0               # padding for tumors near edges
MIN_TUMOR_VOXELS = 100      # skip masks with < 100 voxels (artifacts)


# ─────────────────────────── Helpers ───────────────────────────────

def load_mask(path):
    """Load NIfTI mask as binary uint8 array."""
    img = nib.load(str(path))
    data = img.get_fdata()
    return (data > 0).astype(np.uint8)


def get_tumor_center(mask):
    """Get center of mass of tumor region."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0).astype(int)


def crop_around_center(vol, center, crop_size):
    """Crop a volume around a center point, with zero-padding at edges."""
    d, h, w = crop_size
    cd, ch, cw = center

    # Calculate start/end with padding
    starts = [cd - d // 2, ch - h // 2, cw - w // 2]
    ends = [s + sz for s, sz in zip(starts, crop_size)]

    # Pad if needed
    pad_before = [max(0, -s) for s in starts]
    pad_after = [max(0, e - vol.shape[i]) for i, e in enumerate(ends)]

    starts = [max(0, s) for s in starts]
    ends = [min(vol.shape[i], e) for i, e in enumerate(ends)]

    cropped = vol[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

    # Apply padding
    if any(p > 0 for p in pad_before + pad_after):
        cropped = np.pad(
            cropped,
            [(pad_before[i], pad_after[i]) for i in range(3)],
            mode='constant', constant_values=PAD_VALUE
        )

    return cropped


def find_shared_center(mask1, mask2):
    """Find the center that covers both masks (union center of mass)."""
    union = np.maximum(mask1, mask2)
    center = get_tumor_center(union)
    if center is None:
        center = get_tumor_center(mask1)
    return center


# ─────────────────────────── Main ──────────────────────────────────

def main():
    print("=" * 60)
    print("  SPATIAL PREDICTION DATA PREPARATION")
    print("=" * 60)

    if not RAW_DIR.exists():
        print(f"ERROR: Data directory not found: {RAW_DIR}")
        sys.exit(1)

    # Discover all patients with multiple timepoint masks
    patient_masks = defaultdict(list)

    for p_dir in sorted(RAW_DIR.iterdir()):
        if not p_dir.is_dir():
            continue
        tp_dirs = sorted([d for d in p_dir.iterdir()
                          if d.is_dir() and 'Timepoint' in d.name])
        for tp_dir in tp_dirs:
            mask_files = list(tp_dir.glob('*tumorMask.nii.gz'))
            if mask_files:
                # Extract timepoint number for ordering
                tp_num = int(tp_dir.name.split('_')[-1])
                patient_masks[p_dir.name].append((tp_num, mask_files[0]))

    # Sort timepoints within each patient
    for pid in patient_masks:
        patient_masks[pid].sort(key=lambda x: x[0])

    # Filter to patients with 2+ timepoints
    multi_tp = {pid: tps for pid, tps in patient_masks.items() if len(tps) >= 2}
    print(f"\nPatients with 2+ masks: {len(multi_tp)}")

    # Extract consecutive pairs
    all_inputs = []
    all_targets = []
    pair_meta = []
    skipped = 0

    for pid in sorted(multi_tp.keys()):
        timepoints = multi_tp[pid]

        for i in range(len(timepoints) - 1):
            tp1_num, tp1_path = timepoints[i]
            tp2_num, tp2_path = timepoints[i + 1]

            # Load masks
            mask1 = load_mask(tp1_path)
            mask2 = load_mask(tp2_path)

            # Skip if either mask is too small
            if mask1.sum() < MIN_TUMOR_VOXELS or mask2.sum() < MIN_TUMOR_VOXELS:
                skipped += 1
                continue

            # Find shared center and crop both masks to same region
            center = find_shared_center(mask1, mask2)
            if center is None:
                skipped += 1
                continue

            crop1 = crop_around_center(mask1, center, CROP_SIZE)
            crop2 = crop_around_center(mask2, center, CROP_SIZE)

            all_inputs.append(crop1)
            all_targets.append(crop2)
            pair_meta.append({
                'patient_id': pid,
                'tp_from': int(tp1_num),
                'tp_to': int(tp2_num),
                'vol_from': int(mask1.sum()),
                'vol_to': int(mask2.sum()),
                'center': center.tolist(),
            })

            print(f"  {pid}: T{tp1_num}->T{tp2_num}  "
                  f"vol {mask1.sum():,}->{mask2.sum():,}  "
                  f"crop {crop1.shape}", end="\r")

    print(f"\n\nTotal pairs extracted: {len(all_inputs)}")
    print(f"Skipped (too small / no tumor): {skipped}")

    # Stack into arrays
    inputs = np.stack(all_inputs, axis=0)   # (N, D, H, W) uint8
    targets = np.stack(all_targets, axis=0)  # (N, D, H, W) uint8

    print(f"\nInput shape:  {inputs.shape}  dtype={inputs.dtype}")
    print(f"Target shape: {targets.shape}  dtype={targets.dtype}")
    print(f"Memory: {(inputs.nbytes + targets.nbytes) / 1e6:.1f} MB (uncompressed)")

    # Train/val split (80/20, patient-level to avoid leakage)
    unique_pids = sorted(set(m['patient_id'] for m in pair_meta))
    np.random.seed(42)
    np.random.shuffle(unique_pids)
    split_idx = int(len(unique_pids) * 0.8)
    train_pids = set(unique_pids[:split_idx])
    val_pids = set(unique_pids[split_idx:])

    train_mask = np.array([m['patient_id'] in train_pids for m in pair_meta])
    val_mask = ~train_mask

    print(f"\nSplit: {train_mask.sum()} train, {val_mask.sum()} val "
          f"({len(train_pids)} / {len(val_pids)} patients)")

    # Save
    out_path = OUT_DIR / "spatial_pairs.npz"
    np.savez_compressed(
        out_path,
        inputs=inputs,
        targets=targets,
        train_mask=train_mask,
        val_mask=val_mask,
    )

    # Save metadata
    meta_path = OUT_DIR / "spatial_pairs_meta.json"
    meta = {
        'n_pairs': len(pair_meta),
        'n_train': int(train_mask.sum()),
        'n_val': int(val_mask.sum()),
        'crop_size': list(CROP_SIZE),
        'pairs': pair_meta,
        'train_patients': sorted(train_pids),
        'val_patients': sorted(val_pids),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    file_size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path} ({file_size_mb:.1f} MB)")
    print(f"Meta:  {meta_path}")
    print(f"\n✅ Upload {out_path.name} to Kaggle as a dataset, then run the training notebook.")


if __name__ == "__main__":
    main()
