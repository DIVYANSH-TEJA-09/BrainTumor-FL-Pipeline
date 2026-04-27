"""Quick test: classify BraTS NIfTI data using the QPSO-FL ResNet."""
import sys, glob
sys.path.insert(0, ".")

from models.classifier import classify_image, CLASS_NAMES_3
from utils.preprocessing import extract_classification_slices_for_voting

# Find any NIfTI files in uploads
niftis = glob.glob("uploads/**/*flair*nii*", recursive=True)
if not niftis:
    niftis = glob.glob("uploads/**/*.nii*", recursive=True)

if niftis:
    print(f"Testing with: {niftis[0]}")
    slices, total = extract_classification_slices_for_voting(niftis[0], n_slices=5)
    print(f"Extracted {len(slices)} slices from {total} total")
    for img, idx in slices:
        result = classify_image(img)
        ens = result["ensemble"]
        cn = ens["class_name"]
        cf = ens["confidence"]
        print(f"  Slice {idx}: {cn} ({cf:.1%})")
        for k, v in ens["probabilities"].items():
            print(f"    {k}: {v:.4f}")
else:
    print("No NIfTI files in uploads. Upload BraTS data first via the dashboard.")
