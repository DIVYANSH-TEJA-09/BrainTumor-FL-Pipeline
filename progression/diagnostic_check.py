import json
from pathlib import Path

print("\n=== DIAGNOSTIC CHECK ===\n")

# Check from progression directory
app_file = Path("streamlit_3d_progression.py")
print(f"App file: {app_file}")
print(f"App absolute: {app_file.resolve()}")

DATA_DIR = app_file.parent / "data" / "raw" / "mu_glioma_post"
RESULTS_DIR = app_file.parent / "streamlit_data"
PRED_INDEX_FILE = RESULTS_DIR / "prediction_index.json"

print(f"\nData dir: {DATA_DIR.resolve()}")
print(f"Exists: {DATA_DIR.exists()}")

print(f"\nResults dir: {RESULTS_DIR.resolve()}")
print(f"Exists: {RESULTS_DIR.exists()}")

print(f"\nPrediction index: {PRED_INDEX_FILE.resolve()}")
print(f"Exists: {PRED_INDEX_FILE.exists()}")

if PRED_INDEX_FILE.exists():
    with open(PRED_INDEX_FILE) as f:
        data = json.load(f)
    print(f"Successfully loaded: {data['total_patients']} patients")
else:
    print("ERROR: File not found!")
    # List what's in the directory
    if RESULTS_DIR.exists():
        print(f"\nFiles in {RESULTS_DIR}:")
        for f in list(RESULTS_DIR.glob("*"))[:10]:
            print(f"  - {f.name}")
