########################################################################
# NOTEBOOK 1: DATA PREPARATION
#
# Kaggle Settings:
#   Accelerator: GPU P100
#   Persistence: Files only
#   Internet: ON
#
# Input Datasets (add via + Add Input):
#   - Brain Tumor MRI Dataset (masoudnickparvar)
#   - BRISC 2025 (briscdataset)
#
# Paste each "# ─── Cell N ───" section into a separate Kaggle cell.
########################################################################


# ─── Cell 1: Setup & Imports ─────────────────────────────────────────

import os, random, json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import torch

# ── Seed ──
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed = {seed}")

set_seed(42)

# ── GPU ──
print(f"PyTorch {torch.__version__}  |  CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
else:
    device = torch.device("cpu")
    print("⚠️ No GPU — using CPU")

# ── Directories ──
for d in ['data/processed/client1', 'data/processed/client2',
          'data/processed/client3', 'data/test_set',
          'models', 'results/fedavg', 'results/qpso', 'results/plots', 'logs']:
    os.makedirs(f'/kaggle/working/{d}', exist_ok=True)
print("✅ Directories created")


# ─── Cell 2: Define Preprocessor ─────────────────────────────────────

class BrainTumorPreprocessor:
    VALID_CLASSES = ["glioma", "meningioma", "pituitary"]
    CLASS_TO_IDX  = {"glioma": 0, "meningioma": 1, "pituitary": 2}

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def load_image(self, path):
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize(self.target_size, Image.LANCZOS)
            return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"⚠️ {path}: {e}")
            return None

    def process_dataset(self, dataset_path, client_name,
                        save_dir="/kaggle/working/data/processed",
                        train_ratio=0.70, val_ratio=0.15):
        print(f"\n{'='*60}\nProcessing {client_name} ← {dataset_path}\n{'='*60}")
        images, labels = [], []

        for cls_name in sorted(os.listdir(dataset_path)):
            cls_path = os.path.join(dataset_path, cls_name)
            if not os.path.isdir(cls_path):
                continue
            if cls_name.lower() not in self.VALID_CLASSES:
                print(f"  Skipping: {cls_name}")
                continue

            label = self.CLASS_TO_IDX[cls_name.lower()]
            files = [f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  {cls_name}: {len(files)} images")

            for fname in tqdm(files, desc=f"    {cls_name}", leave=False):
                arr = self.load_image(os.path.join(cls_path, fname))
                if arr is not None:
                    images.append(arr)
                    labels.append(label)

        X, y = np.array(images), np.array(labels)
        print(f"\n  Total: {len(X)}  Shape: {X.shape}")
        print(f"  Classes: {np.bincount(y, minlength=3)}")

        # Stratified split: 70/15/15
        test_ratio = 1.0 - train_ratio - val_ratio
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=42)
        val_adj = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_adj, stratify=y_tmp, random_state=42)

        print(f"  Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

        out = os.path.join(save_dir, client_name)
        os.makedirs(out, exist_ok=True)
        for name, arr in [("X_train", X_train), ("y_train", y_train),
                          ("X_val", X_val), ("y_val", y_val),
                          ("X_test", X_test), ("y_test", y_test)]:
            np.save(os.path.join(out, f"{name}.npy"), arr)
        print(f"  ✅ Saved → {out}")

    @staticmethod
    def create_global_test_set(processed_dir="/kaggle/working/data/processed",
                                save_dir="/kaggle/working/data/test_set"):
        Xs, ys = [], []
        for i in range(1, 4):
            d = os.path.join(processed_dir, f"client{i}")
            Xs.append(np.load(os.path.join(d, "X_test.npy")))
            ys.append(np.load(os.path.join(d, "y_test.npy")))
        X, y = np.concatenate(Xs), np.concatenate(ys)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X_test.npy"), X)
        np.save(os.path.join(save_dir, "y_test.npy"), y)
        print(f"✅ Global test: {len(X)} images → {save_dir}")

print("✅ Preprocessor defined")


# ─── Cell 3: Verify Dataset Paths ────────────────────────────────────

# ⚠️ UPDATE THESE IF YOUR PATHS DIFFER!
# Run: for d in sorted(os.listdir('/kaggle/input')): print(d)
# to see your actual dataset folder names.

CLIENT1_PATH = '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Testing'
CLIENT2_PATH = '/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/classification_task/train'
CLIENT3_PATH = '/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/Training'

for name, path in [("Client1 (Masoud Test)", CLIENT1_PATH),
                   ("Client2 (BRISC)", CLIENT2_PATH),
                   ("Client3 (Masoud Train)", CLIENT3_PATH)]:
    print(f"\n{name}: {path}")
    if os.path.exists(path):
        for d in sorted(os.listdir(path)):
            full = os.path.join(path, d)
            if os.path.isdir(full):
                print(f"  📂 {d}/ ({len(os.listdir(full))} files)")
    else:
        print("  ❌ PATH NOT FOUND!")


# ─── Cell 4: Preprocess All 3 Clients ────────────────────────────────

prep = BrainTumorPreprocessor(target_size=(224, 224))

prep.process_dataset(CLIENT1_PATH, 'client1')
prep.process_dataset(CLIENT2_PATH, 'client2')
prep.process_dataset(CLIENT3_PATH, 'client3')


# ─── Cell 5: Create Global Test Set ──────────────────────────────────

BrainTumorPreprocessor.create_global_test_set()


# ─── Cell 6: Verify Everything ────────────────────────────────────────

import numpy as np
for c in ['client1', 'client2', 'client3']:
    d = f'/kaggle/working/data/processed/{c}'
    for f in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        arr = np.load(f'{d}/{f}.npy')
        print(f"  {c}/{f}: {arr.shape}")

gx = np.load('/kaggle/working/data/test_set/X_test.npy')
gy = np.load('/kaggle/working/data/test_set/y_test.npy')
print(f"\n  Global test: X={gx.shape}  y={gy.shape}  classes={np.bincount(gy)}")
print("\n✅ Notebook 1 complete! Save Version → Save & Run All")
