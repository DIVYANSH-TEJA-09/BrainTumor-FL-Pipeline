"""
BrainTumorPreprocessor — load, filter, resize, split, and save
brain-tumor MRI images for federated learning.

Only 3 classes are kept: glioma, meningioma, pituitary.
'normal' / 'notumor' / 'no_tumor' folders are excluded automatically.
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


class BrainTumorPreprocessor:
    """
    End-to-end preprocessor for one dataset → one FL client.

    Workflow
    -------
    1. Scan dataset_path for class sub-folders.
    2. Filter to valid classes only.
    3. Load each image → RGB → resize → normalise to [0, 1].
    4. Stratified train / val / test split.
    5. Save as .npy arrays.
    """

    VALID_CLASSES = ["glioma", "meningioma", "pituitary"]
    CLASS_TO_IDX = {"glioma": 0, "meningioma": 1, "pituitary": 2}
    IDX_TO_CLASS = {0: "glioma", 1: "meningioma", 2: "pituitary"}

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    # ------------------------------------------------------------------
    def is_valid_class(self, class_name: str) -> bool:
        """Return True if *class_name* is one of the 3 target tumour types."""
        return class_name.lower() in self.VALID_CLASSES

    # ------------------------------------------------------------------
    def load_and_preprocess_image(self, image_path: str):
        """
        Load → RGB → resize → float32 [0, 1].

        Returns
        -------
        np.ndarray of shape (H, W, 3) or None on error.
        """
        try:
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize(self.target_size, Image.LANCZOS)
            return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            return None

    # ------------------------------------------------------------------
    def process_dataset(
        self,
        dataset_path: str,
        client_name: str,
        save_dir: str = "/kaggle/working/data/processed",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        """
        Process an entire dataset folder for one client.

        Parameters
        ----------
        dataset_path : root folder containing class sub-folders.
        client_name  : e.g. 'client1'.
        save_dir     : parent dir; files go into {save_dir}/{client_name}/.
        train_ratio, val_ratio : remaining fraction goes to test.

        Returns
        -------
        dict  {'train': (X, y), 'val': (X, y), 'test': (X, y)}
        """
        print(f"\n{'='*60}")
        print(f"Processing {client_name}  ←  {dataset_path}")
        print(f"{'='*60}")

        images, labels = [], []

        for class_name in sorted(os.listdir(dataset_path)):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
            if not self.is_valid_class(class_name):
                print(f"  Skipping class: {class_name}")
                continue

            label = self.CLASS_TO_IDX[class_name.lower()]
            exts = (".jpg", ".jpeg", ".png")
            files = [f for f in os.listdir(class_path)
                     if f.lower().endswith(exts)]

            print(f"  Loading {class_name} ({len(files)} images) …")
            for fname in tqdm(files, desc=f"    {class_name}", leave=False):
                arr = self.load_and_preprocess_image(
                    os.path.join(class_path, fname)
                )
                if arr is not None:
                    images.append(arr)
                    labels.append(label)

        X = np.array(images)
        y = np.array(labels)
        print(f"\n  Total loaded: {len(X)}  |  shape: {X.shape}")
        print(f"  Label counts: {np.bincount(y, minlength=3)}")

        # --- stratified split ------------------------------------------------
        test_ratio = 1.0 - train_ratio - val_ratio
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=42
        )
        val_adj = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_adj, stratify=y_tmp, random_state=42
        )

        print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

        # --- save -------------------------------------------------------------
        out = os.path.join(save_dir, client_name)
        os.makedirs(out, exist_ok=True)
        for name, arr in [
            ("X_train", X_train), ("y_train", y_train),
            ("X_val", X_val),     ("y_val", y_val),
            ("X_test", X_test),   ("y_test", y_test),
        ]:
            np.save(os.path.join(out, f"{name}.npy"), arr)

        print(f"  ✅ Saved to {out}")
        return {"train": (X_train, y_train),
                "val": (X_val, y_val),
                "test": (X_test, y_test)}

    # ------------------------------------------------------------------
    @staticmethod
    def create_global_test_set(
        processed_dir="/kaggle/working/data/processed",
        save_dir="/kaggle/working/data/test_set",
        num_clients=3,
    ):
        """Combine per-client test sets into a single global test set."""
        Xs, ys = [], []
        for i in range(1, num_clients + 1):
            d = os.path.join(processed_dir, f"client{i}")
            Xs.append(np.load(os.path.join(d, "X_test.npy")))
            ys.append(np.load(os.path.join(d, "y_test.npy")))

        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "X_test.npy"), X)
        np.save(os.path.join(save_dir, "y_test.npy"), y)
        print(f"✅ Global test set: {len(X)} images  →  {save_dir}")
        return X, y
