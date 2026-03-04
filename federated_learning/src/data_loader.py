"""
DataLoader factory for federated learning.
Creates per-client and global-test loaders with proper augmentation.
"""

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.dataset import BrainTumorDataset


# ---------- transforms -------------------------------------------------------

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------- factory -----------------------------------------------------------

def create_data_loaders(
    processed_dir="/kaggle/working/data/processed",
    test_set_dir="/kaggle/working/data/test_set",
    batch_size=32,
    num_workers=2,
):
    """
    Build DataLoaders for all 3 clients + global test set.

    Returns
    -------
    dict with keys 'client1', 'client2', 'client3', 'global_test'.
    Each client entry has sub-keys: 'train', 'val', 'test', 'train_size'.
    """
    loaders = {}

    for i in range(1, 4):
        name = f"client{i}"
        d = f"{processed_dir}/{name}"

        X_tr = np.load(f"{d}/X_train.npy")
        y_tr = np.load(f"{d}/y_train.npy")
        X_va = np.load(f"{d}/X_val.npy")
        y_va = np.load(f"{d}/y_val.npy")
        X_te = np.load(f"{d}/X_test.npy")
        y_te = np.load(f"{d}/y_test.npy")

        ds_tr = BrainTumorDataset(X_tr, y_tr, transform=TRAIN_TRANSFORM)
        ds_va = BrainTumorDataset(X_va, y_va, transform=TEST_TRANSFORM)
        ds_te = BrainTumorDataset(X_te, y_te, transform=TEST_TRANSFORM)

        loaders[name] = {
            "train": DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True),
            "val":   DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True),
            "test":  DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True),
            "train_size": len(ds_tr),
        }
        print(f"{name}: Train={len(ds_tr)}  Val={len(ds_va)}  Test={len(ds_te)}")

    # global test
    gX = np.load(f"{test_set_dir}/X_test.npy")
    gy = np.load(f"{test_set_dir}/y_test.npy")
    gds = BrainTumorDataset(gX, gy, transform=TEST_TRANSFORM)
    loaders["global_test"] = DataLoader(
        gds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"Global test: {len(gds)} images")

    print("✅ All data loaders created")
    return loaders
