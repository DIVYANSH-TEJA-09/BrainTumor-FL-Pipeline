"""
PyTorch Dataset for brain-tumor MRI images stored as .npy arrays.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class BrainTumorDataset(Dataset):
    """
    Wraps pre-processed numpy arrays (N, H, W, 3) in [0, 1]
    into a PyTorch Dataset.

    Parameters
    ----------
    X : np.ndarray  — images, shape (N, 224, 224, 3), float32 in [0, 1].
    y : np.ndarray  — labels, shape (N,), int64.
    transform : optional torchvision transform pipeline.
                If None, applies ImageNet normalisation only.
    """

    _DEFAULT_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = X
        self.y = y
        self.transform = transform or self._DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # numpy [0,1] float → uint8 PIL → transform
        image = Image.fromarray((self.X[idx] * 255).astype(np.uint8))
        image = self.transform(image)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return image, label
