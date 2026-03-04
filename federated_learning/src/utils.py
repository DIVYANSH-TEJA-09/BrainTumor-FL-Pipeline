"""
Utility functions for FL-QPSO experiment.
Seed, GPU check, directory creation, config save/load.
"""

import os
import json
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")


def check_gpu():
    """Print GPU info and return torch device."""
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version:    {torch.version.cuda}")
        print(f"GPU Count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({mem_gb:.2f} GB)")
        device = torch.device("cuda")
    else:
        print("⚠️  No GPU detected — using CPU")
        device = torch.device("cpu")

    return device


def create_directories(base="/kaggle/working"):
    """Create all project directories under *base*."""
    dirs = [
        f"{base}/data/processed/client1",
        f"{base}/data/processed/client2",
        f"{base}/data/processed/client3",
        f"{base}/data/test_set",
        f"{base}/models",
        f"{base}/results/fedavg",
        f"{base}/results/qpso",
        f"{base}/results/plots",
        f"{base}/logs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ Directory structure created under {base}")


def save_experiment_config(config: dict, method_name: str,
                           base="/kaggle/working"):
    """Save experiment config dict as JSON."""
    path = f"{base}/results/{method_name}/config.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved → {path}")


def load_checkpoint(model, checkpoint_path, device="cuda"):
    """Load a model checkpoint. Returns the model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    print(f"✅ Checkpoint loaded ← {checkpoint_path}")
    return model
