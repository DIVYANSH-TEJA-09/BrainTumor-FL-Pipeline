# 2D Brain Tumor Segmentation (BraTS 2020)

Experimental 2D segmentation approaches for brain tumor MRI using the BraTS 2020 dataset.

> **Note:** This is an experimental/legacy module. The primary segmentation pipeline uses the [3D Attention U-Net](../segmentation/) approach.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `BinarySeg.ipynb` | Binary tumor segmentation (tumor vs. background) |
| `MulticlassSeg.ipynb` | Multi-class segmentation (WT, TC, ET) |

## Models Trained

| Model File | Architecture | Purpose |
|------------|-------------|---------|
| `unet_binary.h5` | 2D U-Net | Binary segmentation |
| `unet_balanced.h5` | 2D U-Net | Class-balanced training |
| `unet_focal_dice.h5` | 2D U-Net | Focal + Dice loss |
| `unet_weighted_best.h5` | 2D U-Net | Best weighted model |

> **Note:** Model weight files (`.h5`) are excluded from git via `.gitignore`. Download or retrain locally.
