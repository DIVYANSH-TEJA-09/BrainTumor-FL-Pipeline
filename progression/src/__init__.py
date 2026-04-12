"""
Tumor Progression Module

For FL-QPSO Brain Tumor Management System
Module 3: Longitudinal Progression Forecasting

This package implements:
- Mathematical baseline models (Exponential, Gompertz, Logistic, Linear)
- LSTM-based deep learning enhancement
- 3D visualization of progression
- Integration with Module 1 (Segmentation)

Novel Contribution:
- Hybrid math + DL approach for pre-treatment planning
- Grade-stratified models (LGG vs HGG)
- Clinical interpretability + predictive power
"""

__version__ = "0.1.0"
__author__ = "DIVYANSH-TEJA-09 & Indhumathi L.K."

from .data_loader import (
    ProgressionDataLoader,
    TimeseriesPatient,
    ProgressionDataset,
    create_dataloaders,
)

__all__ = [
    "ProgressionDataLoader",
    "TimeseriesPatient",
    "ProgressionDataset",
    "create_dataloaders",
]
