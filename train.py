import sys

import torch
import torch.nn as nn
import torch.optim as optim

import config
from models.model import PredictiveMaintenanceNet
from utils.preprocessing import preprocess_pipeline
from utils.train_utils import (
    EarlyStopping,
    TrainingHistory,
    plot_training_curves,
    save_model,
    train_one_epoch,
    validate_one_epoch,
)

def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across libraries."""
    torch.manual_seed(seed)
    import numpy as np

    np.random.seed(seed)

def main() -> None:
    """Run the full training pipeline."""
    print("=" * 60)
    print("Predictive Maintenance - Model Training")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print()

    try:
        set_seed()

        # ------------------------------------------------------------------
        # 1. Preprocess data
        # ------------------------------------------------------------------
        print("Step 1: Loading and preprocessing data...")
        data = preprocess_pipeline()
        print(f"Input features: {data.input_size}")
        print(f"Feature names: {data.feature_names}")
        print()