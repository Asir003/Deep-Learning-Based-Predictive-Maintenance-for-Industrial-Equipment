from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import config

class PreprocessedData:

    train_loader: DataLoader
    test_loader: DataLoader
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    input_size: int
    feature_names: list[str]
    scaler: StandardScaler
    type_encoder: LabelEncoder


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    
    path = csv_path or config.DATASET_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please place ai4i2020.csv in the dataset/ folder."
        )

    df = pd.read_csv(path)
    return df

