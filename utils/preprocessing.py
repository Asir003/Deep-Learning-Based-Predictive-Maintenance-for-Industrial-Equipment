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

@dataclass
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

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    
    data = df.copy()

    # Drop columns that are not predictive features
    existing_drop_cols = [col for col in config.COLUMNS_TO_DROP if col in data.columns]
    data = data.drop(columns=existing_drop_cols)

    # Check for missing values and report
    missing_count = data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: Found {missing_count} missing values. Dropping rows.")
        data = data.dropna()
    else:
        print("No missing values detected.")

    return data

def encode_product_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    
    data = df.copy()
    encoder = LabelEncoder()
    data[config.TYPE_COLUMN] = encoder.fit_transform(data[config.TYPE_COLUMN])
    return data, encoder

def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
   
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found.")

    feature_names = [col for col in df.columns if col != config.TARGET_COLUMN]
    X = df[feature_names]
    y = df[config.TARGET_COLUMN]
    return X, y, feature_names

def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def create_dataloaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # CPU-safe on Windows
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader

def preprocess_pipeline(
    csv_path: Path | None = None,
    test_size: float = config.TEST_SIZE,
    random_seed: int = config.RANDOM_SEED,
    batch_size: int = config.BATCH_SIZE,
    
) -> PreprocessedData:
   
    # Load and clean
    raw_df = load_dataset(csv_path)
    print(f"Loaded dataset with {len(raw_df)} rows and {len(raw_df.columns)} columns.")

    cleaned_df = clean_dataset(raw_df)
    encoded_df, type_encoder = encode_product_type(cleaned_df)

    X, y, feature_names = prepare_features_and_target(encoded_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    print(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    print(f"Failure rate (train): {y_train.mean():.4f} | (test): {y_test.mean():.4f}")

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(
        X_train_scaled, X_test_scaled, y_train, y_test, batch_size
    )

    return PreprocessedData(
        train_loader=train_loader,
        test_loader=test_loader,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        input_size=X_train_scaled.shape[1],
        feature_names=feature_names,
        scaler=scaler,
        type_encoder=type_encoder,
    )





    
