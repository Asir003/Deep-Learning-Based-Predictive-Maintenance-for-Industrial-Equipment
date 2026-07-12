from pathlib import Path

# Project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATASET_PATH: Path = PROJECT_ROOT / "dataset" / "ai4i2020.csv"
# Dataset columns

# Training hyperparameters

# Device configuration
DEVICE: str = "cpu"