from pathlib import Path

# Project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATASET_PATH: Path = PROJECT_ROOT / "dataset" / "ai4i2020.csv"
SAVED_MODEL_DIR: Path = PROJECT_ROOT / "saved_model"
MODEL_PATH: Path = SAVED_MODEL_DIR / "best_model.pth"
SCALER_PATH: Path = SAVED_MODEL_DIR / "scaler.joblib"
ENCODER_PATH: Path = SAVED_MODEL_DIR / "type_encoder.joblib"
FEATURE_NAMES_PATH: Path = SAVED_MODEL_DIR / "feature_names.joblib"

PLOTS_DIR: Path = PROJECT_ROOT / "outputs" / "plots"
REPORTS_DIR: Path = PROJECT_ROOT / "outputs" / "reports"

# Dataset columns
TARGET_COLUMN: str = "Machine failure"
COLUMNS_TO_DROP: list[str] = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]
TYPE_COLUMN: str = "Type"
NUMERIC_FEATURES: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Training hyperparameters
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20
BATCH_SIZE: int = 64
LEARNING_RATE: float = 0.001
NUM_EPOCHS: int = 50
DROPOUT_RATE: float = 0.3
EARLY_STOPPING_PATIENCE: int = 10

HIDDEN_LAYERS: list[int] = [128, 64, 32]

# Device configuration
DEVICE: str = "cpu"