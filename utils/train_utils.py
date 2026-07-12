

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config


@dataclass
class TrainingHistory:
    

    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = config.DEVICE,
) -> Tuple[float, float]:
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        predictions = (outputs >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = config.DEVICE,
) -> Tuple[float, float]:
   
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


class EarlyStopping:
    

    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE) -> None:
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state_dict: Dict | None = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
       
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def save_model(
    model: nn.Module,
    input_size: int,
    path: Path | None = None,
) -> None:
    
    save_path = path or config.MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_layers": config.HIDDEN_LAYERS,
        "dropout_rate": config.DROPOUT_RATE,
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to '{save_path}'.")


def load_model(path: Path | None = None) -> Tuple[nn.Module, int]:
   
    from models.model import PredictiveMaintenanceNet

    load_path = path or config.MODEL_PATH
    if not load_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{load_path}'. "
            "Please run train.py first."
        )

    checkpoint = torch.load(load_path, map_location=config.DEVICE, weights_only=False)
    input_size = checkpoint["input_size"]

    model = PredictiveMaintenanceNet(
        input_size=input_size,
        hidden_layers=checkpoint.get("hidden_layers", config.HIDDEN_LAYERS),
        dropout_rate=checkpoint.get("dropout_rate", config.DROPOUT_RATE),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()

    return model, input_size


def plot_training_curves(
    history: TrainingHistory,
    plots_dir: Path | None = None,
) -> None:
    
    output_dir = plots_dir or config.PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.train_losses) + 1)

    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.train_losses, label="Training Loss", marker="o", markersize=3)
    plt.plot(epochs, history.val_losses, label="Validation Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        history.train_accuracies,
        label="Training Accuracy",
        marker="o",
        markersize=3,
    )
    plt.plot(
        epochs,
        history.val_accuracies,
        label="Validation Accuracy",
        marker="s",
        markersize=3,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    print(f"Training curves saved to '{output_dir}'.")
