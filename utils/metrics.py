

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

import config


@dataclass
class EvaluationResults:
    

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = config.DEVICE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    model.eval()
    all_labels: List[float] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            outputs = model(features)
            probs = outputs.cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().flatten().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    return y_true, y_pred, y_prob


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> EvaluationResults:
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Failure", "Failure"],
        zero_division=0,
    )

    return EvaluationResults(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=auc,
        confusion_matrix=cm,
        classification_report=report,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def save_evaluation_report(
    results: EvaluationResults,
    reports_dir: Path | None = None,
) -> None:
    
    output_dir = reports_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_text = (
        "Predictive Maintenance - Evaluation Report\n"
        "=============================================\n\n"
        f"Accuracy  : {results.accuracy:.4f}\n"
        f"Precision : {results.precision:.4f}\n"
        f"Recall    : {results.recall:.4f}\n"
        f"F1 Score  : {results.f1:.4f}\n"
        f"ROC-AUC   : {results.roc_auc:.4f}\n\n"
        "Confusion Matrix:\n"
        f"{results.confusion_matrix}\n\n"
        "Classification Report:\n"
        f"{results.classification_report}\n"
    )

    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(metrics_text)

    print(f"Evaluation report saved to '{report_path}'.")


def plot_confusion_matrix(
    cm: np.ndarray,
    plots_dir: Path | None = None,
) -> None:
   
    output_dir = plots_dir or config.PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Confusion matrix plot saved to '{output_dir / 'confusion_matrix.png'}'.")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    plots_dir: Path | None = None,
) -> None:
    
    output_dir = plots_dir or config.PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Machine Failure Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    print(f"ROC curve saved to '{output_dir / 'roc_curve.png'}'.")
