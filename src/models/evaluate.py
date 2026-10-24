"""Model evaluation metrics, calibration analysis, and visualization generators."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_binary_classifier(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Computes comprehensive performance metrics for a binary placement classifier.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (placed = 1).
        threshold: Decision threshold for classification (default 0.50).

    Returns:
        Dictionary of computed performance metrics and curves.
    """
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Core scalar metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    brier = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (int(x) for x in cm.ravel()) if cm.size == 4 else (0, 0, 0, 0)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "calibration": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }


def evaluate_multiclass_classifier(
    y_true: np.ndarray | list[Any],
    y_pred: np.ndarray | list[Any],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Computes evaluation metrics for multiclass salary band classifier.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        labels: Optional explicit list of class labels.

    Returns:
        Dictionary of multiclass metrics (accuracy, macro-F1, weighted-F1).
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
    }


def plot_roc_pr_curves(
    models_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str | Path = "reports/figures",
) -> None:
    """Generates and saves comparative ROC and PR curves for multiple evaluated models.

    Args:
        models_predictions: dict mapping model_name -> (y_true, y_prob).
        output_dir: Directory where plots should be saved.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curves
    plt.figure(figsize=(8, 6), dpi=150)
    for name, (y_true, y_prob) in models_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.500)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves_comparison.png")
    plt.close()

    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6), dpi=150)
    for name, (y_true, y_prob) in models_predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {ap:.3f})")

    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall (PR) Curve Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curves_comparison.png")
    plt.close()

    logger.info(f"Saved ROC and PR evaluation curves to {out_dir.resolve()}")


def plot_calibration_curves(
    models_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str | Path = "reports/figures",
) -> None:
    """Generates and saves calibration reliability diagrams.

    Args:
        models_predictions: dict mapping model_name -> (y_true, y_prob).
        output_dir: Directory where plot should be saved.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

    for name, (y_true, y_prob) in models_predictions.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_true, y_prob)
        plt.plot(prob_pred, prob_true, marker="o", lw=1.5, label=f"{name} (Brier = {brier:.4f})")

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives", fontsize=11)
    plt.title("Calibration Curves (Reliability Diagram)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curves.png")
    plt.close()

    logger.info(f"Saved calibration curves to {out_dir.resolve() / 'calibration_curves.png'}")
