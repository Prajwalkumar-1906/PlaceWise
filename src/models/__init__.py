"""Model training, tuning, evaluation, inference, and registry."""

from src.models.calibration import (
    ConformalPredictor,
    ProbabilityCalibrator,
    classify_hitl_confidence,
    compute_ece,
)
from src.models.evaluate import evaluate_binary_classifier, evaluate_multiclass_classifier
from src.models.predict import PlacementPredictor
from src.models.registry import ModelRegistry
from src.models.tracking import evaluate_and_promote_model, setup_mlflow

__all__ = [
    "ConformalPredictor",
    "ModelRegistry",
    "PlacementPredictor",
    "ProbabilityCalibrator",
    "classify_hitl_confidence",
    "compute_ece",
    "evaluate_and_promote_model",
    "evaluate_binary_classifier",
    "evaluate_multiclass_classifier",
    "setup_mlflow",
]
