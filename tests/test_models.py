"""Unit tests for model evaluation, registry serialization, and inference."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import (
    evaluate_binary_classifier,
    evaluate_multiclass_classifier,
    plot_calibration_curves,
    plot_roc_pr_curves,
)
from src.models.predict import PlacementPredictor
from src.models.registry import ModelRegistry
from src.models.train import train_models


class DummyPlacementModel(BaseEstimator, ClassifierMixin):
    """Top-level dummy model class that is picklable by joblib."""

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        cgpa_vals = X["cgpa"].values if isinstance(X, pd.DataFrame) else X[:, 0]
        probs = np.where(cgpa_vals >= 7.0, 0.85, 0.25)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return np.where(probs >= 0.50, 1, 0)


def test_evaluate_binary_classifier():
    """Verify binary classification evaluation metrics computation."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])

    metrics = evaluate_binary_classifier(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["brier_score"] < 0.05
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["tn"] == 2


def test_evaluate_multiclass_classifier():
    """Verify multiclass evaluation metrics computation."""
    y_true = ["Low", "Mid", "High", "Mid"]
    y_pred = ["Low", "Mid", "High", "Low"]

    metrics = evaluate_multiclass_classifier(y_true, y_pred)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert metrics["accuracy"] == 0.75


def test_plot_curves(tmp_path):
    """Verify ROC, PR, and calibration curves save successfully."""
    predictions = {
        "Model A": (np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2])),
        "Model B": (np.array([1, 0, 1, 0]), np.array([0.7, 0.3, 0.6, 0.4])),
    }
    plot_roc_pr_curves(predictions, output_dir=tmp_path)
    plot_calibration_curves(predictions, output_dir=tmp_path)

    assert (tmp_path / "roc_curves_comparison.png").exists()
    assert (tmp_path / "pr_curves_comparison.png").exists()
    assert (tmp_path / "calibration_curves.png").exists()


def test_model_registry_save_and_load(tmp_path):
    """Verify pipeline and metadata saving and loading."""
    registry = ModelRegistry(models_dir=tmp_path)

    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)

    meta = {"version": "0.1.0", "metrics": {"accuracy": 1.0}}
    model_path, meta_path = registry.save_model(pipe, "test_model", meta)

    assert model_path.exists()
    assert meta_path.exists()

    loaded_pipe, loaded_meta = registry.load_model("test_model")
    preds = loaded_pipe.predict(X)
    assert np.array_equal(preds, y)
    assert loaded_meta["version"] == "0.1.0"


def test_placement_predictor_with_dummy(tmp_path):
    """Verify PlacementPredictor executes single and batch inference accurately."""
    registry = ModelRegistry(models_dir=tmp_path)
    pipe = Pipeline([("clf", DummyPlacementModel())])
    registry.save_model(pipe, "best_placement_model", {"model_type": "Dummy"})

    predictor = PlacementPredictor(
        placement_model_name="best_placement_model",
        salary_model_name="nonexistent_salary",
        models_dir=tmp_path,
    )

    sample_student = {
        "cgpa": 8.5,
        "tenth_pct": 80.0,
        "twelfth_pct": 78.0,
        "backlogs": 0,
        "dsa_problems_solved": 300,
        "projects": 3,
        "internships": 1,
        "certifications": 1,
        "hackathons": 1,
        "communication_score": 75.0,
        "aptitude_score": 70.0,
        "branch": "Computer Science",
        "college_tier": 1,
        "gender": "Female",
        "work_experience": 1,
    }

    result = predictor.predict_student(sample_student)
    assert result["placement_prediction"] == 1
    assert result["placement_probability"] == 0.85
    assert result["risk_label"] == "High Placement Likelihood"

    # Test batch prediction
    df_batch = pd.DataFrame([sample_student, {**sample_student, "cgpa": 5.5}])
    batch_res = predictor.predict_batch(df_batch)
    assert len(batch_res) == 2
    assert "predicted_placement_probability" in batch_res.columns
    assert "predicted_placement" in batch_res.columns


def test_train_models_execution():
    """Verify that train_models executes end-to-end when tune=False."""
    results = train_models(config_path="config/config.yaml", tune=False)
    assert "test" in results
    assert "xgboost" in results["test"]
    assert "logistic_regression" in results["test"]
    assert "random_forest" in results["test"]
    assert results["test"]["xgboost"]["roc_auc"] > 0.70
