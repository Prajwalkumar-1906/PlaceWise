"""MLflow experiment tracking, nested trial logging, and model registry promotion."""

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_git_commit_hash() -> str:
    """Retrieves the current git commit SHA hash, or 'unavailable' if not in git."""
    try:
        output = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
            )
            .decode("utf-8")
            .strip()
        )
        return output
    except Exception:
        return "unavailable"


def get_data_hash(filepath: str | Path) -> str:
    """Computes a SHA-256 hash prefix for the specified data file.

    Args:
        filepath: Path to dataset file.

    Returns:
        Hexadecimal SHA-256 string prefix (16 chars), or 'unavailable'.
    """
    path = Path(filepath)
    if not path.exists():
        return "unavailable"
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    except Exception as exc:
        logger.warning(f"Could not compute data hash for {filepath}: {exc}")
        return "unavailable"


def setup_mlflow(config: dict[str, Any]) -> str:
    """Configures MLflow tracking URI, local backend store, and active experiment.

    Args:
        config: Loaded project configuration dictionary.

    Returns:
        Active MLflow experiment ID.
    """
    mlflow_cfg = config.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlruns.db")
    experiment_name = mlflow_cfg.get("experiment_name", "PlaceWise-Placement-Prediction")

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    logger.info(
        f"MLflow configured: tracking_uri='{tracking_uri}', experiment='{experiment_name}' (id={experiment.experiment_id})"
    )
    return str(experiment.experiment_id)


def log_optuna_trial_run(trial_number: int, params: dict[str, Any], cv_score: float) -> None:
    """Logs an individual Optuna trial as a nested MLflow run under the parent run.

    Args:
        trial_number: Integer trial index.
        params: Suggested hyperparameters for this trial.
        cv_score: Mean cross-validation ROC-AUC achieved by this trial.
    """
    try:
        if mlflow.active_run() is not None:
            with mlflow.start_run(run_name=f"optuna_trial_{trial_number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_roc_auc", float(cv_score))
                mlflow.set_tag("trial_number", str(trial_number))
                mlflow.set_tag("pipeline", "optuna_tuning")
    except Exception as exc:
        logger.warning(f"Could not log Optuna trial {trial_number} to MLflow: {exc}")


def evaluate_and_promote_model(
    model_name: str,
    run_id: str,
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Registers the model and evaluates promotion rules for Staging vs. Production.

    Promotion requires meeting or exceeding all documented metric thresholds:
      - ROC-AUC >= min_roc_auc (default 0.85)
      - F1 >= min_f1 (default 0.88)
      - Brier Score <= max_brier_score (default 0.12)

    Args:
        model_name: Name of registered model in MLflow Model Registry.
        run_id: Parent MLflow run ID containing the model artifact.
        metrics: Dictionary of test evaluation metrics.
        thresholds: Dictionary of threshold criteria.

    Returns:
        dict containing registration status, assigned stage/alias, and rationale.
    """
    client = MlflowClient()
    min_roc_auc = thresholds.get("min_roc_auc", 0.85)
    min_f1 = thresholds.get("min_f1", 0.88)
    max_brier = thresholds.get("max_brier_score", 0.12)

    roc_auc = float(metrics.get("roc_auc", 0.0))
    f1 = float(metrics.get("f1", 0.0))
    brier = float(metrics.get("brier_score", 1.0))

    passed_auc = roc_auc >= min_roc_auc
    passed_f1 = f1 >= min_f1
    passed_brier = brier <= max_brier
    eligible_for_production = passed_auc and passed_f1 and passed_brier

    target_stage = "Production" if eligible_for_production else "Staging"
    target_alias = "production" if eligible_for_production else "staging"

    reason = (
        f"Promoted to {target_stage}: ROC-AUC={roc_auc:.4f} (min {min_roc_auc}), "
        f"F1={f1:.4f} (min {min_f1}), Brier={brier:.4f} (max {max_brier})"
        if eligible_for_production
        else f"Placed in {target_stage}: Thresholds not fully satisfied (AUC={roc_auc:.4f}, F1={f1:.4f}, Brier={brier:.4f})"
    )

    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Registering model '{model_name}' from URI: {model_uri}")
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Set tags and aliases
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="stage",
        value=target_stage,
    )
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="promotion_reason",
        value=reason,
    )
    client.set_registered_model_alias(
        name=model_name,
        alias=target_alias,
        version=model_version.version,
    )

    # Backward compatibility with MLflow stages API if available
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=target_stage,
            archive_existing_versions=(target_stage == "Production"),
        )
    except Exception as exc:
        logger.debug(f"Stage transition method note: {exc}")

    logger.info(f"Model '{model_name}' version {model_version.version} assigned to {target_stage}.")
    return {
        "model_name": model_name,
        "version": model_version.version,
        "stage": target_stage,
        "alias": target_alias,
        "eligible": eligible_for_production,
        "reason": reason,
        "metrics": {"roc_auc": roc_auc, "f1": f1, "brier_score": brier},
    }
