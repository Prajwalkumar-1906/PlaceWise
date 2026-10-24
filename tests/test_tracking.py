"""Tests for MLflow tracking, nested Optuna runs, and model registry promotion."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.tracking import (
    evaluate_and_promote_model,
    get_data_hash,
    get_git_commit_hash,
    log_optuna_trial_run,
    setup_mlflow,
)


def test_get_git_commit_hash() -> None:
    commit = get_git_commit_hash()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_get_data_hash_existing_file(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.csv"
    test_file.write_text("a,b,c\n1,2,3\n")
    data_hash = get_data_hash(test_file)
    assert isinstance(data_hash, str)
    assert len(data_hash) == 16
    assert data_hash != "unavailable"


def test_get_data_hash_nonexistent_file() -> None:
    data_hash = get_data_hash("nonexistent_path/fake_data.csv")
    assert data_hash == "unavailable"


def test_setup_mlflow(tmp_path: Path) -> None:
    db_path = tmp_path / "test_mlruns.db"
    config = {
        "mlflow": {
            "tracking_uri": f"sqlite:///{db_path}",
            "experiment_name": "Test-Experiment",
        }
    }
    exp_id = setup_mlflow(config)
    assert isinstance(exp_id, str)


def test_log_optuna_trial_run_no_active_run() -> None:
    # When no active MLflow run is present, it should gracefully skip without raising
    log_optuna_trial_run(1, {"max_depth": 5}, 0.88)


@patch("src.models.tracking.mlflow")
@patch("src.models.tracking.MlflowClient")
def test_evaluate_and_promote_model_production(
    mock_client_cls: MagicMock, mock_mlflow: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_version = MagicMock()
    mock_version.version = "1"
    mock_mlflow.register_model.return_value = mock_version

    metrics = {
        "roc_auc": 0.89,
        "f1": 0.92,
        "brier_score": 0.07,
    }
    thresholds = {
        "min_roc_auc": 0.85,
        "min_f1": 0.88,
        "max_brier_score": 0.12,
    }

    result = evaluate_and_promote_model(
        model_name="TestModel",
        run_id="run_123",
        metrics=metrics,
        thresholds=thresholds,
    )

    assert result["eligible"] is True
    assert result["stage"] == "Production"
    assert result["version"] == "1"
    mock_client.set_registered_model_alias.assert_called_with(
        name="TestModel", alias="production", version="1"
    )


@patch("src.models.tracking.mlflow")
@patch("src.models.tracking.MlflowClient")
def test_evaluate_and_promote_model_staging(
    mock_client_cls: MagicMock, mock_mlflow: MagicMock
) -> None:
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_version = MagicMock()
    mock_version.version = "2"
    mock_mlflow.register_model.return_value = mock_version

    # Low AUC below threshold
    metrics = {
        "roc_auc": 0.80,
        "f1": 0.82,
        "brier_score": 0.15,
    }
    thresholds = {
        "min_roc_auc": 0.85,
        "min_f1": 0.88,
        "max_brier_score": 0.12,
    }

    result = evaluate_and_promote_model(
        model_name="TestModel",
        run_id="run_456",
        metrics=metrics,
        thresholds=thresholds,
    )

    assert result["eligible"] is False
    assert result["stage"] == "Staging"
    assert result["version"] == "2"
    mock_client.set_registered_model_alias.assert_called_with(
        name="TestModel", alias="staging", version="2"
    )
