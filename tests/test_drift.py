"""Unit tests for statistical drift detection, cohort simulation, and retraining trigger evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.monitoring.drift import (
    compute_feature_drift,
    compute_prediction_drift,
    evaluate_retraining_trigger,
)
from src.monitoring.simulation import generate_simulated_cohort


@pytest.fixture
def sample_reference_df():
    """Generates a synthetic reference dataset for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "cgpa": np.random.uniform(6.0, 9.5, size=n),
            "tenth_pct": np.random.uniform(60.0, 95.0, size=n),
            "twelfth_pct": np.random.uniform(60.0, 95.0, size=n),
            "dsa_problems_solved": np.random.randint(20, 300, size=n),
            "projects": np.random.randint(1, 5, size=n),
            "internships": np.random.randint(0, 3, size=n),
            "certifications": np.random.randint(0, 4, size=n),
            "hackathons": np.random.randint(0, 4, size=n),
            "communication_score": np.random.uniform(50.0, 95.0, size=n),
            "aptitude_score": np.random.uniform(50.0, 95.0, size=n),
            "backlogs": np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1], size=n),
            "branch": np.random.choice(
                ["Computer Science", "Information Technology", "Mechanical"], size=n
            ),
            "college_tier": np.random.choice([1, 2, 3], size=n),
            "gender": np.random.choice(["Male", "Female"], size=n),
            "work_experience": np.random.choice([0, 1], size=n),
            "placed": np.random.choice([0, 1], size=n),
        }
    )


def test_generate_simulated_cohort(sample_reference_df):
    """Verify cohort generator handles all valid scenarios and rejects unknown."""
    for scenario in ["normal", "cgpa_inflation", "hiring_slump", "curriculum_modernization"]:
        cohort = generate_simulated_cohort(sample_reference_df, scenario=scenario, sample_size=50)
        assert len(cohort) == 50
        assert "cgpa" in cohort.columns
        assert "dsa_problems_solved" in cohort.columns

    with pytest.raises(ValueError):
        generate_simulated_cohort(sample_reference_df, scenario="invalid_scenario")


def test_compute_feature_drift(sample_reference_df):
    """Verify feature drift detection on identical vs shifted data."""
    num_cols = ["cgpa", "tenth_pct", "dsa_problems_solved"]
    cat_cols = ["branch", "gender"]

    # 1. Identical data comparison -> 0 drift
    no_drift = compute_feature_drift(
        sample_reference_df, sample_reference_df.copy(), num_cols, cat_cols
    )
    assert no_drift["drifted_features_count"] == 0
    assert no_drift["dataset_drift_detected"] is False

    # 2. Shifted data (CGPA +3.0) -> drift detected
    shifted_df = sample_reference_df.copy()
    shifted_df["cgpa"] = shifted_df["cgpa"] + 3.0

    drift_res = compute_feature_drift(sample_reference_df, shifted_df, num_cols, cat_cols)
    assert drift_res["features"]["cgpa"]["drift_detected"] is True
    assert drift_res["features"]["cgpa"]["p_value"] < 0.05


def test_compute_prediction_drift(sample_reference_df):
    """Verify prediction drift on model output probabilities."""
    X = sample_reference_df[["cgpa", "dsa_problems_solved"]].values
    y = sample_reference_df["placed"].values
    clf = LogisticRegression()
    clf.fit(X, y)

    pred_drift = compute_prediction_drift(
        clf, sample_reference_df, sample_reference_df, ["cgpa", "dsa_problems_solved"]
    )
    assert "test_name" in pred_drift
    assert "p_value" in pred_drift
    assert pred_drift["p_value"] > 0.90  # Identical predictions


def test_evaluate_retraining_trigger():
    """Verify retraining decision logic based on drift share and output shifts."""
    # Healthy case
    healthy_drift = {"dataset_drift_detected": False, "drift_share_pct": 10.0}
    healthy_pred = {"p_value": 0.45}
    res_healthy = evaluate_retraining_trigger(healthy_drift, healthy_pred)
    assert res_healthy["retraining_recommended"] is False
    assert res_healthy["urgency_level"] == "NORMAL"

    # Drifting case
    drifting = {"dataset_drift_detected": True, "drift_share_pct": 45.0}
    drifting_pred = {"p_value": 0.001}
    res_retrain = evaluate_retraining_trigger(drifting, drifting_pred)
    assert res_retrain["retraining_recommended"] is True
    assert res_retrain["urgency_level"] in ["CRITICAL_RETRAIN", "ATTENTION_REQUIRED"]
