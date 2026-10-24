"""End-to-end integration tests for PlaceWise pipeline components."""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.audit import AUDIT_LOG_FILE, compute_input_hash
from api.main import app
from src.data.schemas import PLACEMENT_SCHEMA
from src.explainability.recourse import IMMUTABLE_FEATURES, RecourseEngine
from src.explainability.shap_local import explain_student_prediction
from src.models.predict import PlacementPredictor
from src.monitoring.drift import (
    compute_feature_drift,
    compute_prediction_drift,
    evaluate_retraining_trigger,
)
from src.monitoring.simulation import generate_simulated_cohort


@pytest.fixture(scope="module")
def sample_applicant():
    return {
        "cgpa": 7.2,
        "tenth_pct": 78.0,
        "twelfth_pct": 74.0,
        "branch": "Mechanical",
        "college_tier": 2,
        "gender": "Female",
        "work_experience": 0,
        "backlogs": 1,
        "dsa_problems_solved": 80,
        "projects": 1,
        "internships": 0,
        "certifications": 1,
        "hackathons": 0,
        "communication_score": 65.0,
        "aptitude_score": 68.0,
    }


def test_end_to_end_data_validation():
    """Verify processed dataset adheres to Pandera schema and manifest exists."""
    manifest_path = Path("reports/data_manifest.json")
    assert manifest_path.exists(), "Data manifest must exist"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert "synthetic_placement_data.csv" in manifest.get("datasets", {})

    df_clean = pd.read_csv("data/synthetic/synthetic_placement_data.csv")
    validated = PLACEMENT_SCHEMA.validate(df_clean)
    assert not validated.empty
    assert len(validated) >= 1000


def test_end_to_end_model_prediction_and_conformal(sample_applicant):
    """Verify model loading, calibrated probability, confidence band, and HITL flag."""
    predictor = PlacementPredictor()

    pred = predictor.predict_student(sample_applicant)
    assert "placement_prediction" in pred
    assert 0.0 <= pred["placement_probability"] <= 1.0
    assert "risk_label" in pred
    assert "confidence_band" in pred
    assert "needs_human_review" in pred
    assert "hitl_recommendation" in pred
    assert "hitl_status" in pred


def test_end_to_end_recourse_and_explanation(sample_applicant):
    """Verify SHAP explanation and actionable recourse maintain strict immutability."""
    predictor = PlacementPredictor()
    model = predictor.placement_pipeline

    numerical_cols = [
        "cgpa",
        "tenth_pct",
        "twelfth_pct",
        "dsa_problems_solved",
        "projects",
        "internships",
        "certifications",
        "hackathons",
        "communication_score",
        "aptitude_score",
        "backlogs",
        "college_tier",
        "work_experience",
    ]
    categorical_cols = ["gender", "branch"]

    explanation = explain_student_prediction(
        pipeline=model,
        student_data=sample_applicant,
        numerical_features=numerical_cols,
        categorical_features=categorical_cols,
    )
    assert "feature_impacts" in explanation
    assert len(explanation["feature_impacts"]) > 0

    engine = RecourseEngine(pipeline=model)
    recourse = engine.generate_recourse_pathways(sample_applicant, target_probability=0.70)

    assert "pathways" in recourse
    assert len(recourse["pathways"]) == 3

    for pathway in recourse["pathways"]:
        interventions = pathway.get("interventions", [])
        for item in interventions:
            feat = item.get("feature")
            assert (
                feat not in IMMUTABLE_FEATURES
            ), f"Immutable feature '{feat}' was modified in recourse!"


def test_end_to_end_api_endpoints_and_audit(sample_applicant):
    """Verify full FastAPI invocation, response contracts, and audit logging."""
    with TestClient(app) as client:
        # 1. Health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        # 2. Predict
        pred_resp = client.post("/v1/predict", json=sample_applicant)
        assert pred_resp.status_code == 200
        pred_data = pred_resp.json()
        assert "placement_probability" in pred_data
        assert "confidence_band" in pred_data
        assert "needs_human_review" in pred_data

        # 3. Explain
        explain_resp = client.post("/v1/explain", json=sample_applicant)
        assert explain_resp.status_code == 200
        explain_data = explain_resp.json()
        assert "top_positive_drivers" in explain_data
        assert "top_negative_drivers" in explain_data

        # 4. Recourse
        recourse_resp = client.post(
            "/v1/recourse", json={"student": sample_applicant, "target_probability": 0.70}
        )
        assert recourse_resp.status_code == 200
        recourse_data = recourse_resp.json()
        assert len(recourse_data["pathways"]) == 3

        # 5. Verify audit log entry
        assert AUDIT_LOG_FILE.exists()
        expected_hash = compute_input_hash(sample_applicant)
        with open(AUDIT_LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()
            assert any(expected_hash in line for line in lines)


def test_end_to_end_monitoring_and_retraining_trigger():
    """Verify shift simulation, drift statistical tests, and automated retraining evaluation."""
    reference_df = pd.read_csv("data/synthetic/synthetic_placement_data.csv")
    current_df = generate_simulated_cohort(reference_df, scenario="cgpa_inflation", sample_size=300)

    drift_results = compute_feature_drift(
        reference_df=reference_df,
        current_df=current_df,
        numerical_features=["cgpa", "aptitude_score", "tenth_pct", "twelfth_pct"],
        categorical_features=["college_tier"],
    )

    predictor = PlacementPredictor()
    all_features = [
        "cgpa",
        "tenth_pct",
        "twelfth_pct",
        "backlogs",
        "dsa_problems_solved",
        "projects",
        "internships",
        "certifications",
        "hackathons",
        "communication_score",
        "aptitude_score",
        "branch",
        "college_tier",
        "gender",
        "work_experience",
    ]
    pred_drift = compute_prediction_drift(
        pipeline=predictor.placement_pipeline,
        reference_df=reference_df,
        current_df=current_df,
        feature_cols=all_features,
    )

    decision = evaluate_retraining_trigger(drift_results, pred_drift)
    assert "retraining_recommended" in decision
    assert "urgency_level" in decision
    assert "triggered_reasons" in decision
    assert decision["retraining_recommended"] is True
