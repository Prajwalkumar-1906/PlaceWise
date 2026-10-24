"""Comprehensive integration tests for PlaceWise FastAPI microservice endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient fixture for FastAPI endpoints."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_student_payload():
    """Sample valid student profile matching StudentPredictionRequest."""
    return {
        "cgpa": 8.4,
        "tenth_pct": 85.0,
        "twelfth_pct": 82.0,
        "branch": "Computer Science",
        "college_tier": 1,
        "gender": "Female",
        "work_experience": 1,
        "backlogs": 0,
        "dsa_problems_solved": 250,
        "projects": 3,
        "internships": 1,
        "certifications": 2,
        "hackathons": 1,
        "communication_score": 80.0,
        "aptitude_score": 82.0,
    }


def test_root_endpoint(client):
    """Verify root / endpoint returns service info and docs link."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "PlaceWise Placement Prediction API"
    assert "/docs" in data["docs"]


def test_health_endpoint(client):
    """Verify healthcheck endpoint returns healthy status and model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data


def test_predict_single_endpoint(client, valid_student_payload):
    """Verify single applicant inference endpoint."""
    response = client.post("/v1/predict", json=valid_student_payload)
    assert response.status_code == 200
    data = response.json()
    assert "placement_probability" in data
    assert 0.0 <= data["placement_probability"] <= 1.0
    assert data["placement_prediction"] in [0, 1]
    assert "confidence_band" in data
    assert "needs_human_review" in data
    assert "salary_band" in data
    assert "input_hash" in data
    assert "latency_ms" in data


def test_predict_validation_error(client, valid_student_payload):
    """Verify that out-of-range or invalid attributes trigger HTTP 422."""
    bad_payload = dict(valid_student_payload)
    bad_payload["cgpa"] = 15.0  # Invalid CGPA > 10.0

    response = client.post("/v1/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


def test_predict_batch_endpoint(client, valid_student_payload):
    """Verify batch inference endpoint."""
    batch_payload = {"students": [valid_student_payload, valid_student_payload]}
    response = client.post("/v1/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert "placed_count" in data
    assert "unplaced_count" in data
    assert "human_review_required_count" in data
    assert len(data["predictions"]) == 2


def test_explain_endpoint(client, valid_student_payload):
    """Verify local SHAP feature contribution explanation endpoint."""
    response = client.post("/v1/explain", json=valid_student_payload)
    assert response.status_code == 200
    data = response.json()
    assert "placement_probability" in data
    assert "top_positive_drivers" in data
    assert "top_negative_drivers" in data
    assert "all_contributions" in data
    assert len(data["all_contributions"]) > 0


def test_recourse_endpoint(client, valid_student_payload):
    """Verify actionable developmental recourse endpoint."""
    recourse_req = {
        "student": valid_student_payload,
        "target_probability": 0.70,
    }
    response = client.post("/v1/recourse", json=recourse_req)
    assert response.status_code == 200
    data = response.json()
    assert "target_already_satisfied" in data
    assert "disclaimer" in data
    assert "DISCLAIMER" in data["disclaimer"]


def test_model_info_endpoint(client):
    """Verify active model metadata and benchmark retrieval."""
    response = client.get("/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "test_performance" in data
    assert "features" in data
    assert data["features"]["sensitive_attributes"] == ["gender", "college_tier"]
