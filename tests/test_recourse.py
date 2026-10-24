"""Unit tests for actionable algorithmic recourse engine and immutability enforcement."""

import pytest

from src.explainability.recourse import (
    ACTIONABLE_CONFIG,
    IMMUTABLE_FEATURES,
    RECOURSE_DISCLAIMER,
    RecourseEngine,
)
from src.models.registry import ModelRegistry


@pytest.fixture
def fitted_pipeline():
    """Loads pre-trained placement pipeline from model registry."""
    registry = ModelRegistry(models_dir="models")
    pipeline, _ = registry.load_model("best_placement_model")
    return pipeline


@pytest.fixture
def borderline_student():
    """Applicant with moderate academic record and low technical metrics."""
    return {
        "cgpa": 6.8,
        "tenth_pct": 72.0,
        "twelfth_pct": 70.0,
        "branch": "Computer Science",
        "college_tier": 2,
        "gender": "Female",
        "work_experience": 0,
        "backlogs": 1,
        "dsa_problems_solved": 40,
        "projects": 1,
        "internships": 0,
        "certifications": 0,
        "hackathons": 0,
        "communication_score": 60.0,
        "aptitude_score": 58.0,
    }


def test_immutable_features_definitions():
    """Verify that demographic and prior schooling attributes are strictly declared immutable."""
    assert "gender" in IMMUTABLE_FEATURES
    assert "college_tier" in IMMUTABLE_FEATURES
    assert "branch" in IMMUTABLE_FEATURES
    assert "tenth_pct" in IMMUTABLE_FEATURES
    assert "twelfth_pct" in IMMUTABLE_FEATURES


def test_recourse_already_satisfied(fitted_pipeline):
    """Verify recourse engine handles applicants who already exceed the target probability."""
    strong_student = {
        "cgpa": 9.5,
        "tenth_pct": 95.0,
        "twelfth_pct": 92.0,
        "branch": "Computer Science",
        "college_tier": 1,
        "gender": "Male",
        "work_experience": 1,
        "backlogs": 0,
        "dsa_problems_solved": 450,
        "projects": 5,
        "internships": 2,
        "certifications": 3,
        "hackathons": 2,
        "communication_score": 90.0,
        "aptitude_score": 92.0,
    }
    engine = RecourseEngine(pipeline=fitted_pipeline)
    recourse = engine.generate_recourse_pathways(strong_student, target_probability=0.65)

    assert recourse["target_already_satisfied"] is True
    assert len(recourse["pathways"]) == 0
    assert "already satisfies" in recourse["message"]
    assert recourse["disclaimer"] == RECOURSE_DISCLAIMER


def test_recourse_generates_valid_pathways(fitted_pipeline, borderline_student):
    """Verify that recourse generates distinct actionable pathways without touching immutable features."""
    engine = RecourseEngine(pipeline=fitted_pipeline)
    recourse = engine.generate_recourse_pathways(borderline_student, target_probability=0.65)

    assert recourse["target_already_satisfied"] is False
    assert len(recourse["pathways"]) > 0
    assert recourse["disclaimer"] == RECOURSE_DISCLAIMER

    for p in recourse["pathways"]:
        assert "pathway_name" in p
        assert "interventions" in p
        assert "feasibility_rating" in p
        assert p["achieved_probability"] > p["baseline_probability"]

        # Immutability check: NO immutable feature is ever in interventions
        for inv in p["interventions"]:
            feat = inv["feature"]
            assert feat not in IMMUTABLE_FEATURES
            assert feat in ACTIONABLE_CONFIG

            # Bound checks
            if feat == "backlogs":
                assert inv["recommended_value"] <= inv["current_value"]
            else:
                assert inv["recommended_value"] >= inv["current_value"]
                assert inv["recommended_value"] <= ACTIONABLE_CONFIG[feat]["max"]
