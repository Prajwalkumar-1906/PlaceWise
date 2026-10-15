"""Unit tests for Pydantic schema and DataFrame validation."""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.validation import StudentRecord, validate_dataframe


def test_valid_student_record():
    """Verify that a valid dictionary parses into a StudentRecord."""
    valid_data = {
        "cgpa": 8.5,
        "tenth_pct": 85.0,
        "twelfth_pct": 82.0,
        "backlogs": 0,
        "dsa_problems_solved": 250,
        "projects": 3,
        "internships": 1,
        "certifications": 2,
        "hackathons": 1,
        "communication_score": 75.0,
        "aptitude_score": 80.0,
        "branch": "Computer Science",
        "college_tier": 1,
        "gender": "Male",
        "work_experience": 0,
        "placed": 1,
        "salary_band": "High",
    }
    record = StudentRecord(**valid_data)
    assert record.cgpa == 8.5
    assert record.branch == "Computer Science"
    assert record.placed == 1


def test_invalid_cgpa_raises():
    """Verify that CGPA out of range [0, 10] raises ValidationError."""
    invalid_data = {
        "cgpa": 11.2,  # Invalid
        "tenth_pct": 85.0,
        "twelfth_pct": 82.0,
        "backlogs": 0,
        "dsa_problems_solved": 250,
        "projects": 3,
        "internships": 1,
        "certifications": 2,
        "hackathons": 1,
        "communication_score": 75.0,
        "aptitude_score": 80.0,
        "branch": "Computer Science",
        "college_tier": 1,
        "gender": "Female",
        "work_experience": 0,
    }
    with pytest.raises(ValidationError):
        StudentRecord(**invalid_data)


def test_invalid_tier_and_branch_raises():
    """Verify that invalid college tier and branch raise validation errors."""
    data = {
        "cgpa": 8.0,
        "tenth_pct": 80.0,
        "twelfth_pct": 80.0,
        "backlogs": 0,
        "dsa_problems_solved": 100,
        "projects": 2,
        "internships": 1,
        "certifications": 1,
        "hackathons": 0,
        "communication_score": 70.0,
        "aptitude_score": 70.0,
        "branch": "Unknown Branch",  # Invalid
        "college_tier": 5,  # Invalid
        "gender": "Female",
        "work_experience": 0,
    }
    with pytest.raises(ValidationError):
        StudentRecord(**data)


def test_validate_dataframe_success():
    """Verify that a properly constructed dataframe passes validation."""
    data = {
        "cgpa": [7.5, 8.2],
        "tenth_pct": [78.0, 85.0],
        "twelfth_pct": [75.0, 80.0],
        "backlogs": [0, 1],
        "dsa_problems_solved": [150, 300],
        "projects": [2, 3],
        "internships": [1, 2],
        "certifications": [1, 2],
        "hackathons": [0, 1],
        "communication_score": [68.0, 75.0],
        "aptitude_score": [70.0, 82.0],
        "branch": ["Computer Science", "Information Technology"],
        "college_tier": [2, 1],
        "gender": ["Male", "Female"],
        "work_experience": [0, 1],
        "placed": [1, 1],
        "salary_band": ["Mid", "High"],
    }
    df = pd.DataFrame(data)
    is_valid, errors = validate_dataframe(df, require_targets=True)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_dataframe_missing_column():
    """Verify that missing column is detected."""
    df = pd.DataFrame({"cgpa": [7.5], "branch": ["Computer Science"]})
    is_valid, errors = validate_dataframe(df, require_targets=False)
    assert is_valid is False
    assert any("Missing required feature columns" in err for err in errors)


def test_validate_dataframe_out_of_bounds():
    """Verify that out-of-bounds values trigger validation failure."""
    data = {
        "cgpa": [-1.0],  # Out of bounds
        "tenth_pct": [105.0],  # Out of bounds
        "twelfth_pct": [75.0],
        "backlogs": [-2],  # Out of bounds
        "dsa_problems_solved": [100],
        "projects": [1],
        "internships": [0],
        "certifications": [0],
        "hackathons": [0],
        "communication_score": [50.0],
        "aptitude_score": [50.0],
        "branch": ["Computer Science"],
        "college_tier": [1],
        "gender": ["Male"],
        "work_experience": [0],
    }
    df = pd.DataFrame(data)
    is_valid, errors = validate_dataframe(df, require_targets=False)
    assert is_valid is False
    assert len(errors) >= 3
