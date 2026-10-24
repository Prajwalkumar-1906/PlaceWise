"""Unit tests for Pandera data schema validation, data quality reporting, and manifest generation."""

import pandas as pd
import pytest

from src.data.manifest import generate_data_manifest
from src.data.schemas import (
    generate_data_quality_report,
    validate_dataset_with_pandera,
)


@pytest.fixture
def valid_dataset():
    """Generates a valid test DataFrame adhering to Pandera schema."""
    return pd.DataFrame(
        {
            "cgpa": [8.5, 7.2, 9.1],
            "tenth_pct": [85.0, 78.0, 92.0],
            "twelfth_pct": [82.0, 75.0, 89.0],
            "backlogs": [0, 1, 0],
            "dsa_problems_solved": [200, 50, 400],
            "projects": [3, 1, 4],
            "internships": [1, 0, 2],
            "certifications": [2, 1, 3],
            "hackathons": [1, 0, 2],
            "communication_score": [80.0, 65.0, 90.0],
            "aptitude_score": [85.0, 70.0, 92.0],
            "branch": ["Computer Science", "Mechanical", "Information Technology"],
            "college_tier": [1, 2, 1],
            "gender": ["Female", "Male", "Male"],
            "work_experience": [1, 0, 1],
            "placed": [1, 0, 1],
            "salary_band": ["High", "Not Placed", "High"],
        }
    )


def test_validate_dataset_with_pandera_valid(valid_dataset):
    """Verify Pandera passes a compliant dataset."""
    is_valid, errors = validate_dataset_with_pandera(valid_dataset, require_targets=True)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_dataset_with_pandera_invalid_bounds(valid_dataset):
    """Verify Pandera catches out-of-range values."""
    bad_df = valid_dataset.copy()
    bad_df.loc[0, "cgpa"] = 11.5  # Out of range [0, 10]
    bad_df.loc[1, "college_tier"] = 5  # Out of allowed [1, 2, 3]

    is_valid, errors = validate_dataset_with_pandera(bad_df, require_targets=True)
    assert is_valid is False
    assert len(errors) > 0


def test_generate_data_quality_report(valid_dataset, tmp_path):
    """Verify Markdown data quality report generation."""
    report_file = tmp_path / "data_quality_report.md"
    generated_path = generate_data_quality_report(valid_dataset, output_path=report_file)

    assert generated_path.exists()
    content = generated_path.read_text(encoding="utf-8")
    assert "Data Quality and Pandera Schema Conformity Report" in content
    assert "CGPA" in content
    assert "100% Schema Compliant" in content


def test_generate_data_manifest(tmp_path):
    """Verify data manifest generation and SHA-256 calculation."""
    # Create dummy csv
    csv_path = tmp_path / "synthetic" / "synthetic_placement_data.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("cgpa,placed\n8.5,1\n7.2,0\n", encoding="utf-8")

    manifest_file = tmp_path / "manifest.json"
    manifest = generate_data_manifest(data_dir=tmp_path, output_path=manifest_file)

    assert manifest_file.exists()
    assert "datasets" in manifest
    assert "synthetic_placement_data.csv" in manifest["datasets"]
    assert len(manifest["datasets"]["synthetic_placement_data.csv"]["sha256"]) == 64
