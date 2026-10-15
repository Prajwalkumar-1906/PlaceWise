"""Schema validation for student placement records using Pydantic and DataFrame validators."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_BRANCHES = (
    "Computer Science",
    "Information Technology",
    "Electronics & Comm",
    "Mechanical",
    "Electrical",
    "Civil",
)
ALLOWED_TIERS = (1, 2, 3)
ALLOWED_GENDERS = ("Male", "Female")
ALLOWED_SALARY_BANDS = ("Low", "Mid", "High", "Not Placed")


class StudentRecord(BaseModel):
    """Pydantic model representing a single student application record."""

    cgpa: float = Field(
        ..., ge=0.0, le=10.0, description="Cumulative Grade Point Average (0-10 scale)"
    )
    tenth_pct: float = Field(..., ge=0.0, le=100.0, description="10th grade percentage")
    twelfth_pct: float = Field(..., ge=0.0, le=100.0, description="12th grade percentage")
    backlogs: int = Field(
        ..., ge=0, le=30, description="Number of active or cleared academic backlogs"
    )
    dsa_problems_solved: int = Field(
        ..., ge=0, le=5000, description="Number of DSA questions solved on competitive platforms"
    )
    projects: int = Field(..., ge=0, le=20, description="Number of technical projects completed")
    internships: int = Field(..., ge=0, le=10, description="Number of completed internships")
    certifications: int = Field(
        ..., ge=0, le=20, description="Number of recognized industry certifications"
    )
    hackathons: int = Field(
        ...,
        ge=0,
        le=20,
        description="Number of hackathons or technical competitions participated in",
    )
    communication_score: float = Field(
        ..., ge=0.0, le=100.0, description="Assessed communication/soft-skills score"
    )
    aptitude_score: float = Field(..., ge=0.0, le=100.0, description="General aptitude test score")
    branch: str = Field(..., description="Engineering discipline / academic branch")
    college_tier: int = Field(
        ..., ge=1, le=3, description="College institutional ranking tier (1, 2, or 3)"
    )
    gender: Literal["Male", "Female"] = Field(..., description="Self-reported gender")
    work_experience: int = Field(
        ..., ge=0, le=1, description="Prior professional work experience flag (0=No, 1=Yes)"
    )

    # Optional outcome fields
    placed: int | None = Field(
        default=None, ge=0, le=1, description="Placement status (0=Not Placed, 1=Placed)"
    )
    salary_band: str | None = Field(default=None, description="Salary compensation tier")

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, v: str) -> str:
        """Validate branch against approved categories."""
        if v not in ALLOWED_BRANCHES:
            raise ValueError(f"branch must be one of {ALLOWED_BRANCHES}, got '{v}'")
        return v

    @field_validator("college_tier")
    @classmethod
    def validate_tier(cls, v: int) -> int:
        """Validate college tier."""
        if v not in ALLOWED_TIERS:
            raise ValueError(f"college_tier must be in {ALLOWED_TIERS}, got {v}")
        return v

    @field_validator("salary_band")
    @classmethod
    def validate_salary(cls, v: str | None) -> str | None:
        """Validate salary band if present."""
        if v is not None and v not in ALLOWED_SALARY_BANDS:
            raise ValueError(f"salary_band must be one of {ALLOWED_SALARY_BANDS}, got '{v}'")
        return v


def validate_dataframe(
    df: pd.DataFrame,
    require_targets: bool = False,
) -> tuple[bool, list[str]]:
    """Performs high-throughput schema and range validation on a pandas DataFrame.

    Args:
        df: Input DataFrame containing student records.
        require_targets: Whether 'placed' and 'salary_band' columns must be present.

    Returns:
        tuple (is_valid, list_of_error_strings)
    """
    errors: list[str] = []

    expected_features = [
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

    missing_features = [col for col in expected_features if col not in df.columns]
    if missing_features:
        errors.append(f"Missing required feature columns: {missing_features}")
        return False, errors

    if require_targets:
        for target in ["placed", "salary_band"]:
            if target not in df.columns:
                errors.append(f"Missing required target column: '{target}'")

    if errors:
        return False, errors

    # Check for empty dataframe
    if len(df) == 0:
        errors.append("DataFrame is empty (0 rows).")
        return False, errors

    # Check nulls in required features
    null_counts = df[expected_features].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        errors.append(f"Columns with null values detected: {cols_with_nulls.to_dict()}")

    # Range and domain validations
    if (df["cgpa"] < 0.0).any() or (df["cgpa"] > 10.0).any():
        errors.append("Column 'cgpa' contains values outside [0.0, 10.0].")

    if (df["tenth_pct"] < 0.0).any() or (df["tenth_pct"] > 100.0).any():
        errors.append("Column 'tenth_pct' contains values outside [0.0, 100.0].")

    if (df["twelfth_pct"] < 0.0).any() or (df["twelfth_pct"] > 100.0).any():
        errors.append("Column 'twelfth_pct' contains values outside [0.0, 100.0].")

    if (df["backlogs"] < 0).any():
        errors.append("Column 'backlogs' contains negative values.")

    if (df["dsa_problems_solved"] < 0).any():
        errors.append("Column 'dsa_problems_solved' contains negative values.")

    if (df["communication_score"] < 0.0).any() or (df["communication_score"] > 100.0).any():
        errors.append("Column 'communication_score' contains values outside [0.0, 100.0].")

    if (df["aptitude_score"] < 0.0).any() or (df["aptitude_score"] > 100.0).any():
        errors.append("Column 'aptitude_score' contains values outside [0.0, 100.0].")

    invalid_branches = set(df["branch"].dropna().unique()) - set(ALLOWED_BRANCHES)
    if invalid_branches:
        errors.append(f"Column 'branch' contains unapproved categories: {invalid_branches}")

    invalid_tiers = set(df["college_tier"].dropna().unique()) - set(ALLOWED_TIERS)
    if invalid_tiers:
        errors.append(f"Column 'college_tier' contains unapproved tiers: {invalid_tiers}")

    invalid_genders = set(df["gender"].dropna().unique()) - set(ALLOWED_GENDERS)
    if invalid_genders:
        errors.append(f"Column 'gender' contains unapproved categories: {invalid_genders}")

    if not df["work_experience"].isin([0, 1]).all():
        errors.append("Column 'work_experience' contains values outside {0, 1}.")

    if "placed" in df.columns:
        if not df["placed"].isin([0, 1]).all():
            errors.append("Column 'placed' contains values outside {0, 1}.")

    if "salary_band" in df.columns:
        invalid_salary_bands = set(df["salary_band"].dropna().unique()) - set(ALLOWED_SALARY_BANDS)
        if invalid_salary_bands:
            errors.append(f"Column 'salary_band' contains invalid values: {invalid_salary_bands}")

    is_valid = len(errors) == 0
    if not is_valid:
        logger.warning(f"DataFrame validation failed with {len(errors)} issues: {errors}")
    return is_valid, errors
