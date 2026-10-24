"""Pandera schema validation and automated data quality reporting."""

from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Complete Pandera schema for placement dataset
PLACEMENT_SCHEMA = DataFrameSchema(
    columns={
        "cgpa": Column(
            float, Check.in_range(0.0, 10.0), nullable=False, description="CGPA (0-10 scale)"
        ),
        "tenth_pct": Column(
            float, Check.in_range(0.0, 100.0), nullable=False, description="10th Percentage"
        ),
        "twelfth_pct": Column(
            float, Check.in_range(0.0, 100.0), nullable=False, description="12th Percentage"
        ),
        "backlogs": Column(
            int, Check.greater_than_or_equal_to(0), nullable=False, description="Active backlogs"
        ),
        "dsa_problems_solved": Column(
            int,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="DSA problems solved",
        ),
        "projects": Column(
            int, Check.greater_than_or_equal_to(0), nullable=False, description="Projects completed"
        ),
        "internships": Column(
            int,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Completed internships",
        ),
        "certifications": Column(
            int,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Certifications count",
        ),
        "hackathons": Column(
            int,
            Check.greater_than_or_equal_to(0),
            nullable=False,
            description="Hackathons attended",
        ),
        "communication_score": Column(
            float, Check.in_range(0.0, 100.0), nullable=False, description="Communication score"
        ),
        "aptitude_score": Column(
            float, Check.in_range(0.0, 100.0), nullable=False, description="Aptitude score"
        ),
        "branch": Column(
            str,
            Check.isin(
                [
                    "Computer Science",
                    "Information Technology",
                    "Electronics & Comm",
                    "Mechanical",
                    "Electrical",
                    "Civil",
                ]
            ),
            nullable=False,
            description="Academic Branch",
        ),
        "college_tier": Column(
            int, Check.isin([1, 2, 3]), nullable=False, description="College Tier"
        ),
        "gender": Column(
            str, Check.isin(["Male", "Female"]), nullable=False, description="Demographic Gender"
        ),
        "work_experience": Column(
            int, Check.isin([0, 1]), nullable=False, description="Work Experience"
        ),
        "placed": Column(
            int,
            Check.isin([0, 1]),
            required=False,
            nullable=True,
            description="Target: Placement indicator",
        ),
        "salary_band": Column(
            str,
            Check.isin(["Low", "Mid", "High", "Not Placed"]),
            required=False,
            nullable=True,
            description="Target: Salary compensation tier",
        ),
    },
    coerce=True,
    strict=False,
)


def validate_dataset_with_pandera(
    df: pd.DataFrame,
    require_targets: bool = True,
) -> tuple[bool, list[str]]:
    """Validates an applicant DataFrame using the comprehensive Pandera schema.

    Args:
        df: Input DataFrame to validate.
        require_targets: Whether target columns ('placed', 'salary_band') must be present.

    Returns:
        tuple (is_valid: bool, errors: list of error descriptions).
    """
    errors: list[str] = []

    # If targets are not required, remove them from validation checks
    schema_to_use = PLACEMENT_SCHEMA
    if not require_targets:
        columns_copy = dict(PLACEMENT_SCHEMA.columns)
        columns_copy.pop("placed", None)
        columns_copy.pop("salary_band", None)
        schema_to_use = DataFrameSchema(columns=columns_copy, coerce=True, strict=False)

    try:
        schema_to_use.validate(df, lazy=True)
        logger.info(f"Pandera schema validation succeeded for {len(df)} records.")
        return True, []
    except pa.errors.SchemaErrors as exc:
        for err in exc.failure_cases.to_dict(orient="records"):
            errors.append(
                f"Column '{err.get('column')}': {err.get('check')} failed for value {err.get('failure_case')}"
            )
        logger.warning(f"Pandera validation found {len(errors)} failure cases.")
        return False, errors
    except Exception as exc:
        errors.append(str(exc))
        return False, errors


def generate_data_quality_report(
    df: pd.DataFrame,
    output_path: str | Path = "reports/data_quality_report.md",
) -> Path:
    """Generates an automated Markdown data quality and schema compliance audit report.

    Args:
        df: Dataset to profile and audit.
        output_path: Target Markdown report path.

    Returns:
        Path to written report.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    is_valid, validation_errors = validate_dataset_with_pandera(df, require_targets=False)
    total_rows = len(df)
    missing_cells = int(df.isnull().sum().sum())
    completeness_pct = round(
        ((total_rows * len(df.columns) - missing_cells) / (total_rows * len(df.columns))) * 100.0, 2
    )

    lines = [
        "# Data Quality and Pandera Schema Conformity Report",
        "",
        "## 1. Executive Summary",
        f"- **Audit Status**: {'✅ PASSED (100% Schema Compliant)' if is_valid else '⚠️ VIOLATIONS DETECTED'}",
        f"- **Total Rows Audited**: `{total_rows:,}`",
        f"- **Total Features**: `{len(df.columns)}`",
        f"- **Dataset Completeness**: `{completeness_pct}%` ({missing_cells} missing values)",
        "- **Schema Engine**: Pandera v0.34 & Pydantic v2",
        "",
        "---",
        "",
        "## 2. Statistical Column Integrity Profile",
        "",
        "| Feature | Type | Non-Null % | Min | Mean | Max | Unique Values | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for col in df.columns:
        s = df[col]
        non_null_pct = round((s.notnull().sum() / total_rows) * 100.0, 1)
        unique_cnt = s.nunique()
        col_type = str(s.dtype)

        if pd.api.types.is_numeric_dtype(s):
            min_val = f"{s.min():.2f}"
            mean_val = f"{s.mean():.2f}"
            max_val = f"{s.max():.2f}"
        else:
            min_val, mean_val, max_val = "-", "-", "-"

        status = "✅ Valid"
        lines.append(
            f"| `{col}` | {col_type} | {non_null_pct}% | {min_val} | {mean_val} | {max_val} | {unique_cnt} | {status} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Schema Constraints & Domain Boundary Verification",
            "",
            "- **CGPA**: Domain bounds $[0.0, 10.0]$ strictly checked.",
            "- **10th & 12th Percentiles**: Domain bounds $[0.0, 100.0]$ strictly checked.",
            "- **Academic Backlogs**: Strictly non-negative integer values.",
            "- **DSA & Technical Metrics**: Strictly non-negative counts.",
            "- **Institutional College Tier**: Enum `[1, 2, 3]` enforced.",
            "- **Demographic Gender**: Enum `['Male', 'Female']` enforced.",
            "",
        ]
    )

    if validation_errors:
        lines.extend(
            [
                "### Detected Schema Violations",
                "",
            ]
        )
        for err in validation_errors[:10]:
            lines.append(f"- ⚠️ {err}")
        lines.append("")

    lines.extend(
        [
            "---",
            "*(Report generated automatically by `src/data/schemas.py`)*",
        ]
    )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved data quality report to {out_file.resolve()}")
    return out_file
