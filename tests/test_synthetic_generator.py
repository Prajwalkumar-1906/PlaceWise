"""Unit tests for the synthetic dataset generator and bias injection."""

import pandas as pd

from src.data.synthetic_generator import SyntheticPlacementGenerator
from src.data.validation import validate_dataframe


def test_generator_shape_and_columns():
    """Verify generator outputs expected number of rows and all expected columns."""
    gen = SyntheticPlacementGenerator()
    df = gen.generate(n_samples=200, random_seed=42, inject_bias=False)

    assert len(df) == 200
    expected_cols = [
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
        "placed",
        "salary_band",
    ]
    for col in expected_cols:
        assert col in df.columns

    is_valid, errors = validate_dataframe(df, require_targets=True)
    assert is_valid is True
    assert len(errors) == 0


def test_generator_reproducibility():
    """Verify that using the same random seed produces identical datasets."""
    gen = SyntheticPlacementGenerator()
    df1 = gen.generate(n_samples=100, random_seed=999, inject_bias=False)
    df2 = gen.generate(n_samples=100, random_seed=999, inject_bias=False)

    pd.testing.assert_frame_equal(df1, df2)


def test_generator_correlations():
    """Verify that generated skill indicators positively correlate with placement."""
    gen = SyntheticPlacementGenerator()
    df = gen.generate(n_samples=1500, random_seed=42, inject_bias=False)

    placed_cgpa = df[df["placed"] == 1]["cgpa"].mean()
    unplaced_cgpa = df[df["placed"] == 0]["cgpa"].mean()
    assert (
        placed_cgpa > unplaced_cgpa
    ), f"Expected placed CGPA ({placed_cgpa}) > unplaced ({unplaced_cgpa})"

    placed_dsa = df[df["placed"] == 1]["dsa_problems_solved"].mean()
    unplaced_dsa = df[df["placed"] == 0]["dsa_problems_solved"].mean()
    assert (
        placed_dsa > unplaced_dsa
    ), f"Expected placed DSA ({placed_dsa}) > unplaced ({unplaced_dsa})"


def test_bias_injection_impact():
    """Verify that bias injection noticeably reduces the placement rate of unprivileged groups."""
    gen = SyntheticPlacementGenerator()
    df_unbiased = gen.generate(n_samples=2500, random_seed=42, inject_bias=False)
    df_biased = gen.generate(n_samples=2500, random_seed=42, inject_bias=True)

    tier1_rate_unbiased = df_unbiased[df_unbiased["college_tier"] == 1]["placed"].mean()
    tier3_rate_unbiased = df_unbiased[df_unbiased["college_tier"] == 3]["placed"].mean()
    unbiased_gap = tier1_rate_unbiased - tier3_rate_unbiased

    tier1_rate_biased = df_biased[df_biased["college_tier"] == 1]["placed"].mean()
    tier3_rate_biased = df_biased[df_biased["college_tier"] == 3]["placed"].mean()
    biased_gap = tier1_rate_biased - tier3_rate_biased

    # Biased gap between tier 1 and tier 3 must be substantially larger due to explicit penalty
    assert biased_gap > unbiased_gap
    assert tier3_rate_biased < tier3_rate_unbiased
