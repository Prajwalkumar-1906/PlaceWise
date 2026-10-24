"""Cohort simulation for post-deployment drift stress testing and monitoring."""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_simulated_cohort(
    base_df: pd.DataFrame,
    scenario: str = "normal",
    sample_size: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generates a realistic post-deployment applicant cohort simulating operational drift.

    Scenarios:
        - 'normal': Natural sampling variance from baseline population. No systemic drift.
        - 'cgpa_inflation': Grade inflation (+0.8 mean CGPA) without skill improvement.
        - 'hiring_slump': Economic contraction; companies raise hiring bars and reduce offers.
        - 'curriculum_modernization': University updates curriculum; massive surge in DSA/cloud skills.

    Args:
        base_df: Baseline reference DataFrame (e.g. training or historical test data).
        scenario: Drift scenario name.
        sample_size: Size of the simulated applicant cohort.
        random_seed: Seed for reproducibility.

    Returns:
        pd.DataFrame: Simulated cohort matching feature schema.
    """
    rng = np.random.default_rng(random_seed)
    n_base = len(base_df)
    indices = rng.choice(n_base, size=sample_size, replace=True)
    cohort = base_df.iloc[indices].copy().reset_index(drop=True)

    logger.info(f"Generating simulated cohort '{scenario}' (n={sample_size})...")

    if scenario == "normal":
        # Slight noise only (zero mean)
        if "communication_score" in cohort.columns:
            cohort["communication_score"] = np.clip(
                cohort["communication_score"] + rng.normal(0, 1.5, size=sample_size), 0.0, 100.0
            )

    elif scenario == "cgpa_inflation":
        # Systematic shift in academic scores without skill gains
        if "cgpa" in cohort.columns:
            cohort["cgpa"] = np.clip(
                cohort["cgpa"] + rng.normal(0.85, 0.25, size=sample_size), 0.0, 10.0
            )
        if "tenth_pct" in cohort.columns:
            cohort["tenth_pct"] = np.clip(
                cohort["tenth_pct"] + rng.normal(4.0, 2.0, size=sample_size), 0.0, 100.0
            )
        if "twelfth_pct" in cohort.columns:
            cohort["twelfth_pct"] = np.clip(
                cohort["twelfth_pct"] + rng.normal(5.0, 2.5, size=sample_size), 0.0, 100.0
            )

    elif scenario == "hiring_slump":
        # Macro hiring freeze: target labels drop, tier-3 applicants face steeper rejections
        if "placed" in cohort.columns:
            # 35% of previously placed candidates are no longer hired
            placed_mask = cohort["placed"] == 1
            flip_candidates = cohort[placed_mask].index
            flips = rng.choice(
                flip_candidates, size=int(len(flip_candidates) * 0.35), replace=False
            )
            cohort.loc[flips, "placed"] = 0

        # Employers require more problem solving for same outcome
        if "dsa_problems_solved" in cohort.columns:
            cohort["dsa_problems_solved"] = np.clip(cohort["dsa_problems_solved"] - 30, 0, 2500)

    elif scenario == "curriculum_modernization":
        # Technical curriculum upgrade: huge boost in coding and certifications
        if "dsa_problems_solved" in cohort.columns:
            cohort["dsa_problems_solved"] = np.clip(
                cohort["dsa_problems_solved"] + rng.integers(120, 250, size=sample_size), 0, 2500
            )
        if "certifications" in cohort.columns:
            cohort["certifications"] = np.clip(
                cohort["certifications"] + rng.integers(1, 3, size=sample_size), 0, 20
            )
        if "projects" in cohort.columns:
            cohort["projects"] = np.clip(
                cohort["projects"] + rng.integers(1, 2, size=sample_size), 0, 20
            )

    else:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Supported: 'normal', 'cgpa_inflation', 'hiring_slump', 'curriculum_modernization'."
        )

    return cohort
