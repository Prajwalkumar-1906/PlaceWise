"""Synthetic dataset generator for campus placement prediction with realistic correlations and configurable bias injection."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validation import validate_dataframe
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

logger = get_logger(__name__)


class SyntheticPlacementGenerator:
    """Generates synthetic student placement data with realistic feature correlations and ground-truth bias injection."""

    def __init__(self, config_path: str | Path = "config/synth_config.yaml") -> None:
        """Initializes generator with configuration parameters.

        Args:
            config_path: Path to synth_config.yaml.
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.gen_cfg = self.config.get("generation", {})
        self.feat_cfg = self.config.get("features", {})
        self.weight_cfg = self.config.get("weights", {})
        self.salary_cfg = self.config.get("salary_bands", {})
        self.bias_cfg = self.config.get("bias_injection", {})

    def generate(
        self,
        n_samples: int | None = None,
        random_seed: int | None = None,
        inject_bias: bool | None = None,
    ) -> pd.DataFrame:
        """Generates the synthetic dataset.

        Args:
            n_samples: Number of records to generate (default from config).
            random_seed: Random seed for reproducibility (default from config).
            inject_bias: Whether to inject institutional penalties for college tier / gender.

        Returns:
            Validated pandas DataFrame of student placement records.
        """
        n = n_samples or self.gen_cfg.get("n_samples", 5000)
        seed = random_seed if random_seed is not None else self.gen_cfg.get("random_seed", 42)
        should_inject_bias = (
            inject_bias if inject_bias is not None else self.bias_cfg.get("enabled", False)
        )

        seed_everything(seed)
        rng = np.random.default_rng(seed)

        logger.info(
            f"Generating {n} synthetic student records (seed={seed}, inject_bias={should_inject_bias})..."
        )

        # 1. Base academic metrics
        cgpa_cfg = self.feat_cfg.get("cgpa", {})
        cgpa = rng.normal(cgpa_cfg.get("mean", 7.65), cgpa_cfg.get("std", 1.05), size=n)
        cgpa = np.clip(cgpa, cgpa_cfg.get("min", 5.0), cgpa_cfg.get("max", 10.0))

        tenth_cfg = self.feat_cfg.get("tenth_pct", {})
        tenth_pct = rng.normal(tenth_cfg.get("mean", 78.5), tenth_cfg.get("std", 9.5), size=n)
        tenth_pct = np.clip(tenth_pct, tenth_cfg.get("min", 50.0), tenth_cfg.get("max", 99.0))

        # Twelfth % positively correlated with 10th % (corr ~ 0.70)
        twelfth_cfg = self.feat_cfg.get("twelfth_pct", {})
        twelfth_base = 0.65 * tenth_pct + rng.normal(
            twelfth_cfg.get("mean", 75.0) * 0.35, twelfth_cfg.get("std", 10.5) * 0.70, size=n
        )
        twelfth_pct = np.clip(
            twelfth_base, twelfth_cfg.get("min", 50.0), twelfth_cfg.get("max", 99.0)
        )

        # 2. Backlogs (poisson / categorical)
        backlog_dist = self.feat_cfg.get(
            "backlogs_dist", {"values": [0, 1, 2, 3, 4], "probs": [0.72, 0.16, 0.07, 0.03, 0.02]}
        )
        backlogs = rng.choice(backlog_dist["values"], p=backlog_dist["probs"], size=n)
        # Students with high backlogs tend to have lower CGPA
        high_backlog_mask = backlogs >= 2
        cgpa[high_backlog_mask] = np.clip(
            cgpa[high_backlog_mask] - rng.uniform(0.3, 0.8, size=high_backlog_mask.sum()), 5.0, 10.0
        )

        # 3. Categorical Demographics & Institutional features
        branch_cfg = self.feat_cfg.get(
            "branches",
            {
                "categories": [
                    "Computer Science",
                    "Information Technology",
                    "Electronics & Comm",
                    "Mechanical",
                    "Electrical",
                    "Civil",
                ],
                "probs": [0.35, 0.20, 0.20, 0.10, 0.10, 0.05],
            },
        )
        branch = rng.choice(branch_cfg["categories"], p=branch_cfg["probs"], size=n)

        tier_cfg = self.feat_cfg.get(
            "college_tiers", {"categories": [1, 2, 3], "probs": [0.15, 0.35, 0.50]}
        )
        college_tier = rng.choice(tier_cfg["categories"], p=tier_cfg["probs"], size=n)

        gender_cfg = self.feat_cfg.get(
            "genders", {"categories": ["Male", "Female"], "probs": [0.58, 0.42]}
        )
        gender = rng.choice(gender_cfg["categories"], p=gender_cfg["probs"], size=n)

        workex_cfg = self.feat_cfg.get("work_experience", {"values": [0, 1], "probs": [0.76, 0.24]})
        work_experience = rng.choice(workex_cfg["values"], p=workex_cfg["probs"], size=n)

        # 4. Technical and Skill Metrics
        # Aptitude & Communication
        apt_cfg = self.feat_cfg.get("aptitude_score", {})
        apt_base = 0.4 * (cgpa / 10.0) * 100 + rng.normal(
            apt_cfg.get("mean", 69.0) * 0.6, apt_cfg.get("std", 14.0) * 0.8, size=n
        )
        aptitude_score = np.clip(apt_base, apt_cfg.get("min", 30.0), apt_cfg.get("max", 100.0))

        comm_cfg = self.feat_cfg.get("communication_score", {})
        comm_base = rng.normal(comm_cfg.get("mean", 71.0), comm_cfg.get("std", 13.0), size=n)
        communication_score = np.clip(
            comm_base, comm_cfg.get("min", 30.0), comm_cfg.get("max", 100.0)
        )

        # DSA problems solved (CS/IT students generally solve more; correlated with aptitude and CGPA)
        dsa_branch_mult = np.where(
            np.isin(branch, ["Computer Science", "Information Technology"]),
            1.45,
            np.where(branch == "Electronics & Comm", 1.05, 0.65),
        )
        dsa_raw = (
            rng.exponential(scale=180.0, size=n)
            * dsa_branch_mult
            * (cgpa / 7.5)
            * (aptitude_score / 70.0)
        )
        dsa_problems_solved = np.clip(np.round(dsa_raw).astype(int), 0, 1200)

        # Projects, Internships, Certifications, Hackathons
        proj_dist = self.feat_cfg.get(
            "projects_dist",
            {"values": [0, 1, 2, 3, 4, 5], "probs": [0.10, 0.25, 0.35, 0.18, 0.08, 0.04]},
        )
        projects = rng.choice(proj_dist["values"], p=proj_dist["probs"], size=n)

        intern_dist = self.feat_cfg.get(
            "internships_dist", {"values": [0, 1, 2, 3], "probs": [0.45, 0.35, 0.15, 0.05]}
        )
        internships = rng.choice(intern_dist["values"], p=intern_dist["probs"], size=n)

        cert_dist = self.feat_cfg.get(
            "certifications_dist",
            {"values": [0, 1, 2, 3, 4], "probs": [0.25, 0.35, 0.25, 0.10, 0.05]},
        )
        certifications = rng.choice(cert_dist["values"], p=cert_dist["probs"], size=n)

        hack_dist = self.feat_cfg.get(
            "hackathons_dist", {"values": [0, 1, 2, 3, 4], "probs": [0.55, 0.25, 0.12, 0.05, 0.03]}
        )
        hackathons = rng.choice(hack_dist["values"], p=hack_dist["probs"], size=n)

        # 5. Target Calculation: Placement Latent Logit
        norm_cgpa = (cgpa - 5.0) / 5.0
        norm_tenth = (tenth_pct - 50.0) / 50.0
        norm_twelfth = (twelfth_pct - 50.0) / 50.0
        norm_dsa = np.clip(dsa_problems_solved / 450.0, 0.0, 2.5)
        norm_apt = aptitude_score / 100.0
        norm_comm = communication_score / 100.0

        w = self.weight_cfg
        logit = (
            w.get("intercept", -4.50)
            + w.get("cgpa", 3.20) * norm_cgpa
            + w.get("dsa_normalized", 2.80) * norm_dsa
            + w.get("internships", 0.85) * internships
            + w.get("projects", 0.50) * projects
            + w.get("aptitude_normalized", 1.60) * norm_apt
            + w.get("communication_normalized", 1.40) * norm_comm
            + w.get("work_experience", 0.90) * work_experience
            + w.get("hackathons", 0.45) * hackathons
            + w.get("certifications", 0.30) * certifications
            + w.get("backlogs", -1.30) * backlogs
            + w.get("tenth_normalized", 0.60) * norm_tenth
            + w.get("twelfth_normalized", 0.70) * norm_twelfth
            + rng.normal(0, w.get("noise_std", 0.40), size=n)
        )

        # Explicit ground-truth bias injection
        if should_inject_bias:
            tier_penalties = self.bias_cfg.get("tier_penalty", {1: 0.0, 2: -0.40, 3: -1.25})
            gender_penalties = self.bias_cfg.get("gender_penalty", {"Male": 0.0, "Female": -0.70})

            tier_penalty_arr = np.array([tier_penalties.get(t, 0.0) for t in college_tier])
            gender_penalty_arr = np.array([gender_penalties.get(g, 0.0) for g in gender])

            logit += tier_penalty_arr + gender_penalty_arr
            logger.info("Ground-truth institutional bias penalties applied to placement logit.")

        # Convert logit to probability and binary target
        placement_prob = 1.0 / (1.0 + np.exp(-logit))
        placed = (rng.uniform(0.0, 1.0, size=n) < placement_prob).astype(int)

        # 6. Salary Band Assignment (only for placed students)
        salary_band = np.full(
            n, self.salary_cfg.get("labels", {}).get("unplaced", "Not Placed"), dtype=object
        )
        placed_indices = np.where(placed == 1)[0]

        if len(placed_indices) > 0:
            comp_score = (
                0.35 * norm_dsa[placed_indices]
                + 0.25 * norm_cgpa[placed_indices]
                + 0.20 * (internships[placed_indices] / 3.0)
                + 0.10 * norm_apt[placed_indices]
                + 0.10 * (work_experience[placed_indices])
                + rng.normal(0, 0.08, size=len(placed_indices))
            )

            # College tier 1 on-campus hiring has higher packages on average
            tier_boost = np.where(
                college_tier[placed_indices] == 1,
                0.20,
                np.where(college_tier[placed_indices] == 2, 0.05, -0.10),
            )
            comp_score += tier_boost

            th_low = self.salary_cfg.get("threshold_low", 0.40)
            th_mid = self.salary_cfg.get("threshold_mid", 0.75)
            labels = self.salary_cfg.get("labels", {})

            bands = np.where(
                comp_score < th_low,
                labels.get("low", "Low"),
                np.where(comp_score < th_mid, labels.get("mid", "Mid"), labels.get("high", "High")),
            )
            salary_band[placed_indices] = bands

        df = pd.DataFrame(
            {
                "cgpa": np.round(cgpa, 2),
                "tenth_pct": np.round(tenth_pct, 2),
                "twelfth_pct": np.round(twelfth_pct, 2),
                "backlogs": backlogs,
                "dsa_problems_solved": dsa_problems_solved,
                "projects": projects,
                "internships": internships,
                "certifications": certifications,
                "hackathons": hackathons,
                "communication_score": np.round(communication_score, 1),
                "aptitude_score": np.round(aptitude_score, 1),
                "branch": branch,
                "college_tier": college_tier,
                "gender": gender,
                "work_experience": work_experience,
                "placed": placed,
                "salary_band": salary_band,
            }
        )

        is_valid, validation_errors = validate_dataframe(df, require_targets=True)
        if not is_valid:
            raise ValueError(
                f"Generated synthetic data failed schema validation: {validation_errors}"
            )

        placement_rate = df["placed"].mean() * 100
        logger.info(
            f"Successfully generated {len(df)} records. Overall placement rate: {placement_rate:.1f}%."
        )
        return df


def main() -> None:
    """CLI entrypoint for generating synthetic placement data."""
    parser = argparse.ArgumentParser(description="Generate synthetic student placement records.")
    parser.add_argument("--samples", type=int, default=5000, help="Number of records to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--inject-bias", action="store_true", help="Flag to inject ground-truth institutional bias."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/synthetic_placement_data.csv",
        help="Output CSV file path.",
    )
    args = parser.parse_args()

    generator = SyntheticPlacementGenerator()
    df = generator.generate(
        n_samples=args.samples, random_seed=args.seed, inject_bias=args.inject_bias
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved dataset ({len(df)} rows) to {out_path.resolve()}")


if __name__ == "__main__":
    main()
