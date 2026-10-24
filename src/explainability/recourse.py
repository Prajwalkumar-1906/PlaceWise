"""Actionable recourse generation with strict feature immutability and developmental pathways."""

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.validation import StudentRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Strictly locked immutable attributes that can never be modified in recourse recommendations
IMMUTABLE_FEATURES = {
    "gender",
    "college_tier",
    "branch",
    "tenth_pct",
    "twelfth_pct",
}

# Actionable attributes with domain-enforced feasibility constraints and realistic effort weights
ACTIONABLE_CONFIG: dict[str, dict[str, Any]] = {
    "dsa_problems_solved": {
        "max": 600,
        "step": 25,
        "effort_weight": 0.02,  # 50 problems ~ 1.0 effort unit
        "unit": "problems",
        "description": "DSA Problem Practice",
    },
    "projects": {
        "max": 8,
        "step": 1,
        "effort_weight": 1.5,
        "unit": "projects",
        "description": "Full-Stack / ML Projects",
    },
    "internships": {
        "max": 4,
        "step": 1,
        "effort_weight": 3.0,
        "unit": "internships",
        "description": "Industry Internships",
    },
    "certifications": {
        "max": 5,
        "step": 1,
        "effort_weight": 1.0,
        "unit": "certifications",
        "description": "Cloud / Domain Certifications",
    },
    "hackathons": {
        "max": 6,
        "step": 1,
        "effort_weight": 1.0,
        "unit": "hackathons",
        "description": "Hackathons & Competitions",
    },
    "communication_score": {
        "max": 100.0,
        "step": 5.0,
        "effort_weight": 0.2,  # 5 points ~ 1.0 effort unit
        "unit": "points",
        "description": "Communication & Interview Prep",
    },
    "aptitude_score": {
        "max": 100.0,
        "step": 5.0,
        "effort_weight": 0.2,
        "unit": "points",
        "description": "Quantitative Aptitude Training",
    },
    "backlogs": {
        "min": 0,
        "step": -1,
        "effort_weight": 2.5,
        "unit": "backlogs cleared",
        "description": "Backlog Clearance",
    },
}

RECOURSE_DISCLAIMER = (
    "DISCLAIMER: Actionable recourse suggestions are mathematical counterfactual simulations "
    "generated from historical placement patterns. They serve as guidance for student skill development "
    "and do NOT constitute an automatic guarantee or contractual commitment of employment by prospective recruiters."
)


class RecourseEngine:
    """Computes realistic, actionable intervention pathways to achieve placement target probabilities."""

    def __init__(self, pipeline: Pipeline) -> None:
        """Initializes recourse engine with fitted pipeline.

        Args:
            pipeline: Fitted scikit-learn placement classification Pipeline.
        """
        self.pipeline = pipeline

    def _predict_prob(self, student_dict: dict[str, Any]) -> float:
        """Helper to predict placement probability for a single dictionary record."""
        record = StudentRecord(**student_dict)
        df_single = pd.DataFrame([record.model_dump()])
        return float(self.pipeline.predict_proba(df_single)[0, 1])

    def generate_recourse_pathways(
        self,
        baseline_student: dict[str, Any],
        target_probability: float = 0.65,
    ) -> dict[str, Any]:
        """Generates multi-attribute recourse pathways respecting immutability constraints.

        Args:
            baseline_student: Original student profile.
            target_probability: Required probability threshold (default 0.65 for high confidence).

        Returns:
            dict with baseline_probability, target_probability, pathways list, and legal disclaimer.
        """
        base_record = StudentRecord(**baseline_student)
        base_dict = base_record.model_dump()
        base_prob = self._predict_prob(base_dict)

        if base_prob >= target_probability:
            return {
                "baseline_probability": round(base_prob, 4),
                "target_probability": round(target_probability, 4),
                "target_already_satisfied": True,
                "message": f"Candidate's baseline placement likelihood ({base_prob*100:.1f}%) already satisfies or exceeds the target ({target_probability*100:.1f}%).",
                "pathways": [],
                "disclaimer": RECOURSE_DISCLAIMER,
            }

        # Define 3 distinct strategic development pathways
        pathway_configs = [
            {
                "name": "Technical Mastery Pathway",
                "description": "Focuses on intensive algorithm practice (DSA), technical project portfolio, and hackathons.",
                "priority_features": [
                    "dsa_problems_solved",
                    "projects",
                    "hackathons",
                    "aptitude_score",
                ],
            },
            {
                "name": "Industry Experience & Credentials Pathway",
                "description": "Focuses on practical internship experience, industry certifications, and clearing backlogs.",
                "priority_features": [
                    "internships",
                    "certifications",
                    "backlogs",
                    "communication_score",
                ],
            },
            {
                "name": "Holistic Balanced Pathway",
                "description": "Distributes developmental effort proportionally across technical coding, aptitude, and soft skills.",
                "priority_features": [
                    "dsa_problems_solved",
                    "aptitude_score",
                    "communication_score",
                    "projects",
                    "certifications",
                    "backlogs",
                ],
            },
        ]

        generated_pathways: list[dict[str, Any]] = []

        for p_cfg in pathway_configs:
            current_dict = dict(base_dict)
            current_prob = base_prob
            interventions: list[dict[str, Any]] = []
            total_effort = 0.0

            # Step iteratively through priority features
            for feat in p_cfg["priority_features"]:
                # Guard: ensure feature is strictly actionable
                if feat in IMMUTABLE_FEATURES or feat not in ACTIONABLE_CONFIG:
                    continue

                cfg = ACTIONABLE_CONFIG[feat]
                start_val = current_dict[feat]

                if feat == "backlogs":
                    # For backlogs, we can only decrease
                    if start_val > 0:
                        clear_count = start_val  # Try clearing all backlogs
                        current_dict[feat] = 0
                        new_prob = self._predict_prob(current_dict)
                        effort = clear_count * cfg["effort_weight"]
                        total_effort += effort
                        interventions.append(
                            {
                                "feature": feat,
                                "description": cfg["description"],
                                "current_value": start_val,
                                "recommended_value": 0,
                                "delta": f"-{clear_count} {cfg['unit']}",
                                "effort_points": round(effort, 1),
                            }
                        )
                        current_prob = new_prob
                else:
                    max_val = cfg["max"]
                    step = cfg["step"]
                    steps_taken = 0
                    max_allowed_steps = 8

                    while (
                        current_prob < target_probability
                        and (current_dict[feat] + step) <= max_val
                        and steps_taken < max_allowed_steps
                    ):
                        current_dict[feat] += step
                        steps_taken += 1
                        current_prob = self._predict_prob(current_dict)

                    if steps_taken > 0:
                        delta_val = steps_taken * step
                        effort = delta_val * cfg["effort_weight"]
                        total_effort += effort
                        interventions.append(
                            {
                                "feature": feat,
                                "description": cfg["description"],
                                "current_value": start_val,
                                "recommended_value": current_dict[feat],
                                "delta": f"+{delta_val} {cfg['unit']}",
                                "effort_points": round(effort, 1),
                            }
                        )

                if current_prob >= target_probability:
                    break

            # Assess feasibility rating based on total effort
            if total_effort <= 3.0:
                feasibility = "High (Achievable in 1-2 months)"
            elif total_effort <= 7.0:
                feasibility = "Moderate (Achievable in 3-5 months)"
            else:
                feasibility = "Challenging (Requires sustained 6+ month plan)"

            if interventions:
                generated_pathways.append(
                    {
                        "pathway_name": p_cfg["name"],
                        "description": p_cfg["description"],
                        "baseline_probability": round(base_prob, 4),
                        "achieved_probability": round(current_prob, 4),
                        "probability_gain_pts": round((current_prob - base_prob) * 100.0, 1),
                        "reaches_target": bool(current_prob >= target_probability),
                        "total_effort_score": round(total_effort, 1),
                        "feasibility_rating": feasibility,
                        "interventions": interventions,
                        "immutable_features_locked": sorted(IMMUTABLE_FEATURES),
                    }
                )

        # Sort pathways by whether they reach target, then lowest effort
        generated_pathways.sort(key=lambda p: (not p["reaches_target"], p["total_effort_score"]))

        return {
            "baseline_probability": round(base_prob, 4),
            "target_probability": round(target_probability, 4),
            "target_already_satisfied": False,
            "pathways_count": len(generated_pathways),
            "pathways": generated_pathways,
            "disclaimer": RECOURSE_DISCLAIMER,
        }
