"""Actionable counterfactual what-if simulation for student profile improvement."""

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.validation import StudentRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)

CONTROLLABLE_FEATURES = {
    "dsa_problems_solved": {"min": 0, "max": 1200, "step": 50, "unit": "problems"},
    "internships": {"min": 0, "max": 4, "step": 1, "unit": "internships"},
    "projects": {"min": 0, "max": 6, "step": 1, "unit": "projects"},
    "certifications": {"min": 0, "max": 5, "step": 1, "unit": "certifications"},
    "hackathons": {"min": 0, "max": 6, "step": 1, "unit": "competitions"},
    "communication_score": {"min": 30.0, "max": 100.0, "step": 5.0, "unit": "points"},
    "aptitude_score": {"min": 30.0, "max": 100.0, "step": 5.0, "unit": "points"},
    "backlogs": {"min": 0, "max": 5, "step": -1, "unit": "backlogs"},
}


class WhatIfSimulator:
    """Simulates counterfactual modifications to controllable student features."""

    def __init__(self, pipeline: Pipeline) -> None:
        """Initializes simulator with a fitted placement pipeline.

        Args:
            pipeline: Fitted scikit-learn Pipeline.
        """
        self.pipeline = pipeline

    def simulate(
        self,
        baseline_student: dict[str, Any],
        modifications: dict[str, Any],
    ) -> dict[str, Any]:
        """Simulates how specific controllable changes alter placement likelihood.

        Args:
            baseline_student: Original student profile dictionary.
            modifications: Changes to apply, e.g. {'dsa_problems_solved': 400, 'internships': 2}.

        Returns:
            dict containing baseline_prob, modified_prob, delta_percentage_points, and outcome shift.
        """
        # Validate baseline
        base_record = StudentRecord(**baseline_student)
        df_base = pd.DataFrame([base_record.model_dump()])

        # Construct and validate modified profile
        mod_dict = dict(baseline_student)
        for feat, val in modifications.items():
            if feat not in StudentRecord.model_fields:
                raise ValueError(f"Unknown feature '{feat}' specified in modifications.")
            mod_dict[feat] = val

        mod_record = StudentRecord(**mod_dict)
        df_mod = pd.DataFrame([mod_record.model_dump()])

        # Predict probabilities
        base_prob = float(self.pipeline.predict_proba(df_base)[0, 1])
        mod_prob = float(self.pipeline.predict_proba(df_mod)[0, 1])

        delta_prob = mod_prob - base_prob
        delta_pts = delta_prob * 100.0

        base_pred = int(base_prob >= 0.50)
        mod_pred = int(mod_prob >= 0.50)

        status_shift = "Unchanged"
        if base_pred == 0 and mod_pred == 1:
            status_shift = "Flipped: Not Placed -> Placed"
        elif base_pred == 1 and mod_pred == 0:
            status_shift = "Flipped: Placed -> Not Placed"

        return {
            "baseline_probability": round(base_prob, 4),
            "modified_probability": round(mod_prob, 4),
            "delta_probability_points": round(delta_pts, 2),
            "baseline_prediction": base_pred,
            "modified_prediction": mod_pred,
            "status_shift": status_shift,
            "modifications_applied": modifications,
        }

    def recommend_interventions(
        self,
        baseline_student: dict[str, Any],
        target_probability: float = 0.65,
    ) -> list[dict[str, Any]]:
        """Identifies high-impact single-feature interventions for candidates below target probability.

        Args:
            baseline_student: Original student record.
            target_probability: Target placement threshold.

        Returns:
            list of recommended adjustments sorted by effectiveness.
        """
        base_record = StudentRecord(**baseline_student)
        df_base = pd.DataFrame([base_record.model_dump()])
        base_prob = float(self.pipeline.predict_proba(df_base)[0, 1])

        if base_prob >= target_probability:
            return []

        recommendations: list[dict[str, Any]] = []

        # Test single-lever improvements
        test_increments: dict[str, list[Any]] = {
            "dsa_problems_solved": [50, 100, 200],
            "internships": [1],
            "projects": [1, 2],
            "certifications": [1],
            "communication_score": [10.0],
            "aptitude_score": [10.0],
        }

        for feat, steps in test_increments.items():
            curr_val = baseline_student.get(feat, 0)
            for step in steps:
                new_val = curr_val + step
                sim_res = self.simulate(baseline_student, {feat: new_val})
                recommendations.append(
                    {
                        "feature": feat,
                        "change": f"+{step}",
                        "new_value": new_val,
                        "new_probability": sim_res["modified_probability"],
                        "gain_points": sim_res["delta_probability_points"],
                        "reaches_target": sim_res["modified_probability"] >= target_probability,
                    }
                )

        # Sort recommendations by highest probability gain
        recommendations.sort(key=lambda x: x["gain_points"], reverse=True)
        return recommendations[:5]
