"""Prediction service for individual student assessments and batch inference."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.validation import StudentRecord, validate_dataframe
from src.models.registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PlacementPredictor:
    """Provides inference methods for placement probability, risk level, and salary compensation bands."""

    def __init__(
        self,
        placement_model_name: str = "best_placement_model",
        salary_model_name: str = "salary_band_model",
        models_dir: str | Path = "models",
    ) -> None:
        self.registry = ModelRegistry(models_dir=models_dir)
        self.placement_pipeline, self.placement_meta = self.registry.load_model(
            placement_model_name
        )

        self.salary_pipeline = None
        self.salary_meta = None
        try:
            self.salary_pipeline, self.salary_meta = self.registry.load_model(salary_model_name)
        except Exception as exc:
            logger.info(f"Salary band model not loaded: {exc}. Predictions will omit salary band.")

    def predict_student(self, student_data: dict[str, Any]) -> dict[str, Any]:
        """Runs validated inference for a single student profile.

        Args:
            student_data: Dictionary matching StudentRecord schema.

        Returns:
            Dictionary containing placement_prob, placement_pred, risk_label, and salary_band.
        """
        # Validate schema
        record = StudentRecord(**student_data)
        df_single = pd.DataFrame([record.model_dump()])

        # Placement prediction
        prob = float(self.placement_pipeline.predict_proba(df_single)[0, 1])
        pred = int(prob >= 0.50)

        # Risk categorization
        if prob >= 0.75:
            risk_label = "High Placement Likelihood"
            risk_color = "green"
        elif prob >= 0.45:
            risk_label = "Competitive / Moderate Risk"
            risk_color = "orange"
        else:
            risk_label = "High Risk / Attention Required"
            risk_color = "red"

        # Salary band prediction (only if placed)
        salary_band = "Not Placed"
        if pred == 1 and self.salary_pipeline is not None:
            try:
                salary_pred = self.salary_pipeline.predict(df_single)[0]
                salary_band = str(salary_pred)
            except Exception as e:
                logger.warning(f"Failed to predict salary band: {e}")
                salary_band = "Pending Evaluation"

        return {
            "placement_probability": round(prob, 4),
            "placement_prediction": pred,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "salary_band": salary_band,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs batch predictions on a DataFrame of student records.

        Args:
            df: DataFrame containing student profiles.

        Returns:
            DataFrame augmented with prediction columns.
        """
        is_valid, errors = validate_dataframe(df, require_targets=False)
        if not is_valid:
            raise ValueError(f"Batch inference DataFrame failed schema validation: {errors}")

        df_out = df.copy()
        probs = self.placement_pipeline.predict_proba(df)[:, 1]
        preds = (probs >= 0.50).astype(int)

        df_out["predicted_placement_probability"] = np.round(probs, 4)
        df_out["predicted_placement"] = preds

        if self.salary_pipeline is not None:
            salary_preds = self.salary_pipeline.predict(df)
            # Assign 'Not Placed' where predicted placement is 0
            df_out["predicted_salary_band"] = np.where(preds == 1, salary_preds, "Not Placed")

        return df_out
