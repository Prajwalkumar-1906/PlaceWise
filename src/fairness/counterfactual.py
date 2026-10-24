"""Counterfactual fairness evaluation: assessing individual prediction invariance under sensitive attribute perturbations."""

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_counterfactual_fairness(
    pipeline: Any,
    test_df: pd.DataFrame,
    sensitive_col: str,
    flip_map: dict[Any, Any] | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluates counterfactual fairness by perturbing a protected attribute while holding all other features fixed.

    A counterfactually fair model ensures that an individual applicant's placement prediction
    and probability remain identical in a counterfactual world where their demographic attribute is changed.

    Args:
        pipeline: Trained scikit-learn Pipeline or estimator supporting predict and predict_proba.
        test_df: Test DataFrame containing applicant profiles.
        sensitive_col: Protected attribute column name to perturb (e.g. 'gender', 'college_tier').
        flip_map: Explicit mapping for perturbation. Defaults to Male<->Female for gender,
            and 1->3, 2->1, 3->1 for college_tier.
        feature_cols: Explicit list of feature columns expected by the pipeline. If None,
            uses pipeline's expected features or numerical+categorical columns.

    Returns:
        dict with flip_rate_pct, mean_prob_shift, max_prob_shift, and counterfactually_fair flag.
    """
    if flip_map is None:
        if sensitive_col == "gender":
            flip_map = {"Male": "Female", "Female": "Male"}
        elif sensitive_col == "college_tier":
            flip_map = {1: 3, 2: 1, 3: 1, "1": "3", "2": "1", "3": "1"}
        else:
            unique_vals = list(test_df[sensitive_col].unique())
            if len(unique_vals) == 2:
                flip_map = {unique_vals[0]: unique_vals[1], unique_vals[1]: unique_vals[0]}
            else:
                flip_map = {
                    v: unique_vals[(i + 1) % len(unique_vals)] for i, v in enumerate(unique_vals)
                }

    # Check if the pipeline uses the sensitive column
    is_unaware = False
    if feature_cols is not None:
        if sensitive_col not in feature_cols:
            is_unaware = True
    elif hasattr(pipeline, "feature_names_in_"):
        if sensitive_col not in pipeline.feature_names_in_:
            is_unaware = True

    # Prepare original input
    X_orig = test_df[feature_cols].copy() if feature_cols is not None else test_df.copy()

    # If the pipeline does not take the sensitive column, prediction shift is identically zero
    if is_unaware:
        return {
            "sensitive_attribute": sensitive_col,
            "perturbation_mapping": {str(k): str(v) for k, v in flip_map.items()},
            "flip_rate_pct": 0.0,
            "mean_prob_shift": 0.0,
            "max_prob_shift": 0.0,
            "is_counterfactually_fair": True,
            "status": "Counterfactually invariant (attribute excluded from input features)",
        }

    # Generate baseline predictions
    try:
        y_pred_orig = pipeline.predict(X_orig)
        if hasattr(pipeline, "predict_proba"):
            y_prob_orig = pipeline.predict_proba(X_orig)[:, 1]
        else:
            y_prob_orig = y_pred_orig.astype(float)
    except Exception as e:
        logger.warning(f"Could not compute original predictions for counterfactual audit: {e}")
        return {
            "sensitive_attribute": sensitive_col,
            "flip_rate_pct": 0.0,
            "mean_prob_shift": 0.0,
            "max_prob_shift": 0.0,
            "is_counterfactually_fair": False,
            "error": str(e),
        }

    # Construct counterfactual dataset
    X_cf = X_orig.copy()
    if sensitive_col in X_cf.columns:
        X_cf[sensitive_col] = X_cf[sensitive_col].map(lambda v: flip_map.get(v, v))

    try:
        y_pred_cf = pipeline.predict(X_cf)
        if hasattr(pipeline, "predict_proba"):
            y_prob_cf = pipeline.predict_proba(X_cf)[:, 1]
        else:
            y_prob_cf = y_pred_cf.astype(float)
    except Exception as e:
        logger.warning(f"Could not compute counterfactual predictions: {e}")
        return {
            "sensitive_attribute": sensitive_col,
            "flip_rate_pct": 0.0,
            "mean_prob_shift": 0.0,
            "max_prob_shift": 0.0,
            "is_counterfactually_fair": False,
            "error": str(e),
        }

    flips = np.sum(y_pred_orig != y_pred_cf)
    total = len(test_df)
    flip_rate = float((flips / total) * 100.0) if total > 0 else 0.0

    prob_shifts = np.abs(y_prob_orig - y_prob_cf)
    mean_shift = float(np.mean(prob_shifts))
    max_shift = float(np.max(prob_shifts))

    is_cf_fair = bool(flip_rate == 0.0 and mean_shift < 1e-4)
    status_str = (
        "Strictly Counterfactually Fair"
        if is_cf_fair
        else f"Counterfactual Disparity Detected ({flip_rate:.1f}% decision flips)"
    )

    return {
        "sensitive_attribute": sensitive_col,
        "perturbation_mapping": {str(k): str(v) for k, v in flip_map.items()},
        "flip_rate_pct": round(flip_rate, 2),
        "mean_prob_shift": round(mean_shift, 4),
        "max_prob_shift": round(max_shift, 4),
        "is_counterfactually_fair": is_cf_fair,
        "status": status_str,
    }
