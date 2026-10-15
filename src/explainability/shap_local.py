"""Local prediction explainability using SHAP waterfall plots, additivity validation, and calibrated probability conversions."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.features.pipeline import get_feature_names
from src.utils.logger import get_logger

logger = get_logger(__name__)


def explain_student_prediction(
    pipeline: Pipeline,
    student_data: dict[str, Any] | pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
    top_n: int = 5,
) -> dict[str, Any]:
    """Generates a complete local explanation for an individual student prediction.

    Validates exact SHAP additivity:
        Model Output (log-odds) = Base Value + sum(phi_i)

    Converts margin log-odds to approximate probability-point contributions (with documented approximation).

    Args:
        pipeline: Fitted scikit-learn Pipeline (preprocessor + XGBoost).
        student_data: Dictionary or single-row DataFrame.
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        top_n: Number of positive and negative drivers to extract.

    Returns:
        dict: Local explanation containing log-odds values, probability deltas, additivity check,
              and plain-English driver statements.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    df_single = (
        pd.DataFrame([student_data]) if isinstance(student_data, dict) else student_data.head(1)
    )
    feature_names = get_feature_names(preprocessor, numerical_features, categorical_features)

    # Transform input
    X_proc = preprocessor.transform(df_single)

    # XGBoost raw margin output (log-odds)
    margin_output = float(clf.predict(X_proc, output_margin=True)[0])
    pred_prob = float(1.0 / (1.0 + np.exp(-margin_output)))

    # Compute SHAP values
    explainer = shap.TreeExplainer(clf)
    explanation = explainer(X_proc)[0]
    explanation.feature_names = feature_names

    base_value = float(explanation.base_values)
    base_prob = float(1.0 / (1.0 + np.exp(-base_value)))
    shap_vals = np.array(explanation.values, dtype=float)

    # Additivity Sanity Check
    sum_shap = float(np.sum(shap_vals))
    reconstructed_margin = base_value + sum_shap
    additivity_error = abs(margin_output - reconstructed_margin)
    is_additive = bool(additivity_error < 1e-4)

    if not is_additive:
        logger.warning(
            f"SHAP additivity error exceeds tolerance: |{margin_output:.5f} - {reconstructed_margin:.5f}| = {additivity_error:.6f}"
        )

    # Probability-Point Conversion (Documented Approximation)
    # The exact nonlinear relationship is: P = sigma(margin).
    # Since individual SHAP values phi_i are in log-odds, their exact probability contribution
    # is non-additive without a path integral. We allocate the total probability delta:
    # Delta_P = P - P_base proportionally across features based on their log-odds share.
    total_prob_delta = pred_prob - base_prob

    if abs(sum_shap) > 1e-6:
        prob_deltas = total_prob_delta * (shap_vals / sum_shap)
    else:
        # Fallback to local Taylor expansion: Delta_P_i ~= phi_i * sigma'(base_value)
        p_mid = 0.5 * (base_prob + pred_prob)
        prob_deltas = shap_vals * (p_mid * (1.0 - p_mid))

    feature_impacts: list[dict[str, Any]] = []
    for name, log_odds, p_delta in zip(feature_names, shap_vals, prob_deltas, strict=False):
        feature_impacts.append(
            {
                "feature": name,
                "shap_log_odds": round(float(log_odds), 4),
                "prob_delta_percentage_points": round(float(p_delta * 100), 2),
            }
        )

    # Sort into positive and negative drivers based on SHAP log-odds direction
    pos_drivers = sorted(
        [f for f in feature_impacts if f["shap_log_odds"] > 0],
        key=lambda x: x["shap_log_odds"],
        reverse=True,
    )[:top_n]

    neg_drivers = sorted(
        [f for f in feature_impacts if f["shap_log_odds"] < 0],
        key=lambda x: x["shap_log_odds"],
    )[:top_n]

    # Generate plain-English explanations
    plain_english: list[str] = []
    for f in pos_drivers:
        pts = f["prob_delta_percentage_points"]
        plain_english.append(f"{f['feature']}: +{pts:.1f} pts (increases placement probability)")

    for f in neg_drivers:
        pts = f["prob_delta_percentage_points"]
        plain_english.append(f"{f['feature']}: {pts:.1f} pts (decreases placement probability)")

    if not plain_english and feature_impacts:
        # Fallback if all impacts are neutral
        top_f = feature_impacts[0]
        plain_english.append(f"{top_f['feature']}: neutral impact (0.0 pts)")

    return {
        "base_value_log_odds": round(base_value, 4),
        "base_probability": round(base_prob, 4),
        "model_margin_log_odds": round(margin_output, 4),
        "model_probability": round(pred_prob, 4),
        "total_prob_delta_points": round(total_prob_delta * 100, 2),
        "additivity_check": {
            "is_additive": is_additive,
            "additivity_error": float(additivity_error),
            "reconstructed_margin": round(reconstructed_margin, 4),
        },
        "feature_impacts": feature_impacts,
        "top_positive_drivers": pos_drivers,
        "top_negative_drivers": neg_drivers,
        "plain_english_summary": plain_english,
        "raw_explanation": explanation,
    }


def save_waterfall_plot(
    explanation: shap.Explanation,
    output_path: str | Path,
    max_display: int = 10,
) -> Path:
    """Renders and saves a SHAP waterfall plot for a single observation.

    Args:
        explanation: Single-row SHAP Explanation object.
        output_path: File path to save PNG plot.
        max_display: Max features to render.

    Returns:
        Path to saved plot.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5), dpi=150)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.title(
        "Local SHAP Waterfall (Feature Contribution to Margin Log-Odds)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved local SHAP waterfall plot to {path.resolve()}")
    return path
