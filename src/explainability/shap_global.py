"""Global model explainability using SHAP TreeExplainer, summary beeswarm, and feature importance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.features.pipeline import get_feature_names
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_shap_explanation(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
) -> tuple[shap.Explanation, list[str]]:
    """Calculates SHAP Explanation object for a fitted pipeline and background dataset.

    Args:
        pipeline: Fitted pipeline with 'preprocessor' and 'classifier'.
        X_background: DataFrame of sample records.
        numerical_features: Numerical column names.
        categorical_features: Categorical column names.

    Returns:
        tuple (shap_explanation_object, transformed_feature_names)
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    feature_names = get_feature_names(preprocessor, numerical_features, categorical_features)
    X_proc = preprocessor.transform(X_background)

    explainer = shap.TreeExplainer(clf)
    explanation = explainer(X_proc)
    explanation.feature_names = feature_names

    return explanation, feature_names


def get_global_feature_importance(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Computes global feature importance ranked by mean absolute SHAP value.

    Args:
        pipeline: Fitted pipeline.
        X_background: DataFrame of student records.
        numerical_features: Numerical feature list.
        categorical_features: Categorical feature list.

    Returns:
        pd.DataFrame: Sorted feature importance table with columns ['feature', 'mean_abs_shap'].
    """
    explanation, feature_names = compute_shap_explanation(
        pipeline, X_background, numerical_features, categorical_features
    )

    vals = np.abs(explanation.values).mean(axis=0)
    df_importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": vals})
        .sort_values(by="mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return df_importance


def generate_global_shap_plots(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
    output_dir: str | Path = "reports/figures",
    max_display: int = 12,
) -> None:
    """Generates and persists global SHAP beeswarm and feature importance bar plots.

    Args:
        pipeline: Fitted pipeline.
        X_background: DataFrame of student records.
        numerical_features: Numerical feature list.
        categorical_features: Categorical feature list.
        output_dir: Directory to save generated plot PNGs.
        max_display: Maximum number of features to display on plots.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    explanation, _ = compute_shap_explanation(
        pipeline, X_background, numerical_features, categorical_features
    )

    # 1. SHAP Beeswarm Summary Plot
    plt.figure(figsize=(9, 6), dpi=150)
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title(
        "Global SHAP Summary (Feature Impact on Margin Log-Odds)", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    beeswarm_path = out_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP beeswarm plot to {beeswarm_path.resolve()}")

    # 2. SHAP Bar Importance Plot
    plt.figure(figsize=(9, 6), dpi=150)
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.title("Global Mean |SHAP| Feature Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    bar_path = out_dir / "shap_feature_importance_bar.png"
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP bar importance plot to {bar_path.resolve()}")
