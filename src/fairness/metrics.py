"""Fairness metric computation across demographic and intersectional sensitive attributes."""

from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_intersectional_attribute(
    df: pd.DataFrame,
    col1: str = "gender",
    col2: str = "college_tier",
) -> pd.Series:
    """Creates a combined intersectional sensitive attribute (e.g. 'Female_Tier3').

    Args:
        df: Input DataFrame containing both sensitive columns.
        col1: First sensitive column name.
        col2: Second sensitive column name.

    Returns:
        pd.Series: Combined intersectional string series.
    """
    return df[col1].astype(str) + "_Tier" + df[col2].astype(str)


def compute_fairness_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_features: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Calculates comprehensive Fairlearn bias and disparate impact metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        sensitive_features: Demographic attribute Series or 1D array.

    Returns:
        dict: Overall parity gaps, ratios, equalized odds, and per-group breakdowns.
    """
    y_t = np.array(y_true).astype(int)
    y_p = np.array(y_pred).astype(int)
    sens = pd.Series(sensitive_features).reset_index(drop=True)

    # Disaggregated metric frame
    metric_fns = {
        "selection_rate": selection_rate,
        "accuracy": accuracy_score,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
    }

    mf = MetricFrame(
        metrics=metric_fns,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=sens,
    )

    # Parity gaps
    dp_diff = float(demographic_parity_difference(y_t, y_p, sensitive_features=sens))
    dp_ratio = float(demographic_parity_ratio(y_t, y_p, sensitive_features=sens))
    eo_diff = float(equalized_odds_difference(y_t, y_p, sensitive_features=sens))

    # Equal Opportunity difference (TPR gap across groups)
    tpr_by_group = mf.by_group["tpr"]
    eq_opp_diff = float(tpr_by_group.max() - tpr_by_group.min())

    # Build detailed per-group dictionary
    by_group_dict: dict[str, dict[str, float]] = {}
    group_counts = sens.value_counts().to_dict()

    for grp in mf.by_group.index:
        grp_str = str(grp)
        by_group_dict[grp_str] = {
            "count": int(group_counts.get(grp, 0)),
            "selection_rate": round(float(mf.by_group.loc[grp, "selection_rate"]), 4),
            "accuracy": round(float(mf.by_group.loc[grp, "accuracy"]), 4),
            "tpr": round(float(mf.by_group.loc[grp, "tpr"]), 4),
            "fpr": round(float(mf.by_group.loc[grp, "fpr"]), 4),
            "fnr": round(float(mf.by_group.loc[grp, "fnr"]), 4),
        }

    return {
        "demographic_parity_difference": round(dp_diff, 4),
        "demographic_parity_ratio": round(dp_ratio, 4),
        "equalized_odds_difference": round(eo_diff, 4),
        "equal_opportunity_difference": round(eq_opp_diff, 4),
        "overall_selection_rate": round(float(selection_rate(y_t, y_p)), 4),
        "overall_accuracy": round(float(accuracy_score(y_t, y_p)), 4),
        "by_group": by_group_dict,
    }
