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


def compute_fairness_bootstrap_cis(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_features: np.ndarray | pd.Series,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculates empirical bootstrap 95% confidence intervals for Fairlearn metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        sensitive_features: Sensitive attribute series or array.
        n_bootstraps: Number of bootstrap iterations.
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        dict containing point estimate, mean, and [lower, upper] empirical bounds.
    """
    y_t = np.array(y_true).astype(int)
    y_p = np.array(y_pred).astype(int)
    sens = np.array(sensitive_features).astype(str)
    n = len(y_t)

    unique_groups = sorted(set(sens))
    rng = np.random.default_rng(seed)

    dp_diffs: list[float] = []
    dp_ratios: list[float] = []
    eo_diffs: list[float] = []
    sel_rates: dict[str, list[float]] = {g: [] for g in unique_groups}

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_b = y_t[idx]
        yp_b = y_p[idx]
        sens_b = sens[idx]

        group_srs: list[float] = []
        group_tprs: list[float] = []
        group_fprs: list[float] = []
        valid = True

        for g in unique_groups:
            g_mask = sens_b == g
            n_g = np.sum(g_mask)
            if n_g == 0:
                valid = False
                break
            sr = float(np.mean(yp_b[g_mask]))
            group_srs.append(sr)

            y1_g = g_mask & (yt_b == 1)
            y0_g = g_mask & (yt_b == 0)
            n_y1 = np.sum(y1_g)
            n_y0 = np.sum(y0_g)
            if n_y1 == 0 or n_y0 == 0:
                valid = False
                break
            group_tprs.append(float(np.mean(yp_b[y1_g])))
            group_fprs.append(float(np.mean(yp_b[y0_g])))

        if not valid:
            continue

        for i, g in enumerate(unique_groups):
            sel_rates[g].append(group_srs[i])

        dp_d = max(group_srs) - min(group_srs)
        max_sr = max(group_srs)
        dp_r = (min(group_srs) / max_sr) if max_sr > 1e-6 else 1.0
        eo_d = max(max(group_tprs) - min(group_tprs), max(group_fprs) - min(group_fprs))

        dp_diffs.append(dp_d)
        dp_ratios.append(dp_r)
        eo_diffs.append(eo_d)

    def _calc_ci(values: list[float]) -> dict[str, float]:
        if not values:
            return {"point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        arr = np.array(values)
        low = float(np.percentile(arr, 100 * (alpha / 2.0)))
        high = float(np.percentile(arr, 100 * (1.0 - alpha / 2.0)))
        return {
            "mean": round(float(np.mean(arr)), 4),
            "ci_lower": round(low, 4),
            "ci_upper": round(high, 4),
        }

    ci_results: dict[str, Any] = {
        "demographic_parity_difference": _calc_ci(dp_diffs),
        "demographic_parity_ratio": _calc_ci(dp_ratios),
        "equalized_odds_difference": _calc_ci(eo_diffs),
        "selection_rates_by_group": {g: _calc_ci(sel_rates[g]) for g in unique_groups},
    }

    return ci_results
