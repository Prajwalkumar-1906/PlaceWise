"""Unit tests for fairness metrics, mitigation methods, audit engine, and report generation."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.fairness.metrics import compute_fairness_metrics, create_intersectional_attribute
from src.fairness.mitigation import (
    MitigationManager,
    apply_reweighing,
    compute_sample_weights,
    fit_exponentiated_gradient,
    fit_threshold_optimizer,
)
from src.fairness.report import generate_fairness_report
from src.features.pipeline import create_preprocessing_pipeline


def test_create_intersectional_attribute():
    """Verify intersectional sensitive feature generation."""
    df = pd.DataFrame({"gender": ["Male", "Female"], "college_tier": [1, 3]})
    inter = create_intersectional_attribute(df, "gender", "college_tier")
    assert inter.iloc[0] == "Male_Tier1"
    assert inter.iloc[1] == "Female_Tier3"


def test_compute_fairness_metrics():
    """Verify Fairlearn disparity metrics calculation."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1])
    sens = pd.Series(["A", "A", "A", "B", "B", "B"])

    metrics = compute_fairness_metrics(y_true, y_pred, sens)

    assert "demographic_parity_difference" in metrics
    assert "demographic_parity_ratio" in metrics
    assert "equalized_odds_difference" in metrics
    assert "equal_opportunity_difference" in metrics
    assert "by_group" in metrics
    assert "A" in metrics["by_group"]
    assert "B" in metrics["by_group"]
    assert 0.0 <= metrics["demographic_parity_difference"] <= 1.0


def test_compute_sample_weights():
    """Verify demographic parity reweighing weights."""
    y = np.array([1, 1, 0, 0, 1, 0])
    sens = np.array(["Tier1", "Tier1", "Tier1", "Tier3", "Tier3", "Tier3"])

    weights = compute_sample_weights(y, sens)
    assert len(weights) == len(y)
    assert (weights > 0).all()
    assert np.isclose(np.mean(weights), 1.0, atol=1e-4)


def test_mitigations_end_to_end():
    """Verify that all three mitigation methods fit and predict correctly."""
    data = {
        "cgpa": [7.0, 8.0, 6.5, 9.0, 7.5, 8.5, 6.0, 9.5, 7.2, 8.8, 6.8, 8.2],
        "tenth_pct": [70.0, 80.0, 65.0, 90.0, 75.0, 85.0, 60.0, 95.0, 72.0, 88.0, 68.0, 82.0],
        "branch": [
            "Computer Science",
            "Information Technology",
            "Civil",
            "Computer Science",
            "Mechanical",
            "Electrical",
            "Civil",
            "Computer Science",
            "Mechanical",
            "Electrical",
            "Civil",
            "Information Technology",
        ],
        "college_tier": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
        "gender": [
            "Male",
            "Female",
            "Male",
            "Female",
            "Male",
            "Female",
            "Female",
            "Male",
            "Female",
            "Male",
            "Female",
            "Male",
        ],
        "work_experience": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "placed": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    num_cols = ["cgpa", "tenth_pct"]
    cat_cols = ["branch", "college_tier", "gender", "work_experience"]

    # 1. Reweighing
    reweight_pipe = apply_reweighing(
        df[num_cols + cat_cols],
        df["placed"],
        df["college_tier"],
        num_cols,
        cat_cols,
    )
    preds_rew = reweight_pipe.predict(df[num_cols + cat_cols])
    assert len(preds_rew) == len(df)

    # 2. ThresholdOptimizer
    preproc = create_preprocessing_pipeline(num_cols, cat_cols)
    clf = XGBClassifier(n_estimators=10, max_depth=2, random_state=42)
    base_pipe = Pipeline([("preprocessor", preproc), ("classifier", clf)])
    base_pipe.fit(df[num_cols + cat_cols], df["placed"])

    thresh_opt = fit_threshold_optimizer(
        base_pipe, df[num_cols + cat_cols], df["placed"], df["college_tier"]
    )
    preds_thresh = thresh_opt.predict(
        df[num_cols + cat_cols], sensitive_features=df["college_tier"]
    )
    assert len(preds_thresh) == len(df)

    # 3. ExponentiatedGradient
    exp_prep, exp_grad = fit_exponentiated_gradient(
        df[num_cols + cat_cols], df["placed"], df["college_tier"], num_cols, cat_cols, eps=0.1
    )
    proc_X = exp_prep.transform(df[num_cols + cat_cols])
    preds_exp = exp_grad.predict(proc_X)
    assert len(preds_exp) == len(df)

    # 4. MitigationManager
    manager = MitigationManager(base_pipe, num_cols, cat_cols)
    manager.fit_all(
        df[num_cols + cat_cols],
        df["placed"],
        df["college_tier"],
        df[num_cols + cat_cols],
        df["placed"],
        df["college_tier"],
    )
    assert len(manager.predict_reweighted(df[num_cols + cat_cols])) == len(df)
    assert len(
        manager.predict_threshold_optimizer(df[num_cols + cat_cols], df["college_tier"])
    ) == len(df)
    assert len(manager.predict_exponentiated_gradient(df[num_cols + cat_cols])) == len(df)


def test_generate_fairness_report(tmp_path):
    """Verify that report generator produces markdown file and figures."""
    dummy_audit = {
        "models": {
            "Baseline (Aware)": {
                "metrics_by_sensitive_attribute": {
                    "college_tier": {
                        "demographic_parity_difference": 0.25,
                        "demographic_parity_ratio": 0.70,
                        "equalized_odds_difference": 0.20,
                        "by_group": {
                            "1": {"selection_rate": 0.90},
                            "2": {"selection_rate": 0.80},
                            "3": {"selection_rate": 0.65},
                        },
                    },
                    "gender": {
                        "demographic_parity_difference": 0.15,
                        "by_group": {
                            "Male": {"selection_rate": 0.85},
                            "Female": {"selection_rate": 0.70},
                        },
                    },
                }
            }
        },
        "tradeoffs": {
            "college_tier": [
                {
                    "model": "Baseline (Aware)",
                    "accuracy": 0.88,
                    "dp_difference": 0.25,
                    "dp_ratio": 0.70,
                    "equalized_odds_diff": 0.20,
                    "bias_reduction_percentage": 0.0,
                }
            ]
        },
    }

    report_md = tmp_path / "fairness_report.md"
    generate_fairness_report(dummy_audit, output_md_path=report_md, figures_dir=tmp_path)

    assert report_md.exists()
    content = report_md.read_text(encoding="utf-8")
    assert "Algorithmic Fairness Audit and Mitigation Report" in content
    assert "Explicit Bias Reduction Formula" in content
    assert (tmp_path / "fairness_selection_rates.png").exists()
    assert (tmp_path / "fairness_tradeoff_pareto.png").exists()
