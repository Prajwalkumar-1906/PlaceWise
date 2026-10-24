"""Unit tests for the rigorous evaluation suite (Repeated CV, Bootstrap CIs, Ablations, Robustness)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation_suite import (
    build_candidate_pipelines,
    compute_bootstrap_cis,
    generate_evaluation_report,
    perform_statistical_comparison,
    plot_evaluation_figures,
    run_feature_ablation_study,
    run_full_evaluation_suite,
    run_repeated_cv,
    run_robustness_stress_tests,
)


@pytest.fixture
def dummy_data() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    rng = np.random.default_rng(42)
    n = 120
    num_cols = [
        "cgpa",
        "tenth_pct",
        "twelfth_pct",
        "backlogs",
        "dsa_problems_solved",
        "projects",
        "internships",
        "certifications",
        "hackathons",
        "communication_score",
        "aptitude_score",
    ]
    cat_cols = ["branch", "college_tier", "gender"]

    data: dict[str, list[float] | list[str] | list[int]] = {}
    for col in num_cols:
        data[col] = rng.normal(50.0, 10.0, size=n).tolist()

    data["branch"] = rng.choice(
        ["Computer Science", "Information Technology", "Mechanical"], size=n
    ).tolist()
    data["college_tier"] = rng.choice([1, 2, 3], size=n).tolist()
    data["gender"] = rng.choice(["Male", "Female"], size=n).tolist()

    y = pd.Series(rng.choice([0, 1], size=n, p=[0.3, 0.7]), name="placed")
    df = pd.DataFrame(data)
    df["placed"] = y
    return df, y, num_cols, cat_cols


def test_build_candidate_pipelines(
    dummy_data: tuple[pd.DataFrame, pd.Series, list[str], list[str]],
) -> None:
    _, _, num_cols, cat_cols = dummy_data
    candidates = build_candidate_pipelines(num_cols, cat_cols, random_seed=42)
    expected_models = {
        "Logistic Regression",
        "Random Forest",
        "LightGBM",
        "CatBoost",
        "Tuned XGBoost",
    }
    assert set(candidates.keys()) == expected_models
    for _name, pipe in candidates.items():
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "predict_proba")


def test_compute_bootstrap_cis() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], size=100, p=[0.25, 0.75])
    y_prob = rng.uniform(0.1, 0.9, size=100)

    cis = compute_bootstrap_cis(y_true, y_prob, n_bootstraps=50, random_seed=42)
    for metric in ["accuracy", "f1", "roc_auc", "pr_auc", "brier_score"]:
        assert metric in cis
        assert "mean" in cis[metric]
        assert "ci_lower" in cis[metric]
        assert "ci_upper" in cis[metric]
        assert cis[metric]["ci_lower"] <= cis[metric]["ci_upper"]


def test_run_repeated_cv_smoke(
    dummy_data: tuple[pd.DataFrame, pd.Series, list[str], list[str]],
) -> None:
    df, y, num_cols, cat_cols = dummy_data
    # Use 2x2 for quick unit test execution
    summary = run_repeated_cv(
        X=df[num_cols + cat_cols],
        y=y,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        n_splits=2,
        n_repeats=2,
        random_seed=42,
    )
    assert summary["total_folds"] == 4
    assert "Tuned XGBoost" in summary["models"]
    xgb_roc = summary["models"]["Tuned XGBoost"]["roc_auc"]
    assert len(xgb_roc["raw_scores"]) == 4
    assert 0.0 <= xgb_roc["mean"] <= 1.0


def test_perform_statistical_comparison() -> None:
    mock_cv_summary = {
        "models": {
            "Tuned XGBoost": {
                "roc_auc": {"raw_scores": [0.88, 0.89, 0.90, 0.87, 0.88]},
                "f1": {"raw_scores": [0.91, 0.92, 0.93, 0.90, 0.91]},
            },
            "Random Forest": {
                "roc_auc": {"raw_scores": [0.84, 0.85, 0.86, 0.83, 0.84]},
                "f1": {"raw_scores": [0.87, 0.88, 0.89, 0.86, 0.87]},
            },
        }
    }
    stats_res = perform_statistical_comparison(mock_cv_summary, baseline_model="Tuned XGBoost")
    assert "Random Forest" in stats_res
    roc_comp = stats_res["Random Forest"]["roc_auc_comparison"]
    assert "wilcoxon_p_value" in roc_comp
    assert "paired_t_p_value" in roc_comp
    assert "cohens_d" in roc_comp
    assert roc_comp["mean_diff"] > 0


def test_run_feature_ablation_study(
    dummy_data: tuple[pd.DataFrame, pd.Series, list[str], list[str]],
) -> None:
    df, _, num_cols, cat_cols = dummy_data
    train_df = df.iloc[:80]
    test_df = df.iloc[80:]

    ablation_res = run_feature_ablation_study(
        train_df=train_df,
        test_df=test_df,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        target_col="placed",
        random_seed=42,
    )
    assert "baseline_full_model" in ablation_res
    assert "groups" in ablation_res
    assert "academic" in ablation_res["groups"]
    assert "skills" in ablation_res["groups"]


def test_run_robustness_stress_tests(
    dummy_data: tuple[pd.DataFrame, pd.Series, list[str], list[str]],
) -> None:
    df, y, num_cols, cat_cols = dummy_data
    train_df = df.iloc[:80]
    test_df = df.iloc[80:]

    candidates = build_candidate_pipelines(num_cols, cat_cols, random_seed=42)
    pipe = candidates["Logistic Regression"]
    pipe.fit(train_df[num_cols + cat_cols], train_df["placed"])

    rob_res = run_robustness_stress_tests(
        pipeline=pipe,
        test_df=test_df,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        target_col="placed",
        random_seed=42,
    )
    assert "baseline_clean" in rob_res
    assert "gaussian_noise" in rob_res
    assert "missing_values" in rob_res
    assert "out_of_range" in rob_res


def test_plot_and_report_generation(tmp_path: Path) -> None:
    mock_cv = {
        "models": {
            "Tuned XGBoost": {
                "accuracy": {"mean": 0.89, "std": 0.01},
                "f1": {"mean": 0.94, "std": 0.01},
                "roc_auc": {"mean": 0.88, "std": 0.02, "raw_scores": [0.88, 0.89, 0.87, 0.88]},
                "pr_auc": {"mean": 0.98, "std": 0.005},
            },
            "Random Forest": {
                "accuracy": {"mean": 0.85, "std": 0.01},
                "f1": {"mean": 0.91, "std": 0.01},
                "roc_auc": {"mean": 0.87, "std": 0.02, "raw_scores": [0.86, 0.87, 0.87, 0.88]},
                "pr_auc": {"mean": 0.97, "std": 0.005},
            },
        }
    }
    mock_boot = {
        "Tuned XGBoost": {
            "accuracy": {"mean": 0.90, "ci_lower": 0.88, "ci_upper": 0.92},
            "f1": {"mean": 0.94, "ci_lower": 0.93, "ci_upper": 0.95},
            "roc_auc": {"mean": 0.87, "ci_lower": 0.84, "ci_upper": 0.90},
            "pr_auc": {"mean": 0.97, "ci_lower": 0.96, "ci_upper": 0.98},
            "brier_score": {"mean": 0.08, "ci_lower": 0.07, "ci_upper": 0.09},
        }
    }
    mock_stats = {
        "Random Forest": {
            "roc_auc_comparison": {
                "mean_diff": 0.01,
                "wilcoxon_stat": 2.0,
                "wilcoxon_p_value": 0.02,
                "paired_t_stat": 2.5,
                "paired_t_p_value": 0.01,
                "cohens_d": 0.6,
                "statistically_significant_005": True,
            }
        }
    }
    mock_abl = {
        "baseline_full_model": {"accuracy": 0.90, "f1": 0.94, "roc_auc": 0.88},
        "groups": {
            "academic": {
                "removed_features": ["cgpa", "backlogs"],
                "accuracy": 0.85,
                "f1": 0.91,
                "roc_auc": 0.75,
                "delta_accuracy": -0.05,
                "delta_f1": -0.03,
                "delta_roc_auc": -0.13,
            }
        },
    }
    mock_rob = {
        "baseline_clean": {"accuracy": 0.90, "roc_auc": 0.88},
        "gaussian_noise": {
            "sigma_0.05": {"accuracy": 0.89, "roc_auc": 0.86, "delta_roc_auc": -0.02},
            "sigma_0.1": {"accuracy": 0.88, "roc_auc": 0.85, "delta_roc_auc": -0.03},
            "sigma_0.2": {"accuracy": 0.87, "roc_auc": 0.84, "delta_roc_auc": -0.04},
        },
        "missing_values": {
            "missing_5pct": {"accuracy": 0.89, "roc_auc": 0.86, "delta_roc_auc": -0.02},
            "missing_10pct": {"accuracy": 0.88, "roc_auc": 0.84, "delta_roc_auc": -0.04},
            "missing_20pct": {"accuracy": 0.87, "roc_auc": 0.83, "delta_roc_auc": -0.05},
        },
    }

    # Test plot generation
    plots = plot_evaluation_figures(mock_cv, mock_abl, mock_rob, output_dir=tmp_path / "figures")
    assert len(plots) == 3
    for p in plots:
        assert Path(p).exists()

    # Test report generation
    report_file = tmp_path / "test_report.md"
    generated_report = generate_evaluation_report(
        cv_summary=mock_cv,
        bootstrap_summary=mock_boot,
        stats_summary=mock_stats,
        ablation_summary=mock_abl,
        robustness_summary=mock_rob,
        output_file=report_file,
    )
    assert generated_report.exists()
    content = generated_report.read_text(encoding="utf-8")
    assert "Comprehensive Model Evaluation" in content
    assert "Tuned XGBoost" in content


def test_run_full_evaluation_suite_fast() -> None:
    results = run_full_evaluation_suite(
        config_path="config/config.yaml",
        cv_splits=2,
        cv_repeats=1,
        n_bootstraps=10,
    )
    assert "repeated_cv_5x5" in results
    assert "bootstrap_cis_test" in results
    assert "statistical_tests" in results
    assert "ablation_study" in results
    assert "robustness_stress_testing" in results
