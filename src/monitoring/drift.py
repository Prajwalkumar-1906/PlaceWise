"""Statistical drift monitoring, Evidently reports, and automated retraining trigger evaluation."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    EVIDENTLY_AVAILABLE = True
except Exception:
    EVIDENTLY_AVAILABLE = False

from src.data.loader import DataLoader
from src.models.registry import ModelRegistry
from src.monitoring.simulation import generate_simulated_cohort
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
    p_threshold: float = 0.05,
) -> dict[str, Any]:
    """Computes statistical feature drift using 2-sample Kolmogorov-Smirnov and Chi-squared tests.

    Args:
        reference_df: Baseline reference data (e.g. training set).
        current_df: Incoming production or monitoring cohort.
        numerical_features: Continuous feature names.
        categorical_features: Categorical feature names.
        p_threshold: Significance cutoff for drift detection (default 0.05).

    Returns:
        dict: Detailed per-feature drift metrics, drift share %, and dataset drift status.
    """
    feature_metrics: dict[str, dict[str, Any]] = {}
    drifted_count = 0

    # 1. Numerical Features: Kolmogorov-Smirnov & Wasserstein Distance
    for feat in numerical_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue

        ref_vals = reference_df[feat].dropna().values
        curr_vals = current_df[feat].dropna().values

        ks_stat, p_val = ks_2samp(ref_vals, curr_vals)
        w_dist = wasserstein_distance(ref_vals, curr_vals)
        is_drifted = bool(p_val < p_threshold)

        if is_drifted:
            drifted_count += 1

        feature_metrics[feat] = {
            "type": "numerical",
            "test_name": "Kolmogorov-Smirnov (2-sample)",
            "statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 5),
            "wasserstein_distance": round(float(w_dist), 4),
            "reference_mean": round(float(np.mean(ref_vals)), 2),
            "current_mean": round(float(np.mean(curr_vals)), 2),
            "mean_shift": round(float(np.mean(curr_vals) - np.mean(ref_vals)), 2),
            "drift_detected": is_drifted,
        }

    # 2. Categorical Features: Chi-Squared Test of Independence
    for feat in categorical_features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue

        ref_counts = reference_df[feat].astype(str).value_counts()
        curr_counts = current_df[feat].astype(str).value_counts()

        # Align categories
        all_cats = sorted(set(ref_counts.index).union(set(curr_counts.index)))
        ref_vec = [ref_counts.get(c, 0) for c in all_cats]
        curr_vec = [curr_counts.get(c, 0) for c in all_cats]

        contingency_table = np.array([ref_vec, curr_vec])
        try:
            chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
            is_drifted = bool(p_val < p_threshold)
        except Exception:
            chi2_stat, p_val = 0.0, 1.0
            is_drifted = False

        if is_drifted:
            drifted_count += 1

        feature_metrics[feat] = {
            "type": "categorical",
            "test_name": "Chi-Squared Independence",
            "statistic": round(float(chi2_stat), 4),
            "p_value": round(float(p_val), 5),
            "drift_detected": is_drifted,
        }

    total_evaluated = len(feature_metrics)
    drift_share = float(drifted_count / total_evaluated) if total_evaluated > 0 else 0.0
    dataset_drift_detected = bool(drift_share >= 0.30)

    return {
        "total_features": total_evaluated,
        "drifted_features_count": drifted_count,
        "drift_share_pct": round(drift_share * 100.0, 1),
        "dataset_drift_detected": dataset_drift_detected,
        "features": feature_metrics,
    }


def compute_prediction_drift(
    pipeline: Any,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Evaluates output distribution shift in model predicted probabilities."""
    ref_probs = pipeline.predict_proba(reference_df[feature_cols])[:, 1]
    curr_probs = pipeline.predict_proba(current_df[feature_cols])[:, 1]

    ks_stat, p_val = ks_2samp(ref_probs, curr_probs)
    ref_sel_rate = float(np.mean(ref_probs >= 0.50))
    curr_sel_rate = float(np.mean(curr_probs >= 0.50))

    is_drifted = bool(p_val < 0.05)
    return {
        "test_name": "Kolmogorov-Smirnov Prediction Probabilities",
        "statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_val), 5),
        "reference_selection_rate": round(ref_sel_rate, 4),
        "current_selection_rate": round(curr_sel_rate, 4),
        "selection_rate_gap": round(abs(curr_sel_rate - ref_sel_rate), 4),
        "prediction_drift_detected": is_drifted,
    }


def evaluate_retraining_trigger(
    drift_results: dict[str, Any],
    pred_drift: dict[str, Any],
    accuracy_drop: float | None = None,
    dp_gap: float | None = None,
) -> dict[str, Any]:
    """Applies mathematically defined retraining rules to incoming monitoring signals."""
    triggered_reasons: list[str] = []

    # Rule 1: Dataset Feature Drift Share >= 30%
    if drift_results.get("dataset_drift_detected", False):
        share = drift_results.get("drift_share_pct", 0.0)
        triggered_reasons.append(
            f"Feature Drift Share ({share}%) exceeds critical threshold of 30.0%."
        )

    # Rule 2: Prediction Output Distribution Shift (p < 0.01)
    if pred_drift.get("p_value", 1.0) < 0.01:
        triggered_reasons.append(
            f"Prediction probability drift detected with high significance (p={pred_drift.get('p_value'):.5f})."
        )

    # Rule 3: Performance Degradation
    if accuracy_drop is not None and accuracy_drop > 0.05:
        triggered_reasons.append(
            f"Held-out candidate accuracy dropped by {accuracy_drop*100:.1f}% (threshold 5.0%)."
        )

    # Rule 4: Demographic Parity Disparity Surge
    if dp_gap is not None and dp_gap > 0.10:
        triggered_reasons.append(
            f"Demographic Parity gap on incoming cohort ({dp_gap:.3f}) exceeds governance tolerance (0.100)."
        )

    retraining_recommended = len(triggered_reasons) > 0

    if len(triggered_reasons) >= 2:
        urgency = "CRITICAL_RETRAIN"
        status_label = "Model Retraining Required Immediately"
        guidance = "Multiple severe distribution shifts detected. Trigger MLflow pipeline re-training and candidate re-weighting."
    elif len(triggered_reasons) == 1:
        urgency = "ATTENTION_REQUIRED"
        status_label = "Retraining Recommended"
        guidance = "Statistical drift threshold breached. Schedule shadow model training on next batch cycle."
    else:
        urgency = "NORMAL"
        status_label = "Model Healthy - Operational"
        guidance = "All feature and prediction distributions remain within acceptable statistical bounds. No retraining needed."

    return {
        "retraining_recommended": retraining_recommended,
        "urgency_level": urgency,
        "status_label": status_label,
        "triggered_reasons": triggered_reasons,
        "guidance": guidance,
    }


def generate_evidently_html_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    output_html_path: str | Path = "reports/drift_report.html",
) -> Path | None:
    """Generates standalone interactive Evidently HTML drift report."""
    if not EVIDENTLY_AVAILABLE:
        logger.warning("Evidently not available. Skipping HTML report generation.")
        return None

    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ref_subset = reference_df[feature_cols].copy()
        curr_subset = current_df[feature_cols].copy()

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=ref_subset, current_data=curr_subset)
        snapshot.save_html(str(out_path))
        logger.info(f"Saved interactive Evidently drift report to {out_path.resolve()}")
        return out_path
    except Exception as e:
        logger.warning(f"Could not generate Evidently HTML report: {e}")
        return None


def run_monitoring_suite(
    config_path: str | Path = "config/config.yaml",
    scenario: str = "cgpa_inflation",
) -> dict[str, Any]:
    """End-to-end monitoring pipeline execution."""
    config = load_config(config_path)
    paths = config.get("paths", {})
    feat_cfg = config.get("features", {})
    num_features = feat_cfg.get("numerical", [])
    cat_features = feat_cfg.get("categorical", [])
    all_features = num_features + cat_features

    # Load baseline reference data (training split)
    loader = DataLoader(config_path=config_path)
    df_raw = loader.load_synthetic_data(biased=False)
    train_df, _, test_df = loader.get_train_val_test_split(df_raw, target_col="placed")

    # Load active production pipeline
    registry = ModelRegistry(models_dir=paths.get("models_dir", "models"))
    pipeline, _ = registry.load_model("best_placement_model")

    # Generate or load monitored cohort
    cohort_df = generate_simulated_cohort(
        base_df=test_df, scenario=scenario, sample_size=600, random_seed=42
    )

    logger.info(
        f"Running drift audit comparing Reference (n={len(train_df)}) vs Current Cohort '{scenario}' (n={len(cohort_df)})..."
    )

    # 1. Feature Drift
    feature_drift = compute_feature_drift(train_df, cohort_df, num_features, cat_features)

    # 2. Prediction Drift
    pred_drift = compute_prediction_drift(pipeline, train_df, cohort_df, all_features)

    # 3. Retraining Evaluation
    retrain_decision = evaluate_retraining_trigger(feature_drift, pred_drift)

    monitoring_summary = {
        "scenario": scenario,
        "cohort_size": len(cohort_df),
        "feature_drift": feature_drift,
        "prediction_drift": pred_drift,
        "retraining_decision": retrain_decision,
    }

    # Save JSON summary
    reports_dir = Path(paths.get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "drift_metrics.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(monitoring_summary, f, indent=2)
    logger.info(f"Saved drift metrics JSON to {json_path.resolve()}")

    # Generate Evidently HTML Report
    html_path = reports_dir / "drift_report.html"
    generate_evidently_html_report(train_df, cohort_df, all_features, output_html_path=html_path)

    logger.info(
        f"\n{'='*70}\n"
        f"DRIFT MONITORING AUDIT SUMMARY ({scenario.upper()}):\n"
        f"Drift Share: {feature_drift['drift_share_pct']}% ({feature_drift['drifted_features_count']}/{feature_drift['total_features']} features drifted)\n"
        f"Dataset Drift Detected: {feature_drift['dataset_drift_detected']}\n"
        f"Prediction Drift (p-val): {pred_drift['p_value']:.5f}\n"
        f"Status: {retrain_decision['status_label']} ({retrain_decision['urgency_level']})\n"
        f"Guidance: {retrain_decision['guidance']}\n"
        f"{'='*70}"
    )

    return monitoring_summary


def main() -> None:
    """CLI entrypoint for drift monitoring."""
    parser = argparse.ArgumentParser(
        description="Run Evidently data and prediction drift monitoring."
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config.yaml."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="cgpa_inflation",
        choices=["normal", "cgpa_inflation", "hiring_slump", "curriculum_modernization"],
        help="Simulated cohort drift scenario.",
    )
    args = parser.parse_args()

    run_monitoring_suite(config_path=args.config, scenario=args.scenario)


if __name__ == "__main__":
    main()
