"""Fairness audit engine evaluating aware, unaware, and mitigated models across gender and college tier."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.loader import DataLoader
from src.fairness.metrics import compute_fairness_metrics, create_intersectional_attribute
from src.fairness.mitigation import MitigationManager
from src.fairness.report import generate_fairness_report
from src.features.encoders import split_features_by_sensitivity
from src.features.pipeline import create_preprocessing_pipeline
from src.models.registry import ModelRegistry
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

logger = get_logger(__name__)


class FairnessAuditor:
    """Orchestrates comprehensive bias audits and mitigation evaluations across single and intersectional groups."""

    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self.paths = self.config.get("paths", {})
        self.feat_cfg = self.config.get("features", {})
        self.registry = ModelRegistry(models_dir=self.paths.get("models_dir", "models"))

    def run_full_audit(self) -> dict[str, Any]:
        """Runs the complete fairness benchmarking suite comparing aware, unaware, and mitigated pipelines.

        Returns:
            dict containing metrics across all models, sensitive attributes, and trade-offs.
        """
        seed = self.config.get("project", {}).get("random_seed", 42)
        seed_everything(seed)

        num_features = self.feat_cfg.get("numerical", [])
        cat_features = self.feat_cfg.get("categorical", [])
        sensitive_cols = self.feat_cfg.get("sensitive_attributes", ["gender", "college_tier"])
        target_col = self.feat_cfg.get("targets", {}).get("placement", "placed")

        # Load biased dataset (ground-truth bias baseline)
        loader = DataLoader(config_path=self.config_path)
        df_biased = loader.load_synthetic_data(biased=True, auto_generate=True)
        train_df, val_df, test_df = loader.get_train_val_test_split(
            df_biased, target_col=target_col, stratify=True, random_seed=seed
        )

        # Create intersectional sensitive attribute
        for d in [train_df, val_df, test_df]:
            d["gender_x_tier"] = create_intersectional_attribute(d, "gender", "college_tier")

        X_train = train_df[num_features + cat_features]
        y_train = train_df[target_col]
        X_val = val_df[num_features + cat_features]
        y_val = val_df[target_col]
        X_test = test_df[num_features + cat_features]
        y_test = test_df[target_col]

        logger.info(f"Loaded biased audit dataset: Test set contains {len(test_df)} observations.")

        # 1. Load or Train Aware Baseline Model
        try:
            baseline_pipeline, _ = self.registry.load_model("best_placement_model_biased")
            logger.info("Loaded pre-trained biased model pipeline from registry.")
        except Exception:
            logger.info("Training fresh biased baseline pipeline...")
            preprocessor = create_preprocessing_pipeline(num_features, cat_features)
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=seed,
            )
            baseline_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
            baseline_pipeline.fit(X_train, y_train)

        # Baseline Predictions
        y_pred_baseline = baseline_pipeline.predict(X_test)
        y_prob_baseline = baseline_pipeline.predict_proba(X_test)[:, 1]
        auc_baseline = float(roc_auc_score(y_test, y_prob_baseline))
        acc_baseline = float(accuracy_score(y_test, y_pred_baseline))
        logger.info(
            f"Baseline Aware Model Test ROC-AUC: {auc_baseline:.4f}, Accuracy: {acc_baseline:.4f}"
        )

        # 2. Train Unawareness Model (excluding gender and college_tier)
        logger.info("Training 'Fairness through Unawareness' model...")
        subsets = split_features_by_sensitivity(num_features, cat_features, sensitive_cols)
        unaware_num = subsets["unaware"]["numerical"]
        unaware_cat = subsets["unaware"]["categorical"]

        unaware_preprocessor = create_preprocessing_pipeline(unaware_num, unaware_cat)
        unaware_clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=seed,
        )
        unaware_pipeline = Pipeline(
            [("preprocessor", unaware_preprocessor), ("classifier", unaware_clf)]
        )
        unaware_pipeline.fit(train_df[unaware_num + unaware_cat], y_train)

        y_pred_unaware = unaware_pipeline.predict(test_df[unaware_num + unaware_cat])
        y_prob_unaware = unaware_pipeline.predict_proba(test_df[unaware_num + unaware_cat])[:, 1]
        auc_unaware = float(roc_auc_score(y_test, y_prob_unaware))
        acc_unaware = float(accuracy_score(y_test, y_pred_unaware))
        logger.info(
            f"Unawareness Blind Model Test ROC-AUC: {auc_unaware:.4f}, Accuracy: {acc_unaware:.4f}"
        )

        # 3. Train Mitigated Models via MitigationManager
        logger.info(
            "Fitting Mitigation Manager (Reweighing, ThresholdOptimizer, ExponentiatedGradient)..."
        )
        mit_manager = MitigationManager(
            unmitigated_pipeline=baseline_pipeline,
            numerical_features=num_features,
            categorical_features=cat_features,
        )
        mit_manager.fit_all(
            X_train=X_train,
            y_train=y_train,
            sensitive_train=train_df["college_tier"],
            X_val=X_val,
            y_val=y_val,
            sensitive_val=val_df["college_tier"],
        )

        y_pred_reweight = mit_manager.predict_reweighted(X_test)
        acc_reweight = float(accuracy_score(y_test, y_pred_reweight))

        y_pred_thresh = mit_manager.predict_threshold_optimizer(
            X_test, sensitive_features=test_df["college_tier"]
        )
        acc_thresh = float(accuracy_score(y_test, y_pred_thresh))

        y_pred_expgrad = mit_manager.predict_exponentiated_gradient(X_test)
        acc_expgrad = float(accuracy_score(y_test, y_pred_expgrad))

        # Dictionary of all evaluated models
        models_preds = {
            "Baseline (Aware)": y_pred_baseline,
            "Unawareness (Blind)": y_pred_unaware,
            "Mitigated (Reweighing)": y_pred_reweight,
            "Mitigated (ThresholdOptimizer)": y_pred_thresh,
            "Mitigated (ExponentiatedGradient)": y_pred_expgrad,
        }
        models_acc = {
            "Baseline (Aware)": acc_baseline,
            "Unawareness (Blind)": acc_unaware,
            "Mitigated (Reweighing)": acc_reweight,
            "Mitigated (ThresholdOptimizer)": acc_thresh,
            "Mitigated (ExponentiatedGradient)": acc_expgrad,
        }

        # 4. Compute Fairness Metrics across Sensitive Attributes
        sensitive_targets = ["gender", "college_tier", "gender_x_tier"]
        audit_results: dict[str, Any] = {
            "models": {},
            "tradeoffs": {},
            "sensitive_attributes": sensitive_targets,
        }

        for model_name, preds in models_preds.items():
            audit_results["models"][model_name] = {
                "accuracy": round(models_acc[model_name], 4),
                "metrics_by_sensitive_attribute": {},
            }
            for sens_attr in sensitive_targets:
                sens_values = test_df[sens_attr]
                f_metrics = compute_fairness_metrics(y_test, preds, sens_values)
                audit_results["models"][model_name]["metrics_by_sensitive_attribute"][
                    sens_attr
                ] = f_metrics

        # 5. Calculate Documented Bias Reduction Percentage
        # Formula: (DP_diff_base - DP_diff_mitigated) / DP_diff_base * 100%
        base_dp_tier = audit_results["models"]["Baseline (Aware)"][
            "metrics_by_sensitive_attribute"
        ]["college_tier"]["demographic_parity_difference"]
        base_eo_tier = audit_results["models"]["Baseline (Aware)"][
            "metrics_by_sensitive_attribute"
        ]["college_tier"]["equalized_odds_difference"]

        tradeoff_summary: list[dict[str, Any]] = []
        for model_name in models_preds:
            m_tier = audit_results["models"][model_name]["metrics_by_sensitive_attribute"][
                "college_tier"
            ]
            dp_diff = m_tier["demographic_parity_difference"]
            eo_diff = m_tier["equalized_odds_difference"]
            acc = models_acc[model_name]

            # Explicit bias reduction percentage
            bias_reduc_pct = 0.0
            if base_dp_tier > 1e-5:
                bias_reduc_pct = float(((base_dp_tier - dp_diff) / base_dp_tier) * 100.0)

            eo_reduc_pct = 0.0
            if base_eo_tier > 1e-5:
                eo_reduc_pct = float(((base_eo_tier - eo_diff) / base_eo_tier) * 100.0)

            tradeoff_summary.append(
                {
                    "model": model_name,
                    "accuracy": round(acc, 4),
                    "dp_difference": round(dp_diff, 4),
                    "dp_ratio": round(m_tier["demographic_parity_ratio"], 4),
                    "equalized_odds_diff": round(eo_diff, 4),
                    "bias_reduction_percentage": round(bias_reduc_pct, 1),
                    "equalized_odds_reduction_pct": round(eo_reduc_pct, 1),
                }
            )

        audit_results["tradeoffs"]["college_tier"] = tradeoff_summary

        logger.info(
            f"\n{'='*70}\n"
            f"FAIRNESS AUDIT & MITIGATION BENCHMARK (College Tier):\n"
            + "\n".join(
                [
                    f"{t['model']:<34} | Acc: {t['accuracy']:.3f} | DP Gap: {t['dp_difference']:.3f} | Bias Reduc: {t['bias_reduction_percentage']:>5.1f}%"
                    for t in tradeoff_summary
                ]
            )
            + f"\n{'='*70}"
        )

        # 6. Generate Reports & Visualizations
        figures_dir = self.paths.get("figures_dir", "reports/figures")
        report_file = self.paths.get("fairness_report_file", "reports/fairness_report.md")
        generate_fairness_report(audit_results, output_md_path=report_file, figures_dir=figures_dir)

        # Save JSON audit results for Streamlit app and programmatic consumption
        audit_json_path = Path(self.paths.get("reports_dir", "reports")) / "fairness_audit.json"
        audit_json_path.parent.mkdir(parents=True, exist_ok=True)

        def _json_serialize(obj: Any) -> Any:
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(audit_json_path, "w", encoding="utf-8") as f:
            json.dump(audit_results, f, indent=2, default=_json_serialize)
        logger.info(f"Saved fairness audit JSON to {audit_json_path.resolve()}")

        return audit_results


def main() -> None:
    """CLI entrypoint for running the fairness audit."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive Fairlearn audit and mitigation analysis."
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config.yaml."
    )
    args = parser.parse_args()

    auditor = FairnessAuditor(config_path=args.config)
    auditor.run_full_audit()


if __name__ == "__main__":
    main()
