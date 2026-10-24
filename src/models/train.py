"""End-to-end model training, hyperparameter optimization, evaluation, and artifact serialization."""

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.loader import DataLoader
from src.features.pipeline import create_preprocessing_pipeline, get_feature_names
from src.models.evaluate import (
    evaluate_binary_classifier,
    evaluate_multiclass_classifier,
    plot_calibration_curves,
    plot_roc_pr_curves,
)
from src.models.registry import ModelRegistry
from src.models.tracking import (
    evaluate_and_promote_model,
    get_data_hash,
    get_git_commit_hash,
    setup_mlflow,
)
from src.models.tune import tune_xgboost
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

logger = get_logger(__name__)


def train_models(
    config_path: str | Path = "config/config.yaml",
    tune: bool = True,
    trials: int | None = None,
    use_biased_data: bool = False,
) -> dict[str, Any]:
    """Orchestrates end-to-end training, Optuna tuning, evaluation, and artifact saving.

    Args:
        config_path: Path to master configuration YAML.
        tune: Whether to run Optuna hyperparameter optimization.
        trials: Number of Optuna trials (overrides config if provided).
        use_biased_data: If True, trains on the biased synthetic dataset.

    Returns:
        dict: Complete evaluation metrics across validation and held-out test sets.
    """
    config = load_config(config_path)
    seed = config.get("project", {}).get("random_seed", 42)
    seed_everything(seed)

    paths = config.get("paths", {})
    feat_cfg = config.get("features", {})
    modeling_cfg = config.get("modeling", {})

    num_features = feat_cfg.get("numerical", [])
    cat_features = feat_cfg.get("categorical", [])
    target_placement = feat_cfg.get("targets", {}).get("placement", "placed")
    target_salary = feat_cfg.get("targets", {}).get("salary", "salary_band")

    # 1. Load Data
    loader = DataLoader(config_path=config_path)
    df = loader.load_synthetic_data(biased=use_biased_data, auto_generate=True)
    logger.info(f"Loaded dataset for training: {len(df)} records (biased={use_biased_data})")

    # 2. Stratified Train / Val / Test Split (Strictly non-leaking)
    train_df, val_df, test_df = loader.get_train_val_test_split(
        df, target_col=target_placement, stratify=True, random_seed=seed
    )

    X_train = train_df[num_features + cat_features]
    y_train = train_df[target_placement]

    X_val = val_df[num_features + cat_features]
    y_val = val_df[target_placement]

    X_test = test_df[num_features + cat_features]
    y_test = test_df[target_placement]

    logger.info(
        f"Data splits prepared: Train={len(X_train)} (Positive rate: {y_train.mean():.1%}), "
        f"Val={len(X_val)} (Positive rate: {y_val.mean():.1%}), "
        f"Test={len(X_test)} (Positive rate: {y_test.mean():.1%})"
    )

    # MLflow Setup & Run Tracking
    mlflow_cfg = config.get("mlflow", {})
    setup_mlflow(config)
    git_commit = get_git_commit_hash()
    data_file_path = Path(paths.get("synthetic_dir", "data/synthetic")) / (
        config.get("data", {}).get("synthetic_biased_file", "synthetic_placement_data_biased.csv")
        if use_biased_data
        else config.get("data", {}).get("synthetic_file", "synthetic_placement_data.csv")
    )
    data_hash = get_data_hash(data_file_path)

    models_predictions_val: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    models_predictions_test: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    results: dict[str, Any] = {"validation": {}, "test": {}, "salary_model": {}}

    run_name = f"xgb_train_{'biased' if use_biased_data else 'unbiased'}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("data_hash", data_hash)
        mlflow.set_tag("dataset_type", "biased" if use_biased_data else "unbiased")
        mlflow.log_params(
            {
                "random_seed": seed,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "numerical_features_count": len(num_features),
                "categorical_features_count": len(cat_features),
            }
        )

        # =========================================================================
        # Model 1: Logistic Regression Baseline (Balanced Class Weights)
        # =========================================================================
        logger.info("Training Baseline 1: Logistic Regression...")
        preprocessor_lr = create_preprocessing_pipeline(num_features, cat_features)
        lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        lr_pipeline = Pipeline([("preprocessor", preprocessor_lr), ("classifier", lr_clf)])
        lr_pipeline.fit(X_train, y_train)

        val_prob_lr = lr_pipeline.predict_proba(X_val)[:, 1]
        test_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

        models_predictions_val["Logistic Regression"] = (y_val.values, val_prob_lr)
        models_predictions_test["Logistic Regression"] = (y_test.values, test_prob_lr)

        results["validation"]["logistic_regression"] = evaluate_binary_classifier(
            y_val, val_prob_lr
        )
        results["test"]["logistic_regression"] = evaluate_binary_classifier(y_test, test_prob_lr)

        # =========================================================================
        # Model 2: Random Forest Baseline
        # =========================================================================
        logger.info("Training Baseline 2: Random Forest Classifier...")
        preprocessor_rf = create_preprocessing_pipeline(num_features, cat_features)
        rf_clf = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=seed, n_jobs=-1
        )
        rf_pipeline = Pipeline([("preprocessor", preprocessor_rf), ("classifier", rf_clf)])
        rf_pipeline.fit(X_train, y_train)

        val_prob_rf = rf_pipeline.predict_proba(X_val)[:, 1]
        test_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

        models_predictions_val["Random Forest"] = (y_val.values, val_prob_rf)
        models_predictions_test["Random Forest"] = (y_test.values, test_prob_rf)

        results["validation"]["random_forest"] = evaluate_binary_classifier(y_val, val_prob_rf)
        results["test"]["random_forest"] = evaluate_binary_classifier(y_test, test_prob_rf)

        # =========================================================================
        # Model 3: Primary Model: XGBoost (with Optuna Hyperparameter Optimization)
        # =========================================================================
        best_xgb_params = modeling_cfg.get("models", {}).get("xgboost", {})
        if tune:
            n_trials = trials or modeling_cfg.get("optuna_trials", 25)
            best_xgb_params, best_cv_score = tune_xgboost(
                X_train=X_train,
                y_train=y_train,
                numerical_features=num_features,
                categorical_features=cat_features,
                n_trials=n_trials,
                cv_folds=modeling_cfg.get("cv_folds", 5),
                random_seed=seed,
            )
            results["optuna_tuning"] = {
                "best_cv_roc_auc": best_cv_score,
                "best_params": best_xgb_params,
            }
            mlflow.log_metric("optuna_best_cv_roc_auc", float(best_cv_score))

        logger.info(f"Training Tuned XGBoost model with parameters: {best_xgb_params}...")
        preprocessor_xgb = create_preprocessing_pipeline(num_features, cat_features)
        xgb_clf = XGBClassifier(**best_xgb_params)
        xgb_pipeline = Pipeline([("preprocessor", preprocessor_xgb), ("classifier", xgb_clf)])
        xgb_pipeline.fit(X_train, y_train)

        val_prob_xgb = xgb_pipeline.predict_proba(X_val)[:, 1]
        test_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

        models_predictions_val["Tuned XGBoost"] = (y_val.values, val_prob_xgb)
        models_predictions_test["Tuned XGBoost"] = (y_test.values, test_prob_xgb)

        results["validation"]["xgboost"] = evaluate_binary_classifier(y_val, val_prob_xgb)
        results["test"]["xgboost"] = evaluate_binary_classifier(y_test, test_prob_xgb)

        # Log performance comparison
        logger.info(
            f"\n{'='*70}\n"
            f"HELD-OUT TEST SET EVALUATION SUMMARY:\n"
            f"Logistic Regression: ROC-AUC={results['test']['logistic_regression']['roc_auc']:.4f}, "
            f"F1={results['test']['logistic_regression']['f1']:.4f}, "
            f"Brier={results['test']['logistic_regression']['brier_score']:.4f}\n"
            f"Random Forest:       ROC-AUC={results['test']['random_forest']['roc_auc']:.4f}, "
            f"F1={results['test']['random_forest']['f1']:.4f}, "
            f"Brier={results['test']['random_forest']['brier_score']:.4f}\n"
            f"Tuned XGBoost:       ROC-AUC={results['test']['xgboost']['roc_auc']:.4f}, "
            f"F1={results['test']['xgboost']['f1']:.4f}, "
            f"Brier={results['test']['xgboost']['brier_score']:.4f}\n"
            f"{'='*70}"
        )

        # =========================================================================
        # Model 4: Salary Band Multiclass Classifier (Placed Candidates Only)
        # =========================================================================
        logger.info("Training Second Model: Multiclass Salary Band Classifier...")
        train_placed = train_df[train_df[target_placement] == 1]
        val_placed = val_df[val_df[target_placement] == 1]
        test_placed = test_df[test_df[target_placement] == 1]

        X_train_sal = train_placed[num_features + cat_features]
        y_train_sal = train_placed[target_salary]
        X_val_sal = val_placed[num_features + cat_features]
        y_val_sal = val_placed[target_salary]
        X_test_sal = test_placed[num_features + cat_features]
        y_test_sal = test_placed[target_salary]

        preprocessor_sal = create_preprocessing_pipeline(num_features, cat_features)
        sal_clf = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=seed)
        salary_pipeline = Pipeline([("preprocessor", preprocessor_sal), ("classifier", sal_clf)])
        salary_pipeline.fit(X_train_sal, y_train_sal)

        val_sal_pred = salary_pipeline.predict(X_val_sal)
        test_sal_pred = salary_pipeline.predict(X_test_sal)

        val_sal_metrics = evaluate_multiclass_classifier(y_val_sal, val_sal_pred)
        test_sal_metrics = evaluate_multiclass_classifier(y_test_sal, test_sal_pred)

        results["salary_model"]["validation"] = val_sal_metrics
        results["salary_model"]["test"] = test_sal_metrics
        logger.info(
            f"Salary Band Classifier Test Metrics: Accuracy={test_sal_metrics['accuracy']:.4f}, "
            f"Macro-F1={test_sal_metrics['macro_f1']:.4f}"
        )

        # =========================================================================
        # Evaluation Figures Generation
        # =========================================================================
        figures_dir = paths.get("figures_dir", "reports/figures")
        plot_roc_pr_curves(models_predictions_test, output_dir=figures_dir)
        plot_calibration_curves(models_predictions_test, output_dir=figures_dir)

        # =========================================================================
        # Model Artifact & Metadata Persistence
        # =========================================================================
        registry = ModelRegistry(models_dir=paths.get("models_dir", "models"))
        fitted_preprocessor = xgb_pipeline.named_steps["preprocessor"]
        transformed_feature_names = get_feature_names(
            fitted_preprocessor, num_features, cat_features
        )

        metadata_placement = {
            "model_type": "XGBoostClassifier",
            "random_seed": seed,
            "features": {
                "numerical": num_features,
                "categorical": cat_features,
            },
            "transformed_feature_names": transformed_feature_names,
            "hyperparameters": best_xgb_params,
            "metrics_test": results["test"]["xgboost"],
            "dataset_type": "biased" if use_biased_data else "unbiased",
            "git_commit": git_commit,
            "data_hash": data_hash,
            "mlflow_run_id": run.info.run_id,
        }
        model_filename = (
            "best_placement_model_biased" if use_biased_data else "best_placement_model"
        )
        registry.save_model(xgb_pipeline, model_filename, metadata_placement)

        metadata_salary = {
            "model_type": "GradientBoostingClassifier",
            "random_seed": seed,
            "features": {
                "numerical": num_features,
                "categorical": cat_features,
            },
            "metrics_test": results["salary_model"]["test"],
            "classes": list(salary_pipeline.classes_),
            "mlflow_run_id": run.info.run_id,
        }
        registry.save_model(salary_pipeline, "salary_band_model", metadata_salary)

        # Save metrics.json
        metrics_path = Path(paths.get("metrics_file", "reports/metrics.json"))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Exported evaluation metrics to {metrics_path.resolve()}")

        # Log artifacts and metrics to MLflow
        for fig_name in ["roc_curves.png", "pr_curves.png", "calibration_curves.png"]:
            fig_path = Path(figures_dir) / fig_name
            if fig_path.exists():
                mlflow.log_artifact(str(fig_path), artifact_path="figures")

        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path), artifact_path="reports")

        mlflow.log_metrics(
            {
                "test_xgb_accuracy": results["test"]["xgboost"]["accuracy"],
                "test_xgb_precision": results["test"]["xgboost"]["precision"],
                "test_xgb_recall": results["test"]["xgboost"]["recall"],
                "test_xgb_f1": results["test"]["xgboost"]["f1"],
                "test_xgb_roc_auc": results["test"]["xgboost"]["roc_auc"],
                "test_xgb_pr_auc": results["test"]["xgboost"]["pr_auc"],
                "test_xgb_brier_score": results["test"]["xgboost"]["brier_score"],
                "val_xgb_roc_auc": results["validation"]["xgboost"]["roc_auc"],
                "test_rf_roc_auc": results["test"]["random_forest"]["roc_auc"],
                "test_lr_roc_auc": results["test"]["logistic_regression"]["roc_auc"],
                "salary_test_accuracy": results["salary_model"]["test"]["accuracy"],
                "salary_test_macro_f1": results["salary_model"]["test"]["macro_f1"],
            }
        )

        # Log model artifact to MLflow using cloudpickle serialization
        mlflow.sklearn.log_model(
            sk_model=xgb_pipeline,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        # Register best model with Staging/Production promotion
        if not use_biased_data:
            model_reg_name = mlflow_cfg.get("registered_model_name", "PlaceWise_XGBoost_Placement")
            promotion_thresholds = mlflow_cfg.get("promotion", {})
            promotion_info = evaluate_and_promote_model(
                model_name=model_reg_name,
                run_id=run.info.run_id,
                metrics=results["test"]["xgboost"],
                thresholds=promotion_thresholds,
            )
            results["model_registry"] = promotion_info
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    return results


def main() -> None:
    """CLI entrypoint for model training."""
    parser = argparse.ArgumentParser(
        description="Train, tune, and evaluate placement prediction models."
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config.yaml."
    )
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna hyperparameter tuning.")
    parser.add_argument("--trials", type=int, default=None, help="Number of Optuna tuning trials.")
    parser.add_argument(
        "--biased", action="store_true", help="Train on the biased synthetic dataset."
    )
    args = parser.parse_args()

    train_models(
        config_path=args.config,
        tune=not args.no_tune,
        trials=args.trials,
        use_biased_data=args.biased,
    )


if __name__ == "__main__":
    main()
