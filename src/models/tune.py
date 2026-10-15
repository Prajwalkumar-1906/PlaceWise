"""Optuna hyperparameter tuning using Stratified Cross-Validation without data leakage."""

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.features.pipeline import create_preprocessing_pipeline
from src.models.evaluate import evaluate_binary_classifier
from src.utils.logger import get_logger

# Suppress Optuna verbose logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger(__name__)


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numerical_features: list[str],
    categorical_features: list[str],
    n_trials: int = 25,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> tuple[dict[str, Any], float]:
    """Tunes XGBoost hyperparameters using Stratified K-Fold CV to maximize ROC-AUC.

    The preprocessing pipeline is fit strictly within each fold training split
    to ensure absolute prevention of data leakage.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        n_trials: Number of Optuna optimization trials.
        cv_folds: Number of stratified folds.
        random_seed: Random seed.

    Returns:
        tuple (best_params_dict, best_mean_cv_roc_auc)
    """
    logger.info(
        f"Starting Optuna tuning for XGBoost ({n_trials} trials, {cv_folds}-fold Stratified CV)..."
    )

    # Compute scale_pos_weight for class imbalance handling
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = float(num_neg / max(num_pos, 1))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.20, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "gamma": trial.suggest_float("gamma", 0.0, 2.5),
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_seed,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        fold_scores: list[float] = []

        for _, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_tr, X_fold_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_tr, y_fold_va = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Fit preprocessor strictly on this fold's training split
            fold_preprocessor = create_preprocessing_pipeline(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
            )
            X_tr_proc = fold_preprocessor.fit_transform(X_fold_tr)
            X_va_proc = fold_preprocessor.transform(X_fold_va)

            clf = XGBClassifier(**params)
            clf.fit(X_tr_proc, y_fold_tr)

            y_va_prob = clf.predict_proba(X_va_proc)[:, 1]
            metrics = evaluate_binary_classifier(y_fold_va, y_va_prob)
            fold_scores.append(metrics["roc_auc"])

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"] = random_seed
    best_params["eval_metric"] = "logloss"
    best_params["n_jobs"] = -1

    logger.info(f"Optuna tuning completed. Best CV ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {best_params}")

    return best_params, float(study.best_value)
