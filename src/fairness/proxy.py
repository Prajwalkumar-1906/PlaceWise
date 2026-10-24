"""Statistical proxy detection: evaluating sensitive attribute leakage from non-sensitive features."""

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from src.features.pipeline import create_preprocessing_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_proxy_leakage(
    df: pd.DataFrame,
    sensitive_col: str,
    feature_cols: list[str],
    random_seed: int = 42,
) -> dict[str, Any]:
    """Trains a gradient boosting model to reconstruct the sensitive attribute from non-sensitive features.

    A high ROC-AUC (> 0.70) demonstrates substantial proxy leakage, confirming that
    'fairness through unawareness' (simply omitting the protected attribute) fails to prevent disparate impact.

    Args:
        df: Dataset containing sensitive and non-sensitive features.
        sensitive_col: Name of the sensitive column (e.g. 'gender', 'college_tier').
        feature_cols: List of non-sensitive feature column names.
        random_seed: Random seed.

    Returns:
        dict containing proxy_roc_auc, accuracy, macro_f1, and top predictive proxy features.
    """
    logger.info(f"Running proxy leakage detection for sensitive attribute '{sensitive_col}'...")

    # Exclude sensitive column if accidentally present in feature_cols
    clean_feature_cols = [c for c in feature_cols if c != sensitive_col]

    # Split numerical and categorical features
    num_features = [c for c in clean_feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_features = [c for c in clean_feature_cols if c not in num_features]

    X = df[clean_feature_cols].copy()
    y = df[sensitive_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_seed, stratify=y
    )

    preprocessor = create_preprocessing_pipeline(num_features, cat_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Determine binary vs multiclass
    unique_classes = np.unique(y)
    is_binary = len(unique_classes) == 2

    clf = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        random_state=random_seed,
        verbosity=-1,
    )
    clf.fit(X_train_proc, y_train)

    y_pred = clf.predict(X_test_proc)
    y_prob = np.asarray(clf.predict_proba(X_test_proc))

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    if is_binary:
        # Binary ROC-AUC
        pos_class = unique_classes[1]
        y_test_bin = (y_test == pos_class).astype(int)
        prob_pos = y_prob[:, 1]
        auc_score = float(roc_auc_score(y_test_bin, prob_pos))
    else:
        # Multiclass One-vs-Rest Macro ROC-AUC
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        try:
            auc_score = float(roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            auc_score = 0.50

    # Extract top proxy features from model importances
    transformed_feature_names: list[str] = []
    try:
        transformed_feature_names = [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        transformed_feature_names = clean_feature_cols

    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:5]
    top_features = [
        {
            "feature": (
                transformed_feature_names[i]
                if i < len(transformed_feature_names)
                else f"feature_{i}"
            ),
            "importance": int(importances[i]),
        }
        for i in top_indices
    ]

    leakage_verdict = (
        "High Proxy Leakage (Fairness through unawareness strictly fails)"
        if auc_score >= 0.70
        else "Moderate / Low Proxy Leakage"
    )

    logger.info(
        f"Proxy Detection for '{sensitive_col}': ROC-AUC = {auc_score:.4f}, Accuracy = {acc:.4f} ({leakage_verdict})"
    )

    return {
        "sensitive_attribute": sensitive_col,
        "is_binary": is_binary,
        "classes": [str(c) for c in unique_classes],
        "proxy_roc_auc": round(auc_score, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "verdict": leakage_verdict,
        "top_predictive_proxies": top_features,
    }
