"""Model explainability via SHAP, probability-point conversions, counterfactual simulators, and actionable recourse."""

from src.explainability.counterfactual import WhatIfSimulator
from src.explainability.recourse import IMMUTABLE_FEATURES, RECOURSE_DISCLAIMER, RecourseEngine
from src.explainability.shap_global import generate_global_shap_plots, get_global_feature_importance
from src.explainability.shap_local import explain_student_prediction

__all__ = [
    "IMMUTABLE_FEATURES",
    "RECOURSE_DISCLAIMER",
    "RecourseEngine",
    "WhatIfSimulator",
    "explain_student_prediction",
    "generate_global_shap_plots",
    "get_global_feature_importance",
]
