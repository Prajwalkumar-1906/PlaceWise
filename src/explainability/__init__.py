"""Model explainability via SHAP, probability-point conversions, and counterfactual simulators."""

from src.explainability.counterfactual import WhatIfSimulator
from src.explainability.shap_global import generate_global_shap_plots, get_global_feature_importance
from src.explainability.shap_local import explain_student_prediction

__all__ = [
    "WhatIfSimulator",
    "explain_student_prediction",
    "generate_global_shap_plots",
    "get_global_feature_importance",
]
