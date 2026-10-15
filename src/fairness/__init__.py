"""Fairness auditing, metric computation, mitigation algorithms, and report generation."""

from src.fairness.audit import FairnessAuditor
from src.fairness.metrics import compute_fairness_metrics, create_intersectional_attribute
from src.fairness.mitigation import (
    MitigationManager,
    apply_reweighing,
    fit_exponentiated_gradient,
    fit_threshold_optimizer,
)
from src.fairness.report import generate_fairness_report

__all__ = [
    "FairnessAuditor",
    "MitigationManager",
    "apply_reweighing",
    "compute_fairness_metrics",
    "create_intersectional_attribute",
    "fit_exponentiated_gradient",
    "fit_threshold_optimizer",
    "generate_fairness_report",
]
