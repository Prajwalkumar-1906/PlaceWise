"""Unit tests for Streamlit application components, charts, and data caching."""

import plotly.graph_objects as go
import pytest

from app.components.cache import (
    get_fairness_audit_data,
    get_metrics_data,
    get_preset_profiles,
)
from app.components.charts import (
    create_gauge_chart,
    create_pareto_curve_chart,
    create_salary_prob_bar,
    create_selection_rates_chart,
    create_shap_waterfall_chart,
    create_what_if_comparison_chart,
)
from app.components.ui import get_risk_badge_html
from src.data.validation import StudentRecord


def test_get_risk_badge_html():
    """Verify risk badge generation across probability thresholds."""
    badge_high = get_risk_badge_html(0.85)
    assert "risk-green" in badge_high
    assert "High Placement Likelihood" in badge_high

    badge_mid = get_risk_badge_html(0.60)
    assert "risk-amber" in badge_mid
    assert "Moderate Likelihood" in badge_mid

    badge_low = get_risk_badge_html(0.35)
    assert "risk-red" in badge_low
    assert "High Risk" in badge_low


def test_preset_profiles_schema():
    """Verify that all pre-configured student personas validate against Pydantic schema."""
    presets = get_preset_profiles()
    assert len(presets) >= 4

    for _name, profile in presets.items():
        record = StudentRecord(**profile)
        assert record.cgpa >= 0.0
        assert record.college_tier in [1, 2, 3]
        assert record.branch is not None


def test_cached_metrics_and_audit():
    """Verify metrics and audit loaders retrieve populated dictionaries."""
    metrics = get_metrics_data()
    assert isinstance(metrics, dict)
    assert "test" in metrics

    audit = get_fairness_audit_data()
    assert isinstance(audit, dict)
    assert "models" in audit


def test_charts_gauge():
    """Verify gauge chart generation."""
    fig = create_gauge_chart(0.82, "Custom Probability")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "indicator"
    assert fig.data[0].value == pytest.approx(82.0)


def test_charts_salary_bar():
    """Verify salary probability bar chart generation."""
    probs = {"Low": 0.20, "Mid": 0.50, "High": 0.30}
    fig = create_salary_prob_bar(probs)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"
    assert len(fig.data[0].x) == 3


def test_charts_shap_waterfall():
    """Verify local SHAP waterfall chart generation."""
    impacts = [
        {"feature": "cgpa (8.5)", "prob_delta_percentage_points": 12.5},
        {"feature": "backlogs (2)", "prob_delta_percentage_points": -8.0},
    ]
    fig = create_shap_waterfall_chart(impacts, base_prob=0.65, final_prob=0.695)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "waterfall"
    assert len(fig.data[0].x) == 4  # Base + 2 features + Final


def test_charts_what_if():
    """Verify counterfactual comparison chart."""
    fig = create_what_if_comparison_chart(0.55, 0.80)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"
    assert len(fig.data[0].x) == 2


def test_charts_pareto_curve():
    """Verify Pareto trade-off curve generation."""
    tradeoffs = [
        {"model": "Model A", "accuracy": 0.88, "dp_difference": 0.12},
        {"model": "Model B", "accuracy": 0.85, "dp_difference": 0.04},
    ]
    fig = create_pareto_curve_chart(tradeoffs)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "scatter"


def test_charts_selection_rates():
    """Verify group selection rates bar chart."""
    by_group = {
        "Tier 1": {"selection_rate": 0.85},
        "Tier 2": {"selection_rate": 0.75},
        "Tier 3": {"selection_rate": 0.70},
    }
    fig = create_selection_rates_chart(by_group, "college_tier")
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"
    assert len(fig.data[0].x) == 3
