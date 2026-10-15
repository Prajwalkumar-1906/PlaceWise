"""Plotly-based interactive charting components for probabilities, SHAP, and fairness."""

from typing import Any

import plotly.graph_objects as go

DARK_THEME_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Plus Jakarta Sans, sans-serif", "color": "#E2E8F0"},
}


def create_gauge_chart(probability: float, title: str = "Placement Probability") -> go.Figure:
    """Renders a sleek semicircular gauge for prediction confidence."""
    pct = probability * 100

    # Color mapping based on probability zones
    if pct >= 75:
        bar_color = "#10B981"
    elif pct >= 50:
        bar_color = "#F59E0B"
    else:
        bar_color = "#F43F5E"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 42, "weight": 700, "color": "#FFFFFF"}},
            title={"text": f"<b>{title}</b>", "font": {"size": 15, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B", "ticks": ""},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255, 255, 255, 0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(244, 63, 94, 0.12)"},
                    {"range": [50, 75], "color": "rgba(245, 158, 11, 0.12)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.12)"},
                ],
                "threshold": {
                    "line": {"color": "#6366F1", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        height=240,
        **DARK_THEME_LAYOUT,
        margin={"t": 30, "b": 10, "l": 20, "r": 20},
    )
    return fig


def create_salary_prob_bar(probabilities: dict[str, float]) -> go.Figure:
    """Renders a modern horizontal bar chart for salary band distribution."""
    bands = ["Low (<4 LPA)", "Mid (4-8 LPA)", "High (>8 LPA)"]
    keys = ["Low", "Mid", "High"]
    values = [probabilities.get(k, 0.0) * 100 for k in keys]
    colors = ["#38BDF8", "#A855F7", "#10B981"]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=bands,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
            textfont=dict(weight=600, color="#FFFFFF"),
        )
    )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=dict(
            text="<b>Predicted Salary Compensation Band</b>", font=dict(size=14, color="#94A3B8")
        ),
        xaxis=dict(
            range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False
        ),
        yaxis=dict(autorange="reversed", showgrid=False),
        height=200,
        margin={"t": 35, "b": 20, "l": 110, "r": 20},
    )
    return fig


def create_shap_waterfall_chart(
    feature_impacts: list[dict[str, Any]],
    base_prob: float,
    final_prob: float,
    max_display: int = 7,
) -> go.Figure:
    """Renders a Plotly waterfall-style bar chart representing local SHAP probability contributions."""
    top_impacts = feature_impacts[:max_display]

    labels = ["Base Rate"] + [f["feature"] for f in top_impacts] + ["Final Prediction"]
    values = [base_prob * 100] + [f["prob_delta_percentage_points"] for f in top_impacts]

    # Calculate measures for waterfall
    measures = ["absolute"] + ["relative"] * len(top_impacts) + ["total"]

    text_labels = [f"{base_prob * 100:.1f}%"]
    for f in top_impacts:
        sign = "+" if f["prob_delta_percentage_points"] >= 0 else ""
        text_labels.append(f"{sign}{f['prob_delta_percentage_points']:.1f} pts")
    text_labels.append(f"{final_prob * 100:.1f}%")

    fig = go.Figure(
        go.Waterfall(
            name="SHAP Driver",
            orientation="v",
            measure=measures,
            x=labels,
            textposition="outside",
            text=text_labels,
            y=values,
            connector={"line": {"color": "rgba(255,255,255,0.2)"}},
            decreasing={"marker": {"color": "#F43F5E"}},
            increasing={"marker": {"color": "#10B981"}},
            totals={"marker": {"color": "#6366F1"}},
            textfont={"size": 11, "color": "#F1F5F9"},
        )
    )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=dict(
            text="<b>SHAP Local Attribution (Impact in Probability Points)</b>",
            font=dict(size=14, color="#94A3B8"),
        ),
        yaxis=dict(title="Probability (%)", range=[0, 115], gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(tickangle=-25),
        height=380,
    )
    return fig


def create_what_if_comparison_chart(base_prob: float, sim_prob: float) -> go.Figure:
    """Renders an interactive comparison between baseline and counterfactual student probability."""
    delta = (sim_prob - base_prob) * 100
    delta_sign = "+" if delta >= 0 else ""

    categories = ["Current Profile", "Simulated What-If"]
    probs = [base_prob * 100, sim_prob * 100]
    colors = ["#64748B", "#10B981" if delta >= 0 else "#F43F5E"]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=probs,
            text=[f"{p:.1f}%" for p in probs],
            textposition="auto",
            textfont=dict(size=14, weight=700, color="#FFFFFF"),
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            width=0.45,
        )
    )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=dict(
            text=f"<b>Counterfactual Impact: {delta_sign}{delta:.1f} percentage points</b>",
            font=dict(size=14, color="#10B981" if delta >= 0 else "#F43F5E"),
        ),
        yaxis=dict(
            title="Placement Probability (%)", range=[0, 110], gridcolor="rgba(255,255,255,0.06)"
        ),
        height=260,
    )
    return fig


def create_pareto_curve_chart(tradeoff_data: list[dict[str, Any]]) -> go.Figure:
    """Renders Accuracy vs. Demographic Parity Gap trade-off scatter plot."""
    models = [d["model"] for d in tradeoff_data]
    accuracies = [d["accuracy"] * 100 for d in tradeoff_data]
    dp_gaps = [d["dp_difference"] * 100 for d in tradeoff_data]

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=dp_gaps,
            y=accuracies,
            mode="markers+text",
            text=models,
            textposition="top center",
            textfont=dict(size=12, color="#F8FAFC", weight=600),
            marker=dict(
                size=14,
                color=["#94A3B8", "#38BDF8", "#10B981", "#F59E0B", "#A855F7"],
                line=dict(color="#FFFFFF", width=1.5),
            ),
            hoverinfo="text",
            hovertext=[
                f"<b>{m}</b><br>Accuracy: {acc:.1f}%<br>Demographic Parity Gap: {gap:.1f}%"
                for m, acc, gap in zip(models, accuracies, dp_gaps)
            ],
        )
    )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=dict(
            text="<b>Accuracy vs. Fairness Disparity Trade-Off (Pareto Frontier)</b>",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="Demographic Parity Gap (%) — (Lower is Fairer)",
            gridcolor="rgba(255,255,255,0.06)",
            autorange="reversed",  # Lower disparity on the right (desirable)
        ),
        yaxis=dict(
            title="Test Accuracy (%) — (Higher is Better)", gridcolor="rgba(255,255,255,0.06)"
        ),
        height=420,
    )
    return fig


def create_selection_rates_chart(
    by_group_dict: dict[str, dict[str, float]], attribute_name: str
) -> go.Figure:
    """Renders a bar chart of selection rates across subgroups with 80% rule reference."""
    groups = list(by_group_dict.keys())
    rates = [by_group_dict[g].get("selection_rate", 0.0) * 100 for g in groups]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=groups,
            y=rates,
            text=[f"{r:.1f}%" for r in rates],
            textposition="auto",
            textfont=dict(weight=600, color="#FFFFFF"),
            marker=dict(
                color=["#6366F1", "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B"][: len(groups)],
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
        )
    )

    max_rate = max(rates) if rates else 100
    four_fifths_threshold = max_rate * 0.80

    fig.add_hline(
        y=four_fifths_threshold,
        line_dash="dot",
        line_color="#F59E0B",
        annotation_text=f"80% Parity Guideline ({four_fifths_threshold:.1f}%)",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color="#F59E0B"),
    )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=dict(
            text=f"<b>Selection Rates by {attribute_name.replace('_', ' ').title()}</b>",
            font=dict(size=14),
        ),
        yaxis=dict(title="Selection Rate (%)", range=[0, 105], gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(title="Subgroup"),
        height=320,
    )
    return fig
