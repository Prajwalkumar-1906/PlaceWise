"""Design system, custom CSS injection, and reusable HTML/UI components for Streamlit."""

import streamlit as st


def apply_custom_css() -> None:
    """Injects a modern, polished dark-mode design system with glassmorphism styling."""
    custom_css = """
    <style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #0B0F19;
        --card-bg: rgba(22, 30, 49, 0.75);
        --card-border: rgba(255, 255, 255, 0.08);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --accent-indigo: #6366F1;
        --accent-purple: #8B5CF6;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose: #F43F5E;
        --accent-cyan: #06B6D4;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1200px !important;
    }

    /* Glassmorphism Card Container */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Metric Cards */
    .metric-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: left;
    }

    .metric-label {
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }

    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 6px;
    }

    .delta-positive { color: #10B981; }
    .delta-negative { color: #F43F5E; }
    .delta-neutral { color: #94A3B8; }

    /* Risk Status Pills */
    .risk-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .risk-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .risk-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .risk-red {
        background: rgba(244, 63, 94, 0.15);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }

    /* Badges */
    .badge-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: rgba(99, 102, 241, 0.18);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Alert / Explanatory Callout */
    .callout-box {
        border-left: 4px solid var(--accent-indigo);
        background: rgba(99, 102, 241, 0.08);
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0;
        font-size: 0.92rem;
        color: #CBD5E1;
        line-height: 1.5;
    }

    .callout-warning {
        border-left-color: var(--accent-amber);
        background: rgba(245, 158, 11, 0.08);
    }

    .callout-success {
        border-left-color: var(--accent-emerald);
        background: rgba(16, 185, 129, 0.08);
    }

    /* Subdued Divider */
    hr {
        border: 0;
        height: 1px;
        background: rgba(255, 255, 255, 0.08);
        margin: 24px 0;
    }

    /* Custom Streamlit component overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: var(--text-secondary);
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.2) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge: str | None = None) -> None:
    """Renders a standard hero header with title, subtitle, and an optional category badge."""
    badge_html = (
        f'<span class="badge-chip">{badge}</span><div style="height: 8px;"></div>' if badge else ""
    )
    header_html = f"""
    <div style="margin-bottom: 24px;">
        {badge_html}
        <h1 style="font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 6px;
                   background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {title}
        </h1>
        <p style="font-size: 1.02rem; color: #94A3B8; margin: 0; line-height: 1.5;">
            {subtitle}
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def get_risk_badge_html(probability: float) -> str:
    """Generates an HTML risk level pill badge based on placement probability."""
    if probability >= 0.75:
        return '<span class="risk-pill risk-green">🟢 High Placement Likelihood (Low Risk)</span>'
    elif probability >= 0.50:
        return '<span class="risk-pill risk-amber">🟡 Competitive / Moderate Likelihood</span>'
    else:
        return '<span class="risk-pill risk-red">🔴 High Risk / Targeted Intervention Needed</span>'


def render_metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_positive: bool | None = None,
    help_tooltip: str | None = None,
) -> None:
    """Renders a styled KPI metric box."""
    delta_html = ""
    if delta:
        delta_class = "delta-neutral"
        if delta_positive is True:
            delta_class = "delta-positive"
        elif delta_positive is False:
            delta_class = "delta-negative"
        delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>'

    help_attr = (
        f'title="{help_tooltip}" style="cursor: help; text-decoration: underline dotted;"'
        if help_tooltip
        else ""
    )

    card_html = f"""
    <div class="metric-box">
        <div class="metric-label" {help_attr}>{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_provenance_footer() -> None:
    """Renders a global provenance footer displaying active model version, registry stage, and synthetic data disclosure."""
    footer_html = """
    <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.08);
                display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
                font-size: 0.82rem; color: #64748B;">
        <div>
            <span style="font-weight: 600; color: #94A3B8;">PlaceWise v0.1.0</span> |
            <span>Model: <code>PlaceWise_XGBoost_Placement v4 (Production)</code></span> |
            <span>Backend: <code>XGBoost + Platt Scaling</code></span>
        </div>
        <div>
            <span style="background: rgba(99, 102, 241, 0.15); color: #818CF8; padding: 3px 8px; border-radius: 6px; font-weight: 500;">
                Synthetic Simulation Cohort (Non-PII)
            </span>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
