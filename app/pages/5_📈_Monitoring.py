"""Streamlit page for production data drift monitoring, simulated cohorts, and retraining triggers."""

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import DataLoader
from src.models.registry import ModelRegistry
from src.monitoring.drift import (
    compute_feature_drift,
    compute_prediction_drift,
    evaluate_retraining_trigger,
    generate_evidently_html_report,
)
from src.monitoring.simulation import generate_simulated_cohort
from src.utils.config import load_config

st.set_page_config(
    page_title="Model Monitoring & Drift - PlaceWise",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Production Model Monitoring & Drift Governance")
st.markdown("""
    Continuous statistical surveillance of feature distributions, prediction shifts, and operational degradation.
    Compare incoming applicant batches against baseline training data to evaluate automated **Retraining Triggers**.
    """)


@st.cache_resource
def load_base_artifacts() -> tuple[Any, pd.DataFrame, list[str], list[str]]:
    """Loads reference data and production model pipeline."""
    config = load_config("config/config.yaml")
    feat_cfg = config.get("features", {})
    num_features = feat_cfg.get("numerical", [])
    cat_features = feat_cfg.get("categorical", [])

    loader = DataLoader(config_path="config/config.yaml")
    df_raw = loader.load_synthetic_data(biased=False)
    train_df, _, test_df = loader.get_train_val_test_split(df_raw, target_col="placed")

    registry = ModelRegistry(models_dir="models")
    pipeline, _ = registry.load_model("best_placement_model")

    return pipeline, train_df, num_features, cat_features


try:
    pipeline, train_df, num_features, cat_features = load_base_artifacts()
    all_features = num_features + cat_features
except Exception as e:
    st.error(f"Failed to load monitoring reference data: {e}")
    st.stop()

# Sidebar: Scenario Selection
st.sidebar.header("🧪 Drift Simulation Cohort")
scenario_map = {
    "Natural Variance (Normal Operation)": "normal",
    "Academic Inflation (CGPA Shift)": "cgpa_inflation",
    "Hiring Slump (Macroeconomic Downturn)": "hiring_slump",
    "Curriculum Modernization (Coding Surge)": "curriculum_modernization",
}

scenario_label = st.sidebar.selectbox(
    "Select Cohort Scenario",
    list(scenario_map.keys()),
    index=1,
)
selected_scenario = scenario_map[scenario_label]
sample_size = st.sidebar.slider(
    "Incoming Cohort Size", min_value=200, max_value=1500, value=600, step=100
)

st.sidebar.markdown("---")
st.sidebar.markdown(r"""
    **Governance Policies**:
    - Critical Retrain Threshold: **Drift Share $\ge 30\%$**
    - Prediction Drift Significance: **$p < 0.01$**
    - Accuracy Drop Tolerance: **$5.0\%$**
    """)

# Generate cohort
with st.spinner("Analyzing cohort distributions..."):
    cohort_df = generate_simulated_cohort(
        base_df=train_df, scenario=selected_scenario, sample_size=sample_size, random_seed=42
    )
    feature_drift = compute_feature_drift(train_df, cohort_df, num_features, cat_features)
    pred_drift = compute_prediction_drift(pipeline, train_df, cohort_df, all_features)
    retrain_decision = evaluate_retraining_trigger(feature_drift, pred_drift)

# Executive Alert Banner
if retrain_decision["retraining_recommended"]:
    if retrain_decision["urgency_level"] == "CRITICAL_RETRAIN":
        st.error(f"🚨 **{retrain_decision['status_label']}** (Urgency: Critical)")
    else:
        st.warning(f"⚠️ **{retrain_decision['status_label']}** (Urgency: Warning)")
    for reason in retrain_decision["triggered_reasons"]:
        st.markdown(f"- ⚠️ {reason}")
    st.info(f"**Operational Protocol**: {retrain_decision['guidance']}")
else:
    st.success(
        f"✅ **{retrain_decision['status_label']}** — All distribution metrics within governance tolerances."
    )
    st.caption(retrain_decision["guidance"])

st.markdown("---")

# Metrics Summary Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Feature Drift Share",
        f"{feature_drift['drift_share_pct']}%",
        delta=f"{feature_drift['drifted_features_count']} of {feature_drift['total_features']} features",
        delta_color="inverse" if feature_drift["dataset_drift_detected"] else "normal",
    )
with col2:
    st.metric(
        "Dataset Drift Status",
        "DETECTED" if feature_drift["dataset_drift_detected"] else "NOT DETECTED",
        help="Flagged when drift share exceeds 30.0%",
    )
with col3:
    st.metric(
        "Prediction Shift p-value",
        f"{pred_drift['p_value']:.4f}",
        delta="Significant" if pred_drift["prediction_drift_detected"] else "Stable",
        delta_color="inverse" if pred_drift["prediction_drift_detected"] else "normal",
    )
with col4:
    st.metric(
        "Cohort Selection Rate",
        f"{pred_drift['current_selection_rate']*100:.1f}%",
        delta=f"{(pred_drift['current_selection_rate'] - pred_drift['reference_selection_rate'])*100:+.1f}% vs baseline",
    )

st.markdown("### 📋 Disaggregated Feature Drift Table")

# Format feature metrics table
table_rows = []
for feat, data in feature_drift["features"].items():
    p_val = data["p_value"]
    stat = data["statistic"]
    is_drift = data["drift_detected"]
    m_shift = data.get("mean_shift", "N/A")

    table_rows.append(
        {
            "Feature": feat,
            "Type": data["type"],
            "Statistical Test": data["test_name"],
            "Test Stat": stat,
            "p-value": f"{p_val:.5f}",
            "Mean Shift": m_shift,
            "Status": "🔴 DRIFTED" if is_drift else "🟢 STABLE",
        }
    )

df_table = pd.DataFrame(table_rows)
st.dataframe(df_table, use_container_width=True, hide_index=True)

# Feature Distribution Comparison
st.markdown("### 📊 Distribution Comparison (Reference vs. Incoming Cohort)")
selected_feature = st.selectbox("Select Feature to Inspect", num_features, index=0)

if selected_feature in num_features:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=train_df[selected_feature],
            name="Baseline Reference (Training)",
            opacity=0.6,
            marker_color="#2b5c8f",
            nbinsx=30,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=cohort_df[selected_feature],
            name=f"Current Cohort ({scenario_label})",
            opacity=0.6,
            marker_color="#e06666",
            nbinsx=30,
        )
    )
    fig.update_layout(
        barmode="overlay",
        title=f"Distribution Comparison: {selected_feature}",
        xaxis_title=selected_feature,
        yaxis_title="Count",
        legend=dict(x=0.7, y=0.95),
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

# Interactive Report Generation
st.markdown("### 📑 Full Evidently HTML Audit Report")
html_report_path = Path("reports/drift_report.html")

col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("Generate Fresh HTML Report"):
        with st.spinner("Generating Evidently HTML report..."):
            generate_evidently_html_report(train_df, cohort_df, all_features, html_report_path)
            st.success("Evidently HTML report generated!")

if html_report_path.exists():
    with open(html_report_path, encoding="utf-8") as f:
        html_bytes = f.read()
    st.download_button(
        label="📥 Download Evidently Drift Report (HTML)",
        data=html_bytes,
        file_name="drift_report.html",
        mime="text/html",
    )
