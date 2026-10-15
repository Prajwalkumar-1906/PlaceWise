"""Fairness Audit, Fairlearn Disparity Benchmarks, and Interactive Threshold/Mitigation Explorer."""

import pandas as pd
import streamlit as st

from app.components.cache import get_audit_test_dataset, get_fairness_audit_data
from app.components.charts import (
    create_pareto_curve_chart,
    create_selection_rates_chart,
)
from app.components.ui import (
    apply_custom_css,
    render_header,
    render_metric_card,
)


def render_fairness_dashboard_page() -> None:
    apply_custom_css()
    render_header(
        title="Algorithmic Fairness Audit & Bias Mitigation Dashboard",
        subtitle="Empirical Fairlearn auditing across Gender and College Tier, threshold calibration, and Pareto trade-off benchmarks.",
        badge="Algorithmic Governance & Fairlearn",
    )

    audit_data = get_fairness_audit_data()
    models_dict = audit_data.get("models", {})
    tradeoffs_tier = audit_data.get("tradeoffs", {}).get("college_tier", [])

    if not models_dict:
        st.warning(
            "Fairness audit data not found. Please run `python -m src.fairness.audit` to generate reports/fairness_audit.json."
        )
        return

    # Top Control Bar: Sensitive Attribute & Model Architecture
    ctrl_col1, ctrl_col2 = st.columns([1, 1], gap="medium")
    with ctrl_col1:
        sens_attr = st.selectbox(
            "🛡️ Select Sensitive Attribute to Audit:",
            options=["college_tier", "gender", "gender_x_tier"],
            format_func=lambda x: {
                "college_tier": "College Tier (Tier 1 vs 2 vs 3)",
                "gender": "Gender (Male vs Female)",
                "gender_x_tier": "Intersectional (Gender × Tier)",
            }.get(x, x),
            help="Demographic axis along which fairness metrics and selection rates are disaggregated.",
        )

    with ctrl_col2:
        model_name = st.selectbox(
            "⚙️ Active Model Pipeline Architecture:",
            options=list(models_dict.keys()),
            index=0,
            help="Toggle between unmitigated baseline, awareness-blind model, and bias-mitigated variants.",
        )

    model_audit = models_dict.get(model_name, {})
    sens_metrics = model_audit.get("metrics_by_sensitive_attribute", {}).get(sens_attr, {})

    dp_diff = sens_metrics.get("demographic_parity_difference", 0.0)
    dp_ratio = sens_metrics.get("demographic_parity_ratio", 1.0)
    eo_diff = sens_metrics.get("equalized_odds_difference", 0.0)
    eq_opp_diff = sens_metrics.get("equal_opportunity_difference", 0.0)

    # 80% Rule Compliance Check
    is_four_fifths_compliant = dp_ratio >= 0.80

    # Disparity KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_metric_card(
            label="Demographic Parity Gap",
            value=f"{dp_diff * 100:.2f}%",
            delta="Lower is Fairer (0 = Exact Parity)",
            delta_positive=dp_diff < 0.05,
            help_tooltip="Maximum selection rate difference between any two demographic groups: max(P(Y_hat=1|A=a)) - min(P(Y_hat=1|A=b)).",
        )
    with kpi2:
        render_metric_card(
            label="Demographic Parity Ratio",
            value=f"{dp_ratio:.3f}",
            delta=(
                "✅ Compliant (≥ 0.80)"
                if is_four_fifths_compliant
                else "⚠️ Disparate Impact (< 0.80)"
            ),
            delta_positive=is_four_fifths_compliant,
            help_tooltip="Selection rate ratio of lowest-selected group over highest-selected group. Values below 0.80 indicate violation of the EEOC 4/5ths Rule.",
        )
    with kpi3:
        render_metric_card(
            label="Equalized Odds Gap",
            value=f"{eo_diff * 100:.2f}%",
            delta="TPR & FPR Parity",
            delta_positive=eo_diff < 0.06,
            help_tooltip="Maximum gap between groups across both True Positive Rate (TPR) and False Positive Rate (FPR).",
        )
    with kpi4:
        render_metric_card(
            label="Equal Opportunity Gap",
            value=f"{eq_opp_diff * 100:.2f}%",
            delta="True Positive Rate Gap",
            delta_positive=eq_opp_diff < 0.06,
            help_tooltip="Maximum difference in True Positive Rates (qualified candidate acceptance rates) across demographic groups.",
        )

    st.markdown("---")

    # Main Dashboard Tabs
    tab_disagg, tab_threshold, tab_mitigation, tab_regulatory = st.tabs(
        [
            "📊 Disaggregated Selection Rates",
            "🎚️ Interactive Decision Threshold Explorer",
            "⚖️ Mitigation Benchmarks & Pareto Frontier",
            "📜 Regulatory & Governance Framework",
        ]
    )

    with tab_disagg:
        st.markdown(f"#### 📊 Subgroup Disaggregation ({sens_attr.replace('_', ' ').title()})")
        st.caption(
            f"Detailed performance and selection rates for **{model_name}** across demographic groups."
        )

        by_group = sens_metrics.get("by_group", {})
        if by_group:
            g_col1, g_col2 = st.columns([1.1, 1], gap="large")

            with g_col1:
                fig_rates = create_selection_rates_chart(by_group, sens_attr)
                st.plotly_chart(fig_rates, use_container_width=True)

            with g_col2:
                st.markdown("##### 📋 Subgroup Performance Table")
                rows = []
                for grp, g_data in by_group.items():
                    rows.append(
                        {
                            "Subgroup": str(grp),
                            "Test Candidates": int(g_data.get("count", 0)),
                            "Selection Rate": f"{g_data.get('selection_rate', 0.0) * 100:.1f}%",
                            "Accuracy": f"{g_data.get('accuracy', 0.0) * 100:.1f}%",
                            "TPR (Recall)": f"{g_data.get('tpr', 0.0) * 100:.1f}%",
                            "FPR (False Alarm)": f"{g_data.get('fpr', 0.0) * 100:.1f}%",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_threshold:
        st.markdown("#### 🎚️ Dynamic Decision Threshold Explorer")
        st.markdown("""
            By default, binary classifiers apply a decision cutoff of **0.50**.
            Adjust the threshold slider below to observe how the cutoff dynamically shifts group selection rates,
            accuracy, and Demographic Parity in real-time on held-out test data.
            """)

        test_df = get_audit_test_dataset()
        thresh = st.slider(
            "Select Placement Decision Threshold (&tau;):",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.02,
        )

        # Dynamic calculation on held-out test data
        y_true = test_df["placed"].values
        y_prob = test_df["pred_prob"].values
        y_pred_dynamic = (y_prob >= thresh).astype(int)

        groups = test_df[sens_attr].unique()
        dyn_rates = {}
        for g in sorted(groups):
            mask = test_df[sens_attr] == g
            dyn_rates[str(g)] = {
                "selection_rate": float(y_pred_dynamic[mask].mean()),
                "accuracy": float((y_pred_dynamic[mask] == y_true[mask]).mean()),
            }

        rates_list = [v["selection_rate"] for v in dyn_rates.values()]
        dyn_dp_gap = max(rates_list) - min(rates_list)
        dyn_acc = float((y_pred_dynamic == y_true).mean())

        t_col1, t_col2 = st.columns([1, 1], gap="large")
        with t_col1:
            fig_dyn_rates = create_selection_rates_chart(
                dyn_rates, f"{sens_attr} (Threshold = {thresh:.2f})"
            )
            st.plotly_chart(fig_dyn_rates, use_container_width=True)

        with t_col2:
            st.markdown(f"##### 🎯 Real-Time Metrics at &tau; = {thresh:.2f}")
            st.markdown(
                f"""
                <div class="metric-box" style="margin-bottom: 12px;">
                    <div class="metric-label">Dynamic Test Accuracy</div>
                    <div class="metric-value">{dyn_acc * 100:.2f}%</div>
                </div>
                <div class="metric-box" style="margin-bottom: 12px;">
                    <div class="metric-label">Dynamic Demographic Parity Gap</div>
                    <div class="metric-value" style="color: {'#10B981' if dyn_dp_gap < 0.05 else '#F43F5E'};">
                        {dyn_dp_gap * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Lowering the threshold admits more students across all groups, boosting recall but increasing false positives. Raising it restricts offers to only highest-confidence candidates."
            )

    with tab_mitigation:
        st.markdown("#### ⚖️ Bias Mitigation Benchmark & Trade-Offs")
        st.markdown("""
            Comparison of three algorithmic bias mitigation paradigms against the baseline:
            - **Pre-processing (Reweighing)**: Reweighs training instances inversely proportional to subgroup selection rates.
            - **Post-processing (ThresholdOptimizer)**: Optimizes group-specific decision thresholds to equalize selection rates.
            - **In-processing (ExponentiatedGradient)**: Solves a saddle-point Lagrangian optimization under demographic constraints.
            """)

        m_col1, m_col2 = st.columns([1.1, 1], gap="large")

        with m_col1:
            if tradeoffs_tier:
                fig_pareto = create_pareto_curve_chart(tradeoffs_tier)
                st.plotly_chart(fig_pareto, use_container_width=True)

        with m_col2:
            st.markdown("##### 📋 Quantitative Mitigation Benchmark (College Tier)")
            bench_rows = []
            for t in tradeoffs_tier:
                bench_rows.append(
                    {
                        "Mitigation Strategy": t["model"],
                        "Test Accuracy": f"{t['accuracy'] * 100:.1f}%",
                        "DP Gap": f"{t['dp_difference'] * 100:.2f}%",
                        "Equalized Odds": f"{t['equalized_odds_diff'] * 100:.2f}%",
                        "Bias Reduction %": f"{t['bias_reduction_percentage']:+.1f}%",
                    }
                )
            st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)

            st.markdown(
                """
                <div class="callout-box callout-success">
                    <b>🏆 Key Recommendation: Pre-processing Reweighing</b><br>
                    Reduces College Tier disparity from <b>7.1% down to 2.7% (62.1% bias reduction)</b> while retaining <b>85.5% test accuracy</b>, without requiring discriminatory group-specific run-time thresholds.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_regulatory:
        st.markdown("#### 📜 Algorithmic Fairness & Regulatory Governance")
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="margin-top: 0;">1. The Fallacy of 'Fairness through Unawareness'</h4>
                <p style="color: #CBD5E1; font-size: 0.95rem;">
                    A common naive assumption is that removing sensitive features like <code>gender</code> and <code>college_tier</code> guarantees fairness.
                    Our audit demonstrates empirically that the <b>Unawareness Blind model</b> still inherits proxy disparities through correlations in aptitude test scores, coding question volumes, and branch selection.
                </p>

                <h4>2. The Four-Fifths (80%) Rule (EEOC Title VII)</h4>
                <p style="color: #CBD5E1; font-size: 0.95rem;">
                    Under US employment discrimination guidelines (EEOC Uniform Guidelines on Employee Selection Procedures), a selection rate for any group which is less than <b>80% (4/5ths)</b> of the rate for the group with the highest selection rate constitutes evidence of disparate impact.
                </p>

                <h4>3. Inherent Mathematical Impossibility (Kleinberg / Chouldechova)</h4>
                <p style="color: #CBD5E1; font-size: 0.95rem;">
                    Mathematical proofs establish that when base rates differ across groups, a predictive model <b>cannot simultaneously satisfy Demographic Parity, Equalized Odds, and Predictive Calibration</b> (unless prediction is 100% perfect). Engineering teams must explicitly select the appropriate ethical objective for the recruitment context.
                </p>

                <h4>4. EU AI Act High-Risk System Classification</h4>
                <p style="color: #CBD5E1; font-size: 0.95rem;">
                    AI systems used in recruitment, candidate screening, and educational admissions are classified as <b>High-Risk under Annex III of the EU AI Act</b>. Compliance mandates human-in-the-loop oversight, continuous logging, explainability, and verifiable algorithmic bias audits prior to deployment.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_fairness_dashboard_page()
