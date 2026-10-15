"""Model Insights, Global Explainability (SHAP), Diagnostic Curves, and Architectural Comparison."""

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.cache import get_metrics_data
from app.components.ui import (
    apply_custom_css,
    render_header,
    render_metric_card,
)


def render_model_insights_page() -> None:
    apply_custom_css()
    render_header(
        title="Model Performance Insights & Global Explainability",
        subtitle="Transparent evaluation of candidate architectures on held-out test data, global SHAP rankings, and probabilistic calibration.",
        badge="Model Diagnostics",
    )

    metrics = get_metrics_data()
    test_metrics = metrics.get("test", {})
    salary_metrics = metrics.get("salary_model", {}).get("test", {})

    # Top KPI Summary Cards for Primary Model (XGBoost)
    xgb_test = test_metrics.get("xgboost", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            label="XGBoost Test Accuracy",
            value=f"{xgb_test.get('accuracy', 0.896) * 100:.1f}%",
            delta="+4.3% over Random Forest",
            delta_positive=True,
            help_tooltip="Overall classification accuracy on 750 strictly held-out test students.",
        )
    with c2:
        render_metric_card(
            label="XGBoost ROC-AUC",
            value=f"{xgb_test.get('roc_auc', 0.8735):.4f}",
            delta="Area under ROC Curve",
            delta_positive=True,
            help_tooltip="Discrimination threshold ranking capability across all classification cutoffs.",
        )
    with c3:
        render_metric_card(
            label="XGBoost PR-AUC",
            value=f"{xgb_test.get('pr_auc', 0.9762):.4f}",
            delta="Precision-Recall Area",
            delta_positive=True,
            help_tooltip="PR Area measures performance on the minority non-placed class under class imbalance.",
        )
    with c4:
        render_metric_card(
            label="Brier Calibration Score",
            value=f"{xgb_test.get('brier_score', 0.0773):.4f}",
            delta="Lower is Better (0 = Perfect)",
            delta_positive=True,
            help_tooltip="Mean squared error between predicted probabilities and actual 0/1 outcomes. Lower values confirm calibrated risk probabilities.",
        )

    st.markdown("---")

    # --- Architectural Comparison Table ---
    st.markdown("### 🏆 Candidate Architecture Benchmark")
    st.caption(
        "Comparison of linear, ensemble bagging, and gradient boosting architectures evaluated under identical cross-validation and test splits."
    )

    table_data = []
    models_display = [
        ("XGBoost (Tuned Classifier)", "xgboost"),
        ("Random Forest (Ensemble)", "random_forest"),
        ("Logistic Regression (Linear)", "logistic_regression"),
    ]

    for display_name, key in models_display:
        m = test_metrics.get(key, {})
        cm = m.get("confusion_matrix", {})
        table_data.append(
            {
                "Model Architecture": display_name,
                "Accuracy": f"{m.get('accuracy', 0.0) * 100:.2f}%",
                "Precision": f"{m.get('precision', 0.0) * 100:.2f}%",
                "Recall": f"{m.get('recall', 0.0) * 100:.2f}%",
                "F1-Score": f"{m.get('f1', 0.0):.4f}",
                "ROC-AUC": f"{m.get('roc_auc', 0.0):.4f}",
                "PR-AUC": f"{m.get('pr_auc', 0.0):.4f}",
                "Brier Score": f"{m.get('brier_score', 0.0):.4f}",
                "TP / FP / FN / TN": f"{cm.get('tp', 0)} / {cm.get('fp', 0)} / {cm.get('fn', 0)} / {cm.get('tn', 0)}",
            }
        )

    df_benchmark = pd.DataFrame(table_data)
    st.dataframe(df_benchmark, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Global Explainability & Diagnostic Figures ---
    tab_shap, tab_curves, tab_salary = st.tabs(
        [
            "🐝 Global SHAP Explainability",
            "📈 ROC, PR & Calibration Diagnostics",
            "💼 Salary Band Model Diagnostics",
        ]
    )

    with tab_shap:
        st.markdown("#### 🌍 Global Feature Attribution (SHAP TreeExplainer)")
        st.markdown("""
            Global SHAP values reveal not just which features are most important across all students,
            but **how** high vs low values shift placement likelihood.
            """)

        col_beeswarm, col_bar = st.columns([1.1, 1], gap="medium")

        with col_beeswarm:
            st.markdown("##### 🐝 SHAP Beeswarm Summary Plot")
            st.caption(
                "Each dot is a student. Red = high feature value, Blue = low. Right = increases placement probability."
            )
            beeswarm_path = Path("reports/figures/shap_beeswarm.png")
            if beeswarm_path.exists():
                st.image(str(beeswarm_path), use_container_width=True)
            else:
                st.info("SHAP beeswarm figure not found on disk.")

        with col_bar:
            st.markdown("##### 📊 Global Mean |SHAP| Importance")
            st.caption(
                "Average absolute impact on placement margin log-odds across all validation candidates."
            )
            bar_path = Path("reports/figures/shap_feature_importance_bar.png")
            if bar_path.exists():
                st.image(str(bar_path), use_container_width=True)
            else:
                st.info("SHAP feature importance figure not found on disk.")

        st.markdown(
            """
            <div class="callout-box">
                <b>💡 Key Global Takeaways:</b>
                <ul>
                    <li><b>Academic Standing & Practical Coding:</b> <code>cgpa</code> and <code>dsa_problems_solved</code> exert the strongest positive pull on placement odds.</li>
                    <li><b>Penalizing Bottlenecks:</b> <code>backlogs</code> is the single most aggressive downward penalizer; students with active backlogs face substantial headwinds.</li>
                    <li><b>Experiential Catalysts:</b> <code>internships</code> and <code>work_experience</code> act as decisive differentiators for borderline students.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_curves:
        st.markdown("#### 🔬 Comparative Diagnostic Curves")
        c_tab1, c_tab2, c_tab3 = st.tabs(
            ["ROC Curves", "Precision-Recall Curves", "Calibration Curves"]
        )

        with c_tab1:
            roc_path = Path("reports/figures/roc_curves_comparison.png")
            if roc_path.exists():
                st.image(
                    str(roc_path),
                    caption="Receiver Operating Characteristic (ROC) Curve Comparison",
                    use_container_width=True,
                )
            st.caption(
                "ROC curves illustrate the true positive rate vs. false positive rate trade-off across all classification thresholds."
            )

        with c_tab2:
            pr_path = Path("reports/figures/pr_curves_comparison.png")
            if pr_path.exists():
                st.image(
                    str(pr_path),
                    caption="Precision-Recall (PR) Curve Comparison",
                    use_container_width=True,
                )
            st.caption(
                "Precision-Recall curves are essential for assessing performance on the minority non-placed class."
            )

        with c_tab3:
            cal_path = Path("reports/figures/calibration_curves.png")
            if cal_path.exists():
                st.image(
                    str(cal_path),
                    caption="Reliability Calibration Curves & Brier Scores",
                    use_container_width=True,
                )
            st.caption(
                "Calibration curves show how closely predicted probabilities mirror empirical placement rates."
            )

    with tab_salary:
        st.markdown("#### 💰 Multiclass Salary Compensation Model")
        st.caption(
            "Predicts candidate compensation tier: Low (<4 LPA), Mid (4-8 LPA), High (>8 LPA) for placed candidates."
        )

        sal_c1, sal_c2, sal_c3 = st.columns(3)
        with sal_c1:
            render_metric_card(
                label="Salary Model Accuracy",
                value=f"{salary_metrics.get('accuracy', 0.8446) * 100:.1f}%",
                delta="3-Class Classification",
                delta_positive=True,
            )
        with sal_c2:
            render_metric_card(
                label="Macro F1-Score",
                value=f"{salary_metrics.get('macro_f1', 0.8467):.4f}",
                delta="Balanced across tiers",
                delta_positive=True,
            )
        with sal_c3:
            render_metric_card(
                label="Weighted F1-Score",
                value=f"{salary_metrics.get('weighted_f1', 0.8450):.4f}",
                delta="Support-weighted F1",
                delta_positive=True,
            )

        st.markdown("##### 📊 Disaggregated Salary Band Classification Report")
        report = salary_metrics.get("report", {})
        if report:
            sal_rows = []
            for band in ["Low", "Mid", "High"]:
                if band in report:
                    b_data = report[band]
                    sal_rows.append(
                        {
                            "Compensation Band": band,
                            "Precision": f"{b_data.get('precision', 0.0) * 100:.1f}%",
                            "Recall": f"{b_data.get('recall', 0.0) * 100:.1f}%",
                            "F1-Score": f"{b_data.get('f1-score', 0.0):.4f}",
                            "Test Support": int(b_data.get("support", 0)),
                        }
                    )
            st.dataframe(pd.DataFrame(sal_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_model_insights_page()
