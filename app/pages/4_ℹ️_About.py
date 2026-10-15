"""About the Project: Data Provenance, Model Card, Architecture, and Governance Guidelines."""

import streamlit as st

from app.components.ui import (
    apply_custom_css,
    render_header,
)


def render_about_page() -> None:
    apply_custom_css()
    render_header(
        title="About the Project: Provenance, Architecture & Governance",
        subtitle="Transparent documentation of data generation, model cards, architectural design, and ethical AI safeguards.",
        badge="System Card & Provenance",
    )

    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top: 0; color: #6366F1;">🌟 Project Mission & Purpose</h3>
            <p style="color: #CBD5E1; font-size: 0.98rem; line-height: 1.6;">
                The <b>Campus Placement Predictor with Explainability and Fairness Audit</b> is an enterprise-grade,
                portfolio ML engineering project. Its goal is to demonstrate how modern machine learning systems in high-stakes
                education and recruitment settings must combine <b>high predictive accuracy</b> with <b>exact local explainability (SHAP)</b>
                and <b>rigorous algorithmic fairness auditing (Fairlearn)</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🗄️ Data Provenance & Honesty")
        st.markdown("""
            This project employs a two-tier data strategy designed for research rigor and honest provenance:

            1. **Real Baseline Dataset**:
               - Sourced from Kaggle: *Campus Recruitment (Academic and Employability Factors)* by Ben Roshan.
               - Contains 215 real undergraduate student records with MBA placement outcomes, secondary percentages, and specializations.
               - Downloadable and verified via `python -m src.data.download`.

            2. **Synthetic Simulation Dataset (5,000 Records)**:
               - Generated via `src.data.synthetic_generator.py` using multivariate distributions with realistic covariance structures.
               - Features include CGPA, competitive DSA problem counts, hackathon participation, communication ratings, and institutional tier.
               - **Configurable Institutional Bias Injection**: To establish a verifiable ground truth for algorithmic bias detection, the generator includes an explicit `--inject-bias` parameter where Tier 3 and female applicants receive systematic penalties at identical skill levels.

            > ⚠️ **Simulation Disclaimer**: All predictions generated in this application are probabilistic simulations and should not be used as sole determinants in real-world recruitment decisions.
            """)

    with col2:
        st.markdown("### 🏗️ Technical Architecture")
        st.markdown("""
            - **ML Core**: Python 3.14+, scikit-learn 1.7+, XGBoost 3.1+, Optuna 4.7+, Fairlearn 0.13+.
            - **Leak-Free Pipelines**: `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`, fitted strictly on training splits.
            - **Explainability**: SHAP `TreeExplainer` providing mathematical additivity checks:
              $$\\text{Model Margin (log-odds)} = \\text{Base Value} + \\sum_{i} \\phi_i$$
            - **Fairness Framework**: Disaggregated auditing via Fairlearn `MetricFrame`, evaluating Demographic Parity, Equalized Odds, and 3 mitigation paradigms (Reweighing, ThresholdOptimizer, ExponentiatedGradient).
            - **Serving**: Interactive Streamlit dashboard with Plotly charts and glassmorphism dark theme.
            - **CI/CD & Testing**: 38 automated unit tests with $\\ge 80\\%$ code coverage, Ruff linting, Black formatting, pre-commit hooks, and GitHub Actions.
            """)

    st.markdown("---")

    # Governance & Model Card Section
    st.markdown("### 🛡️ Model Governance Card & Ethical Boundaries")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            """
            <div class="glass-card" style="min-height: 250px;">
                <h4 style="color: #10B981; margin-top: 0;">✅ Intended Uses</h4>
                <ul style="color: #CBD5E1; font-size: 0.88rem; padding-left: 18px;">
                    <li>Career counselor guidance for university students.</li>
                    <li>Personalized identification of actionable skill improvements (DSA problem targets, internships).</li>
                    <li>Academic research on recruitment fairness auditing.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            """
            <div class="glass-card" style="min-height: 250px;">
                <h4 style="color: #F43F5E; margin-top: 0;">❌ Out-of-Scope Uses</h4>
                <ul style="color: #CBD5E1; font-size: 0.88rem; padding-left: 18px;">
                    <li>Automated resume screening or rejection without human recruiter review.</li>
                    <li>Deployment across non-engineering disciplines without recalibration.</li>
                    <li>Salary compensation bargaining or binding offer determinations.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            """
            <div class="glass-card" style="min-height: 250px;">
                <h4 style="color: #F59E0B; margin-top: 0;">⚠️ Known Limitations</h4>
                <ul style="color: #CBD5E1; font-size: 0.88rem; padding-left: 18px;">
                    <li>Omitted variables: soft skills, portfolio quality, open-source contributions.</li>
                    <li>Macroeconomic hiring freezes or sector-specific volatility not modeled.</li>
                    <li>Potential covariate shift between university curricula.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Artifact & Documentation Links
    st.markdown("### 📚 Repository Artifacts & Documentation")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(
            "📄 **[DATA_CARD.md](file:///d:/Desktop/Projects/AIML/placement-predictor/DATA_CARD.md)**"
        )
        st.caption("Detailed dataset provenance, schemas, and synthetic generation parameters.")
    with d2:
        st.markdown(
            "📄 **[MODEL_CARD.md](file:///d:/Desktop/Projects/AIML/placement-predictor/MODEL_CARD.md)**"
        )
        st.caption(
            "Model architecture, hyperparameters, validation scores, and fairness considerations."
        )
    with d3:
        st.markdown(
            "📄 **[fairness_report.md](file:///d:/Desktop/Projects/AIML/placement-predictor/reports/fairness_report.md)**"
        )
        st.caption("Full algorithmic fairness audit report generated by Fairlearn.")
    with d4:
        st.markdown(
            "📄 **[metrics.json](file:///d:/Desktop/Projects/AIML/placement-predictor/reports/metrics.json)**"
        )
        st.caption("Quantitative test evaluations, confusion matrices, and calibration data.")


if __name__ == "__main__":
    render_about_page()
