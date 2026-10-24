import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from app.components.cache import get_placement_predictor, get_preset_profiles
from app.components.charts import (
    create_gauge_chart,
    create_salary_prob_bar,
    create_shap_waterfall_chart,
    create_what_if_comparison_chart,
)
from app.components.ui import (
    apply_custom_css,
    get_risk_badge_html,
    render_header,
    render_metric_card,
    render_provenance_footer,
)
from src.data.validation import ALLOWED_BRANCHES, StudentRecord
from src.explainability.counterfactual import WhatIfSimulator
from src.explainability.recourse import RecourseEngine
from src.explainability.shap_local import explain_student_prediction
from src.utils.pdf_generator import generate_student_pdf_dossier


def render_predict_page() -> None:
    apply_custom_css()
    render_header(
        title="Student Placement Assessment & Advisory",
        subtitle="Calibrated inference, uncertainty quantification (HITL abstention), local SHAP attribution, and actionable recourse pathways.",
        badge="Inference & Advisory",
    )

    predictor = get_placement_predictor()
    preset_profiles = get_preset_profiles()

    num_features = [
        "cgpa",
        "tenth_pct",
        "twelfth_pct",
        "backlogs",
        "dsa_problems_solved",
        "projects",
        "internships",
        "certifications",
        "hackathons",
        "communication_score",
        "aptitude_score",
    ]
    cat_features = ["branch", "college_tier", "gender", "work_experience"]

    top_mode = st.radio(
        "Assessment Mode",
        ["👤 Single Student Assessment", "📂 Batch CSV Processing"],
        horizontal=True,
    )

    if top_mode == "📂 Batch CSV Processing":
        st.markdown("### 📂 Batch Candidate Processing & Tier Analysis")
        st.caption(
            "Upload a CSV file containing applicant profiles for automated scoring and counselor review routing."
        )

        uploaded_file = st.file_uploader("Upload Applicants CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(df_input)}** records from `{uploaded_file.name}`.")

                with st.spinner("Scoring batch and calculating uncertainty bands..."):
                    df_scored = predictor.predict_batch(df_input)

                placed_count = int(df_scored["predicted_placement"].sum())
                review_count = int(df_scored["needs_human_review"].sum())

                # Batch summary metrics
                b1, b2, b3 = st.columns(3)
                with b1:
                    render_metric_card("Total Applicants", f"{len(df_scored)}")
                with b2:
                    render_metric_card(
                        "Predicted Placed",
                        f"{placed_count}",
                        delta=f"{placed_count/len(df_scored)*100:.1f}%",
                    )
                with b3:
                    render_metric_card(
                        "Counselor Review Required",
                        f"{review_count}",
                        delta="Abstention Zone [35%-65%]",
                        delta_positive=False if review_count > 0 else True,
                    )

                st.markdown("#### Scored Candidate Results")
                st.dataframe(df_scored, use_container_width=True)

                csv_bytes = df_scored.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Scored Batch Results (CSV)",
                    data=csv_bytes,
                    file_name="scored_placement_batch.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error processing batch CSV: {e}")

        render_provenance_footer()
        return

    # --- Single Profile Preset Selector ---
    col_preset, _ = st.columns([2, 1])
    with col_preset:
        selected_preset = st.selectbox(
            "Quick-Fill Profile Benchmark Preset",
            options=list(preset_profiles.keys()),
            index=0,
            help="Pre-loads realistic student archetypes to test model behavior across tiers.",
        )

    profile = preset_profiles[selected_preset]

    # --- Profile Input Form & Live Gauges ---
    col_form, col_results = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("### 📝 Candidate Attributes")
        with st.form("student_profile_form"):
            st.markdown("##### 🏛️ Academic & Institutional Background")
            c1, c2, c3 = st.columns(3)
            with c1:
                college_tier = st.selectbox(
                    "College Tier",
                    options=[1, 2, 3],
                    index=int(profile["college_tier"]) - 1,
                    format_func=lambda x: f"Tier {x}",
                )
            with c2:
                branch = st.selectbox(
                    "Engineering Branch",
                    options=ALLOWED_BRANCHES,
                    index=ALLOWED_BRANCHES.index(profile["branch"]),
                )
            with c3:
                gender = st.selectbox(
                    "Gender",
                    options=["Male", "Female"],
                    index=0 if profile["gender"] == "Male" else 1,
                )

            a1, a2, a3 = st.columns(3)
            with a1:
                cgpa = st.number_input(
                    "CGPA",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(profile["cgpa"]),
                    step=0.05,
                    format="%.2f",
                )
            with a2:
                tenth_pct = st.number_input(
                    "10th Grade %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(profile["tenth_pct"]),
                    step=0.5,
                )
            with a3:
                twelfth_pct = st.number_input(
                    "12th Grade %",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(profile["twelfth_pct"]),
                    step=0.5,
                )

            st.markdown("##### 💻 Technical Skills, Co-Curricular & Experience")
            t1, t2, t3 = st.columns(3)
            with t1:
                dsa_problems = st.number_input(
                    "DSA Problems Solved",
                    min_value=0,
                    max_value=2500,
                    value=int(profile["dsa_problems_solved"]),
                    step=25,
                )
            with t2:
                projects = st.number_input(
                    "Software Projects",
                    min_value=0,
                    max_value=20,
                    value=int(profile["projects"]),
                    step=1,
                )
            with t3:
                internships = st.number_input(
                    "Internships",
                    min_value=0,
                    max_value=10,
                    value=int(profile["internships"]),
                    step=1,
                )

            t4, t5, t6 = st.columns(3)
            with t4:
                aptitude = st.slider(
                    "Aptitude Test Score",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(profile["aptitude_score"]),
                    step=1.0,
                )
            with t5:
                communication = st.slider(
                    "Communication Score",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(profile["communication_score"]),
                    step=1.0,
                )
            with t6:
                work_exp = st.selectbox(
                    "Work Experience",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    index=int(profile["work_experience"]),
                )

            t7, t8, t9 = st.columns(3)
            with t7:
                hackathons = st.number_input(
                    "Hackathons",
                    min_value=0,
                    max_value=20,
                    value=int(profile["hackathons"]),
                    step=1,
                )
            with t8:
                certifications = st.number_input(
                    "Certifications",
                    min_value=0,
                    max_value=20,
                    value=int(profile["certifications"]),
                    step=1,
                )
            with t9:
                backlogs = st.number_input(
                    "Current Backlogs",
                    min_value=0,
                    max_value=20,
                    value=int(profile["backlogs"]),
                    step=1,
                )

            st.form_submit_button(
                "⚡ Run Real-Time Assessment", use_container_width=True, type="primary"
            )

    # Assemble Student Record
    student_dict = {
        "cgpa": cgpa,
        "tenth_pct": tenth_pct,
        "twelfth_pct": twelfth_pct,
        "backlogs": backlogs,
        "dsa_problems_solved": dsa_problems,
        "projects": projects,
        "internships": internships,
        "certifications": certifications,
        "hackathons": hackathons,
        "communication_score": communication,
        "aptitude_score": aptitude,
        "branch": branch,
        "college_tier": college_tier,
        "gender": gender,
        "work_experience": work_exp,
    }

    # Validate Schema
    try:
        StudentRecord(**student_dict)
    except ValidationError as err:
        st.error(f"Input validation error: {err}")
        return

    # Run Prediction
    pred_res = predictor.predict_student(student_dict)
    prob = pred_res["placement_probability"]
    pred_class = pred_res["placement_prediction"]
    salary_band = pred_res["salary_band"]
    needs_review = pred_res.get("needs_human_review", False)
    conf_band = pred_res.get("confidence_band", "High Confidence")

    # Compute Salary Probs if placed
    df_single = pd.DataFrame([student_dict])
    salary_probs = {"Low": 0.33, "Mid": 0.33, "High": 0.34}
    if predictor.salary_pipeline is not None:
        try:
            s_probs = predictor.salary_pipeline.predict_proba(df_single)[0]
            classes = list(predictor.salary_pipeline.classes_)
            salary_probs = {c: float(p) for c, p in zip(classes, s_probs)}
        except Exception:
            pass

    # --- Render Predictions ---
    with col_results:
        st.markdown("### 🎯 Assessment Results")

        # Gauge chart
        fig_gauge = create_gauge_chart(prob)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Status & Risk Badge
        st.markdown(
            f"<div style='text-align: center; margin-bottom: 12px;'>{get_risk_badge_html(prob)}</div>",
            unsafe_allow_html=True,
        )

        # Uncertainty & HITL Banner
        if needs_review:
            st.warning(
                f"⚠️ **Model Abstention Zone ({conf_band})**: Placement likelihood ({prob*100:.1f}%) lies in the uncertain decision interval [35%, 65%]. Academic counselor review recommended."
            )
        else:
            st.success(f"✅ **{conf_band}**: Statistical margin exceeds human review threshold.")

        # Quick Metric Cards
        m1, m2 = st.columns(2)
        with m1:
            render_metric_card(
                label="Placement Status",
                value="Likely Placed" if pred_class == 1 else "Unlikely Placed",
                delta="Cutoff: 50.0%",
                delta_positive=pred_class == 1,
            )
        with m2:
            render_metric_card(
                label="Predicted Salary Band",
                value=f"{salary_band}",
                delta="Low (<4 LPA) | Mid (4-8) | High (>8)",
                delta_positive=True if salary_band in ["Mid", "High"] else None,
            )

        # Salary Band Distribution Chart (if likely placed)
        if pred_class == 1:
            fig_salary = create_salary_prob_bar(salary_probs)
            st.plotly_chart(fig_salary, use_container_width=True)

    st.markdown("---")

    # --- Explainability, What-If & Recourse Tabs ---
    tab_whatif, tab_shap, tab_recourse = st.tabs(
        [
            "💡 What-If Simulator",
            "🔍 SHAP Explainability",
            "🌱 Actionable Recourse Pathways",
        ]
    )

    with tab_whatif:
        st.markdown("#### 🔬 Actionable Counterfactual Simulation")
        st.caption(
            "Adjust actionable student levers to observe how incremental efforts improve placement probability."
        )

        w_col1, w_col2 = st.columns([1, 1], gap="large")
        simulator = WhatIfSimulator(pipeline=predictor.placement_pipeline)

        with w_col1:
            st.markdown("##### 🛠️ Adjust Levers")
            mod_dsa = st.slider(
                "Modify DSA Problems",
                min_value=0,
                max_value=1200,
                value=dsa_problems,
                step=25,
            )
            mod_projects = st.slider(
                "Modify Projects",
                min_value=0,
                max_value=10,
                value=projects,
                step=1,
            )
            mod_internships = st.slider(
                "Modify Internships",
                min_value=0,
                max_value=5,
                value=internships,
                step=1,
            )
            mod_comm = st.slider(
                "Modify Communication Score",
                min_value=0.0,
                max_value=100.0,
                value=float(communication),
                step=2.5,
            )

        with w_col2:
            st.markdown("##### 📈 Simulated Outcome Comparison")
            mods = {
                "dsa_problems_solved": mod_dsa,
                "projects": mod_projects,
                "internships": mod_internships,
                "communication_score": mod_comm,
            }
            sim_res = simulator.simulate(student_dict, mods)

            fig_comp = create_what_if_comparison_chart(
                baseline_prob=sim_res["baseline_probability"],
                modified_prob=sim_res["modified_probability"],
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            delta_sign = "+" if sim_res["delta_probability_points"] >= 0 else ""
            st.markdown(
                f"""
                <div class="callout-box callout-info">
                    <b>Outcome Shift:</b> {sim_res['status_shift']}<br/>
                    <b>Net Probability Impact:</b> <span style="font-weight: 700; color: #818CF8;">{delta_sign}{sim_res['delta_probability_points']:.1f} percentage points</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Compute SHAP explanation
    shap_exp = explain_student_prediction(
        pipeline=predictor.placement_pipeline,
        student_data=student_dict,
        numerical_features=num_features,
        categorical_features=cat_features,
        top_n=4,
    )

    with tab_shap:
        st.markdown("#### 🔍 SHAP Local Attribution (TreeExplainer)")
        st.caption(
            "Exact decomposition of how this student's features shifted the model prediction away from the baseline campus rate."
        )

        additivity = shap_exp["additivity_check"]
        if additivity["is_additive"]:
            st.markdown(
                f"""
                <div class="callout-box callout-success">
                    <b>✅ Mathematical Additivity Verified:</b> Model log-odds ({shap_exp['model_margin_log_odds']:.4f}) equals Base Value ({shap_exp['base_value_log_odds']:.4f}) + &sum; &phi;<sub>i</sub> with numerical error of only {additivity['additivity_error']:.2e}.
                </div>
                """,
                unsafe_allow_html=True,
            )

        s_col1, s_col2 = st.columns([1.1, 1], gap="medium")

        with s_col1:
            fig_waterfall = create_shap_waterfall_chart(
                feature_impacts=shap_exp["feature_impacts"],
                base_prob=shap_exp["base_probability"],
                final_prob=shap_exp["model_probability"],
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

        with s_col2:
            st.markdown("##### 📌 Plain-English Factor Breakdown")
            st.markdown("###### 🟢 Top Positive Drivers (Boosted Probability):")
            for pos in shap_exp["top_positive_drivers"]:
                pts = pos["prob_delta_percentage_points"]
                st.markdown(
                    f"- **{pos['feature']}**: <span style='color: #10B981; font-weight: 600;'>+{pts:.1f} pts</span> ({pos['formatted_value']})",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("###### 🔴 Top Negative Drivers (Penalized Probability):")
            for neg in shap_exp["top_negative_drivers"]:
                pts = neg["prob_delta_percentage_points"]
                st.markdown(
                    f"- **{neg['feature']}**: <span style='color: #F43F5E; font-weight: 600;'>{pts:.1f} pts</span> ({neg['formatted_value']})",
                    unsafe_allow_html=True,
                )

    with tab_recourse:
        st.markdown("#### 🌱 Actionable Developmental Recourse Pathways")
        st.caption(
            "Strategic, feasibility-constrained interventions that strictly lock demographic and institutional attributes."
        )

        recourse_engine = RecourseEngine(pipeline=predictor.placement_pipeline)
        recourse_res = recourse_engine.generate_recourse_pathways(
            student_dict, target_probability=0.65
        )

        if recourse_res["target_already_satisfied"]:
            st.success(f"🎉 {recourse_res['message']}")
        else:
            for p in recourse_res["pathways"]:
                with st.expander(
                    f"🛣️ {p['pathway_name']} — Achieves {p['achieved_probability']*100:.1f}% (+{p['probability_gain_pts']} pts) | Feasibility: {p['feasibility_rating']}",
                    expanded=True,
                ):
                    st.markdown(f"*{p['description']}*")
                    for inv in p["interventions"]:
                        st.markdown(
                            f"- **{inv['description']}**: `{inv['delta']}` (From **{inv['current_value']}** to **{inv['recommended_value']}**) &bull; *Effort: {inv['effort_points']} pts*"
                        )
                    st.caption(
                        "🔒 Immutable features guaranteed locked: Gender, College Tier, Branch, 10th %, 12th %."
                    )

        st.caption(recourse_res["disclaimer"])

    # --- PDF Dossier Download ---
    st.markdown("---")
    st.markdown("### 📄 Formal Student Evaluation Dossier")
    pdf_bytes = generate_student_pdf_dossier(
        student_data=student_dict,
        prediction_result=pred_res,
        explanation_data=shap_exp,
        recourse_data=recourse_res,
    )

    st.download_button(
        label="📥 Download Student Readiness Dossier (PDF)",
        data=pdf_bytes,
        file_name=f"Placement_Dossier_{branch.replace(' ', '_')}_Tier{college_tier}.pdf",
        mime="application/pdf",
    )

    render_provenance_footer()


if __name__ == "__main__":
    render_predict_page()
