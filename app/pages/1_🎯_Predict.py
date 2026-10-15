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
)
from src.data.validation import ALLOWED_BRANCHES, StudentRecord
from src.explainability.counterfactual import WhatIfSimulator
from src.explainability.shap_local import explain_student_prediction


def render_predict_page() -> None:
    apply_custom_css()
    render_header(
        title="Student Placement Likelihood & What-If Simulator",
        subtitle="Real-time multi-task inference (placement probability + salary band), local SHAP attribution, and actionable counterfactual levers.",
        badge="Inference & Explainability",
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

    # --- Profile Preset Selector ---
    col_preset, _ = st.columns([2, 1])
    with col_preset:
        selected_preset = st.selectbox(
            "📋 Load Pre-Configured Student Persona:",
            options=["Custom Student Profile"] + list(preset_profiles.keys()),
            help="Choose an archetype to immediately populate the application form.",
        )

    # Initialize or update active profile from preset
    if selected_preset != "Custom Student Profile":
        profile = preset_profiles[selected_preset]
    else:
        profile = {
            "cgpa": 7.8,
            "tenth_pct": 78.0,
            "twelfth_pct": 74.0,
            "backlogs": 0,
            "dsa_problems_solved": 150,
            "projects": 2,
            "internships": 1,
            "certifications": 1,
            "hackathons": 1,
            "communication_score": 75.0,
            "aptitude_score": 72.0,
            "branch": "Computer Science",
            "college_tier": 2,
            "gender": "Male",
            "work_experience": 0,
        }

    # --- Two Column Main Layout ---
    col_form, col_results = st.columns([1.15, 1], gap="large")

    with col_form:
        st.markdown("### 📝 Student Candidate Profile")
        with st.form("student_profile_form"):
            st.markdown("##### 🎓 Academic Background")
            c1, c2, c3 = st.columns(3)
            with c1:
                cgpa = st.number_input(
                    "CGPA (0 - 10)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(profile["cgpa"]),
                    step=0.1,
                )
            with c2:
                tenth_pct = st.number_input(
                    "10th Grade (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(profile["tenth_pct"]),
                    step=1.0,
                )
            with c3:
                twelfth_pct = st.number_input(
                    "12th Grade (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(profile["twelfth_pct"]),
                    step=1.0,
                )

            c4, c5, c6 = st.columns(3)
            with c4:
                branch = st.selectbox(
                    "Engineering Branch",
                    options=ALLOWED_BRANCHES,
                    index=(
                        ALLOWED_BRANCHES.index(profile["branch"])
                        if profile["branch"] in ALLOWED_BRANCHES
                        else 0
                    ),
                )
            with c5:
                college_tier = st.selectbox(
                    "College Tier",
                    options=[1, 2, 3],
                    index=[1, 2, 3].index(profile["college_tier"]),
                    help="Institutional ranking tier: 1 (Premier), 2 (Standard), 3 (Regional)",
                )
            with c6:
                gender = st.selectbox(
                    "Gender",
                    options=["Male", "Female"],
                    index=["Male", "Female"].index(profile["gender"]),
                )

            st.markdown("##### 💻 Technical & Practical Experience")
            t1, t2, t3 = st.columns(3)
            with t1:
                dsa_problems = st.number_input(
                    "DSA Problems Solved",
                    min_value=0,
                    max_value=3000,
                    value=int(profile["dsa_problems_solved"]),
                    step=25,
                )
            with t2:
                internships = st.number_input(
                    "Internships Completed",
                    min_value=0,
                    max_value=10,
                    value=int(profile["internships"]),
                    step=1,
                )
            with t3:
                projects = st.number_input(
                    "Projects Completed",
                    min_value=0,
                    max_value=20,
                    value=int(profile["projects"]),
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

    # --- Explainability & What-If Tabs ---
    tab_whatif, tab_shap = st.tabs(
        ["💡 Interactive What-If Simulator", "🔍 Local SHAP Explainability & Drivers"]
    )

    with tab_whatif:
        st.markdown("#### 🔬 Actionable Counterfactual Simulation")
        st.caption(
            "Adjust actionable student levers to observe how incremental efforts improve placement probability."
        )

        w_col1, w_col2 = st.columns([1, 1], gap="large")

        with w_col1:
            st.markdown("##### 🛠️ Adjust Controllable Levers")
            add_dsa = st.slider(
                "Solve additional DSA problems:", min_value=0, max_value=500, value=150, step=25
            )
            add_intern = st.slider(
                "Complete additional internships:",
                min_value=0,
                max_value=2,
                value=1 if internships == 0 else 0,
                step=1,
            )
            add_projects = st.slider(
                "Build additional technical projects:", min_value=0, max_value=3, value=1, step=1
            )
            clear_backlogs = st.checkbox(
                "Clear all outstanding academic backlogs", value=True if backlogs > 0 else False
            )
            boost_comm = st.slider(
                "Communication & interview coaching (+ pts):",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=1.0,
            )

        simulator = WhatIfSimulator(predictor.placement_pipeline)
        sim_mods = {
            "dsa_problems_solved": dsa_problems + add_dsa,
            "internships": min(10, internships + add_intern),
            "projects": min(20, projects + add_projects),
            "backlogs": 0 if clear_backlogs else backlogs,
            "communication_score": min(100.0, communication + boost_comm),
        }

        sim_res = simulator.simulate(student_dict, sim_mods)
        sim_prob = sim_res["simulated_probability"]
        prob_delta = sim_res["delta_percentage_points"]

        with w_col2:
            st.markdown("##### 📈 Projected Probability Shift")
            fig_whatif = create_what_if_comparison_chart(prob, sim_prob)
            st.plotly_chart(fig_whatif, use_container_width=True)

            delta_sign = "+" if prob_delta >= 0 else ""
            delta_color = "#10B981" if prob_delta >= 0 else "#F43F5E"
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px 18px;">
                    <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Simulation Outcome</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: {delta_color}; margin-top: 4px;">
                        {delta_sign}{prob_delta:.1f} Percentage Points ({prob*100:.1f}% &rarr; {sim_prob*100:.1f}%)
                    </div>
                    <div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 6px;">
                        {'🎉 Outcome shifted to Placed!' if prob < 0.50 and sim_prob >= 0.50 else 'Consistent placement likelihood trajectory.'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_shap:
        st.markdown("#### 🔍 SHAP Local Attribution (TreeExplainer)")
        st.caption(
            "Exact decomposition of how this student's features shifted the model prediction away from the baseline campus rate."
        )

        with st.spinner("Calculating exact TreeExplainer SHAP values..."):
            shap_exp = explain_student_prediction(
                pipeline=predictor.placement_pipeline,
                student_data=student_dict,
                numerical_features=num_features,
                categorical_features=cat_features,
                top_n=4,
            )

        # Additivity verification banner
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

            if not shap_exp["top_negative_drivers"]:
                st.caption("No significant penalizing factors detected for this candidate.")


if __name__ == "__main__":
    render_predict_page()
