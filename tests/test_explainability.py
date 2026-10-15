"""Unit tests for model explainability (SHAP global/local, additivity check, and what-if simulation)."""

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.explainability.counterfactual import WhatIfSimulator
from src.explainability.shap_global import (
    generate_global_shap_plots,
    get_global_feature_importance,
)
from src.explainability.shap_local import explain_student_prediction, save_waterfall_plot
from src.features.pipeline import create_preprocessing_pipeline


def create_fitted_test_pipeline():
    """Helper creating a trained pipeline on minimal synthetic data."""
    data = {
        "cgpa": [6.0, 7.5, 8.5, 9.2, 5.5, 8.0],
        "tenth_pct": [65.0, 75.0, 85.0, 90.0, 60.0, 80.0],
        "twelfth_pct": [60.0, 70.0, 82.0, 88.0, 58.0, 78.0],
        "backlogs": [2, 0, 0, 0, 3, 0],
        "dsa_problems_solved": [50, 200, 400, 600, 20, 300],
        "projects": [1, 2, 3, 4, 0, 2],
        "internships": [0, 1, 2, 2, 0, 1],
        "certifications": [0, 1, 2, 3, 0, 1],
        "hackathons": [0, 1, 2, 2, 0, 1],
        "communication_score": [50.0, 70.0, 80.0, 90.0, 45.0, 75.0],
        "aptitude_score": [55.0, 70.0, 85.0, 92.0, 50.0, 78.0],
        "branch": [
            "Civil",
            "Mechanical",
            "Computer Science",
            "Information Technology",
            "Civil",
            "Computer Science",
        ],
        "college_tier": [3, 2, 1, 1, 3, 2],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
        "work_experience": [0, 0, 1, 1, 0, 0],
        "placed": [0, 1, 1, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    num_cols = [
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
    cat_cols = ["branch", "college_tier", "gender", "work_experience"]

    preprocessor = create_preprocessing_pipeline(num_cols, cat_cols)
    clf = XGBClassifier(n_estimators=15, max_depth=3, random_state=42)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipeline.fit(df[num_cols + cat_cols], df["placed"])
    return pipeline, df, num_cols, cat_cols


def test_shap_global_computations(tmp_path):
    """Verify global SHAP computation, feature ranking, and plotting."""
    pipeline, df, num_cols, cat_cols = create_fitted_test_pipeline()

    # Feature importance DataFrame
    df_imp = get_global_feature_importance(pipeline, df, num_cols, cat_cols)
    assert isinstance(df_imp, pd.DataFrame)
    assert "feature" in df_imp.columns
    assert "mean_abs_shap" in df_imp.columns
    assert len(df_imp) > 0

    # Global plots generation
    generate_global_shap_plots(pipeline, df, num_cols, cat_cols, output_dir=tmp_path)
    assert (tmp_path / "shap_beeswarm.png").exists()
    assert (tmp_path / "shap_feature_importance_bar.png").exists()


def test_shap_local_additivity_and_probability_conversion(tmp_path):
    """Verify local SHAP additivity: base + sum(phi) == margin, and probability conversion."""
    pipeline, df, num_cols, cat_cols = create_fitted_test_pipeline()
    student = df.iloc[0].to_dict()

    explanation = explain_student_prediction(pipeline, student, num_cols, cat_cols)

    # Sanity check: Exact additivity check
    assert explanation["additivity_check"]["is_additive"] is True
    assert explanation["additivity_check"]["additivity_error"] < 1e-4

    # Probability sanity checks
    assert 0.0 <= explanation["base_probability"] <= 1.0
    assert 0.0 <= explanation["model_probability"] <= 1.0

    # Plain English explanations exist
    assert len(explanation["plain_english_summary"]) > 0

    # Waterfall plot saving
    waterfall_file = tmp_path / "waterfall_test.png"
    save_waterfall_plot(explanation["raw_explanation"], waterfall_file)
    assert waterfall_file.exists()


def test_counterfactual_what_if_simulator():
    """Verify WhatIfSimulator simulates controllable changes accurately."""
    pipeline, df, num_cols, cat_cols = create_fitted_test_pipeline()
    student = df.iloc[4].to_dict()  # Unplaced student with low metrics

    simulator = WhatIfSimulator(pipeline)

    # Simulate boosting DSA and internships
    result = simulator.simulate(
        student,
        {"dsa_problems_solved": 500, "internships": 2, "backlogs": 0},
    )

    assert "baseline_probability" in result
    assert "modified_probability" in result
    assert result["modified_probability"] >= result["baseline_probability"]
    assert "delta_probability_points" in result

    # Test intervention recommendations
    recommendations = simulator.recommend_interventions(student, target_probability=0.70)
    assert isinstance(recommendations, list)
