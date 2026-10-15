"""Cached resource and data loaders for models, evaluation metrics, and audit results."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.models.predict import PlacementPredictor


@st.cache_resource(show_spinner="Loading trained ML models...")
def get_placement_predictor() -> PlacementPredictor:
    """Loads and caches the primary tuned placement model and salary model."""
    return PlacementPredictor(
        placement_model_name="best_placement_model",
        salary_model_name="salary_band_model",
        models_dir="models",
    )


@st.cache_resource(show_spinner="Loading biased baseline audit model...")
def get_biased_placement_predictor() -> PlacementPredictor:
    """Loads and caches the model trained on ground-truth institutional bias data."""
    return PlacementPredictor(
        placement_model_name="best_placement_model_biased",
        salary_model_name="salary_band_model",
        models_dir="models",
    )


@st.cache_data(show_spinner="Loading evaluation metrics...")
def get_metrics_data() -> dict[str, Any]:
    """Loads evaluation metrics from reports/metrics.json."""
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        return {}
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading fairness audit report...")
def get_fairness_audit_data() -> dict[str, Any]:
    """Loads comprehensive Fairlearn audit results from reports/fairness_audit.json."""
    audit_path = Path("reports/fairness_audit.json")
    if not audit_path.exists():
        return {}
    with open(audit_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def get_preset_profiles() -> dict[str, dict[str, Any]]:
    """Curated realistic student personas for instant 1-click loading."""
    return {
        "🌟 Top Performer (Tier 1 CS - High Honors)": {
            "cgpa": 9.2,
            "tenth_pct": 91.0,
            "twelfth_pct": 89.5,
            "backlogs": 0,
            "dsa_problems_solved": 450,
            "projects": 4,
            "internships": 2,
            "certifications": 3,
            "hackathons": 3,
            "communication_score": 88.0,
            "aptitude_score": 92.0,
            "branch": "Computer Science",
            "college_tier": 1,
            "gender": "Female",
            "work_experience": 1,
        },
        "⚡ Competitive Tier 3 Coder (High Technical Skill)": {
            "cgpa": 8.4,
            "tenth_pct": 82.0,
            "twelfth_pct": 79.0,
            "backlogs": 0,
            "dsa_problems_solved": 380,
            "projects": 3,
            "internships": 1,
            "certifications": 2,
            "hackathons": 2,
            "communication_score": 74.0,
            "aptitude_score": 85.0,
            "branch": "Computer Science",
            "college_tier": 3,
            "gender": "Male",
            "work_experience": 0,
        },
        "⚖️ Borderline Candidate (Tier 2 Mech - Needs Polish)": {
            "cgpa": 7.1,
            "tenth_pct": 72.0,
            "twelfth_pct": 70.0,
            "backlogs": 1,
            "dsa_problems_solved": 70,
            "projects": 1,
            "internships": 0,
            "certifications": 1,
            "hackathons": 0,
            "communication_score": 62.0,
            "aptitude_score": 65.0,
            "branch": "Mechanical",
            "college_tier": 2,
            "gender": "Male",
            "work_experience": 0,
        },
        "📈 Aspiring Tech Transitioner (Tier 3 Civil)": {
            "cgpa": 7.8,
            "tenth_pct": 78.0,
            "twelfth_pct": 75.0,
            "backlogs": 0,
            "dsa_problems_solved": 150,
            "projects": 2,
            "internships": 0,
            "certifications": 2,
            "hackathons": 1,
            "communication_score": 75.0,
            "aptitude_score": 78.0,
            "branch": "Civil",
            "college_tier": 3,
            "gender": "Female",
            "work_experience": 0,
        },
    }


@st.cache_data(show_spinner="Precomputing threshold audit data...")
def get_audit_test_dataset() -> pd.DataFrame:
    """Loads held-out test data from biased dataset with precomputed prediction probabilities."""
    from src.data.loader import DataLoader
    from src.fairness.metrics import create_intersectional_attribute

    loader = DataLoader(config_path="config/config.yaml")
    df_biased = loader.load_synthetic_data(biased=True, auto_generate=True)
    _, _, test_df = loader.get_train_val_test_split(
        df_biased, "placed", stratify=True, random_seed=42
    )
    test_df = test_df.copy()
    test_df["gender_x_tier"] = create_intersectional_attribute(test_df, "gender", "college_tier")

    predictor = get_biased_placement_predictor()
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
    probs = predictor.placement_pipeline.predict_proba(test_df[num_features + cat_features])[:, 1]
    test_df["pred_prob"] = probs
    return test_df
