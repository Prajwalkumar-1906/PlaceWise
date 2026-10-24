"""Explainability and actionable recourse endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException

from api.audit import compute_input_hash
from api.routers.v1.predict import get_predictor
from api.schemas.explain import (
    ExplanationResponse,
    FeatureContribution,
    RecoursePathway,
    RecourseRequest,
    RecourseResponse,
)
from api.schemas.predict import StudentPredictionRequest
from src.explainability.recourse import RecourseEngine
from src.explainability.shap_local import explain_student_prediction
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["Explainability & Recourse"])


NUMERICAL_FEATURES = [
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
CATEGORICAL_FEATURES = ["branch", "college_tier", "gender", "work_experience"]


@router.post("/explain", response_model=ExplanationResponse)
def explain_prediction(student: StudentPredictionRequest) -> ExplanationResponse:
    """Generates SHAP feature contribution attributions for a student profile."""
    predictor = get_predictor()
    student_dict = student.model_dump()

    try:
        explanation = explain_student_prediction(
            pipeline=predictor.placement_pipeline,
            student_data=student_dict,
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Explainability computation failed: {str(e)}"
        ) from e

    def _convert_contrib(raw: dict[str, Any]) -> FeatureContribution:
        return FeatureContribution(
            feature=str(raw["feature"]),
            value=raw.get("value"),
            shap_value=round(float(raw.get("shap_value", 0.0)), 4),
            percentage_point_impact=round(float(raw.get("prob_delta_percentage_points", 0.0)), 2),
            direction=str(raw.get("direction", "Neutral")),
        )

    return ExplanationResponse(
        placement_probability=round(float(explanation.get("model_probability", 0.50)), 4),
        base_value_probability=round(float(explanation.get("base_probability", 0.50)), 4),
        top_positive_drivers=[
            _convert_contrib(c) for c in explanation.get("top_positive_drivers", [])
        ],
        top_negative_drivers=[
            _convert_contrib(c) for c in explanation.get("top_negative_drivers", [])
        ],
        all_contributions=[_convert_contrib(c) for c in explanation.get("feature_impacts", [])],
        input_hash=compute_input_hash(student_dict),
    )


@router.post("/recourse", response_model=RecourseResponse)
def generate_recourse(req: RecourseRequest) -> RecourseResponse:
    """Generates feasible developmental recourse pathways with strictly locked immutable demographic features."""
    predictor = get_predictor()
    student_dict = req.student.model_dump()

    try:
        engine = RecourseEngine(pipeline=predictor.placement_pipeline)
        result = engine.generate_recourse_pathways(
            baseline_student=student_dict,
            target_probability=req.target_probability,
        )
    except Exception as e:
        logger.error(f"Recourse pathway generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Recourse optimization failed: {str(e)}"
        ) from e

    pathways = [
        RecoursePathway(
            pathway_name=p["pathway_name"],
            description=p["description"],
            baseline_probability=p["baseline_probability"],
            achieved_probability=p["achieved_probability"],
            probability_gain_pts=p["probability_gain_pts"],
            reaches_target=p["reaches_target"],
            total_effort_score=p["total_effort_score"],
            feasibility_rating=p["feasibility_rating"],
            interventions=p["interventions"],
            immutable_features_locked=p["immutable_features_locked"],
        )
        for p in result.get("pathways", [])
    ]

    return RecourseResponse(
        baseline_probability=result["baseline_probability"],
        target_probability=result["target_probability"],
        target_already_satisfied=result["target_already_satisfied"],
        pathways_count=len(pathways),
        pathways=pathways,
        disclaimer=result["disclaimer"],
    )
