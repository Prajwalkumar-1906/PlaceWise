"""Pydantic schemas for explainability and actionable recourse endpoints."""

from typing import Any

from pydantic import BaseModel, Field

from api.schemas.predict import StudentPredictionRequest


class ExplanationRequest(StudentPredictionRequest):
    """Request schema for local SHAP explanation endpoint."""

    pass


class FeatureContribution(BaseModel):
    """Individual feature SHAP attribution value and impact direction."""

    feature: str = Field(..., description="Feature name")
    value: Any = Field(..., description="Observed applicant feature value")
    shap_value: float = Field(..., description="Log-odds SHAP attribution value")
    percentage_point_impact: float = Field(
        ..., description="Probability-point contribution to final score"
    )
    direction: str = Field(
        ..., description="'Positive' (increases placement) or 'Negative' (decreases placement)"
    )


class ExplanationResponse(BaseModel):
    """SHAP waterfall feature contribution breakdown."""

    placement_probability: float = Field(..., description="Predicted placement probability")
    base_value_probability: float = Field(
        ..., description="Expected baseline population placement probability"
    )
    top_positive_drivers: list[FeatureContribution] = Field(
        ..., description="Top drivers increasing probability"
    )
    top_negative_drivers: list[FeatureContribution] = Field(
        ..., description="Top risk factors decreasing probability"
    )
    all_contributions: list[FeatureContribution] = Field(
        ..., description="All feature attributions sorted by magnitude"
    )
    input_hash: str = Field(..., description="Input provenance hash")


class RecourseRequest(BaseModel):
    """Request schema for actionable developmental recourse pathways."""

    student: StudentPredictionRequest = Field(..., description="Target student profile")
    target_probability: float = Field(
        default=0.65, ge=0.50, le=0.95, description="Desired placement probability threshold"
    )


class InterventionStep(BaseModel):
    """Concrete single-attribute action step."""

    feature: str = Field(..., description="Controllable attribute name")
    description: str = Field(..., description="Human-friendly action label")
    current_value: Any = Field(..., description="Current applicant metric")
    recommended_value: Any = Field(..., description="Target recommended metric")
    delta: str = Field(..., description="Formatted adjustment")
    effort_points: float = Field(..., description="Estimated effort score")


class RecoursePathway(BaseModel):
    """Complete strategic action pathway."""

    pathway_name: str = Field(..., description="Pathway title")
    description: str = Field(..., description="Strategic focus area")
    baseline_probability: float = Field(..., description="Starting probability")
    achieved_probability: float = Field(..., description="Probability after interventions")
    probability_gain_pts: float = Field(..., description="Probability point improvement")
    reaches_target: bool = Field(..., description="True if target threshold is attained")
    total_effort_score: float = Field(..., description="Sum of intervention effort points")
    feasibility_rating: str = Field(..., description="Feasibility category and estimated duration")
    interventions: list[InterventionStep] = Field(..., description="Actionable modification steps")
    immutable_features_locked: list[str] = Field(..., description="Guaranteed locked attributes")


class RecourseResponse(BaseModel):
    """Actionable recourse response containing distinct pathways and legal disclaimer."""

    baseline_probability: float = Field(..., description="Initial candidate placement probability")
    target_probability: float = Field(..., description="Target placement probability")
    target_already_satisfied: bool = Field(
        ..., description="True if candidate already exceeds target"
    )
    pathways_count: int = Field(..., description="Number of generated viable pathways")
    pathways: list[RecoursePathway] = Field(..., description="Ranked developmental pathways")
    disclaimer: str = Field(..., description="Mandatory responsible AI disclaimer")
