"""Pydantic schemas for single and batch placement predictions."""

from typing import Literal

from pydantic import BaseModel, Field


class StudentPredictionRequest(BaseModel):
    """Student profile request schema with strict domain range validation."""

    cgpa: float = Field(
        ..., ge=0.0, le=10.0, description="Cumulative Grade Point Average (0-10)", examples=[8.2]
    )
    tenth_pct: float = Field(
        ..., ge=0.0, le=100.0, description="10th Grade Percentage", examples=[85.0]
    )
    twelfth_pct: float = Field(
        ..., ge=0.0, le=100.0, description="12th Grade Percentage", examples=[82.0]
    )
    branch: Literal[
        "Computer Science",
        "Information Technology",
        "Electrical",
        "Mechanical",
        "Civil",
    ] = Field(..., description="Engineering discipline branch", examples=["Computer Science"])
    college_tier: Literal[1, 2, 3] = Field(
        ..., description="Institutional Tier (1=Premier, 2=Regional, 3=Affiliated)", examples=[1]
    )
    gender: Literal["Male", "Female"] = Field(
        ..., description="Self-reported demographic gender", examples=["Female"]
    )
    work_experience: Literal[0, 1] = Field(
        ..., description="Prior work experience indicator (0=No, 1=Yes)", examples=[1]
    )
    backlogs: int = Field(
        ..., ge=0, le=15, description="Active academic backlogs count", examples=[0]
    )
    dsa_problems_solved: int = Field(
        ..., ge=0, le=2500, description="Coding problems solved on platform", examples=[280]
    )
    projects: int = Field(
        ..., ge=0, le=25, description="Verified software or engineering projects", examples=[3]
    )
    internships: int = Field(
        ..., ge=0, le=10, description="Completed professional internships", examples=[1]
    )
    certifications: int = Field(
        ..., ge=0, le=20, description="Industry recognized certifications", examples=[2]
    )
    hackathons: int = Field(
        ..., ge=0, le=20, description="Hackathons or technical competitions attended", examples=[1]
    )
    communication_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Evaluated soft skills and interview score",
        examples=[78.5],
    )
    aptitude_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="General quantitative and logical aptitude score",
        examples=[80.0],
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema containing up to 500 student records."""

    students: list[StudentPredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of student profiles for batch inference",
    )


class StudentPredictionResponse(BaseModel):
    """Inference response containing prediction, probabilities, salary band, and HITL flags."""

    placement_probability: float = Field(
        ..., description="Calibrated placement likelihood in [0, 1]"
    )
    placement_prediction: int = Field(
        ..., description="Binary classification (1=Placed, 0=Not Placed)"
    )
    risk_label: str = Field(..., description="Qualitative placement risk level")
    risk_color: str = Field(..., description="UI visual color representation")
    salary_band: str = Field(
        ..., description="Predicted compensation band (Low, Mid, High, or Not Placed)"
    )
    confidence_band: str = Field(
        ..., description="Confidence level: High Confidence vs Abstention Zone"
    )
    needs_human_review: bool = Field(
        ..., description="True if profile falls in [0.35, 0.65] abstention zone"
    )
    hitl_recommendation: str = Field(..., description="Counselor guidance recommendation")
    hitl_status: str = Field(..., description="Machine-readable HITL state code")
    input_hash: str = Field(..., description="SHA-256 provenance hash of applicant features")
    model_version: str = Field(default="1.0.0", description="Active model registry version")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Batch inference summary and itemized predictions."""

    total_records: int = Field(..., description="Total student profiles processed")
    placed_count: int = Field(..., description="Total candidates classified as Placed")
    unplaced_count: int = Field(..., description="Total candidates classified as Not Placed")
    human_review_required_count: int = Field(
        ..., description="Count of candidates requiring counselor review"
    )
    batch_latency_ms: float = Field(
        ..., description="Total batch inference latency in milliseconds"
    )
    predictions: list[StudentPredictionResponse] = Field(
        ..., description="Itemized prediction results"
    )
