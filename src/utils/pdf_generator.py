"""Automated PDF dossier generator for student placement readiness assessments."""

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_student_pdf_dossier(
    student_data: dict[str, Any],
    prediction_result: dict[str, Any],
    explanation_data: dict[str, Any] | None = None,
    recourse_data: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> bytes:
    """Generates a professional 1-page PDF assessment dossier for a student profile.

    Args:
        student_data: Student feature profile.
        prediction_result: Inference result with probability, risk, salary, and HITL flags.
        explanation_data: Optional SHAP feature drivers breakdown.
        recourse_data: Optional actionable recourse pathways.
        output_path: Optional path to write PDF file to disk.

    Returns:
        bytes: Raw PDF document content.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer if output_path is None else str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1E293B"),
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748B"),
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("PlaceWise Campus Placement Intelligence", subtitle_style))
    elements.append(Paragraph("Student Readiness & Advisory Dossier", title_style))
    now_str = datetime.now(UTC).strftime("%d %B %Y | %H:%M UTC")
    elements.append(
        Paragraph(
            f"Generated on {now_str} &bull; Model Version: <code>v4.0 (Production)</code>",
            subtitle_style,
        )
    )
    elements.append(Spacer(1, 10))
    elements.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#6366F1"),
            spaceBefore=0,
            spaceAfter=10,
        )
    )

    # 2. Executive Assessment Card (2 columns: Score Box + Status Box)
    prob = prediction_result.get("placement_probability", 0.0)
    pred_str = "Placed" if prediction_result.get("placement_prediction", 0) == 1 else "Not Placed"
    risk_label = prediction_result.get("risk_label", "Moderate Risk")
    salary_band = prediction_result.get("salary_band", "Pending Evaluation")
    conf_band = prediction_result.get("confidence_band", "High Confidence")
    needs_review = prediction_result.get("needs_human_review", False)

    (
        colors.HexColor("#10B981")
        if prob >= 0.75
        else (colors.HexColor("#F59E0B") if prob >= 0.45 else colors.HexColor("#EF4444"))
    )

    assessment_data = [
        [
            Paragraph(f"<b>Placement Likelihood:</b> {prob*100:.1f}% ({pred_str})", heading_style),
            Paragraph(f"<b>Decision Confidence:</b> {conf_band}", heading_style),
        ],
        [
            Paragraph(f"<b>Risk Classification:</b> {risk_label}", body_style),
            Paragraph(
                f"<b>Human Review Flag:</b> {'⚠️ Required (Abstention Zone)' if needs_review else '✅ Standard Assessment'}",
                body_style,
            ),
        ],
        [
            Paragraph(f"<b>Projected Compensation:</b> {salary_band}", body_style),
            Paragraph("<b>Governance Tag:</b> Non-PII Synthetic Evaluation", body_style),
        ],
    ]

    t_assess = Table(assessment_data, colWidths=[270, 270])
    t_assess.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    elements.append(t_assess)
    elements.append(Spacer(1, 12))

    # 3. Candidate Academic & Technical Profile Table
    elements.append(Paragraph("1. Candidate Academic & Technical Profile", heading_style))
    elements.append(Spacer(1, 4))

    profile_rows = [
        [
            "CGPA",
            f"{student_data.get('cgpa', 'N/A')}",
            "DSA Problems Solved",
            f"{student_data.get('dsa_problems_solved', 0)}",
        ],
        [
            "10th / 12th %",
            f"{student_data.get('tenth_pct', 0)}% / {student_data.get('twelfth_pct', 0)}%",
            "Verified Projects",
            f"{student_data.get('projects', 0)}",
        ],
        [
            "Branch",
            f"{student_data.get('branch', 'N/A')}",
            "Internships",
            f"{student_data.get('internships', 0)}",
        ],
        [
            "College Tier",
            f"Tier {student_data.get('college_tier', 'N/A')}",
            "Certifications",
            f"{student_data.get('certifications', 0)}",
        ],
        [
            "Active Backlogs",
            f"{student_data.get('backlogs', 0)}",
            "Hackathons",
            f"{student_data.get('hackathons', 0)}",
        ],
        [
            "Work Experience",
            "Yes" if student_data.get("work_experience") == 1 else "No",
            "Aptitude / Comm. Score",
            f"{student_data.get('aptitude_score', 0)} / {student_data.get('communication_score', 0)}",
        ],
    ]

    t_profile = Table(profile_rows, colWidths=[120, 150, 140, 130])
    t_profile.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#F1F5F9")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#334155")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(t_profile)
    elements.append(Spacer(1, 12))

    # 4. Explainability & Key Drivers (if provided)
    if explanation_data:
        elements.append(
            Paragraph("2. Top Decision Drivers (SHAP Local Attribution)", heading_style)
        )
        elements.append(Spacer(1, 4))
        pos_drivers = explanation_data.get("top_positive_drivers", [])[:3]
        neg_drivers = explanation_data.get("top_negative_drivers", [])[:3]

        driver_rows = [["Factor Type", "Feature", "Observed Value", "Impact on Probability"]]
        for d in pos_drivers:
            driver_rows.append(
                [
                    "Strength (+)",
                    d.get("feature", ""),
                    str(d.get("value", "")),
                    f"+{d.get('prob_delta_percentage_points', d.get('percentage_point_impact', 0.0)):.1f} pts",
                ]
            )
        for d in neg_drivers:
            driver_rows.append(
                [
                    "Risk Factor (-)",
                    d.get("feature", ""),
                    str(d.get("value", "")),
                    f"{d.get('prob_delta_percentage_points', d.get('percentage_point_impact', 0.0)):.1f} pts",
                ]
            )

        if len(driver_rows) > 1:
            t_drivers = Table(driver_rows, colWidths=[110, 160, 120, 150])
            t_drivers.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            elements.append(t_drivers)
            elements.append(Spacer(1, 12))

    # 5. Strategic Developmental Recourse (if provided)
    if recourse_data and recourse_data.get("pathways"):
        elements.append(Paragraph("3. Recommended Developmental Recourse Pathways", heading_style))
        elements.append(Spacer(1, 4))
        for p in recourse_data.get("pathways", [])[:2]:
            p_title = f"<b>{p['pathway_name']}</b> &bull; Achieves {p['achieved_probability']*100:.1f}% (+{p['probability_gain_pts']} pts) &bull; Feasibility: {p['feasibility_rating']}"
            elements.append(Paragraph(p_title, body_style))
            inv_texts = []
            for inv in p.get("interventions", []):
                inv_texts.append(
                    f"&bull; <b>{inv['description']}</b>: {inv['delta']} (current: {inv['current_value']} &rarr; recommended: {inv['recommended_value']})"
                )
            elements.append(Paragraph("<br/>".join(inv_texts), body_style))
            elements.append(Spacer(1, 6))

    # 6. Governance & Ethical Disclaimer
    elements.append(Spacer(1, 10))
    elements.append(
        HRFlowable(
            width="100%",
            thickness=0.8,
            color=colors.HexColor("#CBD5E1"),
            spaceBefore=0,
            spaceAfter=6,
        )
    )
    elements.append(
        Paragraph(
            "<b>ETHICAL & GOVERNANCE NOTICE:</b> This dossier is generated by the PlaceWise AI Decision Support System "
            "as an advisory resource for academic mentors and placement committees. It employs algorithmic counterfactual "
            "optimizations and calibrated machine learning models. Under no circumstances should this evaluation be used to automatically "
            "exclude or reject applicants without qualified human counselor review.",
            disclaimer_style,
        )
    )

    doc.build(elements)

    if output_path is not None:
        logger.info(f"Saved student PDF dossier to {Path(output_path).resolve()}")
        return Path(output_path).read_bytes()

    return buffer.getvalue()
