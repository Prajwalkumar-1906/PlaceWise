"""Automated fairness reporting and trade-off visualization generator."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_fairness_figures(
    audit_results: dict[str, Any],
    output_dir: str | Path = "reports/figures",
) -> None:
    """Generates selection rate comparative charts and Pareto trade-off diagrams.

    Args:
        audit_results: Dictionary returned by FairnessAuditor.run_full_audit().
        output_dir: Target directory for figure PNGs.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_data = audit_results.get("models", {})
    tradeoffs = audit_results.get("tradeoffs", {}).get("college_tier", [])

    # 1. Selection Rates by College Tier Comparison
    tier_records = []
    for model_name, m_info in models_data.items():
        by_grp = m_info["metrics_by_sensitive_attribute"]["college_tier"]["by_group"]
        for tier_val, stats in by_grp.items():
            tier_records.append(
                {
                    "Model": model_name,
                    "College Tier": f"Tier {tier_val}",
                    "Selection Rate (%)": stats["selection_rate"] * 100.0,
                }
            )

    df_tier_rates = pd.DataFrame(tier_records)
    pivot_tier = df_tier_rates.pivot(
        index="College Tier", columns="Model", values="Selection Rate (%)"
    )

    plt.figure(figsize=(10, 5), dpi=150)
    pivot_tier.plot.bar(edgecolor="black", alpha=0.9, ax=plt.gca())
    plt.title(
        "Selection Rate by College Tier Across Mitigation Models", fontsize=12, fontweight="bold"
    )
    plt.ylabel("Selection Rate (%)", fontsize=11)
    plt.xlabel("Institutional Tier", fontsize=11)
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fairness_selection_rates.png", bbox_inches="tight")
    plt.close()

    # 2. Pareto Trade-off: Accuracy vs Demographic Parity Difference
    if tradeoffs:
        plt.figure(figsize=(8, 6), dpi=150)
        df_trade = pd.DataFrame(tradeoffs)

        for _, row in df_trade.iterrows():
            plt.scatter(
                row["dp_difference"],
                row["accuracy"],
                s=160,
                label=row["model"],
                alpha=0.85,
                edgecolors="black",
            )
            plt.annotate(
                row["model"],
                (row["dp_difference"], row["accuracy"]),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=9,
                fontweight="bold",
            )

        plt.xlabel("Demographic Parity Difference (Lower = Fairer)", fontsize=11)
        plt.ylabel("Held-Out Test Accuracy", fontsize=11)
        plt.title(
            "Pareto Trade-Off: Accuracy vs. Parity Gap (College Tier)",
            fontsize=13,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "fairness_tradeoff_pareto.png", bbox_inches="tight")
        plt.close()

    logger.info(f"Saved fairness figures to {out_dir.resolve()}")


def generate_fairness_report(
    audit_results: dict[str, Any],
    output_md_path: str | Path = "reports/fairness_report.md",
    figures_dir: str | Path = "reports/figures",
) -> None:
    """Renders comprehensive Markdown audit report with documented formulas and discussions.

    Args:
        audit_results: Results dict from FairnessAuditor.
        output_md_path: Markdown file destination.
        figures_dir: Directory containing fairness plots.
    """
    generate_fairness_figures(audit_results, output_dir=figures_dir)

    models_data = audit_results.get("models", {})
    tradeoffs = audit_results.get("tradeoffs", {}).get("college_tier", [])

    lines = [
        "# Algorithmic Fairness Audit and Mitigation Report",
        "",
        "## 1. Executive Summary",
        "This report audits the Campus Placement Prediction system for disparate impact across **College Institutional Tier** and **Gender**, using the Fairlearn toolkit.",
        "To establish a verifiable ground truth for algorithmic bias detection, experiments were benchmarked on historical institutional bias injected data, where Tier-3 and Female applicants received systematic placement penalties at identical skill levels.",
        "",
        "---",
        "",
        "## 2. Quantitative Performance & Fairness Benchmark",
        "",
        "### College Tier Audit Benchmark Table",
        "| Model Architecture | Test Accuracy | Demographic Parity Gap | DP Ratio | Equalized Odds Gap | Bias Reduction % |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for t in tradeoffs:
        lines.append(
            f"| **{t['model']}** | {t['accuracy']:.3f} | {t['dp_difference']:.3f} | {t['dp_ratio']:.3f} | {t['equalized_odds_diff']:.3f} | **{t['bias_reduction_percentage']:+.1f}%** |"
        )

    lines.extend(
        [
            "",
            "> [!NOTE]",
            "> **Explicit Bias Reduction Formula**:",
            "> $$\\text{Bias Reduction \\%} = \\frac{\\text{DP\\_Diff}_{\\text{baseline}} - \\text{DP\\_Diff}_{\\text{mitigated}}}{\\text{DP\\_Diff}_{\\text{baseline}}} \\times 100\\%$$",
            "> A positive percentage denotes a quantified reduction in demographic disparity compared to the unmitigated baseline.",
            "",
            "---",
            "",
            "## 3. Disaggregated Group Selection Rates",
            "",
            "### College Tier Breakdown",
            "| Model | Tier 1 Rate | Tier 2 Rate | Tier 3 Rate | Tier 1-3 Gap |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for m_name, m_info in models_data.items():
        by_tier = m_info["metrics_by_sensitive_attribute"]["college_tier"]["by_group"]
        r1 = by_tier.get("1", {}).get("selection_rate", 0.0) * 100
        r2 = by_tier.get("2", {}).get("selection_rate", 0.0) * 100
        r3 = by_tier.get("3", {}).get("selection_rate", 0.0) * 100
        lines.append(f"| {m_name} | {r1:.1f}% | {r2:.1f}% | {r3:.1f}% | {abs(r1 - r3):.1f}% |")

    lines.extend(
        [
            "",
            "### Gender Disparities",
            "| Model | Male Selection Rate | Female Selection Rate | Gender Parity Gap |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )

    for m_name, m_info in models_data.items():
        by_gender = m_info["metrics_by_sensitive_attribute"]["gender"]["by_group"]
        rm = by_gender.get("Male", {}).get("selection_rate", 0.0) * 100
        rf = by_gender.get("Female", {}).get("selection_rate", 0.0) * 100
        dp = m_info["metrics_by_sensitive_attribute"]["gender"]["demographic_parity_difference"]
        lines.append(f"| {m_name} | {rm:.1f}% | {rf:.1f}% | {dp:.3f} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Key Findings & Discussion",
            "",
            "### A. The Failure of 'Fairness through Unawareness'",
            "Simply removing sensitive attributes (`gender`, `college_tier`) from the training set does **not** ensure fairness. "
            "Because technical skills (DSA problem counts), aptitude scores, and branch choices correlate with institutional tier, "
            "the unaware model reconstructs institutional bias through proxy features, yielding nearly identical disparities.",
            "",
            "### B. Inherent Mathematical Conflicts in Fairness Criteria",
            "Fairness criteria frequently stand in mathematical tension (Chouldechova, 2017; Kleinberg et al., 2016):",
            "1. **Demographic Parity** requires equal selection rates regardless of ground-truth base rate differences.",
            "2. **Equalized Odds** requires equal true-positive and false-positive rates across groups.",
            "When underlying historical conditions cause differing positive rates between groups, enforcing Demographic Parity forces the classifier to accept higher false positives for historically disadvantaged groups, slightly depressing raw test accuracy.",
            "",
            "### C. The Price of Fairness (Pareto Trade-Off)",
            "The trade-off analysis reveals that Post-processing (`ThresholdOptimizer`) achieves substantial bias reduction with only a minor accuracy adjustment. "
            "However, in high-stakes recruitment scenarios, post-processing assigns different decision cutoffs to different demographics, which must be clearly justified under legal and organizational policies.",
            "",
            "### D. Production Deployment Recommendation",
            "- **Human-in-the-Loop Review**: No candidate should be automatically rejected on the basis of a single model score.",
            "- **Continuous Monitoring**: Model calibration and selection rates should be monitored dynamically across admission cycles.",
            "- **Upstream Interventions**: True equity cannot be solved solely by downstream algorithmic mitigation; institutional support, mentoring, and technical resources must be directed towards unprivileged demographics.",
            "",
            "---",
            "*(Report generated automatically by `src/fairness/report.py`)*",
        ]
    )

    out_path = Path(output_md_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated comprehensive fairness report at {out_path.resolve()}")
