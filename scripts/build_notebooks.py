"""Generates 01_eda.ipynb, 02_modeling.ipynb, 03_explainability.ipynb, and 04_fairness_audit.ipynb using nbformat."""

from pathlib import Path
import nbformat as nbf


def generate_eda_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 01: Exploratory Data Analysis (EDA)\n"
            "## Campus Placement Predictor\n\n"
            "This notebook explores the dataset distribution, missing values, correlation structures, "
            "and demographic placement rates across gender and college tier."
        ),
        nbf.v4.new_code_cell(
            "%matplotlib inline\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "from src.data.loader import DataLoader\n"
            "from src.utils.config import load_config\n\n"
            "# Configure visual styling\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "plt.rcParams['font.size'] = 11"
        ),
        nbf.v4.new_markdown_cell("### 1. Load Datasets (Unbiased vs Biased)"),
        nbf.v4.new_code_cell(
            "loader = DataLoader()\n"
            "df_unbiased = loader.load_synthetic_data(biased=False)\n"
            "df_biased = loader.load_synthetic_data(biased=True)\n"
            "print(f'Unbiased shape: {df_unbiased.shape}')\n"
            "print(f'Biased shape:   {df_biased.shape}')\n"
            "df_unbiased.head()"
        ),
        nbf.v4.new_markdown_cell("### 2. Missing Values and Schema Summary"),
        nbf.v4.new_code_cell(
            "print('Missing values check:')\n"
            "display(df_unbiased.isnull().sum())\n\n"
            "print('\\nSummary Statistics:')\n"
            "display(df_unbiased.describe().round(2))"
        ),
        nbf.v4.new_markdown_cell("### 3. Class Balance: Placement & Salary Bands"),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
            "# Placement outcome\n"
            "df_unbiased['placed'].value_counts(normalize=True).plot.bar(\n"
            "    ax=axes[0], color=['#2b5c8f', '#e07a5f'], edgecolor='black'\n"
            ")\n"
            "axes[0].set_title('Overall Placement Rate (Unbiased)')\n"
            "axes[0].set_xticklabels(['Placed (1)', 'Not Placed (0)'], rotation=0)\n"
            "axes[0].set_ylabel('Proportion')\n\n"
            "# Salary bands\n"
            "df_unbiased[df_unbiased['placed'] == 1]['salary_band'].value_counts().plot.bar(\n"
            "    ax=axes[1], color='#3d405b', edgecolor='black'\n"
            ")\n"
            "axes[1].set_title('Compensation Bands for Placed Students')\n"
            "axes[1].set_ylabel('Count')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("### 4. Technical Skill Distributions"),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n"
            "sns.histplot(df_unbiased['cgpa'], kde=True, ax=axes[0, 0], color='#2a9d8f')\n"
            "axes[0, 0].set_title('CGPA Distribution')\n\n"
            "sns.histplot(df_unbiased['dsa_problems_solved'], kde=True, ax=axes[0, 1], color='#e76f51')\n"
            "axes[0, 1].set_title('DSA Problems Solved Distribution')\n\n"
            "sns.histplot(df_unbiased['aptitude_score'], kde=True, ax=axes[1, 0], color='#457b9d')\n"
            "axes[1, 0].set_title('Aptitude Score Distribution')\n\n"
            "sns.histplot(df_unbiased['communication_score'], kde=True, ax=axes[1, 1], color='#1d3557')\n"
            "axes[1, 1].set_title('Communication Score Distribution')\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("### 5. Correlation Heatmap"),
        nbf.v4.new_code_cell(
            "numeric_cols = df_unbiased.select_dtypes(include=[np.number]).columns\n"
            "corr = df_unbiased[numeric_cols].corr()\n\n"
            "plt.figure(figsize=(10, 8))\n"
            "sns.heatmap(corr, cmap='vlag', annot=True, fmt='.2f', vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})\n"
            "plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("### 6. Subgroup Placement Rates: Gender and College Tier"),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
            "# Gender\n"
            "gender_comp = pd.DataFrame({\n"
            "    'Unbiased': df_unbiased.groupby('gender')['placed'].mean() * 100,\n"
            "    'Biased': df_biased.groupby('gender')['placed'].mean() * 100\n"
            "})\n"
            "gender_comp.plot.bar(ax=axes[0], color=['#2b5c8f', '#e63946'], edgecolor='black')\n"
            "axes[0].set_title('Placement Rate by Gender (%)')\n"
            "axes[0].set_ylabel('Placement Rate (%)')\n"
            "axes[0].set_xticklabels(gender_comp.index, rotation=0)\n\n"
            "# College Tier\n"
            "tier_comp = pd.DataFrame({\n"
            "    'Unbiased': df_unbiased.groupby('college_tier')['placed'].mean() * 100,\n"
            "    'Biased': df_biased.groupby('college_tier')['placed'].mean() * 100\n"
            "})\n"
            "tier_comp.plot.bar(ax=axes[1], color=['#2b5c8f', '#e63946'], edgecolor='black')\n"
            "axes[1].set_title('Placement Rate by College Tier (%)')\n"
            "axes[1].set_ylabel('Placement Rate (%)')\n"
            "axes[1].set_xticklabels([f'Tier {t}' for t in tier_comp.index], rotation=0)\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    ]

    out_file = Path("notebooks/01_eda.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {out_file}")


def generate_modeling_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 02: Model Training, Evaluation & Benchmarking\n"
            "## Campus Placement Predictor\n\n"
            "This notebook demonstrates the leak-free preprocessing pipeline, baseline models, "
            "Optuna-tuned XGBoost model, multiclass salary classification, and inference."
        ),
        nbf.v4.new_code_cell(
            "%matplotlib inline\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import json\n\n"
            "from src.models.predict import PlacementPredictor\n"
            "from src.models.registry import ModelRegistry\n"
            "from src.utils.config import load_config"
        ),
        nbf.v4.new_markdown_cell("### 1. Load Evaluation Metrics from Registry"),
        nbf.v4.new_code_cell(
            "with open('reports/metrics.json', 'r') as f:\n"
            "    metrics = json.load(f)\n\n"
            "print('Test Set Performance Comparison:')\n"
            "summary = pd.DataFrame({\n"
            "    'Model': ['Logistic Regression', 'Random Forest', 'Tuned XGBoost'],\n"
            "    'ROC-AUC': [\n"
            "        metrics['test']['logistic_regression']['roc_auc'],\n"
            "        metrics['test']['random_forest']['roc_auc'],\n"
            "        metrics['test']['xgboost']['roc_auc'],\n"
            "    ],\n"
            "    'F1 Score': [\n"
            "        metrics['test']['logistic_regression']['f1'],\n"
            "        metrics['test']['random_forest']['f1'],\n"
            "        metrics['test']['xgboost']['f1'],\n"
            "    ],\n"
            "    'Brier Score (Calibration)': [\n"
            "        metrics['test']['logistic_regression']['brier_score'],\n"
            "        metrics['test']['random_forest']['brier_score'],\n"
            "        metrics['test']['xgboost']['brier_score'],\n"
            "    ]\n"
            "})\n"
            "display(summary.round(4))"
        ),
        nbf.v4.new_markdown_cell("### 2. Display Generated ROC & PR Curves"),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "display(Image(filename='reports/figures/roc_curves_comparison.png'))\n"
            "display(Image(filename='reports/figures/pr_curves_comparison.png'))\n"
            "display(Image(filename='reports/figures/calibration_curves.png'))"
        ),
        nbf.v4.new_markdown_cell("### 3. Interactive Student Prediction Walkthrough"),
        nbf.v4.new_code_cell(
            "predictor = PlacementPredictor(\n"
            "    placement_model_name='best_placement_model',\n"
            "    salary_model_name='salary_band_model'\n"
            ")\n\n"
            "sample_student = {\n"
            "    'cgpa': 8.65,\n"
            "    'tenth_pct': 88.0,\n"
            "    'twelfth_pct': 85.5,\n"
            "    'backlogs': 0,\n"
            "    'dsa_problems_solved': 450,\n"
            "    'projects': 4,\n"
            "    'internships': 2,\n"
            "    'certifications': 2,\n"
            "    'hackathons': 2,\n"
            "    'communication_score': 82.0,\n"
            "    'aptitude_score': 86.0,\n"
            "    'branch': 'Computer Science',\n"
            "    'college_tier': 1,\n"
            "    'gender': 'Female',\n"
            "    'work_experience': 1,\n"
            "}\n\n"
            "pred_res = predictor.predict_student(sample_student)\n"
            "print('Prediction Output:')\n"
            "for k, v in pred_res.items():\n"
            "    print(f'  {k}: {v}')"
        ),
    ]

    out_file = Path("notebooks/02_modeling.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {out_file}")


def generate_explainability_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 03: Model Explainability with SHAP & Counterfactual Simulation\n"
            "## Campus Placement Predictor\n\n"
            "This notebook demonstrates:\n"
            "1. Global SHAP feature importances and beeswarm summary plots.\n"
            "2. Local waterfall plots for individual candidate predictions.\n"
            "3. Exact SHAP additivity verification in margin log-odds.\n"
            "4. Calibrated conversion from log-odds to percentage-point probability contributions.\n"
            "5. Actionable what-if counterfactual simulations."
        ),
        nbf.v4.new_code_cell(
            "%matplotlib inline\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from IPython.display import Image, display\n\n"
            "from src.data.loader import DataLoader\n"
            "from src.explainability.counterfactual import WhatIfSimulator\n"
            "from src.explainability.shap_global import get_global_feature_importance\n"
            "from src.explainability.shap_local import explain_student_prediction\n"
            "from src.models.registry import ModelRegistry\n"
            "from src.utils.config import load_config"
        ),
        nbf.v4.new_markdown_cell("### 1. Load Trained Model & Background Data"),
        nbf.v4.new_code_cell(
            "config = load_config('config/config.yaml')\n"
            "num_cols = config['features']['numerical']\n"
            "cat_cols = config['features']['categorical']\n\n"
            "registry = ModelRegistry()\n"
            "pipeline, meta = registry.load_model('best_placement_model')\n\n"
            "loader = DataLoader()\n"
            "df = loader.load_synthetic_data(biased=False)\n"
            "sample_df = df.sample(300, random_state=42)\n"
            "print('Loaded trained model and sample dataset successfully.')"
        ),
        nbf.v4.new_markdown_cell("### 2. Global SHAP Summary & Feature Rankings"),
        nbf.v4.new_code_cell(
            "df_imp = get_global_feature_importance(pipeline, sample_df, num_cols, cat_cols)\n"
            "print('Top 10 Global Features by Mean |SHAP| (Log-Odds Impact):')\n"
            "display(df_imp.head(10))\n\n"
            "display(Image(filename='reports/figures/shap_beeswarm.png'))\n"
            "display(Image(filename='reports/figures/shap_feature_importance_bar.png'))"
        ),
        nbf.v4.new_markdown_cell("### 3. Local Explanation & Mathematical Additivity Verification"),
        nbf.v4.new_code_cell(
            "candidate = df.iloc[10].to_dict()\n"
            "explanation = explain_student_prediction(pipeline, candidate, num_cols, cat_cols)\n\n"
            "print(f'Base Probability:   {explanation[\"base_probability\"]:.1%}')\n"
            "print(f'Model Probability:  {explanation[\"model_probability\"]:.1%}')\n"
            "print(f'Net Prob Delta:     {explanation[\"total_prob_delta_points\"]:+.1f} points')\n"
            "print(f'Additivity Verified: {explanation[\"additivity_check\"][\"is_additive\"]} (Error: {explanation[\"additivity_check\"][\"additivity_error\"]:.2e})\\n')\n\n"
            "print('Plain-English Impact Drivers:')\n"
            "for stmt in explanation['plain_english_summary']:\n"
            "    print(f'  • {stmt}')"
        ),
        nbf.v4.new_markdown_cell("### 4. Actionable Counterfactual What-If Simulation"),
        nbf.v4.new_code_cell(
            "simulator = WhatIfSimulator(pipeline)\n\n"
            "# What if student solves 200 more DSA problems and completes 1 additional internship?\n"
            "what_if_res = simulator.simulate(\n"
            "    candidate,\n"
            "    {'dsa_problems_solved': candidate['dsa_problems_solved'] + 200, 'internships': candidate['internships'] + 1}\n"
            ")\n\n"
            "print('What-If Simulation Results:')\n"
            "print(f'  Baseline Probability: {what_if_res[\"baseline_probability\"]:.1%}')\n"
            "print(f'  Modified Probability: {what_if_res[\"modified_probability\"]:.1%}')\n"
            "print(f'  Net Gain:             {what_if_res[\"delta_probability_points\"]:+.1f} percentage points')\n"
            "print(f'  Outcome Shift:        {what_if_res[\"status_shift\"]}')\n\n"
            "# Recommended interventions\n"
            "recs = simulator.recommend_interventions(candidate, target_probability=0.85)\n"
            "if recs:\n"
            "    print('\\nTop Recommended Interventions to Reach 85% Target:')\n"
            "    display(pd.DataFrame(recs))"
        ),
    ]

    out_file = Path("notebooks/03_explainability.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {out_file}")


def generate_fairness_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 04: Algorithmic Fairness Audit & Mitigation\n"
            "## Campus Placement Predictor\n\n"
            "This notebook audits the placement predictor for disparities across Gender and College Tier using Fairlearn,\n"
            "evaluates 'Fairness through Unawareness', benchmarks three mitigation algorithms (Reweighing, ThresholdOptimizer, ExponentiatedGradient),\n"
            "and visualizes the Pareto accuracy vs fairness frontier."
        ),
        nbf.v4.new_code_cell(
            "%matplotlib inline\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from IPython.display import Image, display\n\n"
            "from src.fairness.audit import FairnessAuditor\n"
            "from src.utils.config import load_config"
        ),
        nbf.v4.new_markdown_cell("### 1. Run or Load Full Fairness Benchmark"),
        nbf.v4.new_code_cell(
            "auditor = FairnessAuditor()\n"
            "audit_results = auditor.run_full_audit()\n"
            "tradeoffs = pd.DataFrame(audit_results['tradeoffs']['college_tier'])\n"
            "print('College Tier Mitigation Trade-Off Summary:')\n"
            "display(tradeoffs)"
        ),
        nbf.v4.new_markdown_cell("### 2. Disaggregated Selection Rates and Pareto Frontier"),
        nbf.v4.new_code_cell(
            "display(Image(filename='reports/figures/fairness_selection_rates.png'))\n"
            "display(Image(filename='reports/figures/fairness_tradeoff_pareto.png'))"
        ),
        nbf.v4.new_markdown_cell("### 3. Read Automated Markdown Fairness Report"),
        nbf.v4.new_code_cell(
            "with open('reports/fairness_report.md', 'r') as f:\n"
            "    report_text = f.read()\n"
            "print(report_text[:1200] + '\\n...\\n[See full report at reports/fairness_report.md]')"
        ),
    ]

    out_file = Path("notebooks/04_fairness_audit.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated {out_file}")


if __name__ == "__main__":
    generate_eda_notebook()
    generate_modeling_notebook()
    generate_explainability_notebook()
    generate_fairness_notebook()
