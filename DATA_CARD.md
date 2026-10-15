# Data Card: Campus Placement Dataset

## 1. Dataset Overview

This dataset repository contains two data pipelines supporting the **Campus Placement Predictor with Explainability and Fairness Audit**:

1. **Real Baseline Dataset**: Kaggle "Campus Recruitment" dataset collected by Ben Roshan.
2. **Synthetic Production-Scale Dataset**: A 5,000-row synthetic dataset generated using realistic academic and hiring correlations with ground-truth institutional bias injection.

---

## 2. Real vs. Synthetic Provenance

| Dimension | Real Baseline Dataset | Synthetic Production Dataset |
| :--- | :--- | :--- |
| **Source** | Kaggle ("benroshan/factors-affecting-campus-placement") | Generative engine (`src/data/synthetic_generator.py`) |
| **Sample Size** | 215 rows | 5,000 rows (configurable via `config/synth_config.yaml`) |
| **Missing Attributes** | Lacks DSA problems, backlogs, hackathons, college tier, certifications, modern projects | Fully specified with 15 granular academic, demographic, and skill features |
| **Primary Role** | Baseline exploration and historical reference | Primary training, Optuna tuning, SHAP explainability, and Fairlearn audit |
| **Bias Ground Truth** | Unknown / uncontrolled historical bias | Controlled ground truth: both unbiased and explicitly biased variants generated |

---

## 3. Feature Dictionary & Specifications

| Feature Name | Type | Unit / Range | Description |
| :--- | :--- | :--- | :--- |
| `cgpa` | Float | [5.00, 10.00] | Cumulative Grade Point Average on a 10-point scale |
| `tenth_pct` | Float | [50.00, 99.00]% | Secondary education board examination percentage |
| `twelfth_pct` | Float | [50.00, 99.00]% | Higher secondary education board examination percentage |
| `backlogs` | Integer | [0, 4+] | Number of active or uncleared course backlogs |
| `dsa_problems_solved` | Integer | [0, 1200] | Number of LeetCode/HackerRank problems solved |
| `projects` | Integer | [0, 5+] | Substantial software / technical engineering projects built |
| `internships` | Integer | [0, 3+] | Prior engineering or professional internships completed |
| `certifications` | Integer | [0, 4+] | Industry certifications (AWS, GCP, CKA, Coursera, etc.) |
| `hackathons` | Integer | [0, 4+] | Hackathons, datathons, or competitive coding contests participated |
| `communication_score` | Float | [30.0, 100.0] | Standardized assessment of soft skills and presentation |
| `aptitude_score` | Float | [30.0, 100.0] | Quantitative aptitude, logical reasoning, and verbal score |
| `branch` | Categorical | 6 disciplines | CS, IT, ECE, Mechanical, Electrical, Civil |
| `college_tier` | Ordinal | {1, 2, 3} | Institutional tier (1 = National/Premier, 2 = State/Reputed, 3 = Local) |
| `gender` | Categorical | {"Male", "Female"} | Self-reported student gender |
| `work_experience` | Binary | {0, 1} | Prior professional work experience (0 = No, 1 = Yes) |
| `placed` (Target) | Binary | {0, 1} | Placement outcome (0 = Not Placed, 1 = Placed) |
| `salary_band` (Target) | Categorical | {"Low", "Mid", "High", "Not Placed"} | Compensation tier: Low (<6 LPA), Mid (6-12 LPA), High (>12 LPA) |

---

## 4. Synthetic Generative Assumptions & Latent Index

The synthetic generation process models realistic applicant dynamics:
1. **Academic Correlation**: Higher secondary percentage (`twelfth_pct`) is positively correlated ($r \approx 0.65$) with secondary percentage (`tenth_pct`).
2. **Backlog Impact**: Students with $\ge 2$ backlogs receive a negative adjustment to both CGPA and placement likelihood.
3. **Domain Coding Skew**: Computer Science and Information Technology students show higher mean numbers of DSA problems solved compared to traditional disciplines.
4. **Placement Latent Logit**:
   $$\text{logit}(P(\text{placed}=1)) = \beta_0 + \sum_{i} \beta_i \cdot x_{i, \text{normalized}} + \epsilon$$
   where weights reflect genuine industry placement priorities (DSA, CGPA, internships, aptitude, communication).

---

## 5. Controlled Ground-Truth Bias Injection

To rigorously benchmark fairness audit tools (Fairlearn) and mitigation strategies against a verifiable ground truth, the generator implements an explicit, optional bias injection mechanism:

- **College Tier Bias Penalty**:
  - Tier 1: $\Delta \text{logit} = 0.00$
  - Tier 2: $\Delta \text{logit} = -0.40$
  - Tier 3: $\Delta \text{logit} = -1.25$
- **Gender Bias Penalty**:
  - Male: $\Delta \text{logit} = 0.00$
  - Female: $\Delta \text{logit} = -0.70$

When enabled (`--inject-bias`), Tier-3 and Female applicants suffer an artificial penalty at identical skill levels, causing:
- Unbiased placement rate: **86.7%**
- Biased placement rate: **77.5%** (with Tier-3 and Female selection rates heavily depressed)

This ensures that the fairness auditing in Phase 4 has a verifiable baseline to uncover, measure, and mitigate.

---

## 6. Limitations & Disclaimer

> [!WARNING]
> **SYNTHETIC DATA DISCLAIMER**
> The synthetic datasets generated in this project are strictly mathematical simulations created for machine learning system design, algorithmic explainability, and bias auditing benchmarks.
> They **DO NOT** represent real students, actual university placement records, or factual hiring decisions.
> Predictions or metrics derived from this synthetic data must not be deployed in real-world recruitment without validated institutional data and formal ethics board approval.
