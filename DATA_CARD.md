# Data Card: PlaceWise Campus Placement Datasets

## 1. Dataset Overview

PlaceWise utilizes a rigorous two-tier data strategy combining real baseline historical data with a production-scale synthetic generator:

1. **Real Baseline Dataset**: Kaggle "Campus Recruitment" dataset collected by Ben Roshan (215 rows). Provides historic baseline context.
2. **Synthetic Production-Scale Dataset**: A 5,000-row synthetic dataset generated using realistic academic and hiring correlations with ground-truth institutional bias injection (`src/data/synthetic_generator.py`).

---

## 2. Real vs. Synthetic Provenance Comparison

| Dimension | Real Baseline Dataset | Synthetic Production Dataset |
| :--- | :--- | :--- |
| **Source** | Kaggle (`benroshan/factors-affecting-campus-placement`) | Generative engine (`src/data/synthetic_generator.py`) |
| **Sample Size** | 215 rows | 5,000 rows (configurable via `config/synth_config.yaml`) |
| **Granular Attributes** | Lacks DSA problems, backlogs, hackathons, college tier, certifications, modern projects | Fully specified with 15 granular academic, demographic, and skill features |
| **Primary Role** | Baseline exploration and historical reference | Primary training, Optuna tuning, SHAP explainability, and Fairlearn audit |
| **Integrity Check** | Unvalidated manual CSV | Validated via Pandera DataFrameSchema and SHA-256 cryptographic manifest |
| **Bias Ground Truth** | Unknown / uncontrolled historical bias | Controlled ground truth: both unbiased and explicitly biased variants generated |

---

## 3. Cryptographic Data Manifest (`reports/data_manifest.json`)

To prevent silent dataset mutations and guarantee complete data provenance, every dataset is cryptographically fingerprinted:

| Dataset File | Rows | Columns | Size | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :--- |
| `data/synthetic/synthetic_placement_data.csv` | 5,000 | 17 | 374 KB | `d4524fbc9f918d75474aa6dd97311a2701634ddeda5958462fe3f883e4f1ffba` |
| `data/synthetic/synthetic_placement_data_biased.csv` | 5,000 | 17 | 378 KB | `9a298a2b87d2e92703d9a60b51953237a16b1ae0a52194a2561f643e70d160ac` |

---

## 4. Feature Dictionary & Pandera Validation Rules

All tabular datasets are validated at ingestion and loading against the `PLACEMENT_SCHEMA` Pandera DataFrameSchema (`src/data/schemas.py`):

| Feature Name | Dtype | Validation Constraints | Description |
| :--- | :---: | :--- | :--- |
| `cgpa` | Float | `Check.in_range(0.0, 10.0)` | Cumulative Grade Point Average on a 10-point scale |
| `tenth_pct` | Float | `Check.in_range(0.0, 100.0)` | Secondary education board examination percentage |
| `twelfth_pct` | Float | `Check.in_range(0.0, 100.0)` | Higher secondary education board examination percentage |
| `backlogs` | Integer | `Check.greater_than_or_equal_to(0)` | Number of active or uncleared course backlogs |
| `dsa_problems_solved` | Integer | `Check.greater_than_or_equal_to(0)` | Number of LeetCode/HackerRank coding problems solved |
| `projects` | Integer | `Check.greater_than_or_equal_to(0)` | Substantial software / technical engineering projects built |
| `internships` | Integer | `Check.greater_than_or_equal_to(0)` | Prior engineering or professional internships completed |
| `certifications` | Integer | `Check.greater_than_or_equal_to(0)` | Industry certifications (AWS, GCP, CKA, Coursera, etc.) |
| `hackathons` | Integer | `Check.greater_than_or_equal_to(0)` | Hackathons, datathons, or competitive coding contests participated |
| `communication_score` | Float | `Check.in_range(0.0, 100.0)` | Standardized assessment of soft skills and presentation |
| `aptitude_score` | Float | `Check.in_range(0.0, 100.0)` | Quantitative aptitude, logical reasoning, and verbal score |
| `branch` | String | `Check.isin([...])` | CS, IT, Electronics & Comm, Mechanical, Electrical, Civil |
| `college_tier` | Integer | `Check.isin([1, 2, 3])` | Institutional tier (1 = Premier, 2 = Reputed, 3 = Local) |
| `gender` | String | `Check.isin(["Male", "Female"])` | Self-reported student demographic gender |
| `work_experience` | Integer | `Check.isin([0, 1])` | Prior professional work experience (0 = No, 1 = Yes) |
| `placed` (Target) | Integer | `Check.isin([0, 1])` | Binary placement indicator (0 = Not Placed, 1 = Placed) |
| `salary_band` (Target) | String | `Check.isin(["Low", "Mid", "High", "Not Placed"])` | Compensation tier (Low <4 LPA, Mid 4-8 LPA, High >8 LPA) |

---

## 5. Synthetic Generative Dynamics & Latent Index

The synthetic generation process models realistic applicant dynamics:
1. **Academic Correlation**: Higher secondary percentage (`twelfth_pct`) is positively correlated ($r \approx 0.65$) with secondary percentage (`tenth_pct`).
2. **Backlog Impact**: Students with $\ge 2$ backlogs receive a negative adjustment to both CGPA and placement likelihood.
3. **Domain Coding Skew**: Computer Science and Information Technology students show higher mean numbers of DSA problems solved compared to traditional disciplines.
4. **Placement Latent Logit**:
   $$\text{logit}(P(\text{placed}=1)) = \beta_0 + \sum_{i} \beta_i \cdot x_{i, \text{normalized}} + \epsilon$$
   where weights reflect genuine industry placement priorities (DSA, CGPA, internships, aptitude, communication).
5. **Conditional Salary Assignment**: Salary bands are strictly modeled conditionally on placed candidates ($Y=1$). Non-placed candidates are labeled `"Not Placed"`.

---

## 6. Controlled Ground-Truth Bias Injection

To benchmark fairness audits (Fairlearn) and mitigation strategies against a verifiable ground truth, the generator implements an explicit bias injection mechanism (`--inject-bias`):

- **College Tier Bias Penalty**:
  - Tier 1: $\Delta \text{logit} = 0.00$
  - Tier 2: $\Delta \text{logit} = -0.40$
  - Tier 3: $\Delta \text{logit} = -1.25$
- **Gender Bias Penalty**:
  - Male: $\Delta \text{logit} = 0.00$
  - Female: $\Delta \text{logit} = -0.70$

When enabled, Tier-3 and Female applicants suffer an artificial penalty at identical skill levels, causing:
- Unbiased placement rate: **86.7%**
- Biased placement rate: **77.5%** (Tier-3 and Female selection rates heavily depressed)

---

## 7. Zero-PII Compliance & Privacy Hardening

- The dataset contains zero Personally Identifiable Information (no student names, roll numbers, IP addresses, email addresses, or phone numbers).
- In the FastAPI inference microservice (`api/audit.py`), applicant provenance is tracked using deterministic SHA-256 hashes of input feature dictionaries rather than personally identifiable tokens.

---

## 8. Limitations & Synthetic Data Disclosure

> [!CAUTION]
> **SYNTHETIC DATA DISCLAIMER**
> The synthetic datasets generated in this project are strictly mathematical simulations created for machine learning system design, algorithmic explainability, and bias auditing benchmarks.
> They **DO NOT** represent real students, actual university placement records, or factual hiring decisions.
> Predictions or metrics derived from this synthetic data must not be deployed in real-world recruitment without validated institutional data and formal ethics board approval.
