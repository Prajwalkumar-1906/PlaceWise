# Data Quality and Pandera Schema Conformity Report

## 1. Executive Summary
- **Audit Status**: ✅ PASSED (100% Schema Compliant)
- **Total Rows Audited**: `5,000`
- **Total Features**: `17`
- **Dataset Completeness**: `100.0%` (0 missing values)
- **Schema Engine**: Pandera v0.34 & Pydantic v2

---

## 2. Statistical Column Integrity Profile

| Feature | Type | Non-Null % | Min | Mean | Max | Unique Values | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `cgpa` | float64 | 100.0% | 5.00 | 7.56 | 10.00 | 486 | ✅ Valid |
| `tenth_pct` | float64 | 100.0% | 50.00 | 78.44 | 99.00 | 2641 | ✅ Valid |
| `twelfth_pct` | float64 | 100.0% | 50.00 | 77.15 | 99.00 | 2688 | ✅ Valid |
| `backlogs` | int64 | 100.0% | 0.00 | 0.49 | 4.00 | 5 | ✅ Valid |
| `dsa_problems_solved` | int64 | 100.0% | 0.00 | 218.71 | 1200.00 | 817 | ✅ Valid |
| `projects` | int64 | 100.0% | 0.00 | 2.04 | 5.00 | 6 | ✅ Valid |
| `internships` | int64 | 100.0% | 0.00 | 0.80 | 3.00 | 4 | ✅ Valid |
| `certifications` | int64 | 100.0% | 0.00 | 1.34 | 4.00 | 5 | ✅ Valid |
| `hackathons` | int64 | 100.0% | 0.00 | 0.77 | 4.00 | 5 | ✅ Valid |
| `communication_score` | float64 | 100.0% | 30.00 | 70.50 | 100.00 | 608 | ✅ Valid |
| `aptitude_score` | float64 | 100.0% | 30.00 | 71.56 | 100.00 | 580 | ✅ Valid |
| `branch` | str | 100.0% | - | - | - | 6 | ✅ Valid |
| `college_tier` | int64 | 100.0% | 1.00 | 2.37 | 3.00 | 3 | ✅ Valid |
| `gender` | str | 100.0% | - | - | - | 2 | ✅ Valid |
| `work_experience` | int64 | 100.0% | 0.00 | 0.25 | 1.00 | 2 | ✅ Valid |
| `placed` | int64 | 100.0% | 0.00 | 0.87 | 1.00 | 2 | ✅ Valid |
| `salary_band` | str | 100.0% | - | - | - | 4 | ✅ Valid |

---

## 3. Schema Constraints & Domain Boundary Verification

- **CGPA**: Domain bounds $[0.0, 10.0]$ strictly checked.
- **10th & 12th Percentiles**: Domain bounds $[0.0, 100.0]$ strictly checked.
- **Academic Backlogs**: Strictly non-negative integer values.
- **DSA & Technical Metrics**: Strictly non-negative counts.
- **Institutional College Tier**: Enum `[1, 2, 3]` enforced.
- **Demographic Gender**: Enum `['Male', 'Female']` enforced.

---
*(Report generated automatically by `src/data/schemas.py`)*