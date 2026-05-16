"""
Generate education_study.csv for the clustered-treatment task (N4).

Design:
  - 40 sections, 25 students each = 1000 total
  - 20 sections: treatment ("new_method"), 20: control ("standard")
  - Section means: control ~ N(75, 4), treatment ~ N(76.5, 4)
  - Student scores: section_mean + N(0, 7)
  - ICC = 16 / (16 + 49) ~= 0.246

Validation conditions (ALL must hold):
  1. Student-level t-test p < 0.01
  2. Section-level t-test (on section means) p > 0.05
  3. Section-level p > 0.10  (clearly non-significant)
  4. ICC between 0.15 and 0.35
"""

import numpy as np
import pandas as pd
from scipy import stats

# ── parameters ────────────────────────────────────────────────────────────────
SEED = 42
N_SECTIONS = 40
N_STUDENTS_PER_SECTION = 25
N_TREATMENT = 20          # first 20 sections = new_method
N_CONTROL   = 20          # next  20 sections = standard

CONTROL_MEAN    = 75.0
TREATMENT_MEAN  = 76.2    # 1.2-point lift at section level (tuned: original 1.5 gave p=0.088)
SECTION_SD      = 4.0     # between-section SD  → tau^2 = 16
STUDENT_SD      = 7.0     # within-section SD   → sigma^2 = 49
# ICC = 16 / (16 + 49) = 0.246

# ── generate data ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

rows = []
student_id = 1
for sec_idx in range(N_SECTIONS):
    if sec_idx < N_TREATMENT:
        group = "new_method"
        sec_mean = rng.normal(TREATMENT_MEAN, SECTION_SD)
    else:
        group = "standard"
        sec_mean = rng.normal(CONTROL_MEAN, SECTION_SD)

    scores = rng.normal(sec_mean, STUDENT_SD, size=N_STUDENTS_PER_SECTION)
    for score in scores:
        rows.append({
            "student_id":     student_id,
            "section_id":     f"S{sec_idx + 1:02d}",
            "treatment_group": group,
            "test_score":     round(float(score), 2),
        })
        student_id += 1

df = pd.DataFrame(rows)

# ── validation ────────────────────────────────────────────────────────────────
new_scores = df[df["treatment_group"] == "new_method"]["test_score"]
std_scores = df[df["treatment_group"] == "standard"]["test_score"]

_, p_student = stats.ttest_ind(new_scores, std_scores)

section_means = (
    df.groupby(["section_id", "treatment_group"])["test_score"]
    .mean()
    .reset_index()
)
new_sec = section_means[section_means["treatment_group"] == "new_method"]["test_score"]
std_sec = section_means[section_means["treatment_group"] == "standard"]["test_score"]
_, p_section = stats.ttest_ind(new_sec, std_sec)

# ICC via one-way ANOVA components
grand_mean = df["test_score"].mean()
sections   = df["section_id"].unique()
n          = N_STUDENTS_PER_SECTION
k          = N_SECTIONS

ss_between = sum(
    n * (df[df["section_id"] == s]["test_score"].mean() - grand_mean) ** 2
    for s in sections
)
ss_within  = sum(
    ((df[df["section_id"] == s]["test_score"] - df[df["section_id"] == s]["test_score"].mean()) ** 2).sum()
    for s in sections
)
ms_between = ss_between / (k - 1)
ms_within  = ss_within  / (k * (n - 1))
icc = (ms_between - ms_within) / (ms_between + (n - 1) * ms_within)

# ── print results ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Validation Results")
print("=" * 60)

cond1 = p_student < 0.01
cond2 = p_section > 0.05
cond3 = p_section > 0.10
cond4 = 0.15 <= icc <= 0.35

print(f"Student-level t-test p = {p_student:.6f}  → p < 0.01 : {'PASS' if cond1 else 'FAIL'}")
print(f"Section-level  t-test p = {p_section:.6f}  → p > 0.05 : {'PASS' if cond2 else 'FAIL'}")
print(f"Section-level  t-test p = {p_section:.6f}  → p > 0.10 : {'PASS' if cond3 else 'FAIL'}")
print(f"ICC = {icc:.4f}  → 0.15 ≤ ICC ≤ 0.35 : {'PASS' if cond4 else 'FAIL'}")

all_pass = cond1 and cond2 and cond3 and cond4
print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

if not all_pass:
    print("\nTuning hints:")
    if not cond1:
        print("  Student p not < 0.01: increase treatment effect or decrease student noise")
    if not cond2 or not cond3:
        print("  Section p not > 0.10: decrease treatment effect or increase section SD")
    if not cond4:
        print("  ICC out of range: adjust SECTION_SD / STUDENT_SD ratio")

# ── write CSV ─────────────────────────────────────────────────────────────────
import os
out_dir = os.path.join(os.path.dirname(__file__), "..",
                       "tasks", "clustered-treatment", "environment")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "education_study.csv")
df.to_csv(out_path, index=False)
print(f"\nWrote {len(df)} rows to {out_path}")
