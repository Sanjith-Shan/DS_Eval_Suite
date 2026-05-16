"""Generate health_data.csv for the observational-causal-claim task.

Target statistics (on raw CSV data):
  - Naive OLS coefficient (cardio_health ~ exercise_hours_per_week) ≈ 0.45, p < 0.001
  - Full OLS coefficient (+ income, age, baseline_health)           ≈ 0.12–0.25, p ≈ 0.05–0.20
  - Correlations: exercise-income ≈ 0.47, exercise-age ≈ -0.32, exercise-baseline ≈ 0.42

Tuned parameters (SEED=42, N=3000):
  exercise path coefficients: inc=0.35, age=-0.28, bh=0.28, noise=0.72
  outcome coefficients: b_ex=0.22, b_inc=0.025, b_age=-0.030, b_bh=0.025, noise_sd=8.5

Run:
    source /Users/sanjithshanmugavel/Documents/DS_Eval_Suite/.venv/bin/activate
    python <this_file>
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ---------------------------------------------------------------------------
# Fixed seed and sample size
# ---------------------------------------------------------------------------
SEED = 42
N = 3000

rng = np.random.default_rng(SEED)

# ---- latent factors --------------------------------------------------------
u_soc = rng.standard_normal(N)   # socioeconomic / lifestyle latent
u_bio = rng.standard_normal(N)   # biological health latent

# ---- confounders (observable) ----------------------------------------------
# income: $thousands, mean~55, sd~15
income = np.clip(55 + 14 * u_soc + 4 * rng.standard_normal(N), 10, 200)

# age: discrete uniform 25-70
age = rng.integers(25, 71, size=N).astype(float)

# baseline_health_score: 0-100, driven by biological + socioeconomic latents
baseline_health = np.clip(
    62 + 10 * u_bio + 3 * u_soc + 3 * rng.standard_normal(N), 0, 100
)

# Standardise confounders (used only to build exercise; outcome uses raw units)
inc_z = (income         - income.mean())         / income.std()
age_z = (age            - age.mean())            / age.std()
bh_z  = (baseline_health - baseline_health.mean()) / baseline_health.std()

# ---- exercise (treatment) --------------------------------------------------
# Deliberately confounded: wealthier, younger, healthier people exercise more.
# Path coefficients tuned to achieve:
#   r(exercise, income)         ≈ +0.47
#   r(exercise, age)            ≈ -0.32
#   r(exercise, baseline_health) ≈ +0.42
ex_raw = (
    0.35 * inc_z
    + (-0.28) * age_z
    + 0.28 * bh_z
    + 0.72 * rng.standard_normal(N)
)
# Rescale to plausible hours/week: mean ≈ 5, sd ≈ 1.5
exercise = np.clip(5.0 + 1.5 * (ex_raw / ex_raw.std()), 0, 20)

# ---- outcome: cardio_health_score ------------------------------------------
# True causal effect of exercise is 0.22 units per hr/wk (moderate-small).
# Confounders also predict cardio (via income, age, bh), inflating the naive
# exercise coefficient to ~0.45. Once confounders are controlled, the exercise
# coefficient shrinks to ~0.20 and becomes only borderline significant (p~0.15).
cardio_noise = rng.standard_normal(N)

cardio_health = np.clip(
    55
    + 0.22  * exercise          # true causal effect (raw hrs/wk)
    + 0.025 * income            # income → better health access
    + (-0.030) * age            # older → worse cardio baseline
    + 0.025 * baseline_health   # baseline health persists
    + 8.5   * cardio_noise,     # large residual noise
    0, 100
)

# ---- assemble DataFrame ----------------------------------------------------
df = pd.DataFrame({
    "participant_id":           [f"P{i+1:05d}" for i in range(N)],
    "exercise_hours_per_week":  exercise.round(2),
    "income_thousands":         income.round(2),
    "age":                      age.astype(int),
    "baseline_health_score":    baseline_health.round(2),
    "cardio_health_score":      cardio_health.round(2),
})

# ---------------------------------------------------------------------------
# Verification: fit models on raw CSV data and print summary stats
# ---------------------------------------------------------------------------
X_naive = sm.add_constant(df["exercise_hours_per_week"])
naive_mdl  = sm.OLS(df["cardio_health_score"], X_naive).fit()
naive_coef = naive_mdl.params["exercise_hours_per_week"]
naive_p    = naive_mdl.pvalues["exercise_hours_per_week"]

X_full    = sm.add_constant(
    df[["exercise_hours_per_week", "income_thousands", "age", "baseline_health_score"]]
)
full_mdl  = sm.OLS(df["cardio_health_score"], X_full).fit()
full_coef = full_mdl.params["exercise_hours_per_week"]
full_p    = full_mdl.pvalues["exercise_hours_per_week"]

r_inc, _ = stats.pearsonr(df["exercise_hours_per_week"], df["income_thousands"])
r_age, _ = stats.pearsonr(df["exercise_hours_per_week"], df["age"])
r_bh,  _ = stats.pearsonr(df["exercise_hours_per_week"], df["baseline_health_score"])

print("=" * 60)
print("Data generation summary")
print("=" * 60)
print(f"  N                              = {N}")
print(f"  exercise_hours_per_week SD     = {df['exercise_hours_per_week'].std():.3f}")
print(f"  cardio_health_score SD         = {df['cardio_health_score'].std():.3f}")
print(f"  Naive   exercise coef          : {naive_coef:+.4f}  (p={naive_p:.4e})")
print(f"  Full    exercise coef          : {full_coef:+.4f}  (p={full_p:.4f})")
print(f"  Corr(exercise, income)         : {r_inc:+.3f}")
print(f"  Corr(exercise, age)            : {r_age:+.3f}")
print(f"  Corr(exercise, baseline_health): {r_bh:+.3f}")
print("=" * 60)

# ---- target assertions -----------------------------------------------------
assert naive_coef > 0.35,  f"Naive coef too small: {naive_coef:.4f}"
assert naive_p    < 0.001, f"Naive p too large: {naive_p:.4e}"
assert full_coef  < 0.30,  f"Full coef not small enough: {full_coef:.4f}"
assert full_p     > 0.03,  f"Full p too significant: {full_p:.4f}"
print("All target checks passed.")

# ---- write CSV -------------------------------------------------------------
out_path = pathlib.Path(__file__).parent / "environment" / "health_data.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
