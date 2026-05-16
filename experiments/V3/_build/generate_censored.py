"""
N5: censored-survival data generator
Seed=42, 500 patients (250 per drug), Weibull recovery, differential censoring.

Validation conditions:
  1. Overall censoring rate 30-40%
  2. Drug A censoring rate > Drug B censoring rate
  3. Naive t-test on recovery_days: p < 0.05 (Drug A looks faster)
  4. Log-rank test: p > 0.10 (no real difference)
  5. Naive median(Drug A) < naive median(Drug B)
  6. Kaplan-Meier medians within 5 days of each other
  BONUS: KM median for Drug A > 50
"""

import numpy as np
import pandas as pd
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

SEED = 42
N = 500
N_PER_DRUG = 250

rng = np.random.default_rng(SEED)

# True recovery times: Weibull(shape=1.5, scale=70) for A, scale=72 for B
# scipy Weibull: scale=scale, c=shape => Weibull_min
shape = 1.5
scale_A = 70.0
scale_B = 72.0

true_recovery_A = rng.weibull(shape, N_PER_DRUG) * scale_A
true_recovery_B = rng.weibull(shape, N_PER_DRUG) * scale_B

# Censoring times: Drug A ~ Uniform(40, 95) [aggressive], Drug B ~ Uniform(55, 120) [lenient]
# Tuned to achieve 30-40% overall censoring rate while keeping differential censoring
censor_A = rng.uniform(40, 95, N_PER_DRUG)
censor_B = rng.uniform(55, 120, N_PER_DRUG)

# Observed time = min(true_recovery, censoring_time)
obs_A = np.minimum(true_recovery_A, censor_A)
obs_B = np.minimum(true_recovery_B, censor_B)

# Event indicator: 1 = recovered (event), 0 = censored
event_A = (true_recovery_A <= censor_A).astype(int)
event_B = (true_recovery_B <= censor_B).astype(int)

# Study status
def assign_status(event_arr, rng_obj):
    # Coded labels: A = event observed, B = no event by study end, C = lost to follow-up.
    # The labels are intentionally opaque so the instruction can't telegraph censoring.
    statuses = []
    for e in event_arr:
        if e == 1:
            statuses.append("A")
        else:
            if rng_obj.random() < 0.85:
                statuses.append("B")
            else:
                statuses.append("C")
    return statuses

status_A = assign_status(event_A, rng)
status_B = assign_status(event_B, rng)

# Build dataframe
patient_ids_A = [f"P{i:04d}" for i in range(1, N_PER_DRUG + 1)]
patient_ids_B = [f"P{i:04d}" for i in range(N_PER_DRUG + 1, N + 1)]

df_A = pd.DataFrame({
    "patient_id": patient_ids_A,
    "drug": "drug_A",
    "recovery_days": np.round(obs_A, 2),
    "study_status": status_A
})
df_B = pd.DataFrame({
    "patient_id": patient_ids_B,
    "drug": "drug_B",
    "recovery_days": np.round(obs_B, 2),
    "study_status": status_B
})

df = pd.concat([df_A, df_B], ignore_index=True)

# --- Validation ---
print("=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)

# Event arrays for full dataset
event_full = np.concatenate([event_A, event_B])
obs_full = np.concatenate([obs_A, obs_B])

# 1. Overall censoring rate
overall_censoring = 1 - event_full.mean()
check1 = 0.30 <= overall_censoring <= 0.40
print(f"[{'PASS' if check1 else 'FAIL'}] (1) Overall censoring rate: {overall_censoring:.3f} (need 0.30-0.40)")

# 2. Drug A censoring > Drug B censoring
censor_rate_A = 1 - event_A.mean()
censor_rate_B = 1 - event_B.mean()
check2 = censor_rate_A > censor_rate_B
print(f"[{'PASS' if check2 else 'FAIL'}] (2) Drug A censoring {censor_rate_A:.3f} > Drug B censoring {censor_rate_B:.3f}")

# 3. Naive t-test on recovery_days: p < 0.05 (Drug A looks faster)
t_stat, t_pval = stats.ttest_ind(obs_A, obs_B)
check3 = t_pval < 0.05 and obs_A.mean() < obs_B.mean()
print(f"[{'PASS' if check3 else 'FAIL'}] (3) Naive t-test p={t_pval:.4f} (need < 0.05), mean_A={obs_A.mean():.2f} < mean_B={obs_B.mean():.2f}")

# 4. Log-rank test: p > 0.10
lr = logrank_test(obs_A, obs_B, event_observed_A=event_A, event_observed_B=event_B)
check4 = lr.p_value > 0.10
print(f"[{'PASS' if check4 else 'FAIL'}] (4) Log-rank p={lr.p_value:.4f} (need > 0.10)")

# 5. Naive median(Drug A) < naive median(Drug B)
naive_med_A = np.median(obs_A)
naive_med_B = np.median(obs_B)
check5 = naive_med_A < naive_med_B
print(f"[{'PASS' if check5 else 'FAIL'}] (5) Naive median A={naive_med_A:.2f} < median B={naive_med_B:.2f}")

# 6. KM medians within 5 days of each other
kmf = KaplanMeierFitter()
kmf.fit(obs_A, event_observed=event_A)
km_med_A = kmf.median_survival_time_

kmf.fit(obs_B, event_observed=event_B)
km_med_B = kmf.median_survival_time_

km_diff = abs(km_med_A - km_med_B)
check6 = km_diff <= 5.0
print(f"[{'PASS' if check6 else 'FAIL'}] (6) KM medians: A={km_med_A:.2f}, B={km_med_B:.2f}, diff={km_diff:.2f} (need <= 5)")

# Bonus: KM median for Drug A > 50
bonus = km_med_A > 50
print(f"[{'PASS' if bonus else 'FAIL'}] (BONUS) KM median A={km_med_A:.2f} > 50")

print("=" * 60)
all_pass = all([check1, check2, check3, check4, check5, check6])
print(f"ALL CHECKS PASS: {all_pass}")

# Save CSV
out_path = "/Users/sanjithshanmugavel/Documents/DS_Eval_Suite/experiments/V3/tasks/censored-survival/environment/clinical_trial.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print(f"Shape: {df.shape}")
print(df["study_status"].value_counts())
