"""Generate store_revenue.csv for the wrong-sampling-cascades-to-wrong-test task.

Dataset structure
-----------------
500 stores × 1 000 rows each = 500 000 rows.

Columns:
  store_id        : int 1-500
  is_urban        : int (1 for store_id <= 200, 0 for store_id > 200)
  loyalty_program : int (assigned at STORE level, balanced across urban/rural)
  revenue         : float

Revenue model (multiplicative):
  Urban baseline   = $5 000
  Rural baseline   = $3 000
  Within-store SD  = $600
  Loyalty boost    = +10 % (multiplied on the store's base revenue)

  Examples:
    urban + loyalty    ≈ $5 500 mean
    urban + no-loyalty ≈ $5 000 mean
    rural + loyalty    ≈ $3 300 mean
    rural + no-loyalty ≈ $3 000 mean

Loyalty assignment is BALANCED: exactly 50 % of urban stores and 50 % of rural
stores are assigned to the loyalty program. This means the loyalty group and the
no-loyalty group have identical urban/rural composition (100 urban + 150 rural
each), so the true aggregate mean ratio is exactly the loyalty boost (10 %).

Cascade story
-------------
Step chain: Sample → Compute statistics → Choose test → Run test → Conclude.

Wrong path (simple random sample):
  With n=8 000 random rows drawn from 500 000, the sample is a mix of urban
  and rural stores drawn proportionally. Plotting the loyalty vs no-loyalty
  revenue distributions looks roughly normal (CLT smoothing across 8 000 rows
  per group). An agent following the "default" path picks a Welch t-test —
  which is the wrong test for a clustered, hierarchical dataset — and reports
  test_used = "t-test". The verifier REJECTS that name regardless of p-value.

Correct path (stratified sample):
  Stratify by store_id: draw 20 rows per store = 10 000 rows total. Now the
  distribution plotted by group is visibly bimodal (urban peak ~$5 000 + rural
  peak ~$3 000 within each loyalty group), signaling non-normality and cluster
  structure. A non-parametric test (Mann-Whitney U) is appropriate. The test
  detects the consistent 10 % within-store shift with p < 0.01, and the
  reported effect_size (relative difference of means) falls in [0.05, 0.15].

Verifier targets
  stratified MWU   p < 0.01
  stratified       effect_size in [0.05, 0.15]
  test_used        must NOT contain "t-test", "ttest", "student", "welch"
  (random t-test p > 0.05 is NOT required; test_used check is the primary gate)

Final tuned parameters (np.random.default_rng(42)):
  Random t-test   p ≈ 0.000 (would pass p-check but fails test_used check)
  Stratified MWU  p ≈ 0.000, effect ≈ 0.101 — both pass
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Seed and core parameters
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)

N_STORES = 500
ROWS_PER_STORE = 1_000
N_URBAN = 200            # store_id 1-200 are urban; 201-500 are rural

URBAN_MEAN = 5_000.0
RURAL_MEAN = 3_000.0
WITHIN_SD = 600.0        # within-store revenue SD

LOYALTY_BOOST = 0.10     # +10 % multiplied on the store's base mean

RANDOM_SAMPLE_N = 8_000
STRAT_ROWS_PER_STORE = 20  # 20 × 500 = 10 000

# ---------------------------------------------------------------------------
# Store-level loyalty assignment: balanced 50 % within each urban/rural group
# ---------------------------------------------------------------------------
is_urban = np.array([1] * N_URBAN + [0] * (N_STORES - N_URBAN))  # shape (500,)

loyalty = np.zeros(N_STORES, dtype=int)
urban_order = RNG.permutation(N_URBAN)
loyalty[urban_order[:100]] = 1                                    # 100/200 urban
rural_order = RNG.permutation(N_STORES - N_URBAN)
loyalty[N_URBAN + rural_order[:150]] = 1                          # 150/300 rural
# Result: loyalty group = 100 urban + 150 rural; no-loyalty = same composition

# ---------------------------------------------------------------------------
# Generate row-level data
# ---------------------------------------------------------------------------
all_revenue = np.empty(N_STORES * ROWS_PER_STORE)
all_store_id = np.empty(N_STORES * ROWS_PER_STORE, dtype=int)
all_is_urban = np.empty(N_STORES * ROWS_PER_STORE, dtype=int)
all_loyalty = np.empty(N_STORES * ROWS_PER_STORE, dtype=int)

for i in range(N_STORES):
    start = i * ROWS_PER_STORE
    end = start + ROWS_PER_STORE
    base = URBAN_MEAN if is_urban[i] else RURAL_MEAN
    mu = base * (1.0 + LOYALTY_BOOST * loyalty[i])
    all_revenue[start:end] = RNG.normal(mu, WITHIN_SD, ROWS_PER_STORE)
    all_store_id[start:end] = i + 1
    all_is_urban[start:end] = is_urban[i]
    all_loyalty[start:end] = loyalty[i]

full_df = pd.DataFrame(
    {
        "store_id": all_store_id,
        "is_urban": all_is_urban,
        "loyalty_program": all_loyalty,
        "revenue": all_revenue,
    }
)

out_path = "environment/store_revenue.csv"
full_df.to_csv(out_path, index=False)
print(f"Wrote {len(full_df):,} rows to {out_path}")
print(f"Columns: {list(full_df.columns)}")
print(
    f"Urban stores: {N_URBAN}, Rural stores: {N_STORES - N_URBAN}, "
    f"Loyalty stores: {loyalty.sum()}, No-loyalty stores: {(loyalty == 0).sum()}"
)
print(
    f"Urban loyalty: {loyalty[:N_URBAN].sum()}/{N_URBAN}, "
    f"Rural loyalty: {loyalty[N_URBAN:].sum()}/{N_STORES - N_URBAN}"
)

# ---------------------------------------------------------------------------
# Cascade 1: simple random sample → Welch t-test (the WRONG path)
# ---------------------------------------------------------------------------
rand_sample = full_df.sample(n=RANDOM_SAMPLE_N, random_state=0)
loy_r = rand_sample[rand_sample["loyalty_program"] == 1]["revenue"].values
noloy_r = rand_sample[rand_sample["loyalty_program"] == 0]["revenue"].values
t_stat, t_p = stats.ttest_ind(loy_r, noloy_r, equal_var=False)

mean_loy_r = loy_r.mean()
mean_noloy_r = noloy_r.mean()
effect_r = (mean_loy_r - mean_noloy_r) / mean_noloy_r

pooled_sd = np.sqrt((loy_r.std() ** 2 + noloy_r.std() ** 2) / 2.0)
cohens_d_r = (mean_loy_r - mean_noloy_r) / pooled_sd

print(
    f"\n--- Cascade 1: simple random sample (n={RANDOM_SAMPLE_N}) + Welch t-test ---"
)
print(f"  Loyalty group:    n={len(loy_r)}, mean=${mean_loy_r:,.2f}")
print(f"  No-loyalty group: n={len(noloy_r)}, mean=${mean_noloy_r:,.2f}")
print(f"  t={t_stat:.4f}, p={t_p:.4f}")
print(f"  Relative effect (mean diff / no-loyalty mean): {effect_r:.4f}")
print(f"  Cohen's d: {cohens_d_r:.4f}")
print(
    f"  Conclusion: {'SIGNIFICANT (p<0.05)' if t_p < 0.05 else 'NOT significant (p>=0.05)'}"
)
print("  [Verifier outcome: FAIL — test_used='t-test' is rejected by name check]")

# ---------------------------------------------------------------------------
# Cascade 2: stratified sample (20 rows/store) → Mann-Whitney U (the RIGHT path)
# ---------------------------------------------------------------------------
strat_frames: list[pd.DataFrame] = []
for sid in range(1, N_STORES + 1):
    store_rows = full_df[full_df["store_id"] == sid]
    strat_frames.append(store_rows.sample(n=STRAT_ROWS_PER_STORE, random_state=sid))
strat_sample = pd.concat(strat_frames, ignore_index=True)

loy_s = strat_sample[strat_sample["loyalty_program"] == 1]["revenue"].values
noloy_s = strat_sample[strat_sample["loyalty_program"] == 0]["revenue"].values
mwu_stat, mwu_p = stats.mannwhitneyu(loy_s, noloy_s, alternative="two-sided")

mean_loy_s = loy_s.mean()
mean_noloy_s = noloy_s.mean()
effect_s = (mean_loy_s - mean_noloy_s) / mean_noloy_s

print(
    f"\n--- Cascade 2: stratified sample "
    f"({STRAT_ROWS_PER_STORE} rows/store × {N_STORES} = "
    f"{STRAT_ROWS_PER_STORE * N_STORES:,}) + Mann-Whitney U ---"
)
print(f"  Loyalty group:    n={len(loy_s)}, mean=${mean_loy_s:,.2f}")
print(f"  No-loyalty group: n={len(noloy_s)}, mean=${mean_noloy_s:,.2f}")
print(f"  U={mwu_stat:.2f}, p={mwu_p:.8f}")
print(f"  Relative effect (mean diff / no-loyalty mean): {effect_s:.4f}")
print(
    f"  Conclusion: {'SIGNIFICANT (p<0.05)' if mwu_p < 0.05 else 'NOT significant'}"
)

# ---------------------------------------------------------------------------
# Verifier target check
# ---------------------------------------------------------------------------
print("\n=== Verifier target check ===")
print(
    f"  [Primary gate] test_used != 't-test' type:         "
    f"Cascade path enforces this by instruction design"
)
print(
    f"  Stratified MWU p < 0.01:                           "
    f"{mwu_p:.8f} => {'PASS' if mwu_p < 0.01 else 'FAIL'}"
)
print(
    f"  Stratified effect_size in [0.05, 0.15]:            "
    f"{effect_s:.4f} => {'PASS' if 0.05 <= effect_s <= 0.15 else 'FAIL'}"
)
print(
    f"  Random t-test p > 0.05 (optional):                 "
    f"{t_p:.4f} => {'PASS' if t_p > 0.05 else 'N/A (test_used gate handles cascade)'}"
)
