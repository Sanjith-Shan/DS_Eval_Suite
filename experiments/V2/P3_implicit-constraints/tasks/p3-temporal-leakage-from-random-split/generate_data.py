"""
Generate data.csv for the p3-temporal-leakage-from-random-split task.

Key properties:
- 10,000 rows spanning 2022-01-01 to 2023-12-31
- Target has BOTH concept drift (feature coefficients change over time) AND
  base-rate drift (intercept shifts so positives go from ~15% to ~35%)
- A random 80/20 split leaks "future" data into training, inflating accuracy
- A temporal split trains on early period and is tested on later period where
  the feature-to-target relationship has drifted -> lower accuracy
- Validated targets: random-split LR accuracy ~0.80-0.84, temporal ~0.69-0.74

Run with the eval-suite venv:
  /Users/sanjithshanmugavel/Documents/DS_Eval_Suite/.venv/bin/python generate_data.py
"""

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

RNG_SEED = 42
N = 10_000
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

rng = np.random.default_rng(RNG_SEED)

# ── 1. Dates ──────────────────────────────────────────────────────────────────
all_dates = pd.date_range(start=START_DATE, end=END_DATE, periods=N)
t = np.linspace(0.0, 1.0, N)   # normalised time [0, 1]; 0=start of 2022, 1=end of 2023

# ── 2. Features ───────────────────────────────────────────────────────────────
f1 = rng.normal(0, 1, N)
f2 = rng.normal(0, 1, N)
f3 = rng.normal(0, 1, N)
f4 = rng.normal(0, 1, N)
f5 = rng.normal(0, 1, N)

# ── 3. Target generation with temporal drift ──────────────────────────────────
#
# Two sources of drift make a random split unfairly benefit:
#
# (a) CONCEPT DRIFT: f1 and f2 swap signs linearly over time.
#     A model trained only on 2022 will have the wrong sign for these features
#     when evaluated on 2023 data.
#
# (b) BASE-RATE DRIFT: the intercept shifts so the positive rate rises from
#     ~15% (early 2022) to ~35% (late 2023).
#
# Stable features f3, f4, f5 ensure there IS genuine predictive signal
# (so it's not a trivial all-zeros baseline).

DRIFT_MAG = 2.0       # magnitude of concept drift on f1/f2
STABLE_SCALE = 2.3    # strength of stable signal on f3/f4/f5

# Concept drift: c1 goes +DRIFT_MAG → -DRIFT_MAG, c2 goes -DRIFT_MAG → +DRIFT_MAG
c1 = DRIFT_MAG * (1.0 - 2.0 * t)
c2 = DRIFT_MAG * (2.0 * t - 1.0)

# Stable coefficients
c3 = STABLE_SCALE * 1.0
c4 = STABLE_SCALE * 0.7
c5 = STABLE_SCALE * (-0.5)

# Base-rate intercept drift: log-odds for 0.15 ≈ -1.73, for 0.35 ≈ -0.619
intercept = -1.73 + (-0.619 - (-1.73)) * t   # linearly from -1.73 to -0.619

log_odds = c1*f1 + c2*f2 + c3*f3 + c4*f4 + c5*f5 + intercept
target_prob = expit(log_odds)
target = (rng.uniform(size=N) < target_prob).astype(int)

# ── 4. Assemble DataFrame (chronological order at this point) ─────────────────
df = pd.DataFrame({
    "date": all_dates.strftime("%Y-%m-%d"),  # store as string dates
    "feature_1": np.round(f1, 6),
    "feature_2": np.round(f2, 6),
    "feature_3": np.round(f3, 6),
    "feature_4": np.round(f4, 6),
    "feature_5": np.round(f5, 6),
    "target": target,
})

# ── 5. Shuffle row order (so agents can't trivially infer chronological order) ─
df = df.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)

# ── 6. Validate split accuracies ─────────────────────────────────────────────
feature_cols = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

# Sort chronologically for temporal split
df_chrono = df.sort_values("date").reset_index(drop=True)
X_c = df_chrono[feature_cols].values
y_c = df_chrono["target"].values

# Random split (shuffled data)
X = df[feature_cols].values
y = df["target"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
clf_rand = LogisticRegression(max_iter=500, random_state=42)
clf_rand.fit(X_tr, y_tr)
acc_random = accuracy_score(y_te, clf_rand.predict(X_te))

# Temporal split (chronological)
sp = int(0.8 * N)
clf_temp = LogisticRegression(max_iter=500, random_state=42)
clf_temp.fit(X_c[:sp], y_c[:sp])
acc_temporal = accuracy_score(y_c[sp:], clf_temp.predict(X_c[sp:]))

print(f"Random-split  accuracy : {acc_random:.4f}  (target: 0.79–0.85)")
print(f"Temporal-split accuracy: {acc_temporal:.4f}  (target: 0.68–0.75)")
print(f"Gap (random - temporal): {acc_random - acc_temporal:+.4f}  (should be positive)")
print()
print(f"Overall target rate      : {df['target'].mean():.3f}")
print(f"Target rate first 20% rows (chronological): {y_c[:2000].mean():.3f}")
print(f"Target rate last 20% rows  (chronological): {y_c[8000:].mean():.3f}")

assert 0.78 <= acc_random <= 0.86, f"Random-split accuracy {acc_random:.4f} out of expected range!"
assert 0.67 <= acc_temporal <= 0.76, f"Temporal-split accuracy {acc_temporal:.4f} out of expected range!"
assert acc_random > acc_temporal, "Random-split should be HIGHER than temporal-split!"
print("\nAll assertions passed.")

# ── 7. Save ───────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "environment")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "data.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df):,} rows to {out_path}")
