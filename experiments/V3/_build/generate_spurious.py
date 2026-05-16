"""
generate_spurious.py
--------------------
Generates marketing_data.csv for the spurious-regression task.

Spec (CONTEXT.md N6):
  Seed: 42
  N: 120 months ("2015-01" through "2024-12")

  X[0] = 1000, Y[0] = 500
  X[t] = X[t-1] + 5.0 + N(0, 15)   -- random walk, independent
  Y[t] = Y[t-1] + 3.0 + N(0, 10)   -- random walk, independent

  X and Y share NO common noise component.

VALIDATION (all 10 must pass before writing the CSV):
  1.  Naive OLS (Y on X levels): R² > 0.60
  2.  Naive OLS p-value < 0.001
  3.  ADF on X: p > 0.05  (non-stationary)
  4.  ADF on Y: p > 0.05  (non-stationary)
  5.  ADF on diff(X): p < 0.05  (stationary)
  6.  ADF on diff(Y): p < 0.05  (stationary)
  7.  OLS on differenced series: R² < 0.10
  8.  OLS on differenced series: p > 0.10
  9.  Correlation(X, Y levels) > 0.70
  10. Correlation(diff(X), diff(Y)) < 0.20
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
import os, sys

# ──────────────────────────────────────────────
# Parameters (tune here if validation fails)
# ──────────────────────────────────────────────
SEED       = 42
N          = 120
X0, Y0     = 1000.0, 500.0
DRIFT_X    = 5.0
DRIFT_Y    = 3.0
NOISE_X    = 15.0
NOISE_Y    = 10.0
OUT_PATH   = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "tasks", "spurious-regression", "environment", "marketing_data.csv"
)

rng = np.random.default_rng(SEED)

# ──────────────────────────────────────────────
# Generate two INDEPENDENT random walks
# ──────────────────────────────────────────────
innovations_x = rng.normal(0, NOISE_X, N)   # independent draw 1
innovations_y = rng.normal(0, NOISE_Y, N)   # independent draw 2  ← no shared noise

X = np.zeros(N)
Y = np.zeros(N)
X[0] = X0
Y[0] = Y0
for t in range(1, N):
    X[t] = X[t-1] + DRIFT_X + innovations_x[t]
    Y[t] = Y[t-1] + DRIFT_Y + innovations_y[t]

months = pd.period_range("2015-01", periods=N, freq="M").strftime("%Y-%m")

df = pd.DataFrame({
    "month": months,
    "social_media_mentions": X,
    "monthly_revenue": Y,
})

# ──────────────────────────────────────────────
# Run all 10 validation checks
# ──────────────────────────────────────────────
results = {}

# 1 & 2: Naive OLS on levels
X_sm = sm.add_constant(X)
ols_levels = sm.OLS(Y, X_sm).fit()
r2_levels = ols_levels.rsquared
p_levels  = ols_levels.pvalues[1]   # p-value for X coefficient
results[1]  = ("Naive OLS R² > 0.60",          r2_levels > 0.60,  f"R²={r2_levels:.4f}")
results[2]  = ("Naive OLS p < 0.001",           p_levels < 0.001,  f"p={p_levels:.6f}")

# 3 & 4: ADF on levels (expect NON-stationary → p > 0.05)
adf_x = adfuller(X, autolag="AIC")
adf_y = adfuller(Y, autolag="AIC")
results[3] = ("ADF X p > 0.05 (non-stationary)", adf_x[1] > 0.05, f"p={adf_x[1]:.4f}")
results[4] = ("ADF Y p > 0.05 (non-stationary)", adf_y[1] > 0.05, f"p={adf_y[1]:.4f}")

# 5 & 6: ADF on differences (expect stationary → p < 0.05)
dX = np.diff(X)
dY = np.diff(Y)
adf_dx = adfuller(dX, autolag="AIC")
adf_dy = adfuller(dY, autolag="AIC")
results[5] = ("ADF diff(X) p < 0.05 (stationary)", adf_dx[1] < 0.05, f"p={adf_dx[1]:.4f}")
results[6] = ("ADF diff(Y) p < 0.05 (stationary)", adf_dy[1] < 0.05, f"p={adf_dy[1]:.4f}")

# 7 & 8: OLS on differenced series
dX_sm = sm.add_constant(dX)
ols_diff = sm.OLS(dY, dX_sm).fit()
r2_diff = ols_diff.rsquared
p_diff  = ols_diff.pvalues[1]
results[7] = ("Diff OLS R² < 0.10",  r2_diff < 0.10, f"R²={r2_diff:.4f}")
results[8] = ("Diff OLS p > 0.10",   p_diff  > 0.10, f"p={p_diff:.4f}")

# 9: Correlation of levels
corr_levels, _ = pearsonr(X, Y)
results[9] = ("Corr(X,Y levels) > 0.70", corr_levels > 0.70, f"r={corr_levels:.4f}")

# 10: Correlation of differences (must be LOW — confirms independence)
corr_diff, _ = pearsonr(dX, dY)
results[10] = ("Corr(diff(X),diff(Y)) < 0.20", abs(corr_diff) < 0.20, f"r={corr_diff:.4f}")

# ──────────────────────────────────────────────
# Print results
# ──────────────────────────────────────────────
print("\n===== Validation Results =====")
all_pass = True
for i in range(1, 11):
    label, ok, detail = results[i]
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{status}] {i:2d}. {label}  ({detail})")

print()
print(f"Key gap: naive R²={r2_levels:.4f}  vs  differenced R²={r2_diff:.4f}")
print(f"Key gap: naive p={p_levels:.6f}  vs  differenced p={p_diff:.4f}")
print(f"Correlation levels={corr_levels:.4f}  diff={corr_diff:.4f}")

if not all_pass:
    print("\n*** One or more validation checks FAILED. Tune DRIFT/NOISE parameters. ***")
    sys.exit(1)

# ──────────────────────────────────────────────
# Write CSV
# ──────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"\nCSV written to: {OUT_PATH}")
print(f"Rows: {len(df)}")
