"""
Generate synthetic dataset for mnar-imputation-destroys-signal task.

Data spec:
- 2000 rows, 8 features (X1..X8), 1 target Y
- True coefficients: X3=0.45, X1=0.30, X5=-0.35, rest near zero
- X3 and X6 are correlated (r≈0.81): X3 = 0.82*X6 + sqrt(1-0.82^2)*Z
- MNAR deletion: X3 is missing when X6 > median(X6) with 80% probability
  This creates ~40% missing in X3, all driven by an observed auxiliary variable X6.
  Mean imputation of X3 makes rows with high X6 (and thus high true X3) appear
  to have an average X3, drastically attenuating the X3-Y relationship.
  Correct approach: detect missingness correlates with X6, use model-based imputation
  or add a missingness indicator to capture the structural shift.
- Fixed seed: 20260516

Output: train.csv (written to tasks/mnar-imputation-destroys-signal/environment/train.csv)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 20260516
N = 2000
N_FEATURES = 8
CORR_X3_X6 = 0.82   # X3 and X6 are correlated
DEL_PROB = 0.80       # 80% deletion when X6 > median

# Output path
OUT_DIR = Path(__file__).parent.parent / "tasks" / "mnar-imputation-destroys-signal" / "environment"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "train.csv"

rng = np.random.default_rng(SEED)

# Generate base features: standardized normals
X_base = rng.standard_normal((N, N_FEATURES))

# Make X3 and X6 correlated (r ≈ 0.82)
X6 = X_base[:, 5].copy()
X3 = CORR_X3_X6 * X6 + np.sqrt(1 - CORR_X3_X6**2) * X_base[:, 2]
X_base[:, 2] = X3

actual_corr = np.corrcoef(X3, X6)[0, 1]
print(f"Corr(X3, X6): {actual_corr:.4f}")

# True coefficients: X3 (index 2) = 0.45, X1 (index 0) = 0.30, X5 (index 4) = -0.35
true_coefs = np.array([0.30, 0.05, 0.45, 0.08, -0.35, 0.06, 0.04, 0.07])

# Generate target Y = X @ coefs + noise
noise_std = 0.7
Y = X_base @ true_coefs + rng.normal(0, noise_std, N)

# Apply MNAR deletion: X3 is missing when X6 > median(X6), with 80% probability
# This simulates a realistic scenario where missingness is driven by an auxiliary variable
q50_X6 = np.percentile(X6, 50)
high_X6 = X6 > q50_X6
deletion_mask = high_X6 & (rng.uniform(0, 1, N) < DEL_PROB)

x3_mnar = X3.copy()
x3_mnar[deletion_mask] = np.nan

missing_pct = np.isnan(x3_mnar).mean()
print(f"X3 missing percentage: {missing_pct:.1%}  ({deletion_mask.sum()} rows)")
print(f"X6 > median(X6): {high_X6.sum()} rows, deleted {deletion_mask.sum()} ({deletion_mask.mean()/high_X6.mean():.0%})")

# Verify separation exists
import statsmodels.api as sm

# Naive: mean impute
df_check = pd.DataFrame(X_base, columns=[f"X{i+1}" for i in range(N_FEATURES)])
df_check["X3"] = x3_mnar
df_check["Y"] = Y
df_naive = df_check.copy()
df_naive["X3"] = df_naive["X3"].fillna(df_naive["X3"].mean())
model_naive = sm.OLS(df_naive["Y"], sm.add_constant(df_naive.drop("Y", axis=1))).fit()
c_naive = model_naive.params["X3"]
p_naive = model_naive.pvalues["X3"]
naive_pass = (p_naive < 0.05) and (0.30 <= c_naive <= 0.60)

# Oracle: indicator + median impute
df_oracle = df_check.copy()
missing_mask = df_oracle["X3"].isna()
df_oracle["M_X3"] = missing_mask.astype(int)
df_oracle["X3"] = df_oracle["X3"].fillna(df_oracle["X3"].median())
model_oracle = sm.OLS(df_oracle["Y"], sm.add_constant(df_oracle.drop("Y", axis=1))).fit()
c_oracle = model_oracle.params["X3"]
p_oracle = model_oracle.pvalues["X3"]
oracle_pass = (p_oracle < 0.05) and (0.30 <= c_oracle <= 0.60)

print(f"\nSelf-check:")
print(f"  NAIVE   X3_coef={c_naive:.4f}  X3_pvalue={p_naive:.6f}  pass={naive_pass}")
print(f"  ORACLE  X3_coef={c_oracle:.4f}  X3_pvalue={p_oracle:.6f}  pass={oracle_pass}")
print(f"  SEPARATION: {oracle_pass and not naive_pass}")
assert oracle_pass and not naive_pass, "FAIL: separation not achieved"

# Build final DataFrame (without ground truth info)
cols = [f"X{i+1}" for i in range(N_FEATURES)]
df = pd.DataFrame(X_base, columns=cols)
df["X3"] = x3_mnar  # MNAR version
df["Y"] = Y

# Save
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}  ({len(df)} rows, {missing_pct:.1%} missing in X3)")
print(f"True coefficients: X1={true_coefs[0]}, X3={true_coefs[2]}, X5={true_coefs[4]}")
