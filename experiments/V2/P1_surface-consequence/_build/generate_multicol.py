"""
generate_multicol.py
====================
Generates the synthetic dataset for the multicollinearity-after-log-transform task.

Final calibrated parameters (SEED=42, N=1500):
  rho = 0.987       → raw corr(X1,X2) ≈ 0.987, VIF > 35 in log-space OLS
  noise = 0.65      → enough variance that X2's small coef (0.15) is hard to isolate

Target behaviour:
  Raw OLS:            BP p < 0.01  (heteroscedastic)              → FAILS verifier
  log(Y)+plain OLS:   BP p > 0.05, X2_p ≈ 0.12 (> 0.10)         → FAILS verifier
  log(Y)+Ridge(α=5):  BP p > 0.05, X1/X2/X3 all p < 0.05        → PASSES

Saves to: tasks/multicollinearity-after-log-transform/environment/data.csv
Also prints diagnostics for all three regimes.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from pathlib import Path

SEED = 42
N = 1500
RHO = 0.987

rng = np.random.default_rng(SEED)

# Features
X1 = rng.standard_normal(N)
X2 = RHO * X1 + np.sqrt(1 - RHO**2) * rng.standard_normal(N)
X3 = rng.standard_normal(N)
X4 = rng.standard_normal(N)
X5 = rng.standard_normal(N)
X6 = rng.standard_normal(N)

# log(Y) is linear + homoscedastic in log space
logY_true = 0.3 * X1 + 0.15 * X2 + 0.4 * X3 + 0.65 * rng.standard_normal(N)
Y = np.exp(logY_true)

df = pd.DataFrame({
    "X1": X1, "X2": X2, "X3": X3,
    "X4": X4, "X5": X5, "X6": X6,
    "Y": Y,
})

out_path = Path(__file__).parent.parent / "tasks/multicollinearity-after-log-transform/environment/data.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")

# ---- Diagnostics ----
X = np.column_stack([X1, X2, X3, X4, X5, X6])
Xc = sm.add_constant(X)
logY_data = np.log(Y)

print(f"\nRaw correlation X1-X2: {np.corrcoef(X1,X2)[0,1]:.4f}")

# Raw OLS on Y
ols_raw = sm.OLS(Y, Xc).fit()
bp_raw = het_breuschpagan(ols_raw.resid, ols_raw.model.exog)
print(f"\n[Raw OLS] BP p: {bp_raw[1]:.4e}")
for i, name in enumerate(["X1","X2","X3","X4","X5","X6"], start=1):
    print(f"  {name}: p={ols_raw.pvalues[i]:.4f}")

# log(Y) OLS (named fix only)
ols_log = sm.OLS(logY_data, Xc).fit()
bp_log = het_breuschpagan(ols_log.resid, ols_log.model.exog)
print(f"\n[log(Y)+plain OLS] BP p: {bp_log[1]:.4e}")
for i, name in enumerate(["X1","X2","X3","X4","X5","X6"], start=1):
    print(f"  {name}: coef={ols_log.params[i]:.4f}, p={ols_log.pvalues[i]:.4f}")

# VIFs in log space
print("\nVIFs in log(Y) OLS:")
for j, name in enumerate(["X1","X2","X3","X4","X5","X6"]):
    others = np.delete(X, j, axis=1)
    r2 = sm.OLS(X[:,j], sm.add_constant(others)).fit().rsquared
    print(f"  VIF({name}) = {1/(1-r2):.2f}")

# Oracle: ridge alpha=5 with bootstrap
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ridge = Ridge(alpha=5.0)
ridge.fit(X_scaled, logY_data)

rng2 = np.random.default_rng(123)
n_boot = 3000
boot_coefs = np.zeros((n_boot, 6))
for b in range(n_boot):
    idx = rng2.integers(0, N, size=N)
    r = Ridge(alpha=5.0)
    r.fit(X_scaled[idx], logY_data[idx])
    boot_coefs[b] = r.coef_

yhat = ridge.predict(X_scaled)
resid = logY_data - yhat
bp_ridge = het_breuschpagan(resid, sm.add_constant(X))
print(f"\n[log(Y)+Ridge(α=5)+Bootstrap] BP p: {bp_ridge[1]:.4e}")
for j, name in enumerate(["X1","X2","X3","X4","X5","X6"]):
    p = 2 * np.mean(boot_coefs[:, j] * ridge.coef_[j] <= 0)
    ci_lo = np.percentile(boot_coefs[:, j], 2.5)
    ci_hi = np.percentile(boot_coefs[:, j], 97.5)
    print(f"  {name}: coef={ridge.coef_[j]:.4f}, p={p:.4f}, 95%CI=[{ci_lo:.4f},{ci_hi:.4f}]")
