"""
named_fix_only_multicol.py
==========================
Simulates the "named fix only" agent: applies the log-transform to Y (correct
fix for heteroscedasticity) but then uses plain OLS — failing to notice or
address the severe X1-X2 multicollinearity that emerges in log-space.

Expected outcome:
  bp_pvalue > 0.05   (heteroscedasticity fixed) ✓
  X1_pvalue < 0.05   ✓
  X2_pvalue > 0.10   ✗  ← fails the verifier
  X3_pvalue < 0.05   ✓

Run: python _build/named_fix_only_multicol.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "tasks/multicollinearity-after-log-transform/environment/data.csv"

df = pd.read_csv(DATA_PATH)
feature_cols = ["X1", "X2", "X3", "X4", "X5", "X6"]

X = df[feature_cols].values
Y = df["Y"].values
logY = np.log(Y)

Xc = sm.add_constant(X)
ols = sm.OLS(logY, Xc).fit()
bp = het_breuschpagan(ols.resid, ols.model.exog)

result = {
    "X1_pvalue": float(ols.pvalues[1]),
    "X2_pvalue": float(ols.pvalues[2]),
    "X3_pvalue": float(ols.pvalues[3]),
    "X4_pvalue": float(ols.pvalues[4]),
    "X5_pvalue": float(ols.pvalues[5]),
    "X6_pvalue": float(ols.pvalues[6]),
    "bp_pvalue": float(bp[1]),
}

print("=== Named-fix-only: log(Y) + plain OLS ===")
for k, v in result.items():
    flag = ""
    if k in ("X1_pvalue", "X2_pvalue", "X3_pvalue"):
        flag = " ✓ PASS" if v < 0.05 else " ✗ FAIL"
    if k == "bp_pvalue":
        flag = " ✓ PASS" if v > 0.05 else " ✗ FAIL"
    print(f"  {k}: {v:.6f}{flag}")

passes = (
    result["X1_pvalue"] < 0.05
    and result["X2_pvalue"] < 0.05
    and result["X3_pvalue"] < 0.05
    and result["bp_pvalue"] > 0.05
)
print(f"\nWould pass verifier: {passes}  (expected: False)")
