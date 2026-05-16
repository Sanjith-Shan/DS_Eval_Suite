"""
Naive stub: mean imputation + plain OLS (no missingness indicator, no MNAR awareness).

This is what a naive practitioner does: they see missing values in X3 and fill them
with the column mean, then run a standard OLS regression. They do not investigate
WHY X3 is missing or how the missingness pattern relates to other features.

Expected behavior:
- X3_coef ≈ 0.27 (attenuated by MNAR-induced confounding with X6)
- X3_pvalue < 0.05 but coef is BELOW the verifier's lower bound of 0.30
- would_pass_verifier = False  (coef outside [0.30, 0.60])
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit(data_path: str) -> dict:
    df = pd.read_csv(data_path)

    # Naive approach: fill missing X3 values with the column mean
    df["X3"] = df["X3"].fillna(df["X3"].mean())

    # Assemble features and target
    feature_cols = [c for c in df.columns if c != "Y"]
    X = df[feature_cols].copy()
    y = df["Y"].values

    # Plain OLS with statsmodels
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    x3_coef = float(model.params["X3"])
    x3_pvalue = float(model.pvalues["X3"])

    print(f"[naive] X3_coef={x3_coef:.4f}  X3_pvalue={x3_pvalue:.4f}")
    return {"X3_coef": x3_coef, "X3_pvalue": x3_pvalue}


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/sanjithshanmugavel/Documents/DS_Eval_Suite/experiments/"
        "P1_surface-consequence/tasks/mnar-imputation-destroys-signal/environment/train.csv"
    )
    result = fit(data_path)
    print(f"Result: {result}")
    pval = result["X3_pvalue"]
    coef = result["X3_coef"]
    would_pass = (pval < 0.05) and (0.30 <= coef <= 0.60)
    print(f"Would pass verifier: {would_pass}  (need pval<0.05 AND 0.30<=coef<=0.60)")
    if coef < 0.30:
        print(f"  -> coef={coef:.4f} is BELOW lower bound 0.30 (MNAR confounding attenuated it)")
