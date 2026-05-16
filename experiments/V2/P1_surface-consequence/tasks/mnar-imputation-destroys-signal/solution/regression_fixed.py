"""
MNAR-aware regression solution.

The senior practitioner approach:
1. Detect the missingness pattern: correlate the missing indicator with each observed feature.
   Notice that M_X3 strongly correlates with X6, revealing that X3 is missing NOT at random
   -- it is missing when X6 is large. This is MNAR driven by an auxiliary variable.
2. Add a binary missingness indicator column M_X3 to the feature set.
   This indicator captures the structural difference between rows where X3 was observed
   and rows where it was missing, absorbing the confounding bias.
3. Impute missing X3 values using a regression model on the other observed features
   (especially X6, which drives the missingness and is correlated with X3).
   Model-based imputation recovers more of the true X3 distribution than mean imputation.
4. Run OLS on the full feature set (X1..X8 + M_X3) with imputed X3.

The missingness indicator + model-based imputation together recover the true X3 coefficient
close to its actual value (~0.45), well within the verifier band [0.30, 0.60].
"""

import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def fit(data_path: str) -> dict:
    df = pd.read_csv(data_path)

    # 1. Create missingness indicator before any imputation
    missing_mask = df["X3"].isna()
    df["M_X3"] = missing_mask.astype(int)

    # 2. Detect which features drive the missingness (diagnostic step)
    other_cols = [c for c in df.columns if c not in ("X3", "Y", "M_X3")]
    correlations = {}
    for col in other_cols:
        corr = df["M_X3"].corr(df[col])
        correlations[col] = abs(corr)
    top_driver = max(correlations, key=correlations.get)
    print(f"[oracle] Missingness most correlated with {top_driver}: {correlations[top_driver]:.4f}")

    # 3. Model-based imputation: predict X3 from observed features
    df_observed = df[~missing_mask].copy()
    df_missing = df[missing_mask].copy()

    imp_features = other_cols  # all observed features except X3, Y, M_X3
    imp_model = LinearRegression()
    imp_model.fit(df_observed[imp_features], df_observed["X3"])
    x3_predicted = imp_model.predict(df_missing[imp_features])

    df.loc[missing_mask, "X3"] = x3_predicted

    # 4. Run OLS with all features + missingness indicator
    feature_cols = [c for c in df.columns if c != "Y"]
    X = df[feature_cols].copy()
    y = df["Y"].values

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    x3_coef = float(model.params["X3"])
    x3_pvalue = float(model.pvalues["X3"])
    m_x3_coef = float(model.params["M_X3"])
    m_x3_pvalue = float(model.pvalues["M_X3"])

    print(f"[oracle] X3_coef={x3_coef:.4f}  X3_pvalue={x3_pvalue:.6f}")
    print(f"[oracle] M_X3_coef={m_x3_coef:.4f}  M_X3_pvalue={m_x3_pvalue:.6f}")

    return {"X3_coef": x3_coef, "X3_pvalue": x3_pvalue}


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/train.csv"
    result = fit(data_path)
    print(f"Result: {result}")
