"""
Oracle solution for multicollinearity-after-log-transform.

Strategy:
1. Log-transform Y to fix heteroscedasticity (Breusch-Pagan passes after this).
2. Detect multicollinearity: VIF(X1) and VIF(X2) are both >> 10.
3. Address multicollinearity via ridge regression (alpha=5, features standardised).
4. Compute p-values via parametric bootstrap (3000 resamples) because ridge has
   no closed-form p-values.
5. Return p-values for X1..X6 (ridge) and bp_pvalue from the ridge model residuals.

The bootstrap p-value is a two-sided test: 2 * min(P(coef_boot <= 0), P(coef_boot >= 0))
when the point estimate is positive (and vice versa). Equivalently,
p = 2 * mean(sign(coef_boot) != sign(coef_hat)) capped at 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def analyze(data_path: str) -> dict:
    """Read CSV, fix heteroscedasticity, handle multicollinearity, return p-values."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(data_path)
    feature_cols = ["X1", "X2", "X3", "X4", "X5", "X6"]

    X = df[feature_cols].values
    Y = df["Y"].values

    # Step 1: log-transform Y (fixes multiplicative heteroscedasticity)
    logY = np.log(Y)

    # Step 2: standardize features for ridge (ridge penalises coefficients; scaling
    # ensures the penalty is fair across predictors with different scales)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: ridge regression with alpha=5 to stabilise X1/X2 coefficients
    alpha = 5.0
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_scaled, logY)

    # Step 4: bootstrap p-values (3000 resamples, fixed seed for reproducibility)
    n_boot = 3000
    n = len(logY)
    rng = np.random.default_rng(123)
    boot_coefs = np.zeros((n_boot, len(feature_cols)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = Ridge(alpha=alpha, fit_intercept=True)
        r.fit(X_scaled[idx], logY[idx])
        boot_coefs[b] = r.coef_

    # Two-sided p-value: fraction of bootstrap coefs that have opposite sign
    # to the point estimate (× 2 for two-sided)
    pvalues = {}
    for j, name in enumerate(feature_cols):
        coef = ridge.coef_[j]
        p = 2.0 * float(np.mean(boot_coefs[:, j] * coef <= 0))
        pvalues[name] = min(p, 1.0)

    # Step 5: Breusch-Pagan on ridge residuals (using original unscaled X)
    yhat = ridge.predict(X_scaled)
    residuals = logY - yhat
    Xc = sm.add_constant(X)
    _, bp_p, _, _ = het_breuschpagan(residuals, Xc)

    return {
        "X1_pvalue": pvalues["X1"],
        "X2_pvalue": pvalues["X2"],
        "X3_pvalue": pvalues["X3"],
        "X4_pvalue": pvalues["X4"],
        "X5_pvalue": pvalues["X5"],
        "X6_pvalue": pvalues["X6"],
        "bp_pvalue": float(bp_p),
    }
