import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import json, os

def main():
    df = pd.read_csv("business_data.csv")
    X = df[["X1", "X2", "X3", "X4"]]
    y = df["Y"]
    X_const = sm.add_constant(X)

    # Fit OLS
    ols = sm.OLS(y, X_const).fit()

    # Check Durbin-Watson
    dw = durbin_watson(ols.resid)
    print(f"Durbin-Watson: {dw:.3f}")

    # Refit with HAC standard errors
    robust = ols.get_robustcov_results(cov_type="HAC", maxlags=int(np.ceil(len(df)**0.25)))

    # robust.params and robust.pvalues may be numpy arrays; use OLS param_names for mapping
    param_names = ols.model.exog_names  # ['const', 'X1', 'X2', 'X3', 'X4']
    params_s = pd.Series(robust.params, index=param_names)
    pvalues_s = pd.Series(robust.pvalues, index=param_names)

    predictors = ["X1", "X2", "X3", "X4"]
    coefficients = {p: float(params_s[p]) for p in predictors}
    p_values = {p: float(pvalues_s[p]) for p in predictors}
    significant = [p for p in predictors if p_values[p] < 0.05]

    result = {
        "significant_predictors": significant,
        "coefficients": coefficients,
        "p_values": p_values,
        "r_squared": float(ols.rsquared)
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
