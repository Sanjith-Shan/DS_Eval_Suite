import pandas as pd
import numpy as np
import statsmodels.api as sm
import json, os

def main():
    df = pd.read_csv("advertising_data.csv")
    X = df[["advertising_spend"]]
    y = df["sales_revenue"]
    X_const = sm.add_constant(X)

    model = sm.OLS(y, X_const).fit()

    # Check Cook's distance
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(df)
    influential = cooks_d > threshold

    df_clean = df[~influential]
    X_clean = sm.add_constant(df_clean[["advertising_spend"]])
    y_clean = df_clean["sales_revenue"]
    model_clean = sm.OLS(y_clean, X_clean).fit()

    result = {
        "coefficient": float(model_clean.params["advertising_spend"]),
        "r_squared": float(model_clean.rsquared),
        "p_value": float(model_clean.pvalues["advertising_spend"]),
        "significant": bool(model_clean.pvalues["advertising_spend"] < 0.05),
        "n_observations_used": int(len(df_clean))
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
