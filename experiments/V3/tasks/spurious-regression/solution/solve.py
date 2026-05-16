import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import json, os

def main():
    df = pd.read_csv("marketing_data.csv")
    X = df["social_media_mentions"]
    Y = df["monthly_revenue"]

    # Check stationarity
    adf_x = adfuller(X)
    adf_y = adfuller(Y)
    print(f"ADF X: p={adf_x[1]:.4f}, ADF Y: p={adf_y[1]:.4f}")

    # Difference
    dX = X.diff().dropna()
    dY = Y.diff().dropna()

    # Regress differences
    model = sm.OLS(dY, sm.add_constant(dX)).fit()

    result = {
        "coefficient": float(model.params.iloc[1]),
        "r_squared": float(model.rsquared),
        "p_value": float(model.pvalues.iloc[1]),
        "significant": bool(model.pvalues.iloc[1] < 0.05)
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
