"""
verify.py — Harbor verifier for p3-units-mismatch-across-columns.

Scoring:
  - reward 1  if agent RMSE <= 1.3 × baseline_rmse
  - reward 0  otherwise

Baseline is computed here: load /workspace/properties.csv, one-hot encode
zip_code and school_district, train LinearRegression on 80/20 split
(random_state=0), compute RMSE on the test set.

A naive agent that treats zip_code and school_district as plain numeric
features will have RMSE ~56 000, which is ~3.7× the baseline (~15 000).
That is well above the 1.3× threshold and earns reward 0.

An agent that one-hot encodes (or otherwise treats them as categorical)
will have RMSE close to the baseline and earns reward 1.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

RESULTS_PATH = "/output/results.json"
DATA_PATH    = "/workspace/properties.csv"
REWARD_PATH  = "/logs/verifier/reward.txt"
LOG_PATH     = "/logs/verifier/output.log"


def compute_baseline_rmse():
    """One-hot-encode zip_code and school_district, train LinearRegression, return test RMSE."""
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["zip_code", "school_district", "sqft", "bedrooms",
                    "bathrooms", "age_years", "garage_spaces"]
    df_enc = pd.get_dummies(df[feature_cols + ["price"]],
                            columns=["zip_code", "school_district"])
    X = df_enc.drop(columns=["price"])
    y = df_enc["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return float(root_mean_squared_error(y_test, preds))


def write_result(reward, reason):
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(REWARD_PATH, "w") as f:
        f.write(str(int(reward)))
    with open(LOG_PATH, "w") as f:
        f.write(f"reward={reward}\nreason={reason}\n")
    print(f"reward={reward}  reason={reason}")
    sys.exit(0)


def main():
    # 1. Check output file exists
    if not os.path.exists(RESULTS_PATH):
        write_result(0, f"Missing output file: {RESULTS_PATH}")

    # 2. Parse JSON
    try:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    except Exception as e:
        write_result(0, f"Could not parse {RESULTS_PATH}: {e}")

    # 3. Validate required fields
    rmse_val  = results.get("rmse")
    model_t   = results.get("model_type")

    if rmse_val is None:
        write_result(0, "Missing field 'rmse' in results.json")
    if model_t is None:
        write_result(0, "Missing field 'model_type' in results.json")

    try:
        rmse_float = float(rmse_val)
    except (TypeError, ValueError):
        write_result(0, f"'rmse' is not a valid float: {rmse_val!r}")

    if not (rmse_float > 0 and rmse_float < 1e9):
        write_result(0, f"'rmse' value {rmse_float} is out of expected range")

    # 4. Compute dynamic RMSE threshold
    try:
        baseline_rmse = compute_baseline_rmse()
    except Exception as e:
        baseline_rmse = 15500.0
        print(f"Warning: could not compute baseline RMSE ({e}); using fallback {baseline_rmse}")

    threshold = 1.3 * baseline_rmse

    # 5. Pass/fail
    if rmse_float <= threshold:
        write_result(
            1,
            f"PASS: agent RMSE={rmse_float:,.1f} <= 1.3 × baseline {threshold:,.1f} "
            f"(baseline={baseline_rmse:,.1f}); model_type={model_t!r}"
        )
    else:
        write_result(
            0,
            f"FAIL: agent RMSE={rmse_float:,.1f} exceeds 1.3 × baseline ({threshold:,.1f}). "
            f"Baseline RMSE={baseline_rmse:,.1f}. "
            "Likely cause: zip_code and school_district were used as raw numeric features "
            "instead of being treated as categorical (non-monotone integer codes)."
        )


if __name__ == "__main__":
    main()
