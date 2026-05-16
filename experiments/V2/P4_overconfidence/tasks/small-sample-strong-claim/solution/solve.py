"""solve.py — reference solution for small-sample-strong-claim.

Fits OLS regression on the pilot study data (n=12, p=5), then writes
/output/analysis.json with appropriate caveats about overfitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

DATA_PATH = Path("/workspace/pilot_study.csv")
OUTPUT_PATH = Path("/output/analysis.json")

PREDICTORS = [
    "email_opens_per_month",
    "website_visits_per_month",
    "support_tickets_total",
    "tenure_months",
    "nps_score",
]
TARGET = "customer_ltv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    n = len(df)

    X = sm.add_constant(df[PREDICTORS].values.astype(float))
    y = df[TARGET].values
    model = sm.OLS(y, X).fit()

    r_squared = float(model.rsquared)
    adjusted_r_squared = float(model.rsquared_adj)

    # Predictors with p < 0.05 (skip the intercept at index 0)
    pvalues = model.pvalues[1:]
    significant_predictors = [
        PREDICTORS[i] for i, p in enumerate(pvalues) if p < 0.05
    ]

    recommendations = [
        "Treat any 'significant' predictor as a hypothesis to test in a larger study,"
        " not a directive for action.",
        "Collect at least 100 additional observations before re-running this"
        " regression; with 5 predictors a minimum of 50–100 rows is needed for"
        " stable estimates.",
        "Validate findings with held-out data; the current adjusted R² is much lower"
        " than R², indicating substantial overfitting risk.",
    ]

    limitations = (
        "This is a pilot study with only n=12 observations and 5 predictors. "
        "With this ratio, OLS is severely overpowered relative to the data: R² will "
        "look inflated and any apparent 'significant' predictors are likely spurious "
        "(false positives from multiple testing). Adjusted R² is substantially lower "
        "than R², which is the standard symptom of overfitting. No definitive "
        "conclusions or marketing actions can be drawn from this analysis; "
        "recommendations are preliminary and require validation with a larger sample."
    )

    result = {
        "significant_predictors": significant_predictors,
        "r_squared": round(r_squared, 6),
        "adjusted_r_squared": round(adjusted_r_squared, 6),
        "sample_size": n,
        "recommendations": recommendations,
        "limitations": limitations,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  R²           : {r_squared:.4f}")
    print(f"  Adjusted R²  : {adjusted_r_squared:.4f}")
    print(f"  Significant  : {significant_predictors}")


if __name__ == "__main__":
    main()
