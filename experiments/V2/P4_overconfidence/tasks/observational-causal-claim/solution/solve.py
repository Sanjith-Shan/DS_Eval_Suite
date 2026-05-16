"""Reference solution for observational-causal-claim.

Fits a naive OLS (cardio_health ~ exercise) and a full OLS controlling for
income, age, and baseline_health. Observes that the exercise coefficient
shrinks substantially (from ~0.45 to ~0.20) and loses significance (p~0.15)
after controlling for confounders. Concludes that this observational data
cannot establish a causal effect of exercise on cardiovascular health.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


DATA_PATH   = Path("/workspace/health_data.csv")
OUTPUT_PATH = Path("/output/analysis.json")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # ---- naive regression: cardio_health ~ exercise -------------------------
    X_naive = sm.add_constant(df["exercise_hours_per_week"])
    naive_mdl  = sm.OLS(df["cardio_health_score"], X_naive).fit()
    naive_coef = naive_mdl.params["exercise_hours_per_week"]
    naive_p    = naive_mdl.pvalues["exercise_hours_per_week"]

    # ---- full regression: cardio_health ~ exercise + confounders ------------
    confounders = ["income_thousands", "age", "baseline_health_score"]
    X_full    = sm.add_constant(df[["exercise_hours_per_week"] + confounders])
    full_mdl  = sm.OLS(df["cardio_health_score"], X_full).fit()
    full_coef = full_mdl.params["exercise_hours_per_week"]
    full_p    = full_mdl.pvalues["exercise_hours_per_week"]

    # ---- write output -------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "causal_effect": False,
        "effect_size": round(float(full_coef), 4),
        "methodology": (
            f"OLS regression of cardio_health_score on exercise_hours_per_week "
            f"alone (naive coef={naive_coef:.3f}, p={naive_p:.4f}), then re-fit "
            f"controlling for income_thousands, age, and baseline_health_score "
            f"(controlled coef={full_coef:.3f}, p={full_p:.4f}). "
            f"The exercise coefficient shrank from {naive_coef:.2f} to "
            f"{full_coef:.2f} and lost significance once confounders were "
            f"included in the multivariate regression, indicating substantial "
            f"confounding in the naive estimate."
        ),
        "conclusion": (
            "This is observational data. Exercise is confounded with income, "
            "age, and baseline health: wealthier, younger, and healthier "
            "participants exercise more AND have better cardiovascular health "
            "for reasons unrelated to exercise. Once these confounders are "
            "controlled via adjusted regression, the exercise association "
            "shrinks considerably and is no longer statistically significant. "
            "We can describe an association between exercise and cardiovascular "
            "health, but this observational study cannot establish causation. "
            "A randomised controlled trial (RCT) would be needed to make a "
            "causal claim."
        ),
        "confounders_considered": confounders,
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Wrote analysis to {OUTPUT_PATH}")
    print(f"  naive exercise coef : {naive_coef:+.4f} (p={naive_p:.4e})")
    print(f"  full  exercise coef : {full_coef:+.4f} (p={full_p:.4f})")
    print(f"  causal_effect       : {result['causal_effect']}")


if __name__ == "__main__":
    main()
