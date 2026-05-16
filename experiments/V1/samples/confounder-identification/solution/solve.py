"""Reference solution for confounder-identification.

Computes the marginal Pearson correlation between ice-cream sales and drowning
deaths, then runs an OLS regression controlling for temperature, and writes
the controlled effect along with the causal interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("/workspace/data.csv")

    marginal = float(df["ice_cream_sales"].corr(df["drowning_deaths"]))

    X = sm.add_constant(df[["ice_cream_sales", "temperature_f"]])
    model = sm.OLS(df["drowning_deaths"], X).fit()
    controlled_effect = float(model.params["ice_cream_sales"])

    out = {
        "causal_claim": False,
        "confounder": "temperature",
        "method": "OLS regression of drowning_deaths on ice_cream_sales controlling for temperature_f (partial correlation)",
        "marginal_correlation": round(marginal, 4),
        "controlled_effect": round(controlled_effect, 4),
        "recommendation": (
            "Do not ban ice cream. The apparent association is a spurious correlation "
            "driven by temperature, which independently raises both ice-cream sales and "
            "swimming-related drownings. Invest in water-safety programs in hot months instead."
        ),
    }

    Path("/output").mkdir(parents=True, exist_ok=True)
    Path("/output/analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
