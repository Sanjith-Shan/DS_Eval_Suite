"""Reference solution for simpsons-paradox.

Computes aggregate and stratified success rates, detects the inversion
(Simpson's paradox), and recommends Treatment A.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    df = pd.read_csv("/workspace/outcomes.csv")
    df["success"] = (df["outcome"] == "success").astype(int)

    agg = df.groupby("treatment")["success"].mean()
    strat = df.groupby(["severity", "treatment"])["success"].mean().unstack("treatment")

    paradox = bool(
        (strat.loc["mild", "A"] > strat.loc["mild", "B"]) and
        (strat.loc["severe", "A"] > strat.loc["severe", "B"]) and
        (agg["A"] < agg["B"])
    )

    out = {
        "better_treatment": "A",
        "aggregate_rate_a": round(float(agg["A"]), 4),
        "aggregate_rate_b": round(float(agg["B"]), 4),
        "stratified_rates": {
            "mild":   {"A": round(float(strat.loc["mild", "A"]), 4),   "B": round(float(strat.loc["mild", "B"]), 4)},
            "severe": {"A": round(float(strat.loc["severe", "A"]), 4), "B": round(float(strat.loc["severe", "B"]), 4)},
        },
        "stratified_analysis": True,
        "paradox_identified": paradox,
        "explanation": (
            "This is Simpson's Paradox: Treatment B looks better in aggregate (82.6% vs 78.0%), "
            "but Treatment A is more effective within both severity strata (93.1% vs 86.7% for mild; "
            "73.0% vs 68.8% for severe). Treatment B was preferentially assigned to milder, easier-"
            "to-cure cases, inflating its aggregate success rate. Severity is the confounder."
        ),
    }

    Path("/output").mkdir(parents=True, exist_ok=True)
    Path("/output/analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
