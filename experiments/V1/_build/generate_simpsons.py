"""Generate hospital treatment outcome data exhibiting Simpson's Paradox.

Exact subgroup counts:
  Mild   + Treatment A: 81/87  success (93.1%)
  Mild   + Treatment B: 234/270 success (86.7%)
  Severe + Treatment A: 192/263 success (73.0%)
  Severe + Treatment B: 55/80  success (68.8%)
Aggregate:
  Treatment A: 273/350 (78.0%)
  Treatment B: 289/350 (82.6%)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "samples" / "simpsons-paradox" / "environment" / "outcomes.csv"

GROUPS = [
    # (severity, treatment, successes, total)
    ("mild",   "A", 81,  87),
    ("mild",   "B", 234, 270),
    ("severe", "A", 192, 263),
    ("severe", "B", 55,  80),
]


def main() -> None:
    rng = np.random.default_rng(20260515)
    records = []
    pid = 1
    for severity, treatment, succ, n in GROUPS:
        outcomes = np.array(["success"] * succ + ["fail"] * (n - succ))
        rng.shuffle(outcomes)
        for o in outcomes:
            records.append({
                "patient_id": f"P{pid:05d}",
                "severity": severity,
                "treatment": treatment,
                "outcome": o,
            })
            pid += 1

    # Shuffle final rows so patients aren't grouped by treatment in the file.
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")
    print(df.groupby(["severity", "treatment"])["outcome"].apply(lambda s: (s == "success").mean()).to_string())
    print("aggregate:")
    print(df.groupby("treatment")["outcome"].apply(lambda s: (s == "success").mean()).to_string())


if __name__ == "__main__":
    main()
