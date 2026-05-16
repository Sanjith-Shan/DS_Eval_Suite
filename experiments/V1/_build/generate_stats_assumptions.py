"""Generate customer-satisfaction scores for 4 store locations.

Distributions are lognormal with unequal variances and unequal group sizes
so that ANOVA assumptions are decisively violated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

OUT = Path(__file__).resolve().parents[1] / "samples" / "statistical-test-assumptions" / "environment" / "satisfaction.csv"

SPECS = {
    "A": dict(n=200, mu=2.0, sigma=0.3),
    "B": dict(n=50, mu=2.1, sigma=0.8),
    "C": dict(n=180, mu=1.9, sigma=0.3),
    "D": dict(n=30, mu=2.5, sigma=1.2),
}


def main() -> None:
    rng = np.random.default_rng(20260515)
    records = []
    for store, params in SPECS.items():
        samples = rng.lognormal(mean=params["mu"], sigma=params["sigma"], size=params["n"])
        for v in samples:
            records.append({"store_id": store, "satisfaction_score": round(float(v), 4)})

    df = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # Diagnostics
    print(f"Total rows: {len(df)}")
    for store in SPECS:
        g = df[df["store_id"] == store]["satisfaction_score"]
        sw_stat, sw_p = stats.shapiro(g)
        print(
            f"Store {store}: n={len(g)} mean={g.mean():.3f} median={g.median():.3f} "
            f"std={g.std():.3f} shapiro_p={sw_p:.4f}"
        )

    groups = [df[df["store_id"] == s]["satisfaction_score"].values for s in SPECS]
    levene_stat, levene_p = stats.levene(*groups, center="median")
    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"Levene  p={levene_p:.4e}  (need < 0.05 to fail equal-variance)")
    print(f"Kruskal p={kw_p:.4e}")

    # Pairwise: Mann-Whitney for direction; we'll embed the ground-truth pairs
    # whose medians differ at alpha=0.05/6 (Bonferroni for 6 pairs).
    stores = list(SPECS.keys())
    alpha = 0.05 / 6
    pairs_sig = {}
    for i, s1 in enumerate(stores):
        for s2 in stores[i + 1:]:
            g1 = df[df["store_id"] == s1]["satisfaction_score"].values
            g2 = df[df["store_id"] == s2]["satisfaction_score"].values
            u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            if p < alpha:
                direction = "greater" if np.median(g1) > np.median(g2) else "less"
                pairs_sig[f"{s1}_vs_{s2}"] = (p, direction)
    print("Pairs significant (Bonferroni 6 comparisons):")
    for k, (p, d) in pairs_sig.items():
        print(f"  {k}: p={p:.2e}  direction (first vs second): {d}")


if __name__ == "__main__":
    main()
