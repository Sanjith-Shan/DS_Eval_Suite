"""Reference solution for statistical-test-assumptions.

1. Checks normality (Shapiro-Wilk) per group and equal variance (Levene).
2. Both fail, so we use a Kruskal-Wallis omnibus test followed by pairwise
   Mann-Whitney with a Bonferroni-Holm correction (acts as a Dunn-equivalent
   post-hoc).
3. Writes the verdict to /output/analysis.json.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ALPHA = 0.05
STORES = ["A", "B", "C", "D"]


def holm_threshold(rank: int, m: int, alpha: float) -> float:
    return alpha / (m - rank + 1)


def main() -> None:
    df = pd.read_csv("/workspace/satisfaction.csv")
    groups = {s: df[df["store_id"] == s]["satisfaction_score"].values for s in STORES}

    # Normality per group
    shapiro_ps = {s: stats.shapiro(g).pvalue for s, g in groups.items()}
    normality_violated = any(p < ALPHA for p in shapiro_ps.values())

    # Equal variance
    levene_p = stats.levene(*groups.values(), center="median").pvalue
    equal_variance_violated = levene_p < ALPHA

    # Omnibus: Kruskal-Wallis
    kw_stat, kw_p = stats.kruskal(*groups.values())

    # Pairwise: Mann-Whitney with Holm-Bonferroni
    pairs = list(itertools.combinations(STORES, 2))
    raw = []
    for a, b in pairs:
        u, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        raw.append({"pair": (a, b), "p": float(p)})
    raw.sort(key=lambda d: d["p"])
    m = len(raw)
    significant_pairs: list[str] = []
    for i, item in enumerate(raw):
        threshold = holm_threshold(i + 1, m, ALPHA)
        if item["p"] >= threshold:
            break
        a, b = item["pair"]
        if np.median(groups[a]) > np.median(groups[b]):
            significant_pairs.append(f"{a}>{b}")
        else:
            significant_pairs.append(f"{b}>{a}")

    out = {
        "assumptions_checked": True,
        "normality_violated": bool(normality_violated),
        "equal_variance_violated": bool(equal_variance_violated),
        "test_used": "Kruskal-Wallis",
        "test_p_value": round(float(kw_p), 6),
        "post_hoc_test": "Mann-Whitney U with Holm-Bonferroni correction (Dunn-equivalent)",
        "group_medians": {s: round(float(np.median(groups[s])), 4) for s in STORES},
        "significant_pairs": significant_pairs,
    }

    Path("/output").mkdir(parents=True, exist_ok=True)
    Path("/output/analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
