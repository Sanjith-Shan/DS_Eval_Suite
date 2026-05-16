"""Reference solution for wrong-sampling-cascades-to-wrong-test.

Strategy
--------
1. Load the full store_revenue.csv (500 000 rows, four columns: store_id,
   is_urban, loyalty_program, revenue).
2. Take a STRATIFIED sample: draw exactly 20 rows per store_id → 10 000 rows.
   This preserves the urban/rural composition within each loyalty group and
   makes within-store variance visible.
3. Plot (mentally): the revenue distribution within each loyalty group is
   bimodal (urban cluster ~$5 000, rural cluster ~$3 000), violating the
   normality assumption required by a t-test.
4. Apply Mann-Whitney U test (non-parametric, rank-based) — appropriate for
   non-normal, clustered data.
5. Compute effect_size as the relative difference of means:
   (mean_loyalty - mean_no_loyalty) / mean_no_loyalty.
6. Write /output/analysis.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path("/workspace/store_revenue.csv")
OUTPUT_PATH = Path("/output/analysis.json")
STRAT_ROWS_PER_STORE = 20
RANDOM_STATE = 42


def main() -> None:
    # --- Load full dataset ---
    df = pd.read_csv(DATA_PATH)

    # --- Stratified sample: 20 rows per store_id ---
    strat_frames: list[pd.DataFrame] = []
    for store_id, group in df.groupby("store_id"):
        sampled = group.sample(
            n=min(STRAT_ROWS_PER_STORE, len(group)), random_state=RANDOM_STATE
        )
        strat_frames.append(sampled)
    sample = pd.concat(strat_frames, ignore_index=True)

    # --- Split by loyalty group ---
    loy = sample[sample["loyalty_program"] == 1]["revenue"].values
    noloy = sample[sample["loyalty_program"] == 0]["revenue"].values

    # --- Mann-Whitney U test (non-parametric, two-sided) ---
    mwu_stat, p_value = stats.mannwhitneyu(loy, noloy, alternative="two-sided")

    # --- Effect size: relative difference of means ---
    mean_loy = float(np.mean(loy))
    mean_noloy = float(np.mean(noloy))
    effect_size = (mean_loy - mean_noloy) / mean_noloy

    significant = bool(p_value < 0.05)

    result = {
        "sample_size": int(len(sample)),
        "test_used": "Mann-Whitney U",
        "p_value": round(float(p_value), 8),
        "significant": significant,
        "effect_size": round(effect_size, 6),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Written to {OUTPUT_PATH}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
