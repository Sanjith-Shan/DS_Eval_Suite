"""Generate data.csv for the outlier-removal-kills-minority-class task.

Design
------
* 3000 rows total, binary target (0/1).
* Class 0 (majority): 2760 rows (~92 %)
* Class 1 (minority):  240 rows (~8 %)

Predictive signal
-----------------
Class 1 has shifted means on features 3, 4, 5 (+1.5, +1.2, +1.0 sigma
above the class-0 mean). A well-tuned balanced classifier can achieve
class-1 recall ~0.80 on a 20 % held-out test set.

Outlier structure (the trap)
-----------------------------
* Class 1 has TWO subpopulations:
    - "normal"  (60 %, 144 rows): all features drawn from N(0, 1) + signal.
    - "extreme" (40 %, 96 rows): features 0, 1, 2 drawn from N(0, 1) plus
      a shift of 4-6 standard deviations. These represent LEGITIMATE
      extreme values, NOT errors. They also carry the same signal on
      features 3-5.
* Class 0 has 15 PLANTED ERRORS: features 0-4 replaced with
  Uniform(8, 12). These are genuine data-entry mistakes.

Result with naive global |z| > 3 removal
-----------------------------------------
* Drops ~38 of 240 class-1 rows (36-40 % loss) — the legitimate extremes
  are mistakenly flagged.
* Drops ~38-41 of 2760 class-0 rows (1.4 % loss) — the actual 15 errors
  plus a handful of random outliers from the tails.
* After removal, training a plain (unweighted) LogisticRegression gives
  class-1 recall ~0.42 on the standard test split (SEED 42) — FAILS the
  verifier threshold of 0.55.

Oracle approach
---------------
1. Inspect which class has z-score flagged rows.
2. Remove only the class-0 errors (they are clearly spurious).
3. Keep all class-1 rows (their extreme values are legitimate).
4. Train LogisticRegression with class_weight='balanced'.
=> class-1 recall ~0.81 — PASSES.

Output
------
Writes data_outlier.csv (rename to data.csv and copy into
tasks/outlier-removal-kills-minority-class/environment/).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

SEED = 42
rng = np.random.default_rng(SEED)

N_TOTAL = 3000
N_CLASS1 = 240
N_CLASS0 = N_TOTAL - N_CLASS1   # 2760
N_FEATURES = 8
N_CLASS1_EXTREME = 96    # 40 % of class-1; legitimately extreme on feat 0-2
N_CLASS1_NORMAL  = N_CLASS1 - N_CLASS1_EXTREME   # 144
N_ERRORS_CLASS0  = 15   # planted errors in class 0

# -----------------------------------------------------------------------
# Class 0 — all normal, with 15 planted errors
# -----------------------------------------------------------------------
X0 = rng.standard_normal((N_CLASS0, N_FEATURES))
y0 = np.zeros(N_CLASS0, dtype=int)

error_idx = rng.choice(N_CLASS0, size=N_ERRORS_CLASS0, replace=False)
X0[error_idx, :5] = rng.uniform(8.0, 12.0, size=(N_ERRORS_CLASS0, 5))

# -----------------------------------------------------------------------
# Class 1 — normal subpopulation (with signal on features 3-5)
# -----------------------------------------------------------------------
X1n = rng.standard_normal((N_CLASS1_NORMAL, N_FEATURES))
X1n[:, 3] += 1.5
X1n[:, 4] += 1.2
X1n[:, 5] += 1.0
y1n = np.ones(N_CLASS1_NORMAL, dtype=int)

# -----------------------------------------------------------------------
# Class 1 — legitimately extreme subpopulation
#   features 0-2: shifted 4-6 sigma (the "outlier" values the task traps on)
#   features 3-5: same signal shift as the normal subpop
# -----------------------------------------------------------------------
X1e = rng.standard_normal((N_CLASS1_EXTREME, N_FEATURES))
X1e[:, 0] += rng.uniform(4.0, 5.5, size=N_CLASS1_EXTREME)
X1e[:, 1] += rng.uniform(3.5, 5.0, size=N_CLASS1_EXTREME)
X1e[:, 2] += rng.uniform(4.0, 6.0, size=N_CLASS1_EXTREME)
X1e[:, 3] += 1.5
X1e[:, 4] += 1.2
X1e[:, 5] += 1.0
y1e = np.ones(N_CLASS1_EXTREME, dtype=int)

# -----------------------------------------------------------------------
# Combine and shuffle
# -----------------------------------------------------------------------
X = np.vstack([X0, X1n, X1e])
y = np.concatenate([y0, y1n, y1e])

perm = rng.permutation(N_TOTAL)
X, y = X[perm], y[perm]

feature_cols = [f"feature_{i}" for i in range(N_FEATURES)]
df = pd.DataFrame(X, columns=feature_cols)
df["target"] = y

out_path = Path(__file__).parent / "data_outlier.csv"
df.to_csv(out_path, index=False)

# -----------------------------------------------------------------------
# Sanity report
# -----------------------------------------------------------------------
print(f"Wrote {len(df)} rows to {out_path}")
print(f"  class 0 : {(df.target == 0).sum()} rows")
print(f"  class 1 : {(df.target == 1).sum()} rows")

z = np.abs(sp_stats.zscore(df[feature_cols].values))
naive_mask = (z > 3).any(axis=1)
for cls in [0, 1]:
    m = df.target == cls
    nd = (m & naive_mask).sum()
    print(f"  class {cls}: naive |z|>3 drops {nd}/{m.sum()} ({100 * nd / m.sum():.1f} %)")
