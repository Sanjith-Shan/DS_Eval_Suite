"""
Generate synthetic dataset for normalization-destroys-temporal-feature task.

Fixed seed: 20260516
N = 4000 rows, N_TRAIN = 3000

DGP:
  - 8 numeric features (f0-f7) + days_since_first_purchase (monotone, 30->1500)
  - Target driven by numeric features in training AND test, plus a mild temporal
    component in training only (zeroed for test => test labels ~50% balanced)
  - train = rows 0:3000, test = rows 3000:4000

Key numbers (with SEED=20260516, num_scale=1.2, t_dgp=0.8):
  - Buggy   (StandardScaler fit on ALL data  + LR):           ~0.748
  - Fix-only (StandardScaler fit on TRAIN + LR, no clip):     ~0.746
  - Oracle  (train-only scaler + LR + clip temporal):         ~0.790
  - Oracle  (train-only scaler + LR + bin temporal):          ~0.803
  - Verifier band: [0.75, 0.86]  (clip/bin oracle PASSES; buggy/fix-only FAIL)

This script writes:
  - tasks/normalization-destroys-temporal-feature/environment/data.csv
  - tasks/normalization-destroys-temporal-feature/environment/pipeline.py (buggy)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
import textwrap

SEED = 20260516
N = 4000
N_TRAIN = 3000
N_TEST  = 1000

OUT_DIR = Path(__file__).parent.parent / "tasks" / "normalization-destroys-temporal-feature" / "environment"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH      = OUT_DIR / "data.csv"
PIPELINE_PATH = OUT_DIR / "pipeline.py"

rng = np.random.default_rng(SEED)

# ── Temporal feature ──────────────────────────────────────────────────────────
days_base = np.linspace(30, 1500, N)
days = days_base + rng.normal(0, 3, N)
days = np.maximum(days, 1.0)

train_days = days[:N_TRAIN]
train_mean  = train_days.mean()
train_std   = train_days.std(ddof=0)
days_ts_fix = (days - train_mean) / train_std          # fix-only-scaled temporal
# Train: [-1.74, 1.73]; Test: [1.72, 2.90]

# ── 8 numeric features ────────────────────────────────────────────────────────
F = rng.standard_normal((N, 8))

# ── DGP ───────────────────────────────────────────────────────────────────────
# temporal zeroed for test => test labels ~50%; training has mild temporal signal
num_scale = 1.2
t_dgp     = 0.8

days_ts_for_dgp = days_ts_fix.copy()
days_ts_for_dgp[N_TRAIN:] = 0.0         # test labels driven by numerics only

numeric_log_odds = (
    num_scale * 2.0 * F[:, 0]
    + num_scale * 1.3 * F[:, 1]
    + num_scale * 0.9 * F[:, 2]
    + num_scale * 0.6 * F[:, 3]
    - num_scale * 0.4 * F[:, 4]
    + num_scale * 0.3 * F[:, 5]
)
full_log_odds = numeric_log_odds + t_dgp * days_ts_for_dgp
prob   = 1 / (1 + np.exp(-full_log_odds))
target = rng.binomial(1, prob, N)

print(f"Target: train={target[:N_TRAIN].mean():.3f}  test={target[N_TRAIN:].mean():.3f}")

# ── Build arrays ──────────────────────────────────────────────────────────────
feature_cols = [f"f{i}" for i in range(8)] + ["days_since_first_purchase"]
df = pd.DataFrame(F, columns=[f"f{i}" for i in range(8)])
df["days_since_first_purchase"] = days
df["target"] = target

X = df[feature_cols].values
y = df["target"].values

X_train_raw = X[:N_TRAIN];  X_test_raw = X[N_TRAIN:]
y_train      = y[:N_TRAIN];  y_test     = y[N_TRAIN:]
temporal_idx = feature_cols.index("days_since_first_purchase")

# ── Scenario 1: BUGGY — full-data StandardScaler ─────────────────────────────
sc_buggy = StandardScaler()
X_all_scaled  = sc_buggy.fit_transform(X)
X_train_buggy = X_all_scaled[:N_TRAIN]
X_test_buggy  = X_all_scaled[N_TRAIN:]

print(f"\nBUGGY train temporal: [{X_train_buggy[:, temporal_idx].min():.2f}, {X_train_buggy[:, temporal_idx].max():.2f}]")
print(f"BUGGY test  temporal: [{X_test_buggy[:, temporal_idx].min():.2f}, {X_test_buggy[:, temporal_idx].max():.2f}]")

lr_buggy = LogisticRegression(max_iter=1000, random_state=42)
lr_buggy.fit(X_train_buggy, y_train)
acc_buggy = accuracy_score(y_test, lr_buggy.predict(X_test_buggy))
print(f"BUGGY accuracy: {acc_buggy:.4f}  (temporal coef: {lr_buggy.coef_[0][temporal_idx]:.3f})")

# ── Scenario 2: FIX-ONLY — train-only StandardScaler, no temporal handling ───
sc_fix = StandardScaler()
sc_fix.fit(X_train_raw)
X_train_fix = sc_fix.transform(X_train_raw)
X_test_fix  = sc_fix.transform(X_test_raw)

print(f"\nFIX-ONLY train temporal: [{X_train_fix[:, temporal_idx].min():.2f}, {X_train_fix[:, temporal_idx].max():.2f}]")
print(f"FIX-ONLY test  temporal: [{X_test_fix[:, temporal_idx].min():.2f}, {X_test_fix[:, temporal_idx].max():.2f}]")

lr_fix = LogisticRegression(max_iter=1000, random_state=42)
lr_fix.fit(X_train_fix, y_train)
acc_fix = accuracy_score(y_test, lr_fix.predict(X_test_fix))
print(f"FIX-ONLY accuracy: {acc_fix:.4f}")

# ── Scenario 3a: ORACLE — clip temporal to training range ────────────────────
t_train_min = X_train_fix[:, temporal_idx].min()
t_train_max = X_train_fix[:, temporal_idx].max()

X_train_clip = X_train_fix.copy()
X_test_clip  = X_test_fix.copy()
X_test_clip[:, temporal_idx] = np.clip(X_test_clip[:, temporal_idx], t_train_min, t_train_max)

lr_clip = LogisticRegression(max_iter=1000, random_state=42)
lr_clip.fit(X_train_clip, y_train)
acc_clip = accuracy_score(y_test, lr_clip.predict(X_test_clip))
print(f"\nORACLE-clip accuracy: {acc_clip:.4f}")

# ── Scenario 3b: ORACLE — quantile-transform temporal ────────────────────────
qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=42)
qt.fit(X_train_raw[:, temporal_idx:temporal_idx+1])
X_train_qt = X_train_fix.copy()
X_test_qt  = X_test_fix.copy()
X_train_qt[:, temporal_idx:temporal_idx+1] = qt.transform(X_train_raw[:, temporal_idx:temporal_idx+1])
X_test_qt[:, temporal_idx:temporal_idx+1]  = qt.transform(X_test_raw[:, temporal_idx:temporal_idx+1])

lr_qt = LogisticRegression(max_iter=1000, random_state=42)
lr_qt.fit(X_train_qt, y_train)
acc_qt = accuracy_score(y_test, lr_qt.predict(X_test_qt))
print(f"ORACLE-quantile accuracy: {acc_qt:.4f}")

# ── Scenario 3c: ORACLE — bin temporal feature ───────────────────────────────
kbd = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
kbd.fit(X_train_raw[:, temporal_idx:temporal_idx+1])
raw_t_min = X_train_raw[:, temporal_idx].min()
raw_t_max = X_train_raw[:, temporal_idx].max()

X_train_bin = X_train_fix.copy()
X_test_bin  = X_test_fix.copy()
X_train_bin[:, temporal_idx] = kbd.transform(X_train_raw[:, temporal_idx:temporal_idx+1]).ravel()
X_test_bin[:, temporal_idx]  = kbd.transform(
    np.clip(X_test_raw[:, temporal_idx:temporal_idx+1], raw_t_min, raw_t_max)
).ravel()

lr_bin = LogisticRegression(max_iter=1000, random_state=42)
lr_bin.fit(X_train_bin, y_train)
acc_bin = accuracy_score(y_test, lr_bin.predict(X_test_bin))
print(f"ORACLE-bin accuracy: {acc_bin:.4f}")

# ── Verifier band check ───────────────────────────────────────────────────────
BAND_LOW, BAND_HIGH = 0.75, 0.86

print(f"\n{'='*65}")
print(f"SELF-CHECK (verifier band [{BAND_LOW}, {BAND_HIGH}]):")
print(f"  Buggy   : {acc_buggy:.4f}  in band: {BAND_LOW <= acc_buggy <= BAND_HIGH}")
print(f"  Fix-only: {acc_fix:.4f}  below {BAND_LOW}: {acc_fix < BAND_LOW}")
print(f"  Oracle-clip: {acc_clip:.4f}  in band: {BAND_LOW <= acc_clip <= BAND_HIGH}")
print(f"  Oracle-qt:   {acc_qt:.4f}  in band: {BAND_LOW <= acc_qt <= BAND_HIGH}")
print(f"  Oracle-bin:  {acc_bin:.4f}  in band: {BAND_LOW <= acc_bin <= BAND_HIGH}")
print(f"{'='*65}")

oracle_ok = (
    (BAND_LOW <= acc_clip <= BAND_HIGH)
    or (BAND_LOW <= acc_qt   <= BAND_HIGH)
    or (BAND_LOW <= acc_bin  <= BAND_HIGH)
)

if not oracle_ok:
    print("\nFAIL: no oracle approach lands in band")
    import sys; sys.exit(1)
else:
    print("\nALL ORACLE CHECKS PASSED")

# ── Write CSV ─────────────────────────────────────────────────────────────────
df.to_csv(CSV_PATH, index=False)
print(f"\nSaved {len(df)} rows to {CSV_PATH}")

# ── Write buggy pipeline.py ───────────────────────────────────────────────────
PIPELINE_CODE = textwrap.dedent("""\
    \"\"\"
    pipeline.py  —  feature scaling bug: scaler is fit on the FULL dataset (train+test).
    This is data leakage: statistics computed from test rows contaminate the training scaler.

    Usage
    -----
    python pipeline.py          # prints test accuracy
    \"\"\"
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    DATA_PATH = "/workspace/data.csv"
    N_TRAIN   = 3000

    def load_and_run(data_path: str = DATA_PATH) -> float:
        df = pd.read_csv(data_path)
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values
        y = df["target"].values

        X_train_raw, X_test_raw = X[:N_TRAIN], X[N_TRAIN:]
        y_train,     y_test     = y[:N_TRAIN], y[N_TRAIN:]

        # ── BUG: scaler is fit on ALL rows (train + test) ──────────────────────
        scaler = StandardScaler()
        X_all_scaled  = scaler.fit_transform(X)         # leaks test statistics
        X_train = X_all_scaled[:N_TRAIN]
        X_test  = X_all_scaled[N_TRAIN:]

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        return acc

    if __name__ == "__main__":
        acc = load_and_run()
        print(f"test_accuracy = {acc:.4f}")
""")

PIPELINE_PATH.write_text(PIPELINE_CODE)
print(f"Wrote buggy pipeline to {PIPELINE_PATH}")
