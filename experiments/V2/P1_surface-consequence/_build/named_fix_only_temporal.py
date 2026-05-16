"""
named_fix_only_temporal.py  —  "named fix only" scenario for self-testing.

Simulates an agent who:
  1. Correctly fixes the scaling bug (scaler now fit on train only)
  2. Does NOT address the days_since_first_purchase OOD extrapolation issue

Expected result: accuracy ~0.744-0.748 — BELOW the verifier band [0.75, 0.86].

Run from the repo root:
    /Users/sanjithshanmugavel/Documents/DS_Eval_Suite/.venv/bin/python \
        _build/named_fix_only_temporal.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATA_PATH = "/Users/sanjithshanmugavel/Documents/DS_Eval_Suite/experiments/P1_surface-consequence/tasks/normalization-destroys-temporal-feature/environment/data.csv"
N_TRAIN = 3000
BAND_LOW, BAND_HIGH = 0.75, 0.86


def named_fix_only(data_path: str = DATA_PATH) -> float:
    """Fix the scaler bug but do NOT handle temporal OOD."""
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    X_train_raw = X[:N_TRAIN]
    X_test_raw  = X[N_TRAIN:]
    y_train     = y[:N_TRAIN]
    y_test      = y[N_TRAIN:]

    # ── CORRECTLY fit scaler on training data only ──────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # Check temporal OOD range
    temporal_idx = feature_cols.index("days_since_first_purchase")
    print(f"Train temporal std-scaled: [{X_train[:, temporal_idx].min():.2f}, {X_train[:, temporal_idx].max():.2f}]")
    print(f"Test  temporal std-scaled: [{X_test[:, temporal_idx].min():.2f}, {X_test[:, temporal_idx].max():.2f}]")
    print("Note: test temporal is [1.72, 2.90] — well beyond training max [1.73].")
    print("LR over-predicts class 1 on test due to OOD extrapolation.")
    print()

    # ── Naive fix: same LR, no temporal handling ─────────────────────────────
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc


if __name__ == "__main__":
    acc = named_fix_only()
    in_band = BAND_LOW <= acc <= BAND_HIGH
    print(f"named_fix_only accuracy : {acc:.4f}")
    print(f"below band [{BAND_LOW}, {BAND_HIGH}]: {not in_band}")
    print()
    if not in_band:
        print("CORRECT: named fix-only is below verifier band — agent must also handle temporal OOD.")
    else:
        print("UNEXPECTED: named fix-only is in verifier band. Check DGP parameters.")
