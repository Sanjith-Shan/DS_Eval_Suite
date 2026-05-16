"""
pipeline_fixed.py  —  reference oracle solution.

Fixes applied:
  1. StandardScaler is fit on TRAINING data only (eliminates data leakage).
  2. days_since_first_purchase is clipped to the training range after scaling,
     neutralising the out-of-distribution extrapolation that occurs because test
     rows are chronologically later (scaled values [1.72, 2.90] vs training max
     [1.73]).

Test accuracy: ~0.79 (in verifier band [0.75, 0.86]).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATA_PATH = "/workspace/data.csv"
N_TRAIN   = 3000


def train_and_evaluate(data_path: str = DATA_PATH) -> float:
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    X_train_raw = X[:N_TRAIN]
    X_test_raw  = X[N_TRAIN:]
    y_train     = y[:N_TRAIN]
    y_test      = y[N_TRAIN:]

    # ── FIX 1: fit scaler on training data only ──────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # ── FIX 2: clip days_since_first_purchase to training range ─────────────
    temporal_col = "days_since_first_purchase"
    temporal_idx = feature_cols.index(temporal_col)

    t_min = X_train[:, temporal_idx].min()
    t_max = X_train[:, temporal_idx].max()
    X_test[:, temporal_idx] = np.clip(X_test[:, temporal_idx], t_min, t_max)

    # ── Train classifier ─────────────────────────────────────────────────────
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc


if __name__ == "__main__":
    acc = train_and_evaluate()
    print(f"test_accuracy = {acc:.4f}")
