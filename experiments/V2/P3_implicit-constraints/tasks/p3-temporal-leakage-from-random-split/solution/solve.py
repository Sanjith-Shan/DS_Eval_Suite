"""
Reference solution for p3-temporal-leakage-from-random-split.

Key steps:
1. Load /workspace/data.csv.
2. Sort by the 'date' column to establish chronological order.
3. Split 80% train / 20% test CHRONOLOGICALLY (no random shuffle).
4. Train a LogisticRegression classifier on the feature columns.
5. Report test accuracy and model_type.

A senior data scientist would notice the date column and the temporal drift
in EDA and choose a chronological split. This solution demonstrates that.
"""

import json
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ── Paths (configurable via env vars for local testing) ───────────────────────
WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

DATA_PATH = os.path.join(WORKSPACE, "data.csv")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# ── Chronological split ───────────────────────────────────────────────────────
# Sort by date so we train on past, test on future — no temporal leakage.
df = df.sort_values("date").reset_index(drop=True)

feature_cols = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
X = df[feature_cols].values
y = df["target"].values

split_idx = int(0.8 * len(df))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ── Train classifier ──────────────────────────────────────────────────────────
clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
accuracy = float(accuracy_score(y_test, y_pred))

print(f"Temporal-split accuracy: {accuracy:.4f}")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")

# ── Write results ─────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = {
    "accuracy": round(accuracy, 6),
    "model_type": "LogisticRegression",
}
with open(RESULTS_PATH, "w") as fh:
    json.dump(results, fh, indent=2)

print(f"Results written to {RESULTS_PATH}")
print(json.dumps(results, indent=2))
