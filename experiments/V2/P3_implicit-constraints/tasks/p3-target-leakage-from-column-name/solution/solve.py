"""
Reference solution: identifies and drops the two leaky columns, then trains
a LogisticRegression classifier and writes /output/results.json.

Leaky columns (post-outcome, only populated when the customer has already churned):
  - account_closure_reason
  - last_contact_before_churn_days
"""

import json
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "/workspace/customers.csv"
OUT_PATH = "/output/results.json"

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# ── Drop non-predictive / leaky columns ─────────────────────────────────────
# customer_id   → identifier only
# account_closure_reason          → LEAKY: only filled after a churn event
# last_contact_before_churn_days  → LEAKY: only recorded for customers who churned
DROP_COLS = ["customer_id", "account_closure_reason", "last_contact_before_churn_days"]
df = df.drop(columns=DROP_COLS)

# ── Feature engineering ──────────────────────────────────────────────────────
TARGET = "churned"
y = df[TARGET]
X = df.drop(columns=[TARGET])

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=["contract_type", "payment_method"])

# ── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# ── Model ────────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = float(accuracy_score(y_test, y_pred))

print(f"Accuracy: {accuracy:.4f}")

# ── Write results ────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
results = {
    "accuracy":   accuracy,
    "model_type": "LogisticRegression",
}
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results written to {OUT_PATH}")
