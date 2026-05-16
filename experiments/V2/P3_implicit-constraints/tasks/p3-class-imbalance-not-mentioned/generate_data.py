"""
Generate train.csv, holdout.csv, and tests/holdout_labels.csv
for the class-imbalance task.

- 8000 total rows, 97/3 imbalance, 10 features, id column (0..7999).
- Stratified split: rows 0..6999 -> train.csv (features + target)
                    rows 7000..7999 -> holdout.csv (features + id, NO target)
                                    -> tests/holdout_labels.csv (id + target)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR  = os.path.join(TASK_DIR, "environment")
TEST_DIR = os.path.join(TASK_DIR, "tests")

rng = np.random.default_rng(42)

N = 8000
N_MINORITY = 240   # 3% of 8000
N_MAJORITY = N - N_MINORITY  # 7760

# --- Majority class (target=0) ---
X0_signal = rng.normal(loc=0.0, scale=1.0, size=(N_MAJORITY, 4))
X0_noise  = rng.normal(loc=0.0, scale=1.0, size=(N_MAJORITY, 6))
X0 = np.hstack([X0_signal, X0_noise])

# --- Minority class (target=1) ---
X1_signal = rng.normal(loc=0.8, scale=1.0, size=(N_MINORITY, 4))
X1_noise  = rng.normal(loc=0.0, scale=1.0, size=(N_MINORITY, 6))
X1 = np.hstack([X1_signal, X1_noise])

X = np.vstack([X0, X1])
y = np.array([0] * N_MAJORITY + [1] * N_MINORITY)

# Shuffle
idx = rng.permutation(N)
X, y = X[idx], y[idx]

cols = [f"feature_{i}" for i in range(1, 11)]
df = pd.DataFrame(X, columns=cols)
df["target"] = y
df.insert(0, "id", range(N))   # id column 0..7999

# --- Stratified split: 7000 train / 1000 holdout ---
train_df, holdout_df = train_test_split(
    df, test_size=1000, random_state=42, stratify=df["target"]
)

# Re-sort by id for tidiness (not strictly necessary)
train_df   = train_df.sort_values("id").reset_index(drop=True)
holdout_df = holdout_df.sort_values("id").reset_index(drop=True)

# --- Write train.csv (id + features + target) ---
train_path = os.path.join(ENV_DIR, "train.csv")
train_df.to_csv(train_path, index=False)
print(f"Wrote {len(train_df)} rows -> {train_path}")

# --- Write holdout.csv (id + features, NO target) ---
feature_cols = ["id"] + cols
holdout_features = holdout_df[feature_cols]
holdout_path = os.path.join(ENV_DIR, "holdout.csv")
holdout_features.to_csv(holdout_path, index=False)
print(f"Wrote {len(holdout_features)} rows -> {holdout_path}")

# --- Write tests/holdout_labels.csv (id + target) ---
holdout_labels = holdout_df[["id", "target"]]
labels_path = os.path.join(TEST_DIR, "holdout_labels.csv")
holdout_labels.to_csv(labels_path, index=False)
print(f"Wrote {len(holdout_labels)} rows -> {labels_path}")

# --- Validation ---
print("\n--- Class balance ---")
train_rate   = train_df["target"].mean()
holdout_rate = holdout_df["target"].mean()
print(f"  train   rows={len(train_df)},   minority_rate={train_rate:.4f}")
print(f"  holdout rows={len(holdout_df)}, minority_rate={holdout_rate:.4f}")

assert len(train_df) == 7000,   f"Expected 7000 train rows, got {len(train_df)}"
assert len(holdout_df) == 1000, f"Expected 1000 holdout rows, got {len(holdout_df)}"
assert "target" not in holdout_features.columns, "target must NOT be in holdout.csv"
assert "target" in train_df.columns, "target must be in train.csv"
assert set(holdout_labels["id"]) == set(holdout_features["id"]), "id mismatch between holdout files"
assert 0.025 <= train_rate   <= 0.035, f"Train minority rate out of band: {train_rate}"
assert 0.025 <= holdout_rate <= 0.035, f"Holdout minority rate out of band: {holdout_rate}"

# Quick sanity: balanced LogReg on train, predict holdout (using true labels locally)
X_train = train_df[cols].values
y_train = train_df["target"].values
X_holdout = holdout_df[cols].values
y_holdout = holdout_df["target"].values

clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_holdout)
y_prob = clf.predict_proba(X_holdout)[:, 1]

f1  = f1_score(y_holdout, y_pred)
auc = roc_auc_score(y_holdout, y_prob)
print(f"\nBalanced LogReg on train -> holdout: F1={f1:.4f}, AUC={auc:.4f}")
assert f1  > 0.15, f"F1 too low: {f1}"
assert auc > 0.65, f"AUC too low: {auc}"

# Confirm unbalanced (default) LogReg predicts all zeros -> F1=0
clf_naive = LogisticRegression(max_iter=1000, random_state=0)
clf_naive.fit(X_train, y_train)
y_pred_naive = clf_naive.predict(X_holdout)
f1_naive = f1_score(y_holdout, y_pred_naive)
print(f"Naive (unbalanced) LogReg -> holdout: F1={f1_naive:.4f}")
all_zeros = (y_pred_naive == 0).all()
print(f"All-zeros prediction: {all_zeros}")
assert f1_naive <= 0.15, f"Naive (no class_weight) F1 should be <=0.15 but got {f1_naive} — signal too strong"

print("\nAll validation checks passed.")
