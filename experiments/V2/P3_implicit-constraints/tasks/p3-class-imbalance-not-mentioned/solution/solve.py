"""
Reference solution for p3-class-imbalance-not-mentioned.

A senior data scientist notices the 97/3 class imbalance and uses
class_weight='balanced' so the model actually predicts minority-class positives.
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression

# --- Load training data ---
train_df = pd.read_csv("/workspace/train.csv")

feature_cols = [c for c in train_df.columns if c not in ("id", "target")]
X_train = train_df[feature_cols].values
y_train = train_df["target"].values

# --- Train with balanced class weights to handle imbalance ---
clf = LogisticRegression(class_weight="balanced", random_state=0, max_iter=1000)
clf.fit(X_train, y_train)

# --- Load holdout (no target column) ---
holdout_df = pd.read_csv("/workspace/holdout.csv")
X_holdout = holdout_df[feature_cols].values
ids = holdout_df["id"].values

# --- Predict ---
y_pred = clf.predict(X_holdout)

# --- Write predictions ---
os.makedirs("/output", exist_ok=True)
output = pd.DataFrame({"id": ids, "predicted_class": y_pred.astype(int)})
output.to_csv("/output/predictions.csv", index=False)

print(f"Wrote /output/predictions.csv ({len(output)} rows)")
print(f"Predicted class distribution:\n{output['predicted_class'].value_counts()}")
