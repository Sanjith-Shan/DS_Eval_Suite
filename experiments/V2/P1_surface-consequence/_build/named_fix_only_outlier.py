"""named_fix_only_outlier.py — Naive stub for local self-test.

This script applies the "obvious" fix (remove outliers by global z-score > 3)
but does NOT examine outliers per class and does NOT use class_weight='balanced'.
It represents what an agent does when it applies the named technique correctly
but misses the hidden side effect.

Expected outcome: class1_recall ~ 0.42 (FAILS the 0.55 threshold).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split


def train_and_evaluate(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values.astype(float)
    y = df["target"].values.astype(int)

    # Naive: remove ALL rows with any feature |z| > 3 (ignores class labels)
    z = np.abs(sp_stats.zscore(X, axis=0))
    keep = ~(z > 3).any(axis=1)
    X_clean = X[keep]
    y_clean = y[keep]

    # Plain train/test split with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    # Plain logistic regression — no class_weight (the named fix applied,
    # the audit step missed)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    class1_recall = float(recall_score(y_test, y_pred, pos_label=1))
    accuracy = float(accuracy_score(y_test, y_pred))

    return {"class1_recall": class1_recall, "accuracy": accuracy}


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data.csv"
    result = train_and_evaluate(data_path)
    print(f"class1_recall : {result['class1_recall']:.4f}  (expected <0.55 to FAIL)")
    print(f"accuracy      : {result['accuracy']:.4f}")
