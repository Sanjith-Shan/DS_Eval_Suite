"""Reference solution: classify_fixed.py

Strategy
--------
1. Load the data.
2. Compute per-row z-scores on the full dataset to find candidate outliers.
3. Inspect outliers BY CLASS:
   - Rows flagged in class 1 are legitimately extreme values (real data).
   - Rows flagged in class 0 are genuine data-entry errors.
4. Remove ONLY the class-0 rows flagged by |z| > 3.
5. Train a LogisticRegression with class_weight='balanced' to handle the
   remaining class imbalance.
6. Evaluate on train_test_split(test_size=0.2, random_state=42, stratify=y).
7. Return {"class1_recall": float, "accuracy": float}.

Why LogisticRegression?
-----------------------
With the signal on features 3-5 and balanced weighting, LR achieves a
stable class-1 recall of ~0.81, well above the 0.55 pass threshold.
RandomForest with balanced weights gives a lower and more variable recall
(~0.46 on the fixed seed) because the extreme class-1 subpopulation creates
large-margin errors that RF struggles to compensate for without the
implicit calibration that balanced LR provides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split


def train_and_evaluate(data_path: str) -> dict:
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values.astype(float)
    y = df["target"].values.astype(int)

    # ------------------------------------------------------------------
    # Outlier detection: compute global z-scores
    # ------------------------------------------------------------------
    z = np.abs(sp_stats.zscore(X, axis=0))
    outlier_flag = (z > 3).any(axis=1)

    # ------------------------------------------------------------------
    # Examine outliers per class (senior practitioner step)
    # Class 1 outliers are legitimate extreme values — do NOT remove them.
    # Class 0 outliers are actual data-entry errors — remove them.
    # ------------------------------------------------------------------
    remove_mask = (y == 0) & outlier_flag
    keep_mask = ~remove_mask

    X_clean = X[keep_mask]
    y_clean = y[keep_mask]

    # ------------------------------------------------------------------
    # Train / test split (fixed seed for reproducibility)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    # ------------------------------------------------------------------
    # Train balanced logistic regression
    # ------------------------------------------------------------------
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    class1_recall = float(recall_score(y_test, y_pred, pos_label=1))
    accuracy = float(accuracy_score(y_test, y_pred))

    return {"class1_recall": class1_recall, "accuracy": accuracy}


if __name__ == "__main__":
    result = train_and_evaluate("/workspace/data.csv")
    print(f"class1_recall : {result['class1_recall']:.4f}")
    print(f"accuracy      : {result['accuracy']:.4f}")
