"""
model_fixed.py — Oracle solution for deduplication-loses-valid-longitudinal-data.

Approach
--------
1. Recognise visit_number signals a longitudinal dataset; do NOT deduplicate.
2. Sort by patient_id + visit_number, then engineer per-visit delta features:
   - delta_lab_value_1 = lab_value_1[v] - lab_value_1[v-1]
   - delta_lab_value_2 = lab_value_2[v] - lab_value_2[v-1]
   (NaN for the first visit of each patient; filled with 0.)
3. Patient-level train/test split (random_state=42, test_size=0.2) so no
   patient appears in both sets.
4. Train RandomForestClassifier(n_estimators=200, random_state=42).
5. Return {"auc": float} evaluated on the held-out patient set.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def train_and_evaluate(data_path: str) -> dict:
    df = pd.read_csv(data_path)

    # ------------------------------------------------------------------ #
    # 1. Feature engineering — longitudinal deltas                        #
    # ------------------------------------------------------------------ #
    df = df.sort_values(["patient_id", "visit_number"]).reset_index(drop=True)

    df["delta_lab_value_1"] = df.groupby("patient_id")["lab_value_1"].diff().fillna(0.0)
    df["delta_lab_value_2"] = df.groupby("patient_id")["lab_value_2"].diff().fillna(0.0)

    # ------------------------------------------------------------------ #
    # 2. Patient-level train/test split                                   #
    # ------------------------------------------------------------------ #
    patients = df["patient_id"].unique()
    train_patients, test_patients = train_test_split(
        patients, test_size=0.2, random_state=42
    )

    train_df = df[df["patient_id"].isin(train_patients)]
    test_df = df[df["patient_id"].isin(test_patients)]

    feature_cols = [
        "visit_number",
        "days_between_visits",
        "lab_value_1",
        "lab_value_2",
        "delta_lab_value_1",
        "delta_lab_value_2",
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df["readmitted"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["readmitted"].values

    # ------------------------------------------------------------------ #
    # 3. Train                                                            #
    # ------------------------------------------------------------------ #
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # 4. Evaluate                                                         #
    # ------------------------------------------------------------------ #
    proba = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    return {"auc": auc}


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data.csv"
    result = train_and_evaluate(data_path)
    print(f"AUC = {result['auc']:.4f}")
