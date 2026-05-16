"""
named_fix_only_dedup.py
=======================
Simulates the "named fix only" agent approach:
- Sees multiple rows per patient_id → drops duplicates (keep='last').
- Trains a RandomForest on the single-row-per-patient data.
- Does a patient-level split (which is now equivalent to a row-level split
  since each patient has one row, but we still use it for fairness).
- Reports AUC — expected to fail (~0.55-0.65).

This is NOT the oracle.  It is a calibration check to confirm that the
naive dedup approach cannot cross the 0.72 threshold.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def train_and_evaluate(data_path: str) -> dict:
    df = pd.read_csv(data_path)

    # Named "fix": drop apparent duplicates, keep the last visit per patient
    df_dedup = df.drop_duplicates(subset="patient_id", keep="last").reset_index(drop=True)

    patients = df_dedup["patient_id"].unique()
    train_patients, test_patients = train_test_split(
        patients, test_size=0.2, random_state=42
    )

    train_df = df_dedup[df_dedup["patient_id"].isin(train_patients)]
    test_df = df_dedup[df_dedup["patient_id"].isin(test_patients)]

    feature_cols = [
        "visit_number",
        "days_between_visits",
        "lab_value_1",
        "lab_value_2",
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df["readmitted"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["readmitted"].values

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    return {"auc": auc}


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "tasks/deduplication-loses-valid-longitudinal-data/environment/data.csv"
    result = train_and_evaluate(data_path)
    print(f"AUC (named_fix_only / dedup keep-last) = {result['auc']:.4f}")
