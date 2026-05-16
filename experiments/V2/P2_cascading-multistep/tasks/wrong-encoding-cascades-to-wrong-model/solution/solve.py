"""Reference solution for wrong-encoding-cascades-to-wrong-model.

Key decision: recognise that education_level has a natural ordinal ordering:
    high_school < bachelors < masters < phd
and encode it as a single integer (0–3) rather than using one-hot encoding.

With ordinal encoding, a RandomForestClassifier sees education_level as a
continuous predictor and assigns it the highest feature importance. Reporting
top features by RF importance then yields "education_level" (the bare column
name), which the verifier requires. With one-hot encoding, importance is split
across four dummy columns and the model reports "education_level_phd" etc.

Paths default to /workspace/dataset.csv and /output/results.json so the
solution works inside the Docker environment. Override via env vars for local
testing:
    DATASET_PATH=/path/to/dataset.csv OUTPUT_DIR=/tmp/out python solve.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


DATASET_PATH = Path(os.environ.get("DATASET_PATH", "/workspace/dataset.csv"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    # ------------------------------------------------------------------ #
    # Step 1: Identify feature types                                        #
    # ------------------------------------------------------------------ #
    numeric_cols = [
        "age",
        "income",
        "years_experience",
        "subscription_months",
        "support_tickets_last_year",
        "n_logins_last_30d",
        "last_login_days_ago",
    ]
    # Nominal categoricals: no natural ordering — use one-hot
    nominal_cat_cols = ["region", "account_type", "marketing_channel", "device_type"]

    # education_level: ORDINAL — high_school < bachelors < masters < phd.
    # Encoding as integers 0–3 preserves the monotonic relationship with the
    # target and keeps the signal concentrated in a single feature.
    education_order = [["high_school", "bachelors", "masters", "phd"]]

    # ------------------------------------------------------------------ #
    # Step 2: Build column transformer                                      #
    # ------------------------------------------------------------------ #
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            (
                "nom",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                nominal_cat_cols,
            ),
            (
                "edu",
                OrdinalEncoder(categories=education_order),
                ["education_level"],
            ),
        ]
    )

    # ------------------------------------------------------------------ #
    # Step 3: Encode and split                                             #
    # ------------------------------------------------------------------ #
    X_enc = ct.fit_transform(X)

    nom_names = list(
        ct.named_transformers_["nom"].get_feature_names_out(nominal_cat_cols)
    )
    feature_names = numeric_cols + nom_names + ["education_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------ #
    # Step 4: Train classifier                                             #
    # ------------------------------------------------------------------ #
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = float(model.score(X_test, y_test))

    # ------------------------------------------------------------------ #
    # Step 5: Extract top-5 features by importance                         #
    # ------------------------------------------------------------------ #
    importances = model.feature_importances_
    top5_idx = np.argsort(importances)[::-1][:5]
    top_features = [feature_names[i] for i in top5_idx]

    # ------------------------------------------------------------------ #
    # Step 6: Write results                                                #
    # ------------------------------------------------------------------ #
    results = {
        "accuracy": round(accuracy, 4),
        "top_features": top_features,
        "model_type": type(model).__name__,
    }
    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote results to {out_path}")
    print(f"  accuracy    = {accuracy:.4f}")
    print(f"  top_features = {top_features}")
    print(f"  model_type  = {type(model).__name__}")


if __name__ == "__main__":
    main()
