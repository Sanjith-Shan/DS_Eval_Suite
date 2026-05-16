"""Reference fix for the leakage-laden pipeline.

Three problems with the original /workspace/pipeline.py:

1. StandardScaler was fitted on the FULL feature matrix before the train/test
   split, so test-row statistics influenced train-time normalisation. Fix:
   fit only on training data via a sklearn Pipeline.
2. The categorical was target-encoded using full-dataset means, so each row's
   encoding included its own label. Because the categorical here is high-card
   (nearly unique per row), this collapses to giving the model the answer.
   Fix: drop the categorical (or use cross-fitted target encoding) — here we
   drop it because it carries no genuine signal.
3. mutual_info_classif for feature selection was computed on the full data,
   which is itself a form of supervised leakage. Fix: run feature selection
   inside the Pipeline so it only sees training fold data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(data_path: str = "/workspace/data.csv") -> float:
    df = pd.read_csv(data_path)
    y = df.pop("target")

    # Drop the leaky high-cardinality categorical (no real signal in it).
    df = df.drop(columns=["customer_segment"])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=mutual_info_classif, k=10)),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)
    return float(accuracy_score(y_test, pipe.predict(X_test)))


if __name__ == "__main__":
    acc = train_and_evaluate()
    print(f"TEST_ACCURACY={acc:.4f}")
