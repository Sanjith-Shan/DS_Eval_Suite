"""Customer churn prediction pipeline.

Recent runs hit ~95% test accuracy, which the team flagged as suspicious.
Review the methodology and produce a corrected version at
/output/pipeline_fixed.py. The corrected script must expose the same
``train_and_evaluate(data_path: str) -> float`` entrypoint and the same return
contract (test accuracy as a float).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(data_path: str = "/workspace/data.csv") -> float:
    df = pd.read_csv(data_path)
    y = df.pop("target")

    # Scale numeric features.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Target-encode high-cardinality categoricals.
    for col in df.select_dtypes(include="object").columns:
        means = df.assign(_t=y.values).groupby(col)["_t"].mean()
        df[col + "_te"] = df[col].map(means)
        df = df.drop(columns=[col])

    # Pick the most informative features.
    mi = mutual_info_classif(df, y, random_state=42)
    top_idx = np.argsort(mi)[-10:]
    df = df.iloc[:, top_idx]

    # Train / test split.
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return float(accuracy_score(y_test, model.predict(X_test)))


if __name__ == "__main__":
    acc = train_and_evaluate()
    print(f"TEST_ACCURACY={acc:.4f}")
