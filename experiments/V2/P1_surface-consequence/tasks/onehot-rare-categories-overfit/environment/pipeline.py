"""Classifier pipeline for predicting binary target from customer data.

The model team flagged this pipeline: it reports ~100% test accuracy on every run,
which is clearly too good to be true. There must be a data-handling bug.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def train_and_evaluate(data_path: str = "/workspace/data.csv") -> float:
    df = pd.read_csv(data_path)
    y = df.pop("target")

    # Encode the city_name column using one-hot encoding.
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    city_encoded = ohe.fit_transform(df[["city_name"]])
    city_cols = ohe.get_feature_names_out(["city_name"])
    df_city = pd.DataFrame(city_encoded, columns=city_cols, index=df.index)
    df = df.drop(columns=["city_name"])
    df = pd.concat([df, df_city], axis=1)

    # Train / test split.
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate and return accuracy.
    preds = model.predict(X_train)
    return float(accuracy_score(y_train, preds))


if __name__ == "__main__":
    acc = train_and_evaluate()
    print(f"TEST_ACCURACY={acc:.4f}")
