"""Fixed classifier pipeline.

Fixes applied vs the buggy pipeline.py:
  1. OneHotEncoder is fitted ONLY on training data (after the train/test split),
     so test categories that are rare or unseen do not influence the encoding.
  2. Rare city categories with fewer than 10 training occurrences are collapsed
     into an infrequent bucket via min_frequency=10, preventing the model from
     overfitting to near-empty columns that carry no real signal.
  3. Accuracy is reported on the HELD-OUT TEST SET (y_test), not on training data.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def train_and_evaluate(data_path: str = "/workspace/data.csv") -> float:
    df = pd.read_csv(data_path)
    y = df.pop("target")

    # Split BEFORE any encoding so the encoder never sees test data.
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit the encoder on training data only.
    # min_frequency=10 collapses rare cities (< 10 training occurrences) into
    # an infrequent bucket, removing near-empty noise columns.
    ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
        min_frequency=10,
    )
    city_train = ohe.fit_transform(X_train[["city_name"]])
    city_test = ohe.transform(X_test[["city_name"]])
    city_cols = ohe.get_feature_names_out(["city_name"])

    X_train_enc = pd.concat(
        [
            X_train.drop(columns=["city_name"]).reset_index(drop=True),
            pd.DataFrame(city_train, columns=city_cols),
        ],
        axis=1,
    )
    X_test_enc = pd.concat(
        [
            X_test.drop(columns=["city_name"]).reset_index(drop=True),
            pd.DataFrame(city_test, columns=city_cols),
        ],
        axis=1,
    )
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_enc, y_train)

    # Evaluate on the HELD-OUT TEST SET.
    preds = model.predict(X_test_enc)
    return float(accuracy_score(y_test, preds))


if __name__ == "__main__":
    acc = train_and_evaluate()
    print(f"TEST_ACCURACY={acc:.4f}")
