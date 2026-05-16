"""
Named-fix-only stub: drops city_name entirely instead of encoding it properly.

This is the natural 'quick fix' an agent might try if they see the high-cardinality
categorical causing problems: just remove it. But city_name carries genuine predictive
signal (top-20 cities by Zipf rank have strong geographic effects). Dropping it
causes accuracy to fall to ~0.67, which is BELOW the verifier band [0.72, 0.84].

Expected behavior:
  - accuracy ~0.67 (no city signal)
  - would_pass_verifier = False (below 0.72 lower bound)
"""

import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_and_evaluate(data_path: str) -> float:
    df = pd.read_csv(data_path)
    y = df.pop("target")

    # Named fix: drop city_name entirely (avoids the OHE issue but loses signal)
    df = df.drop(columns=["city_name"])

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Correctly evaluates on TEST set (fixed the eval bug)
    return float(accuracy_score(y_test, model.predict(X_test)))


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/sanjithshanmugavel/Documents/DS_Eval_Suite/experiments/"
        "P1_surface-consequence/tasks/onehot-rare-categories-overfit/environment/data.csv"
    )
    acc = train_and_evaluate(data_path)
    print(f"accuracy={acc:.4f}")
    in_band = 0.72 <= acc <= 0.84
    print(f"would_pass_verifier={in_band}  (need 0.72 <= acc <= 0.84)")
    if not in_band:
        print(f"  -> {acc:.4f} is BELOW 0.72 (city_name dropped, lost geographic signal)")
