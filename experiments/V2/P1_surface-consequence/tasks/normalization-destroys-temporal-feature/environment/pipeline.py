"""
pipeline.py  —  feature scaling bug: scaler is fit on the FULL dataset (train+test).
This is data leakage: statistics computed from test rows contaminate the training scaler.

Usage
-----
python pipeline.py          # prints test accuracy
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATA_PATH = "/workspace/data.csv"
N_TRAIN   = 3000

def load_and_run(data_path: str = DATA_PATH) -> float:
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    X_train_raw, X_test_raw = X[:N_TRAIN], X[N_TRAIN:]
    y_train,     y_test     = y[:N_TRAIN], y[N_TRAIN:]

    # ── BUG: scaler is fit on ALL rows (train + test) ──────────────────────
    scaler = StandardScaler()
    X_all_scaled  = scaler.fit_transform(X)         # leaks test statistics
    X_train = X_all_scaled[:N_TRAIN]
    X_test  = X_all_scaled[N_TRAIN:]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc

if __name__ == "__main__":
    acc = load_and_run()
    print(f"test_accuracy = {acc:.4f}")
