"""Generate the dataset and buggy pipeline for the data-leakage-detection task.

Design goals:
- Clean pipeline reaches ~75-82% test accuracy.
- Buggy pipeline (with the three planted leakages) reaches >=92% test accuracy
  on the same train/test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "samples" / "data-leakage-detection" / "environment"
CSV_OUT = OUT_DIR / "data.csv"
PIPELINE_OUT = OUT_DIR / "pipeline.py"

SEED = 20260515
N_ROWS = 10_000
N_NUMERIC = 14


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Numeric features. Modest, noisy real signal so the clean baseline lands
    # comfortably in the 73-80% band.
    X = rng.normal(0, 1, size=(N_ROWS, N_NUMERIC))
    weights = np.array([0.7, -0.55, 0.45, -0.35] + [0.0] * (N_NUMERIC - 4))
    logits = X @ weights + rng.normal(0, 1.0, size=N_ROWS)
    proba = 1.0 / (1.0 + np.exp(-logits))
    target = (proba > 0.5).astype(int)

    # High-cardinality categorical that is independent of target. Effectively
    # one row per category (with a small amount of accidental collision), so
    # target encoding on the full dataset is a near-perfect self-leak.
    customer_segment = np.array([f"seg_{i:05d}" for i in rng.integers(0, N_ROWS * 4, size=N_ROWS)])

    df = pd.DataFrame(X, columns=[f"feat_{i:02d}" for i in range(N_NUMERIC)])
    df["customer_segment"] = customer_segment
    df["target"] = target

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT} ({len(df)} rows, target mean={target.mean():.3f})")

    # --- Buggy reference pipeline shipped to the agent ---
    PIPELINE_OUT.write_text(BUGGY_PIPELINE)
    print(f"wrote {PIPELINE_OUT}")


BUGGY_PIPELINE = '''"""Customer churn prediction pipeline.

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
'''


if __name__ == "__main__":
    main()
