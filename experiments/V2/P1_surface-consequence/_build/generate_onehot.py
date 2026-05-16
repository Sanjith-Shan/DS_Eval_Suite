"""Generate data.csv and buggy pipeline.py for onehot-rare-categories-overfit.

Data spec:
  - 5000 rows
  - 10 numeric features with real signal (X0..X9) - moderate predictive power
  - city_name: 120 unique values, Zipf-distributed (many rare categories)
  - Binary target driven by BOTH numeric features AND city_name geographic effect
    (top-20 cities by Zipf rank have strong signal; cities 20-120 have no signal)

The BUGGY pipeline has two issues:
  1. Fits OneHotEncoder on the FULL dataset before the train/test split
  2. Evaluates accuracy on the TRAINING set instead of the test set
  -> Reports ~100% accuracy (training accuracy, clearly a bug)

Calibrated accuracy bands:
  - Buggy (eval on training data):           ~1.00 (clearly wrong, fails verifier)
  - Named fix (drop city_name entirely):     ~0.67 (fails band [0.72, 0.84])
  - Oracle (OHE after split, min_freq=10):   ~0.77 (passes band [0.72, 0.84])

Fixed seed: 20260516
"""

import numpy as np
import pandas as pd
import pathlib

SEED = 20260516
RNG = np.random.default_rng(SEED)

N_ROWS = 5000
N_NUMERIC = 10
N_CITIES = 120
ZIPF_ALPHA = 1.8

# ── 1. City effects (only top-20 cities have real signal) ───────────────────

city_effects = np.zeros(N_CITIES)
city_effects[:20] = RNG.uniform(-3.0, 3.0, 20)  # strong geographic effect

# ── 2. City Zipf distribution ────────────────────────────────────────────────

cities = [f"city_{i:03d}" for i in range(N_CITIES)]
ranks = np.arange(1, N_CITIES + 1, dtype=float)
zipf_weights = ranks ** (-ZIPF_ALPHA)
zipf_weights /= zipf_weights.sum()
city_indices = RNG.choice(N_CITIES, size=N_ROWS, p=zipf_weights)
city_col = [cities[i] for i in city_indices]

# ── 3. Numeric features ──────────────────────────────────────────────────────

coefs = np.array([0.25, 0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02])
X_num = RNG.standard_normal((N_ROWS, N_NUMERIC))

# ── 4. Binary target ─────────────────────────────────────────────────────────

city_contribution = np.array([city_effects[i] for i in city_indices])
raw_logit = X_num @ coefs + city_contribution * 0.9
# Center logit so target is ~balanced
intercept = -np.median(raw_logit)
logit = raw_logit + intercept + RNG.standard_normal(N_ROWS) * 0.5
prob = 1 / (1 + np.exp(-logit))
y = (prob > 0.5).astype(int)

# ── 5. Assemble DataFrame ────────────────────────────────────────────────────

col_names = [f"X{i}" for i in range(N_NUMERIC)]
df = pd.DataFrame(X_num, columns=col_names)
df["city_name"] = city_col
df["target"] = y

city_counts = pd.Series(city_col).value_counts()
print(f"Dataset shape: {df.shape}")
print(f"Target distribution: {df['target'].value_counts().to_dict()}")
print(f"Unique cities: {df['city_name'].nunique()}")
print(f"Cities with < 5 rows: {(city_counts < 5).sum()} / {N_CITIES}")
print(f"Cities with < 10 rows: {(city_counts < 10).sum()} / {N_CITIES}")
print(f"Cities with >= 10 rows: {(city_counts >= 10).sum()} / {N_CITIES}")

# ── 6. Write data.csv ────────────────────────────────────────────────────────

out_dir = (
    pathlib.Path(__file__).parent.parent
    / "tasks"
    / "onehot-rare-categories-overfit"
    / "environment"
)
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "data.csv"
df.to_csv(csv_path, index=False)
print(f"Written: {csv_path}")

# ── 7. Write buggy pipeline.py ───────────────────────────────────────────────

buggy_pipeline = '''"""Classifier pipeline for predicting binary target from customer data.

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
'''

pipeline_path = out_dir / "pipeline.py"
pipeline_path.write_text(buggy_pipeline)
print(f"Written: {pipeline_path}")
print("\nVerify buggy pipeline reports ~1.0 (training accuracy):")
print(f"  python {pipeline_path}")
