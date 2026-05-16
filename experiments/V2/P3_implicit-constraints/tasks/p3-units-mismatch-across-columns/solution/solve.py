"""
solve.py — Reference solution for p3-units-mismatch-across-columns.

Key insight: zip_code and school_district are stored as integers but they are
categorical codes (30 zip codes, 15 districts) with non-monotone effects on
price. Treating them as numeric causes LinearRegression to mis-model the
non-linear relationship. One-hot encoding them captures each category's
individual premium and produces an RMSE near the noise floor (~15k).
"""

import json
import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

DATA_PATH    = "/workspace/properties.csv"
OUTPUT_PATH  = "/output/results.json"
RANDOM_STATE = 0
TEST_SIZE    = 0.2

# Load data
df = pd.read_csv(DATA_PATH)

# Inspect columns
print("Columns:", df.columns.tolist())
print("dtypes:\n", df.dtypes)
print("\nSample:\n", df.head())
print("\nUnique counts:\n", df.nunique())

# Notice: zip_code has 30 unique integers; school_district has 15 unique integers.
# Both have small, bounded integer ranges — treat them as categorical.

feature_cols = ["zip_code", "school_district", "sqft", "bedrooms",
                "bathrooms", "age_years", "garage_spaces"]
target_col   = "price"

# One-hot encode the integer-coded categorical columns
df_enc = pd.get_dummies(df[feature_cols + [target_col]],
                        columns=["zip_code", "school_district"])

X = df_enc.drop(columns=[target_col])
y = df_enc[target_col]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Fit LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse  = float(root_mean_squared_error(y_test, preds))
print(f"\nTest RMSE: {rmse:,.2f}")

# Write output
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
results = {
    "rmse":       round(rmse, 4),
    "model_type": "LinearRegression",
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results written to {OUTPUT_PATH}")
print(json.dumps(results, indent=2))
