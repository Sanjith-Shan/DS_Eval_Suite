"""
generate_data.py

Generates environment/properties.csv for the p3-units-mismatch-across-columns task.

New design: categorical-encoded-as-numeric
  zip_code        — int in {10001..10030}, stored as int (actually categorical)
  school_district — int in {1..15},        stored as int (actually categorical)

Both columns have NON-MONOTONE lookup tables mapping integer code -> price premium.
A naive LinearRegression treats them as numeric (wrong).
An aware model one-hot encodes them first (correct).

True price relationship:
  price = zip_premium[zip_code]
        + district_quality[school_district]
        + 150 * sqft
        + 12000 * bedrooms
        + 9000 * bathrooms
        - 500 * age_years
        + 8000 * garage_spaces
        + noise ~ N(0, 15000)

Validation: NAIVE RMSE / AWARE RMSE >= 1.5 (target 2x).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os

SEED = 42
N = 5000
rng = np.random.default_rng(SEED)

# ---------- non-monotone lookup tables ----------
# Assign zip premiums: non-monotone in integer value (seeded random assignment)
_zip_rng = np.random.default_rng(SEED + 1)
zip_codes_list = list(range(10001, 10031))  # 30 zip codes
zip_premiums_arr = _zip_rng.uniform(-60000, 90000, len(zip_codes_list))
zip_premium = dict(zip(zip_codes_list, zip_premiums_arr))

# Assign district quality: non-monotone in integer value
_dist_rng = np.random.default_rng(SEED + 2)
district_list = list(range(1, 16))  # 15 districts
district_quality_arr = _dist_rng.uniform(-40000, 70000, len(district_list))
district_quality = dict(zip(district_list, district_quality_arr))

# ---------- feature generation ----------
zip_code        = rng.choice(zip_codes_list, N)
school_district = rng.choice(district_list, N)
sqft            = rng.normal(1800, 500, N).clip(800, 4000)
bedrooms        = rng.integers(1, 7, N)        # 1-6
bathrooms       = rng.integers(1, 5, N)        # 1-4
age_years       = rng.integers(0, 81, N)       # 0-80
garage_spaces   = rng.integers(0, 4, N)        # 0-3

# ---------- target ----------
zip_prem_vec  = np.array([zip_premium[z]      for z in zip_code])
dist_qual_vec = np.array([district_quality[d] for d in school_district])

noise = rng.normal(0, 15000, N)

price = (
    zip_prem_vec
    + dist_qual_vec
    + 150   * sqft
    + 12000 * bedrooms
    + 9000  * bathrooms
    - 500   * age_years
    + 8000  * garage_spaces
    + noise
)

# ---------- build dataframe ----------
property_id = [f"P{i:05d}" for i in range(1, N + 1)]

df = pd.DataFrame({
    "property_id":     property_id,
    "zip_code":        zip_code.astype(int),
    "school_district": school_district.astype(int),
    "sqft":            sqft.round(1),
    "bedrooms":        bedrooms.astype(int),
    "bathrooms":       bathrooms.astype(int),
    "age_years":       age_years.astype(int),
    "garage_spaces":   garage_spaces.astype(int),
    "price":           price.round(2),
})

# ---------- save ----------
out_dir = os.path.join(os.path.dirname(__file__), "environment")
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "properties.csv")
df.to_csv(csv_path, index=False)
print(f"Saved {len(df)} rows to {csv_path}")

# ---------- summary stats ----------
print("\n=== Summary Statistics ===")
print(df.describe(include="all").to_string())

# ---------- validation models ----------
feature_cols = ["zip_code", "school_district", "sqft", "bedrooms",
                "bathrooms", "age_years", "garage_spaces"]
target_col = "price"

y = df[target_col]

# Naive: treat zip_code and school_district as numeric
X_naive = df[feature_cols]

# Aware: one-hot encode zip_code and school_district
df_encoded = pd.get_dummies(df[feature_cols + [target_col]],
                            columns=["zip_code", "school_district"])
X_aware = df_encoded.drop(columns=[target_col])

def fit_rmse(X, y, label, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"  {label:50s}  RMSE = {rmse:,.0f}")
    return rmse

print("\n=== Model Comparison ===")
rmse_naive = fit_rmse(X_naive, y, "NAIVE (zip_code & school_district as numeric)")
rmse_aware = fit_rmse(X_aware, y, "AWARE (zip_code & school_district one-hot encoded)")

ratio = rmse_naive / rmse_aware
print(f"\n  NAIVE / AWARE ratio = {ratio:.2f}x")

# ---------- validation checks ----------
print("\n=== Validation Assertions ===")

assert rmse_naive > 25000, f"FAIL: NAIVE RMSE={rmse_naive:,.0f} is surprisingly low (< 25k)"
assert rmse_aware < 25000, f"FAIL: AWARE RMSE={rmse_aware:,.0f} should be < 25k (noise floor ~15k)"
assert ratio >= 1.5, (
    f"FAIL: NAIVE/AWARE ratio={ratio:.2f} < 1.5 — gap not large enough to discriminate.\n"
    "Increase the magnitude of zip_premium / district_quality lookup tables and re-run."
)

print(f"  NAIVE RMSE  = {rmse_naive:,.0f}")
print(f"  AWARE RMSE  = {rmse_aware:,.0f}")
print(f"  Ratio       = {ratio:.2f}x  (>= 1.5 required)")
print("  All assertions passed.")
