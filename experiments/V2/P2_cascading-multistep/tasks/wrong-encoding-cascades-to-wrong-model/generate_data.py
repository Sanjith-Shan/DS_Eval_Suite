"""generate_data.py — produce dataset.csv for wrong-encoding-cascades-to-wrong-model.

Run from the task root directory. Writes environment/dataset.csv.
Validates both a one-hot pipeline and an ordinal pipeline, confirming:
  - Ordinal: accuracy >= 0.79, 'education_level' is top feature
  - One-hot: same accuracy range (also >= 0.79), but top features contain
    'education_level_hs' / 'education_level_phd' NOT the bare 'education_level'

Cascade mechanism:
  The ba-group (bachelors) has P(y=1) that depends almost entirely on
  subscription_months. With one-hot encoding, the 'education_level_bachelors'
  dummy has near-zero mutual information and near-zero RF importance (because
  the average P(y=1|ba) ≈ overall marginal). This means one-hot RF reports
  edu_hs / edu_phd / edu_ma as important, NEVER 'education_level'.
  With ordinal encoding, the single 'education_level' integer captures the
  full monotonic ordering and consistently ranks as the #1 important feature.

np.random.seed(42) for reproducibility.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)

N = 5000
EDUCATION_LEVELS = ["high_school", "bachelors", "masters", "phd"]
EDU_PROBS = [0.35, 0.35, 0.20, 0.10]


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    # Education level — the ordinal trap
    edu_idx = rng.choice(4, size=N, p=EDU_PROBS)
    education_level = np.array(EDUCATION_LEVELS)[edu_idx]

    # Subscription months: 1-60
    subscription_months = rng.integers(1, 61, size=N)
    sub_norm = (subscription_months - 1.0) / 59.0

    # Other numeric features (mostly noise for this task's cascade)
    age = rng.integers(18, 71, size=N)
    income = np.round(np.exp(rng.normal(10.8, 0.6, size=N)), 2)
    years_experience = np.clip(
        (age - 18) * 0.65 + rng.integers(-5, 6, size=N), 0, 40
    ).astype(int)
    support_tickets_last_year = rng.integers(0, 21, size=N)
    n_logins_last_30d = rng.integers(0, 101, size=N)
    last_login_days_ago = rng.integers(0, 366, size=N)

    # Nominal categoricals (no ordinality, true one-hot candidates)
    region = rng.choice(
        ["north", "south", "east", "west"], size=N, p=[0.25, 0.25, 0.25, 0.25]
    )
    account_type = rng.choice(
        ["basic", "premium", "enterprise"], size=N, p=[0.50, 0.35, 0.15]
    )
    marketing_channel = rng.choice(
        ["organic", "paid", "referral", "social"], size=N, p=[0.30, 0.30, 0.20, 0.20]
    )
    device_type = rng.choice(
        ["mobile", "desktop", "tablet"], size=N, p=[0.50, 0.35, 0.15]
    )

    # Target probability — cascade design:
    # hs (idx=0): very low probability, determined by education alone
    # ba (idx=1): probability ENTIRELY depends on subscription_months
    #             P(y=1|ba, sub_norm) = 0.20 + sub_norm * 0.60
    #             avg P(y=1|ba) = 0.50 ≈ overall marginal -> edu_ba dummy has ~0 MI
    # ma (idx=2): moderately high, education alone nearly sufficient
    # phd (idx=3): very high probability, education alone determines outcome
    p_map = {
        0: lambda sn: 0.03,              # hs: always ~0
        1: lambda sn: 0.20 + sn * 0.60, # ba: 0.20–0.80, avg 0.50
        2: lambda sn: 0.75,              # ma: always ~0.75
        3: lambda sn: 0.95,              # phd: always ~0.95
    }
    prob_raw = np.array([p_map[e](sn) for e, sn in zip(edu_idx, sub_norm)])
    noise = rng.normal(0, 0.015, size=N)
    prob = np.clip(prob_raw + noise, 0.01, 0.99)
    target = (rng.uniform(0, 1, size=N) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "years_experience": years_experience,
        "region": region,
        "account_type": account_type,
        "marketing_channel": marketing_channel,
        "subscription_months": subscription_months,
        "support_tickets_last_year": support_tickets_last_year,
        "n_logins_last_30d": n_logins_last_30d,
        "last_login_days_ago": last_login_days_ago,
        "device_type": device_type,
        "education_level": education_level,
        "target": target,
    })
    return df


def run_pipeline(
    df: pd.DataFrame,
    use_ordinal_for_education: bool,
) -> tuple[float, list[str]]:
    """Encode → train RF on all features → report accuracy and top-5 feature names.

    The top-5 feature names come from RF feature importances.
    This is the natural thing an agent does:
    - Encode all categoricals
    - Train a RandomForestClassifier
    - Report feature importances as top features
    - Report test accuracy

    With one-hot: 'education_level' is split into 4 binary dummies.
      RF importance goes to the extreme-signal dummies (hs, phd, ma) but NOT
      'education_level' as a single name -> the agent reports 'education_level_hs',
      'education_level_phd', etc. rather than 'education_level'.
    With ordinal: 'education_level' (0-3 integer) is a single feature.
      RF ranks it #1 -> the agent reports 'education_level' in top features.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    numeric_cols = [
        "age", "income", "years_experience", "subscription_months",
        "support_tickets_last_year", "n_logins_last_30d", "last_login_days_ago",
    ]
    nominal_cat_cols = ["region", "account_type", "marketing_channel", "device_type"]
    ordinal_col = ["education_level"]
    edu_categories = [["high_school", "bachelors", "masters", "phd"]]

    if use_ordinal_for_education:
        ct = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_cat_cols),
            ("edu", OrdinalEncoder(categories=edu_categories), ordinal_col),
        ])
        X_enc = ct.fit_transform(X)
        nom_names = list(ct.named_transformers_["nom"].get_feature_names_out(nominal_cat_cols))
        feature_names = numeric_cols + nom_names + ["education_level"]
    else:
        ct = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             nominal_cat_cols + ordinal_col),
        ])
        X_enc = ct.fit_transform(X)
        cat_names = list(
            ct.named_transformers_["cat"].get_feature_names_out(nominal_cat_cols + ordinal_col)
        )
        feature_names = numeric_cols + cat_names

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = rf.score(X_test, y_test)

    top5_idx = np.argsort(rf.feature_importances_)[::-1][:5]
    top5_names = [feature_names[i] for i in top5_idx]

    return accuracy, top5_names


def evaluate_pipelines(df: pd.DataFrame) -> None:
    acc_oh, top5_oh = run_pipeline(df, use_ordinal_for_education=False)
    acc_ord, top5_ord = run_pipeline(df, use_ordinal_for_education=True)

    print("=" * 60)
    print("ONE-HOT PIPELINE (wrong choice — default encoding)")
    print(f"  Test accuracy:  {acc_oh:.4f}")
    print(f"  Top 5 features: {top5_oh}")
    print()
    print("ORDINAL PIPELINE (correct choice)")
    print(f"  Test accuracy:  {acc_ord:.4f}")
    print(f"  Top 5 features: {top5_ord}")
    print("=" * 60)

    # Cascade property 1: ordinal accuracy >= 0.79 (verifier threshold)
    assert 0.79 <= acc_ord <= 0.87, \
        f"Ordinal accuracy {acc_ord:.4f} not in [0.79, 0.87]"

    # Cascade property 2: ordinal top-5 contains 'education_level' (bare name)
    assert "education_level" in top5_ord, \
        f"'education_level' not in ordinal top-5: {top5_ord}"

    # Cascade property 3: one-hot top-5 does NOT contain bare 'education_level'
    assert "education_level" not in top5_oh, \
        f"Bare 'education_level' should NOT be in one-hot top-5: {top5_oh}"

    # Cascade property 4: one-hot top-5 contains fragmented dummy names
    edu_fragments = [f for f in top5_oh if "education_level_" in f]
    assert edu_fragments, \
        f"One-hot top-5 should contain an education_level_* dummy: {top5_oh}"

    print("All cascade assertions passed.")
    print(f"  One-hot fragmented edu features in top-5: {edu_fragments}")
    print(f"  Ordinal accuracy {acc_ord:.4f} >= 0.79 threshold: True")


if __name__ == "__main__":
    df = generate()
    print(f"Generated {len(df)} rows.")
    print(f"Target rate: {df['target'].mean():.3f}")
    print(f"Education distribution:\n{df['education_level'].value_counts()}")
    print()

    evaluate_pipelines(df)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "environment", "dataset.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
