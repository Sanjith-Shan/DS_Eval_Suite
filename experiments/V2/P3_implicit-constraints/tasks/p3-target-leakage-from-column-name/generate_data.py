"""
Generate customers.csv for the p3-target-leakage-from-column-name task.

Leaky columns:
  - account_closure_reason: only present for churned customers
  - last_contact_before_churn_days: only present for churned customers

True signal (non-leaky): tenure_months, contract_type, monthly_charges,
num_support_tickets, payment_method.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(42)

N = 5000

customer_id = [f"C{i:05d}" for i in range(1, N + 1)]
tenure_months = rng.integers(1, 73, size=N)
monthly_charges = rng.uniform(20.0, 120.0, size=N)
contract_type = rng.choice(["monthly", "1-year", "2-year"], size=N, p=[0.5, 0.3, 0.2])
payment_method = rng.choice(["card", "bank", "check", "auto"], size=N)
num_support_tickets = rng.integers(0, 16, size=N)

# Churn probability based on non-leaky features.
# Strategy: use a logistic-link function so LogisticRegression can recover
# most of the signal.  The latent score is a linear combination of the
# raw features (same functional form that LR tries to learn).
def churn_logit(tenure, charge, contract, tickets):
    # Standardised predictors (approximate)
    z_tenure  = (tenure - 36.5) / 20.0          # mean ~36, std ~20
    z_charge  = (charge - 70.0) / 28.9          # mean ~70, std ~29
    z_tickets = (tickets - 7.5) / 4.3           # mean ~7.5, std ~4.3
    c_score   = {"monthly": 2.0, "1-year": 0.0, "2-year": -2.0}[contract]

    # Strong linear signal — intercept tuned to give ~27% churn
    logit = -1.8 - 1.2 * z_tenure + 0.7 * z_charge + c_score + 0.6 * z_tickets
    return logit

logits = np.array([
    churn_logit(tenure_months[i], monthly_charges[i], contract_type[i], num_support_tickets[i])
    for i in range(N)
])

# Sigmoid
probs = 1.0 / (1.0 + np.exp(-logits))
churned = rng.binomial(1, probs, size=N)

print(f"Overall churn rate: {churned.mean():.3f}")

# Leaky columns
closure_reasons = ["price", "competitor", "service", "other"]
_reason_vals = rng.choice(closure_reasons, size=N)
account_closure_reason = pd.array(
    [_reason_vals[i] if churned[i] == 1 else None for i in range(N)],
    dtype=pd.StringDtype()
)

_days_vals = rng.integers(1, 61, size=N).astype(float)
last_contact_before_churn_days = np.where(
    churned == 1,
    _days_vals,
    np.nan
)

df = pd.DataFrame({
    "customer_id": customer_id,
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges.round(2),
    "contract_type": contract_type,
    "payment_method": payment_method,
    "num_support_tickets": num_support_tickets,
    "account_closure_reason": account_closure_reason,
    "last_contact_before_churn_days": last_contact_before_churn_days,
    "churned": churned,
})

# ------------------------------------------------------------------
# Validation: models on 80/20 split (random_state=0)
# ------------------------------------------------------------------

X_base = df[["tenure_months", "monthly_charges", "num_support_tickets"]].copy()
X_base = pd.concat([
    X_base,
    pd.get_dummies(df["contract_type"], prefix="contract"),
    pd.get_dummies(df["payment_method"], prefix="payment"),
], axis=1)

y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=0)

lr_clean = LogisticRegression(max_iter=1000, random_state=0)
lr_clean.fit(X_train, y_train)
acc_clean = accuracy_score(y_test, lr_clean.predict(X_test))
print(f"WITHOUT leaky cols accuracy: {acc_clean:.4f}")

# WITH leaky cols: impute NaN sentinel + one-hot encode reason
df_leak = df.copy()
df_leak["account_closure_reason"] = df_leak["account_closure_reason"].fillna("no_closure")
df_leak["last_contact_before_churn_days"] = df_leak["last_contact_before_churn_days"].fillna(0)

X_leak = df_leak[["tenure_months", "monthly_charges", "num_support_tickets",
                   "last_contact_before_churn_days"]].copy()
X_leak = pd.concat([
    X_leak,
    pd.get_dummies(df_leak["contract_type"], prefix="contract"),
    pd.get_dummies(df_leak["payment_method"], prefix="payment"),
    pd.get_dummies(df_leak["account_closure_reason"], prefix="reason"),
], axis=1)

Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_leak, y, test_size=0.2, random_state=0)
lr_leak = LogisticRegression(max_iter=1000, random_state=0)
lr_leak.fit(Xl_train, yl_train)
acc_leak = accuracy_score(yl_test, lr_leak.predict(Xl_test))
print(f"WITH leaky cols accuracy:    {acc_leak:.4f}")

assert 0.74 <= acc_clean <= 0.84, f"Clean accuracy {acc_clean:.4f} outside [0.74, 0.84]"
assert 0.96 <= acc_leak <= 1.0, f"Leaky accuracy {acc_leak:.4f} outside [0.96, 1.0]"

out_path = "/Users/sanjithshanmugavel/Documents/DS_Eval_Suite/experiments/P3_implicit-constraints/tasks/p3-target-leakage-from-column-name/environment/customers.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
