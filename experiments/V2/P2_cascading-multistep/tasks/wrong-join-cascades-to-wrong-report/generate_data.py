"""
Generate synthetic data for the wrong-join-cascades-to-wrong-report task.

Produces three CSVs into environment/:
  transactions.csv  – 10 000 rows
  customers.csv     – 7 500 rows
  regions.csv       – 50 rows

Design:
  - 7 500 "matched" transactions (customer_ids 1..7500):
      Q1: 2000, Q2: 2000, Q3: 2000, Q4: 1500
  - 2 500 "orphan" transactions (customer_ids 100000-200000, no customer record):
      Q1: 333, Q2: 333, Q3: 334, Q4: 1500  (60% of orphans in Q4)
  - All amounts drawn from lognormal(mu=6.52, sigma=0.70), E[amount] ~ $864

Expected output (verified at the bottom of this script):
  True annual revenue  ~ $8.4 M  (all transactions, target [7.98M, 8.82M])
  Q4 true revenue      ~ $2.65 M  (NOT underperforming)
  INNER-join annual    ~ $6.5 M   (orphans dropped)
  INNER-join Q4        ~ $1.32 M  (~19% below INNER avg of $1.63 M -> flagged)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(TASK_DIR, "environment")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MU_AMOUNT    = 6.52   # lognormal mu; E[amount] = exp(6.52 + 0.70^2/2) ~ $864
SIGMA_AMOUNT = 0.70

N_CUSTOMERS = 7_500
N_REGIONS   = 50

# Matched txn counts per quarter (total = 7500)
N_MATCHED_Q1 = 2000
N_MATCHED_Q2 = 2000
N_MATCHED_Q3 = 2000
N_MATCHED_Q4 = 1500

# Orphan txn counts per quarter (total = 2500, Q4 = 60%)
N_ORPHAN_Q1 = 333
N_ORPHAN_Q2 = 333
N_ORPHAN_Q3 = 334
N_ORPHAN_Q4 = 1500

ORPHAN_ID_LOW  = 100_000
ORPHAN_ID_HIGH = 200_000

Q_RANGES = {
    "Q1": ("2024-01-01", "2024-03-31"),
    "Q2": ("2024-04-01", "2024-06-30"),
    "Q3": ("2024-07-01", "2024-09-30"),
    "Q4": ("2024-10-01", "2024-12-31"),
}


def random_dates(quarter: str, n: int) -> list[str]:
    start = pd.Timestamp(Q_RANGES[quarter][0])
    end   = pd.Timestamp(Q_RANGES[quarter][1])
    ndays = (end - start).days + 1
    offsets = rng.integers(0, ndays, size=n)
    return [(start + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in offsets]


def make_txn_block(customer_ids: list[int], quarter: str, n: int) -> pd.DataFrame:
    cids    = rng.choice(customer_ids, size=n, replace=True)
    amounts = rng.lognormal(mean=MU_AMOUNT, sigma=SIGMA_AMOUNT, size=n).round(2)
    dates   = random_dates(quarter, n)
    return pd.DataFrame({
        "customer_id": cids,
        "amount":      amounts,
        "date":        dates,
        "quarter":     quarter,
    })


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------
matched_ids = list(range(1, N_CUSTOMERS + 1))
orphan_ids  = list(range(ORPHAN_ID_LOW, ORPHAN_ID_HIGH + 1))

blocks = [
    make_txn_block(matched_ids, "Q1", N_MATCHED_Q1),
    make_txn_block(matched_ids, "Q2", N_MATCHED_Q2),
    make_txn_block(matched_ids, "Q3", N_MATCHED_Q3),
    make_txn_block(matched_ids, "Q4", N_MATCHED_Q4),
    make_txn_block(orphan_ids,  "Q1", N_ORPHAN_Q1),
    make_txn_block(orphan_ids,  "Q2", N_ORPHAN_Q2),
    make_txn_block(orphan_ids,  "Q3", N_ORPHAN_Q3),
    make_txn_block(orphan_ids,  "Q4", N_ORPHAN_Q4),
]

txns = pd.concat(blocks, ignore_index=True)
txns = txns.sample(frac=1, random_state=42).reset_index(drop=True)
txns.insert(0, "transaction_id", range(1, len(txns) + 1))

# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
signup_start = pd.Timestamp("2018-01-01")
signup_end   = pd.Timestamp("2023-12-31")
ndays_range  = (signup_end - signup_start).days

signup_offsets = rng.integers(0, ndays_range, size=N_CUSTOMERS)
signup_dates   = [
    (signup_start + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
    for d in signup_offsets
]

customers = pd.DataFrame({
    "customer_id": range(1, N_CUSTOMERS + 1),
    "region_id":   rng.integers(1, N_REGIONS + 1, size=N_CUSTOMERS),
    "signup_date": signup_dates,
})

# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
REGION_NAMES = [
    "Northwest", "Northeast", "Southwest", "Southeast", "Midwest",
    "Plains", "Mountain", "Pacific", "Atlantic", "Gulf",
    "Great_Lakes", "New_England", "Mid_Atlantic", "South_Central", "North_Central",
    "West_Coast", "East_Coast", "Interior", "Border", "Capital",
    "Lakefront", "Coastal", "Highland", "Lowland", "Peninsula",
    "Delta", "Valley", "Canyon", "Prairie", "Tundra",
    "Cascade", "Appalachian", "Ozark", "Sierra", "Rockies",
    "Columbia", "Mississippi", "Ohio", "Hudson", "Tennessee",
    "Chesapeake", "Puget", "Mojave", "Sonoran", "Everglades",
    "Piedmont", "Panhandle", "Heartland", "Frontier", "Metro",
]
COUNTRIES = ["USA"] * 40 + ["Canada"] * 10

regions = pd.DataFrame({
    "region_id":   range(1, N_REGIONS + 1),
    "region_name": REGION_NAMES,
    "country":     COUNTRIES,
})

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
print("=== DATA GENERATION VERIFICATION ===\n")
print(f"Total transactions: {len(txns)}")
print(f"  Orphan txns: {(txns['customer_id'] >= ORPHAN_ID_LOW).sum()} "
      f"({(txns['customer_id'] >= ORPHAN_ID_LOW).sum() / len(txns) * 100:.1f}%)")

print("\n--- True quarterly revenue (all transactions) ---")
true_qrev   = txns.groupby("quarter")["amount"].sum()
true_annual = true_qrev.sum()
for q in ["Q1", "Q2", "Q3", "Q4"]:
    print(f"  {q}: ${true_qrev[q]:>12,.0f}")
print(f"  Annual: ${true_annual:>10,.0f}")

print("\n--- INNER-join quarterly revenue (orphans dropped) ---")
inner_txns  = txns[txns["customer_id"] <= N_CUSTOMERS]
inner_qrev  = inner_txns.groupby("quarter")["amount"].sum()
inner_annual = inner_qrev.sum()
inner_avg   = inner_annual / 4
for q in ["Q1", "Q2", "Q3", "Q4"]:
    drop_pct = (inner_avg - inner_qrev[q]) / inner_avg * 100
    flag = " <- FLAGGED (>10% below avg)" if inner_qrev[q] < 0.90 * inner_avg else ""
    print(f"  {q}: ${inner_qrev[q]:>12,.0f}  (vs avg ${inner_avg:,.0f}, {drop_pct:+.1f}%){flag}")
print(f"  Annual: ${inner_annual:>10,.0f}")

print("\n--- Cascade check ---")
in_bounds = 7_980_000 <= true_annual <= 8_820_000
print(f"  True annual in [7.98M, 8.82M]: {true_annual:,.0f} -> {'PASS' if in_bounds else 'FAIL'}")
q4_flagged = inner_qrev["Q4"] < 0.90 * inner_avg
print(f"  Q4 INNER flagged (>10% drop):  {q4_flagged} -> {'PASS' if q4_flagged else 'FAIL'}")
q4_true_ok = true_qrev["Q4"] >= 0.90 * (true_annual / 4)
print(f"  Q4 true NOT underperforming:   {q4_true_ok} -> {'PASS' if q4_true_ok else 'FAIL'}")

# ---------------------------------------------------------------------------
# Write CSVs
# ---------------------------------------------------------------------------
txns.to_csv(os.path.join(OUT_DIR, "transactions.csv"), index=False)
customers.to_csv(os.path.join(OUT_DIR, "customers.csv"), index=False)
regions.to_csv(os.path.join(OUT_DIR, "regions.csv"), index=False)

print(f"\nWrote environment/transactions.csv ({len(txns)} rows)")
print(f"Wrote environment/customers.csv ({len(customers)} rows)")
print(f"Wrote environment/regions.csv ({len(regions)} rows)")
