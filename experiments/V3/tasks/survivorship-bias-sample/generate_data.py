"""
Generate companies.csv and README.md for the survivorship-bias task.

Key design decisions:
- ~2000 companies, all currently active (no is_active column)
- Founding years 2010-2024 with survivorship-induced skew: older cohorts are sparse
- Survival probability = 0.95 ^ (2024 - founding_year)
- revenue_growth_pct is the regression target
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

TARGET_N = 2000
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(TASK_DIR, "environment")
os.makedirs(ENV_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Sample founding years with survivorship decay
# ---------------------------------------------------------------------------
# Draw many candidates from uniform [2010, 2024], then accept each with
# probability 0.95 ^ (2024 - founding_year).
founding_years: list[int] = []
while len(founding_years) < TARGET_N:
    candidates = rng.integers(2010, 2025, size=TARGET_N * 10)  # 2025 exclusive
    for yr in candidates:
        p_survive = 0.85 ** (2024 - int(yr))
        if rng.random() < p_survive:
            founding_years.append(int(yr))
        if len(founding_years) >= TARGET_N:
            break

founding_years = founding_years[:TARGET_N]
n = len(founding_years)

# ---------------------------------------------------------------------------
# 2. Other features
# ---------------------------------------------------------------------------
industries = ["tech", "retail", "manufacturing", "services", "healthcare"]
industry = rng.choice(industries, size=n)

# employees: log-normal, clipped to [5, 500]
employees = np.clip(
    np.round(rng.lognormal(mean=3.5, sigma=1.0, size=n)).astype(int), 5, 500
)

# initial_funding_usd: log-normal, $10k – $50M
funding_raw = rng.lognormal(mean=13.0, sigma=2.0, size=n)  # ln(~$440k) center
initial_funding_usd = np.clip(funding_raw, 10_000, 50_000_000).astype(int)

# marketing_spend_pct: uniform 0–30
marketing_spend_pct = np.round(rng.uniform(0, 30, size=n), 2)

# ---------------------------------------------------------------------------
# 3. Revenue growth target (signal + noise)
# ---------------------------------------------------------------------------
# Coefficients (log-scale for funding to tame huge range):
#   baseline ≈ 10%
#   employees: slight negative (larger firms grow slower)
#   log_funding: positive
#   marketing_spend_pct: positive
#   industry: tech +15, healthcare +10, services +5, retail 0, manufacturing -5
industry_effect = {
    "tech": 15.0,
    "healthcare": 10.0,
    "services": 5.0,
    "retail": 0.0,
    "manufacturing": -5.0,
}

log_funding = np.log(initial_funding_usd)
growth = (
    10.0
    + (-0.04) * employees
    + 1.2 * log_funding
    + 0.5 * marketing_spend_pct
    + np.array([industry_effect[ind] for ind in industry])
    + rng.normal(0, 8, size=n)
)
revenue_growth_pct = np.clip(np.round(growth, 2), -20, 80)

# ---------------------------------------------------------------------------
# 4. Assemble DataFrame
# ---------------------------------------------------------------------------
company_ids = [f"CO{i:05d}" for i in range(1, n + 1)]

df = pd.DataFrame(
    {
        "company_id": company_ids,
        "founding_year": founding_years,
        "industry": industry,
        "employees": employees,
        "initial_funding_usd": initial_funding_usd,
        "marketing_spend_pct": marketing_spend_pct,
        "revenue_growth_pct": revenue_growth_pct,
    }
)

out_csv = os.path.join(ENV_DIR, "companies.csv")
df.to_csv(out_csv, index=False)
print(f"Saved {n} rows to {out_csv}")

# ---------------------------------------------------------------------------
# 5. Write README.md (implicit survivorship signal, no explicit hint)
# ---------------------------------------------------------------------------
readme_text = """# Company Dataset

This dataset was collected from active companies registered in the 2024 business registry.
Founding years range from 2010 to 2024. The data captures current operating metrics
and revenue growth figures reported for the 2024 fiscal year.
"""

out_readme = os.path.join(ENV_DIR, "README.md")
with open(out_readme, "w") as f:
    f.write(readme_text)
print(f"Saved README to {out_readme}")

# ---------------------------------------------------------------------------
# 6. Print founding-year histogram to confirm survivorship skew
# ---------------------------------------------------------------------------
print("\nFounding-year distribution (survivorship skew should be visible):")
counts = df["founding_year"].value_counts().sort_index()
for yr, cnt in counts.items():
    bar = "#" * (cnt // 5)
    print(f"  {yr}: {cnt:4d}  {bar}")
