"""generate_data.py — host-side script for extrapolation-beyond-training.

Generates 6 annual observations (2018-2023) following a logarithmic growth
curve and writes them to ./environment/historical.csv.

Run from the repo root or from this file's directory:
    python generate_data.py
"""

from __future__ import annotations

import math
import os
import csv
from pathlib import Path

import numpy as np

# ── reproducibility ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── parameters ───────────────────────────────────────────────────────────────
# value(t) = a * log(t - 2017) + b + noise
# t in {2018, ..., 2023}  =>  (t - 2017) in {1, ..., 6}
# We want ~60 at t=2018 and ~110 at t=2023.
#   a * log(1) + b = 60  =>  b = 60   (log(1) = 0)
#   a * log(6) + b = 110 =>  a = 50 / log(6) ≈ 27.87

A = 50.0 / math.log(6)  # ≈ 27.87
B = 60.0
NOISE_STD = 0.8

years = list(range(2018, 2024))  # 2018..2023 inclusive

noise = rng.normal(0, NOISE_STD, len(years))
values = [A * math.log(y - 2017) + B + n for y, n in zip(years, noise)]

# ── print table ──────────────────────────────────────────────────────────────
print("Generated data (2018-2023):")
print(f"  {'year':>4}  {'value':>10}")
for y, v in zip(years, values):
    print(f"  {y:>4}  {v:>10.4f}")

# ── linear extrapolation to 2028 ─────────────────────────────────────────────
x = np.array(years, dtype=float)
y_arr = np.array(values)

coeffs_lin = np.polyfit(x, y_arr, 1)  # slope, intercept
lin_2028 = np.polyval(coeffs_lin, 2028.0)
print(f"\nLinear extrapolation to 2028 : {lin_2028:.2f}")

# ── log extrapolation to 2028 ────────────────────────────────────────────────
# Fit: value = a_fit * log(t - 2017) + b_fit  via polyfit on transformed x
log_x = np.log(x - 2017)
coeffs_log = np.polyfit(log_x, y_arr, 1)  # slope, intercept on log scale
log_2028 = np.polyval(coeffs_log, math.log(2028 - 2017))
print(f"Log    extrapolation to 2028 : {log_2028:.2f}")
print(f"\nTrue   approx value  at 2028 : ~125")
print(f"Linear over-shoot            : +{lin_2028 - 125:.1f} above true")
print(f"Log    over-shoot            : +{log_2028 - 125:.1f} above true")

# ── write CSV ────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent / "environment"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "historical.csv"

with open(out_path, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["year", "value"])
    for y, v in zip(years, values):
        writer.writerow([y, round(v, 6)])

print(f"\nWrote {len(years)} rows to {out_path}")
