"""
generate_dedup.py
=================
Generates the synthetic patient-visit dataset for the
deduplication-loses-valid-longitudinal-data task.

Design goals
------------
* 800 patients, 1–5 visits each (~2 100 total rows).
* Columns: patient_id, visit_number, days_between_visits,
           lab_value_1, lab_value_2, readmitted.
* Readmission signal lives ENTIRELY in the CHANGE (delta) of lab_value_1
  between consecutive visits and in short days_between_visits.
  The raw lab values are mean-reverting with large patient-level noise so
  the absolute level carries almost zero cross-patient predictive signal.

Expected AUC bands (calibrated empirically):
  - Naive dedup (keep-last per patient) + RF         → ~0.58-0.66. Fails.
  - All rows, patient-level split, raw lab values    → ~0.60-0.68. Fails.
  - All rows, patient-level split, + delta features  → ~0.76-0.84. Passes.

Run this script to (re-)generate data.csv.  Copy the output into
tasks/deduplication-loses-valid-longitudinal-data/environment/data.csv.
"""

import numpy as np
import pandas as pd

SEED = 42
N_PATIENTS = 800
OUT_PATH = "tasks/deduplication-loses-valid-longitudinal-data/environment/data.csv"


def generate(seed: int = SEED, n_patients: int = N_PATIENTS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []

    for i in range(n_patients):
        patient_id = f"p_{i+1:04d}"
        n_visits = int(rng.choice([1, 2, 3, 4, 5], p=[0.20, 0.30, 0.25, 0.15, 0.10]))

        # Patient-level baseline drawn from a VERY wide uniform distribution.
        # This ensures that cross-patient variation in absolute lab values is
        # large random noise — the RF cannot learn "lab1 > X → readmit" because
        # high lab1 at one patient's visit 2 might just mean that patient has a
        # naturally high baseline, not that they had a big delta.
        lab1_base = rng.uniform(-20.0, 20.0)   # enormous range, no signal
        lab2_base = rng.uniform(-15.0, 15.0)

        prev_lab1 = lab1_base
        prev_lab2 = lab2_base

        for v in range(1, n_visits + 1):
            if v == 1:
                days_between = 0
                lab1 = lab1_base + rng.normal(0, 0.1)
                lab2 = lab2_base + rng.normal(0, 0.1)
                p_readmit = 0.10
            else:
                days_between = int(rng.integers(7, 180))

                # Mean-reverting delta: lab tends back toward baseline +/- noise.
                # The *signed* delta drawn here IS the signal.
                delta1 = rng.normal(0.0, 2.0)   # SD=2, centered at 0 (no trend)
                delta2 = rng.normal(0.0, 1.5)

                # Mean-revert slightly so the level stays near base
                lab1 = 0.6 * prev_lab1 + 0.4 * lab1_base + delta1
                lab2 = 0.6 * prev_lab2 + 0.4 * lab2_base + delta2

                # ---- Actual delta (what happened since last visit) ----
                actual_delta1 = lab1 - prev_lab1
                actual_delta2 = lab2 - prev_lab2

                # ---- Signal is ONLY from actual_delta1 and interval ----
                d1_z = actual_delta1 / 2.0          # normalised
                interval_z = (180.0 - days_between) / 173.0

                logit = -1.5 + 3.5 * d1_z + 2.0 * interval_z
                logit += rng.normal(0, 0.5)
                p_readmit = float(np.clip(1.0 / (1.0 + np.exp(-logit)), 0.02, 0.98))

            readmitted = int(rng.random() < p_readmit)

            rows.append({
                "patient_id": patient_id,
                "visit_number": v,
                "days_between_visits": days_between,
                "lab_value_1": round(lab1, 4),
                "lab_value_2": round(lab2, 4),
                "readmitted": readmitted,
            })

            prev_lab1 = lab1
            prev_lab2 = lab2

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate()
    print(f"Total rows      : {len(df)}")
    print(f"Unique patients : {df['patient_id'].nunique()}")
    print(f"Readmit rate    : {df['readmitted'].mean():.3f}")
    print(f"Visit dist      :\n{df.groupby('visit_number').size()}")
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")
