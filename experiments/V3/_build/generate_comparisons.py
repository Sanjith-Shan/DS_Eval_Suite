"""
Data generator for N2: multiple-comparisons task.

8 drug formulations A-H, 50 patients each, 400 total.
Group D has a genuinely higher mean (108 vs 100 for all others), SD=15.

Validation conditions:
  1. Uncorrected pairwise t-tests at alpha=0.05 yield >= 3 significant pairs
  2. At least one D-vs-other pair has p < 0.001
  3. After Bonferroni (alpha=0.05/28), significant pairs <= 2
  4. At least one D-vs-other pair survives Bonferroni

Tuning: if < 3 uncorrected, try seeds 43, 44, ...
         if real pair doesn't survive Bonferroni, increase D mean to 110.
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import os


def generate_data(seed, d_mean=108):
    rng = np.random.default_rng(seed)

    groups = list("ABCDEFGH")
    means = {g: 100.0 for g in groups}
    means["D"] = float(d_mean)
    sd = 15.0
    n_per_group = 50

    records = []
    patient_id = 1
    for g in groups:
        scores = rng.normal(loc=means[g], scale=sd, size=n_per_group)
        for score in scores:
            records.append({
                "patient_id": patient_id,
                "formulation": g,
                "efficacy_score": round(float(score), 4),
            })
            patient_id += 1

    df = pd.DataFrame(records)
    return df


def validate(df):
    formulations = sorted(df["formulation"].unique())
    pairs = list(combinations(formulations, 2))
    n_pairs = len(pairs)
    alpha = 0.05
    alpha_bonf = alpha / n_pairs

    p_values = []
    for a, b in pairs:
        da = df[df["formulation"] == a]["efficacy_score"]
        db = df[df["formulation"] == b]["efficacy_score"]
        _, p = stats.ttest_ind(da, db)
        p_values.append((a, b, p))

    # Check 1: uncorrected significant pairs >= 3
    uncorr_sig = [(a, b, p) for a, b, p in p_values if p < alpha]
    check1 = len(uncorr_sig) >= 3

    # Check 2: at least one D-vs-other pair has p < 0.001
    d_pairs_uncorr = [(a, b, p) for a, b, p in p_values
                      if ("D" in (a, b)) and p < 0.001]
    check2 = len(d_pairs_uncorr) >= 1

    # Check 3: after Bonferroni, significant pairs <= 2
    bonf_sig = [(a, b, p) for a, b, p in p_values if p < alpha_bonf]
    check3 = len(bonf_sig) <= 2

    # Check 4: at least one D-vs-other pair survives Bonferroni
    d_pairs_bonf = [(a, b, p) for a, b, p in bonf_sig if "D" in (a, b)]
    check4 = len(d_pairs_bonf) >= 1

    return {
        "check1_pass": check1,
        "check2_pass": check2,
        "check3_pass": check3,
        "check4_pass": check4,
        "n_uncorr_sig": len(uncorr_sig),
        "uncorr_sig_pairs": [(a, b, round(p, 6)) for a, b, p in uncorr_sig],
        "n_bonf_sig": len(bonf_sig),
        "bonf_sig_pairs": [(a, b, round(p, 6)) for a, b, p in bonf_sig],
        "alpha_bonf": alpha_bonf,
    }


def main():
    # Try seed=42 first; then 43, 44, ... if validation fails
    # Try D mean=108 first; if real pair doesn't survive Bonferroni, use 110
    best_seed = None
    best_result = None
    best_df = None

    for d_mean in [108, 110]:
        for seed in range(42, 100):
            df = generate_data(seed=seed, d_mean=d_mean)
            result = validate(df)

            all_pass = all([
                result["check1_pass"],
                result["check2_pass"],
                result["check3_pass"],
                result["check4_pass"],
            ])

            if all_pass:
                best_seed = seed
                best_result = result
                best_df = df
                best_d_mean = d_mean
                break

        if best_seed is not None:
            break

    if best_seed is None:
        print("ERROR: Could not find valid seed in range 42-99!")
        return

    print(f"\n=== Data Generator: N2 multiple-comparisons ===")
    print(f"Seed used      : {best_seed}")
    print(f"D mean used    : {best_d_mean}")
    print(f"Bonferroni alpha: {best_result['alpha_bonf']:.6f}  (0.05 / {28})")
    print()
    print(f"Uncorrected significant pairs ({best_result['n_uncorr_sig']} total):")
    for a, b, p in best_result["uncorr_sig_pairs"]:
        d_tag = " <-- D pair" if "D" in (a, b) else "         <-- FALSE POSITIVE"
        print(f"  {a} vs {b}: p={p:.6f}{d_tag}")
    print()
    print(f"Bonferroni significant pairs ({best_result['n_bonf_sig']} total):")
    for a, b, p in best_result["bonf_sig_pairs"]:
        print(f"  {a} vs {b}: p={p:.6f}")
    print()

    checks = [
        ("1", "Uncorrected >= 3 sig pairs", best_result["check1_pass"]),
        ("2", "D-vs-other pair p < 0.001", best_result["check2_pass"]),
        ("3", "After Bonferroni <= 2 sig pairs", best_result["check3_pass"]),
        ("4", "D-vs-other survives Bonferroni", best_result["check4_pass"]),
    ]
    for num, desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  Check {num} [{status}]: {desc}")

    all_pass = all(c[2] for c in checks)
    print()
    print(f"Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    if all_pass:
        # Write CSV to the task environment directory
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tasks", "multiple-comparisons", "environment"
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "drug_trial.csv")
        best_df.to_csv(out_path, index=False)
        print(f"\nCSV written to: {out_path}")


if __name__ == "__main__":
    main()
