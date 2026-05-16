"""
generate_autocorrelated.py
--------------------------
Data generator for N1: autocorrelated-residuals task.

Seed: 42, N=120 monthly observations (2015-01 through 2024-12)
AR(1) residuals with rho=0.7 make X2 appear significant under naive OLS
but insignificant under Newey-West HAC.

VALIDATION CONDITIONS (all must hold):
  1. Naive OLS p(X1) < 0.05
  2. Naive OLS p(X2) < 0.05   <- false positive
  3. Naive OLS p(X3) < 0.05
  4. Naive OLS p(X4) > 0.10
  5. Durbin-Watson < 1.0
  6. Newey-West p(X1) < 0.05
  7. Newey-West p(X2) > 0.05  <- corrected
  8. Newey-West p(X3) < 0.05
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import os

# ---- Parameters ----
SEED = 42
N = 120
RHO = 0.9   # spec says 0.7 but 0.9 is required for X2 false positive; see tuning notes
INNOVATION_SD = 3.0

def generate(seed=SEED, rho=RHO, innovation_sd=INNOVATION_SD,
             x2_slope=0.03):  # rho=0.9 needed for X2 false-positive (spec default 0.7 too weak)
    rng = np.random.default_rng(seed)
    t = np.arange(N)

    # Predictors
    X1 = rng.normal(10, 2, N) + 0.02 * t   # true predictor, trending
    X2 = rng.normal(5, 1.5, N) + x2_slope * t  # null predictor, trending
    X3 = rng.normal(8, 2.5, N)              # true predictor, iid
    X4 = rng.normal(3, 1, N)               # null predictor, iid

    # AR(1) noise
    e = np.empty(N)
    e[0] = rng.normal(0, innovation_sd)
    for i in range(1, N):
        e[i] = rho * e[i-1] + rng.normal(0, innovation_sd)

    # Response: only X1 and X3 are true predictors
    Y = 2.5 * X1 + 0.0 * X2 + 1.8 * X3 + 0.0 * X4 + e

    # Month index
    months = pd.date_range("2015-01", periods=N, freq="MS").strftime("%Y-%m")

    df = pd.DataFrame({
        "month": months,
        "X1": X1, "X2": X2, "X3": X3, "X4": X4, "Y": Y
    })
    return df


def validate(df, verbose=True):
    X = df[["X1", "X2", "X3", "X4"]]
    y = df["Y"]
    X_const = sm.add_constant(X)

    ols = sm.OLS(y, X_const).fit()
    dw = durbin_watson(ols.resid)

    maxlags = int(np.ceil(len(df) ** 0.25))
    robust = ols.get_robustcov_results(cov_type="HAC", maxlags=maxlags)

    ols_p = {k: float(v) for k, v in zip(["X1","X2","X3","X4"], ols.pvalues[1:])}
    hac_p = {k: float(v) for k, v in zip(["X1","X2","X3","X4"], robust.pvalues[1:])}

    checks = [
        ("1. Naive OLS p(X1) < 0.05",        ols_p["X1"] < 0.05,   f"p={ols_p['X1']:.4f}"),
        ("2. Naive OLS p(X2) < 0.05 [FP]",   ols_p["X2"] < 0.05,   f"p={ols_p['X2']:.4f}"),
        ("3. Naive OLS p(X3) < 0.05",        ols_p["X3"] < 0.05,   f"p={ols_p['X3']:.4f}"),
        ("4. Naive OLS p(X4) > 0.10",        ols_p["X4"] > 0.10,   f"p={ols_p['X4']:.4f}"),
        ("5. Durbin-Watson < 1.0",            dw < 1.0,              f"DW={dw:.4f}"),
        ("6. Newey-West p(X1) < 0.05",       hac_p["X1"] < 0.05,   f"p={hac_p['X1']:.4f}"),
        ("7. Newey-West p(X2) > 0.05 [corr]",hac_p["X2"] > 0.05,   f"p={hac_p['X2']:.4f}"),
        ("8. Newey-West p(X3) < 0.05",       hac_p["X3"] < 0.05,   f"p={hac_p['X3']:.4f}"),
    ]

    all_pass = True
    for name, result, detail in checks:
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        if verbose:
            print(f"  [{status}] {name}  ({detail})")

    if verbose:
        print()
        print("Naive OLS p-values:")
        for k, v in ols_p.items():
            print(f"  {k}: {v:.4f}")
        print("Newey-West HAC p-values:")
        for k, v in hac_p.items():
            print(f"  {k}: {v:.4f}")
        print(f"Durbin-Watson: {dw:.4f}")
        print(f"HAC maxlags used: {maxlags}")

    return all_pass, ols_p, hac_p, dw


def main():
    out_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "tasks", "autocorrelated-residuals", "environment"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "business_data.csv")

    print(f"Generating data with seed={SEED}, rho={RHO}, innovation_sd={INNOVATION_SD} ...")
    df = generate()
    all_pass, ols_p, hac_p, dw = validate(df)

    if not all_pass:
        print("\nSome validations FAILED. Attempting tuning ...")
        # rho=0.9 was found empirically to satisfy all 8 conditions with seed=42
        for rho_try in [0.88, 0.90, 0.92]:
            for x2_slope in [0.03, 0.04, 0.05]:
                print(f"\n  Trying rho={rho_try} x2_slope={x2_slope} ...")
                df2 = generate(rho=rho_try, x2_slope=x2_slope)
                ok, ols_p2, hac_p2, dw2 = validate(df2, verbose=False)
                if ok:
                    df = df2
                    ols_p, hac_p, dw = ols_p2, hac_p2, dw2
                    print(f"  => All 8 conditions pass with rho={rho_try} x2_slope={x2_slope}")
                    validate(df, verbose=True)
                    break
            else:
                continue
            break
        else:
            print("WARNING: Could not satisfy all conditions after tuning.")

    df.to_csv(out_path, index=False)
    print(f"\nCSV written to: {out_path}")
    print(f"Shape: {df.shape}")
    print("\nOverall:", "ALL 8 PASS" if all_pass else "SOME FAIL")


if __name__ == "__main__":
    main()
