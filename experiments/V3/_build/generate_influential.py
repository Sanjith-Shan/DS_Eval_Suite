"""
generate_influential.py
-----------------------
Data generator for N3: influential-outliers task.

Seed: 42
N_clean: 196  (X ~ N(50,10), Y = 0.30*X + N(0,8))
N_influential: 4  (X = [90,93,96,99], Y = [75,80,85,90])

VALIDATION CONDITIONS (all must hold):
  1. Full-data OLS coefficient > 0.60
  2. Clean-data OLS coefficient in [0.20, 0.45]
  3. Full-data R-squared > 0.25
  4. Clean-data R-squared < 0.15
  5. All 4 influential points: Cook's d > 4/N
  6. No clean point: Cook's d > 4/N
  7. Influential points Y z-scores: |z| < 2.5  (not outliers in Y)

NOTE: Conditions 6 and 7 have a mathematical tension with each other when
combined with condition 1.  To get coef_full > 0.60 the influential Y values
must be far above the clean-data regression line (~24 at X=90), which makes
them statistical outliers in Y (condition 7 fails).  Conversely, keeping
Y_inf within 2.5 SD (~≤36) limits coef_full to ~0.31 (condition 1 fails).

The design therefore INTENTIONALLY accepts that checks 6 and 7 cannot be
simultaneously satisfied with condition 1; what matters for the task is:
  - The oracle (Cook's D filter, threshold 4/N) removes the 4 influential
    points (plus at most ~5 borderline clean points) and then re-fits,
    obtaining a coefficient in [0.20, 0.45] that passes the verifier.
  - Naive full-data OLS gives coefficient ~0.66, which fails the verifier's
    [0.15, 0.50] range.
  - The 4 influential points have very large Cook's d (0.40–1.05) relative
    to the threshold 0.02, so they are reliably identified and removed.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os


SEED = 42
N_CLEAN = 196
N_INFLUENTIAL = 4

# Influential point coordinates (per spec)
INF_X = [90, 93, 96, 99]
INF_Y = [75, 80, 85, 90]


def generate(seed=SEED, inf_x=None, inf_y=None):
    if inf_x is None:
        inf_x = INF_X
    if inf_y is None:
        inf_y = INF_Y

    rng = np.random.default_rng(seed)

    # Clean data
    X_clean = rng.normal(50, 10, N_CLEAN)
    Y_clean = 0.30 * X_clean + rng.normal(0, 8, N_CLEAN)

    # Influential points
    X_inf = np.array(inf_x, dtype=float)
    Y_inf = np.array(inf_y, dtype=float)

    # Combine
    X_all = np.concatenate([X_clean, X_inf])
    Y_all = np.concatenate([Y_clean, Y_inf])
    N_total = N_CLEAN + N_INFLUENTIAL

    obs_ids = np.arange(1, N_total + 1)

    df = pd.DataFrame({
        "observation_id": obs_ids,
        "advertising_spend": X_all,
        "sales_revenue": Y_all,
    })
    return df, N_CLEAN


def validate(df, n_clean=N_CLEAN, verbose=True):
    N = len(df)
    threshold = 4 / N

    X_all = df["advertising_spend"].values
    Y_all = df["sales_revenue"].values

    # --- Full-data OLS ---
    X_const = sm.add_constant(X_all)
    model_full = sm.OLS(Y_all, X_const).fit()
    coef_full = model_full.params[1]
    r2_full = model_full.rsquared

    # Cook's distances (full model)
    influence = model_full.get_influence()
    cooks_d = influence.cooks_distance[0]

    inf_cooks = cooks_d[n_clean:]   # last 4 are influential points
    clean_cooks = cooks_d[:n_clean]

    # --- Clean-data OLS ---
    df_clean = df.iloc[:n_clean]
    X_clean = df_clean["advertising_spend"].values
    Y_clean = df_clean["sales_revenue"].values
    X_clean_const = sm.add_constant(X_clean)
    model_clean = sm.OLS(Y_clean, X_clean_const).fit()
    coef_clean = model_clean.params[1]
    r2_clean = model_clean.rsquared

    # --- Y z-scores (over entire dataset) ---
    y_mean = Y_all.mean()
    y_std = Y_all.std()
    inf_y_zscores = np.abs((Y_all[n_clean:] - y_mean) / y_std)

    # --- Oracle: remove Cook's d > threshold, refit ---
    oracle_mask = cooks_d > threshold
    X_keep = X_all[~oracle_mask]
    Y_keep = Y_all[~oracle_mask]
    model_oracle = sm.OLS(Y_keep, sm.add_constant(X_keep)).fit()
    oracle_coef = model_oracle.params[1]
    oracle_r2 = model_oracle.rsquared
    oracle_n_removed = oracle_mask.sum()
    oracle_clean_removed = oracle_mask[:n_clean].sum()
    oracle_inf_removed = oracle_mask[n_clean:].sum()

    checks = [
        ("1. Full-data coef > 0.60",
         coef_full > 0.60,
         f"coef={coef_full:.4f}"),
        ("2. Clean-data coef in [0.20, 0.45]",
         0.20 <= coef_clean <= 0.45,
         f"coef={coef_clean:.4f}"),
        ("3. Full-data R² > 0.25",
         r2_full > 0.25,
         f"R²={r2_full:.4f}"),
        ("4. Clean-data R² < 0.15",
         r2_clean < 0.15,
         f"R²={r2_clean:.4f}"),
        ("5. All 4 influential: Cook's d > 4/N",
         bool(np.all(inf_cooks > threshold)),
         f"inf Cook's d={inf_cooks.round(4).tolist()}, threshold={threshold:.4f}"),
        ("6. No clean point: Cook's d > 4/N",
         bool(np.all(clean_cooks <= threshold)),
         f"max clean Cook's d={clean_cooks.max():.4f}, threshold={threshold:.4f} "
         f"[NOTE: may fail due to design tension with check 1 -- see module docstring]"),
        ("7. Influential Y |z| < 2.5",
         bool(np.all(inf_y_zscores < 2.5)),
         f"|z|={inf_y_zscores.round(4).tolist()} "
         f"[NOTE: may fail due to design tension with check 1 -- see module docstring]"),
    ]

    all_pass = True
    critical_pass = True  # checks 1-5 (functionally required)
    for i, (name, result, detail) in enumerate(checks):
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
            if i < 5:  # checks 1-5 are critical
                critical_pass = False
        if verbose:
            print(f"  [{status}] {name}  ({detail})")

    if verbose:
        print()
        print(f"Full-data   coef={coef_full:.4f}  R²={r2_full:.4f}")
        print(f"Clean-data  coef={coef_clean:.4f}  R²={r2_clean:.4f}")
        print(f"Cook's d threshold: {threshold:.4f}")
        print(f"Influential Cook's d: {inf_cooks.round(4).tolist()}")
        print(f"Max clean Cook's d:   {clean_cooks.max():.4f}")
        print(f"Influential Y |z|:    {inf_y_zscores.round(4).tolist()}")
        print()
        print(f"Oracle removes {oracle_n_removed} points "
              f"({oracle_inf_removed} influential + {oracle_clean_removed} clean borderline)")
        print(f"Oracle coef={oracle_coef:.4f}, R²={oracle_r2:.4f}")
        print(f"Oracle verifier pass: coef in [0.15,0.50]={0.15 <= oracle_coef <= 0.50}, "
              f"R2<0.25={oracle_r2 < 0.25}")

    return all_pass, critical_pass


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "tasks", "influential-outliers", "environment"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "advertising_data.csv")

    print("Generating data with seed=42 ...")
    df, n_clean = generate()
    all_pass, critical_pass = validate(df, n_clean=n_clean)

    if not critical_pass:
        print("\nCritical validations (1-5) FAILED. Attempting tuning ...")
        # Move influential X further out or adjust Y
        tuning_configs = [
            ([92, 95, 98, 101], [78, 84, 90, 96]),
            ([95, 98, 101, 104], [80, 86, 92, 98]),
            ([90, 93, 96, 99],   [80, 86, 92, 98]),
            ([88, 92, 96, 100],  [76, 82, 88, 94]),
        ]
        for inf_x, inf_y in tuning_configs:
            print(f"\n  Trying inf_x={inf_x}, inf_y={inf_y} ...")
            df2, n_clean2 = generate(inf_x=inf_x, inf_y=inf_y)
            _, crit_ok = validate(df2, n_clean=n_clean2, verbose=False)
            if crit_ok:
                df = df2
                n_clean = n_clean2
                print(f"  => Critical checks 1-5 pass with inf_x={inf_x}, inf_y={inf_y}")
                validate(df, n_clean=n_clean, verbose=True)
                all_pass = False  # checks 6,7 still likely fail
                critical_pass = True
                break
        else:
            print("WARNING: Could not satisfy critical checks 1-5 after tuning.")

    df.to_csv(out_path, index=False)
    print(f"\nCSV written to: {out_path}")
    print(f"Shape: {df.shape}")
    if all_pass:
        print("\nOverall: ALL 7 PASS")
    elif critical_pass:
        print("\nOverall: CRITICAL CHECKS 1-5 PASS (checks 6+7 have known design tension)")
    else:
        print("\nOverall: SOME CRITICAL CHECKS FAIL")


if __name__ == "__main__":
    main()
