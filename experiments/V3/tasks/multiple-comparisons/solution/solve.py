import pandas as pd
from scipy import stats
from itertools import combinations
import json, os

def main():
    df = pd.read_csv("drug_trial.csv")
    formulations = sorted(df["formulation"].unique())
    pairs = list(combinations(formulations, 2))

    p_values = []
    results = {}
    for a, b in pairs:
        da = df[df["formulation"] == a]["efficacy_score"]
        db = df[df["formulation"] == b]["efficacy_score"]
        t, p = stats.ttest_ind(da, db)
        results[f"{a}_vs_{b}"] = {"t_statistic": float(t), "p_value": float(p)}
        p_values.append((a, b, p))

    # Bonferroni correction
    alpha_corrected = 0.05 / len(pairs)
    significant = []
    details = {}
    for a, b, p in p_values:
        if p < alpha_corrected:
            significant.append([a, b])
            details[f"{a}_vs_{b}"] = results[f"{a}_vs_{b}"]

    output = {
        "significant_pairs": significant,
        "total_comparisons": len(pairs),
        "test_used": f"t-test with Bonferroni correction (alpha={alpha_corrected:.6f})",
        "details": details
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
