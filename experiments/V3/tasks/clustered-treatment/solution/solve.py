import pandas as pd
from scipy import stats
import json, os

def main():
    df = pd.read_csv("education_study.csv")

    # Aggregate to section means
    section_means = df.groupby(["section_id", "treatment_group"])["test_score"].mean().reset_index()
    new = section_means[section_means["treatment_group"] == "new_method"]["test_score"]
    std = section_means[section_means["treatment_group"] == "standard"]["test_score"]

    t, p = stats.ttest_ind(new, std)

    result = {
        "significant": bool(p < 0.05),
        "p_value": float(p),
        "effect_size": float(new.mean() - std.mean()),
        "test_used": "t-test on section means",
        "mean_new_method": float(new.mean()),
        "mean_standard": float(std.mean())
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
