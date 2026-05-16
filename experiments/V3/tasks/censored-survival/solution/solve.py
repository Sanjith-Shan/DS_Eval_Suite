import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import json, os

def main():
    df = pd.read_csv("clinical_trial.csv")
    df["event"] = (df["study_status"] == "A").astype(int)

    a = df[df["drug"] == "drug_A"]
    b = df[df["drug"] == "drug_B"]

    kmf = KaplanMeierFitter()
    kmf.fit(a["recovery_days"], event_observed=a["event"])
    median_a = kmf.median_survival_time_

    kmf.fit(b["recovery_days"], event_observed=b["event"])
    median_b = kmf.median_survival_time_

    lr = logrank_test(a["recovery_days"], b["recovery_days"],
                      event_observed_A=a["event"], event_observed_B=b["event"])

    result = {
        "median_recovery_drug_a": float(median_a),
        "median_recovery_drug_b": float(median_b),
        "significant_difference": bool(lr.p_value < 0.05),
        "p_value": float(lr.p_value),
        "test_used": "log-rank test"
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
