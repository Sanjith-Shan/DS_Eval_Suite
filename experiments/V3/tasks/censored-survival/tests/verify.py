import json, sys, os

def main():
    reward_dir = "/logs/verifier"
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward.txt")

    try:
        output_path = "/output/analysis.json"
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file not found: {output_path}")

        with open(output_path) as f:
            result = json.load(f)

        # Check 1: Required fields present
        required = ["significant_difference", "p_value", "median_recovery_drug_a"]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Check 2: p_value > 0.05 OR significant_difference == False
        # KEY DISCRIMINATOR: naive t-test gives p ~0.01 -> fails; log-rank gives p ~0.79 -> passes
        p_value = float(result["p_value"])
        significant = result["significant_difference"]

        if not (p_value > 0.05 or significant == False):
            raise ValueError(
                f"Key discriminator failed: p_value={p_value:.4f} must be > 0.05 OR "
                f"significant_difference must be False (got significant_difference={significant}). "
                f"Naive t-test gives p ~0.01 which fails this check. "
                f"Use survival analysis (e.g., log-rank test) to account for censored observations."
            )

        # Check 3: median_recovery_drug_a > 50
        # KM median ~54; naive median ~48 -> fails (recoveries only)
        median_a = float(result["median_recovery_drug_a"])
        if median_a <= 50:
            raise ValueError(
                f"median_recovery_drug_a={median_a:.2f} must be > 50. "
                f"Naive median using only recovered patients underestimates true recovery time."
            )

        with open(reward_path, "w") as f:
            f.write("1")
        print("reward=1 reason=all checks passed")
        return True

    except Exception as e:
        with open(reward_path, "w") as f:
            f.write("0")
        print(f"reward=0 reason={e}")
        return False

if __name__ == "__main__":
    main()
    sys.exit(0)  # always exit 0; reward file carries the signal
