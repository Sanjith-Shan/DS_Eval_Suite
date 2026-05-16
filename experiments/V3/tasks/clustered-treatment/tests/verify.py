import json, sys, os


def main():
    reward_dir = "/logs/verifier"
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward.txt")

    try:
        output_path = os.environ.get("ANALYSIS_JSON", "/output/analysis.json")
        with open(output_path) as f:
            result = json.load(f)

        # Check 1: required fields
        for field in ("significant", "p_value"):
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        p_value = float(result["p_value"])
        significant = result["significant"]

        # Check 2: KEY DISCRIMINATOR
        # Student-level t-test gives p ~0.000014 → fails (significant and p < 0.05)
        # Section-level t-test gives p ~0.13 → passes (p > 0.05 OR significant==false)
        if not (p_value > 0.05 or significant == False):
            raise ValueError(
                f"p_value={p_value:.6f} and significant={significant}: "
                "analysis reports a significant result with p <= 0.05. "
                "This indicates treatment was compared at the student level "
                "(inflated N) rather than the section level. "
                "Expected p > 0.05 OR significant=false."
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
