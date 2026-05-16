import json, sys, os

def main():
    reward_dir = "/logs/verifier"
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward.txt")

    try:
        # 1. Load output
        output_path = "/output/analysis.json"
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"{output_path} not found")

        with open(output_path) as f:
            result = json.load(f)

        # 2. Required fields
        required = ["significant_predictors", "coefficients", "p_values", "r_squared"]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        sig = result["significant_predictors"]
        r2 = result["r_squared"]

        # 3. X1 in significant_predictors (true positive)
        if "X1" not in sig:
            raise ValueError("X1 should be in significant_predictors (true positive)")

        # 4. X3 in significant_predictors (true positive)
        if "X3" not in sig:
            raise ValueError("X3 should be in significant_predictors (true positive)")

        # 5. X2 NOT in significant_predictors (KEY DISCRIMINATOR)
        if "X2" in sig:
            raise ValueError(
                "X2 should NOT be in significant_predictors -- "
                "it is a null predictor that only appears significant under naive OLS "
                "(false positive due to autocorrelated residuals)"
            )

        # 6. X4 NOT in significant_predictors (true negative)
        if "X4" in sig:
            raise ValueError("X4 should NOT be in significant_predictors (null predictor)")

        # 7. r_squared is a valid proportion
        if not (0 < r2 < 1):
            raise ValueError(f"r_squared={r2} must be strictly between 0 and 1")

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
