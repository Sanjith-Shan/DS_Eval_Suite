import json, sys, os


def main():
    reward_dir = "/logs/verifier"
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward.txt")

    output_path = os.environ.get("OUTPUT_PATH", "/output/analysis.json")

    try:
        # ------------------------------------------------------------------
        # Check 1: output file exists
        # ------------------------------------------------------------------
        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"Output file not found: {output_path}"
            )

        with open(output_path) as f:
            result = json.load(f)

        # ------------------------------------------------------------------
        # Check 2: required fields present
        # ------------------------------------------------------------------
        required = ["coefficient", "r_squared", "p_value", "significant",
                    "n_observations_used"]
        missing = [k for k in required if k not in result]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        coefficient = float(result["coefficient"])
        r_squared = float(result["r_squared"])
        significant = bool(result["significant"])

        # ------------------------------------------------------------------
        # Check 3: coefficient in [0.15, 0.50]  (KEY DISCRIMINATOR)
        # Naive full-data OLS gives ~0.66, which fails this check.
        # After removing high-Cook's-d points the clean coefficient is ~0.30.
        # ------------------------------------------------------------------
        if not (0.15 <= coefficient <= 0.50):
            raise ValueError(
                f"coefficient={coefficient:.4f} is not in [0.15, 0.50]. "
                "Naive full-data OLS (coef≈0.66) fails this check; "
                "influence diagnostics are required."
            )

        # ------------------------------------------------------------------
        # Check 4: r_squared < 0.25
        # ------------------------------------------------------------------
        if r_squared >= 0.25:
            raise ValueError(
                f"r_squared={r_squared:.4f} is not < 0.25. "
                "The clean dataset (without influential points) has R²≈0.09."
            )

        # ------------------------------------------------------------------
        # Check 5: significant == true
        # The relationship is real (coef≈0.23 in clean data, p<0.05).
        # ------------------------------------------------------------------
        if not significant:
            raise ValueError(
                "significant must be true; the advertising–sales relationship "
                "is statistically significant even after cleaning the data."
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
