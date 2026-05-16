import json, sys, os


def main():
    reward_dir = "/logs/verifier"
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward.txt")

    try:
        # Check 1: load output file
        output_path = "/output/analysis.json"
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file not found: {output_path}")

        with open(output_path) as f:
            data = json.load(f)

        # Check 2: required fields present
        required = ["significant_pairs", "total_comparisons"]
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Check 3: normalize pairs to alphabetical order
        raw_pairs = data["significant_pairs"]
        if not isinstance(raw_pairs, list):
            raise ValueError("significant_pairs must be a list")

        normalized = []
        for pair in raw_pairs:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError(f"Each pair must be a list of 2 elements, got: {pair}")
            a, b = str(pair[0]), str(pair[1])
            normalized.append(tuple(sorted([a, b])))

        # Check 4: at least one pair involving "D" is present (real signal detected)
        d_pairs = [p for p in normalized if "D" in p]
        if len(d_pairs) == 0:
            raise ValueError(
                "No pair involving formulation D found in significant_pairs. "
                "The real signal (formulation D) was not detected."
            )

        # Check 5: len(significant_pairs) <= 2  -- KEY DISCRIMINATOR
        # Uncorrected at alpha=0.05 yields many pairs; correction trims to <= 2
        if len(raw_pairs) > 2:
            raise ValueError(
                f"Too many significant pairs reported: {len(raw_pairs)}. "
                f"Expected at most 2 after appropriate correction. "
                f"Apply a multiple comparison correction (e.g. Bonferroni) "
                f"before reporting."
            )

        # Check 6: total_comparisons >= 20 (sanity: 8C2 = 28)
        total = data["total_comparisons"]
        if not isinstance(total, (int, float)) or total < 20:
            raise ValueError(
                f"total_comparisons must be >= 20 (expected 28), got {total}"
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
