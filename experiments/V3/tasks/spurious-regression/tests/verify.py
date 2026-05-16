import json, sys, os

def main():
    reward_dir = "/logs/verifier"
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward.txt")

    try:
        # ── 1. Load output ──────────────────────────────────────────────
        output_path = "/output/analysis.json"
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file not found: {output_path}")

        with open(output_path) as f:
            result = json.load(f)

        # ── 2. Required fields ─────────────────────────────────────────
        for field in ("r_squared", "p_value"):
            if field not in result:
                raise KeyError(f"Missing required field: {field}")

        r_squared = float(result["r_squared"])
        p_value   = float(result["p_value"])
        significant = result.get("significant", None)

        # ── 3. KEY DISCRIMINATOR: r_squared < 0.15 ────────────────────
        # Naive OLS on levels gives R² ~0.80 → FAILS this check.
        # Regression on differenced series gives R² ~0.01 → PASSES.
        if r_squared >= 0.15:
            raise ValueError(
                f"r_squared={r_squared:.4f} >= 0.15 "
                f"(expected differenced-series R² < 0.15; "
                f"naive OLS on levels gives ~0.80)"
            )

        # ── 4. p_value > 0.05 OR significant == False ─────────────────
        if significant is None:
            # fall back to p_value alone
            if p_value <= 0.05:
                raise ValueError(
                    f"p_value={p_value:.4f} <= 0.05; "
                    f"the differenced regression should be non-significant"
                )
        else:
            sig = bool(significant)
            if p_value <= 0.05 and sig:
                raise ValueError(
                    f"p_value={p_value:.4f} and significant={sig}; "
                    f"both cannot indicate significance after correct analysis"
                )

        # ── All checks passed ──────────────────────────────────────────
        with open(reward_path, "w") as f:
            f.write("1")
        print(f"reward=1 reason=all checks passed  r_squared={r_squared:.4f}  p_value={p_value:.4f}")
        return True

    except Exception as e:
        with open(reward_path, "w") as f:
            f.write("0")
        print(f"reward=0 reason={e}")
        return False


if __name__ == "__main__":
    main()
    sys.exit(0)  # always exit 0; reward file carries the signal
