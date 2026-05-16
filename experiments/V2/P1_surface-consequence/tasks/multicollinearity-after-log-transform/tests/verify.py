"""Verifier for multicollinearity-after-log-transform.

Imports the agent's /output/regression.py, calls analyze(data_path) -> dict,
and checks that:
  1. X1_pvalue < 0.05
  2. X2_pvalue < 0.05   (hard: plain OLS after log-transform gives ~0.12)
  3. X3_pvalue < 0.05
  4. bp_pvalue > 0.05   (Breusch-Pagan on final model residuals — must be homoscedastic)

No AST/keyword checks. Pure numeric outcome.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
REGRESSION_PATH = Path("/output/regression.py")
DATA_PATH = "/workspace/data.csv"


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> None:
    if not REGRESSION_PATH.exists():
        write_reward(0, "missing /output/regression.py")
        return

    # Import agent module
    spec = importlib.util.spec_from_file_location("regression", REGRESSION_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        write_reward(0, f"failed to import regression.py: {exc}")
        return

    if not hasattr(module, "analyze"):
        write_reward(0, "regression.py is missing the analyze() function")
        return

    try:
        result = module.analyze(DATA_PATH)
    except Exception as exc:
        write_reward(0, f"analyze() raised: {exc}")
        return

    # Validate return type
    if not isinstance(result, dict):
        write_reward(0, f"analyze() must return a dict, got {type(result)}")
        return

    required_keys = ["X1_pvalue", "X2_pvalue", "X3_pvalue", "bp_pvalue"]
    for key in required_keys:
        if key not in result:
            write_reward(0, f"result dict missing key: {key!r}")
            return
        try:
            result[key] = float(result[key])
        except (TypeError, ValueError):
            write_reward(0, f"result[{key!r}] is not numeric: {result[key]!r}")
            return

    x1_p = result["X1_pvalue"]
    x2_p = result["X2_pvalue"]
    x3_p = result["X3_pvalue"]
    bp_p = result["bp_pvalue"]

    print(
        f"X1_pvalue={x1_p:.6f}  X2_pvalue={x2_p:.6f}  "
        f"X3_pvalue={x3_p:.6f}  bp_pvalue={bp_p:.6f}",
        file=sys.stderr,
    )

    if x1_p >= 0.05:
        write_reward(0, f"X1 not significant: p={x1_p:.6f} >= 0.05")
        return
    if x2_p >= 0.05:
        write_reward(0, f"X2 not significant: p={x2_p:.6f} >= 0.05")
        return
    if x3_p >= 0.05:
        write_reward(0, f"X3 not significant: p={x3_p:.6f} >= 0.05")
        return
    if bp_p <= 0.05:
        write_reward(0, f"Residuals still heteroscedastic: BP p={bp_p:.6f} <= 0.05")
        return

    write_reward(
        1,
        f"all checks passed: X1_p={x1_p:.4f}, X2_p={x2_p:.4f}, "
        f"X3_p={x3_p:.4f}, BP_p={bp_p:.4f}",
    )


if __name__ == "__main__":
    main()
