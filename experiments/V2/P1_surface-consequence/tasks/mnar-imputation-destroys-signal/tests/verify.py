"""
Verifier for mnar-imputation-destroys-signal.

Imports the agent's /output/regression.py, calls fit('/workspace/train.csv'),
and checks that the returned X3 coefficient and p-value satisfy:
  - X3_pvalue < 0.05   (X3 is a statistically significant predictor)
  - 0.30 <= X3_coef <= 0.60   (coefficient is within the plausible recovery band)

A naive mean-imputation approach yields X3_coef ≈ 0.27 (fails the lower bound).
An MNAR-aware approach (missingness indicator + model-based or median imputation)
yields X3_coef in [0.34, 0.50] and passes both checks.

No keyword or AST checks are performed -- only the numeric output is evaluated.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REWARD_PATH = Path("/logs/verifier/reward.txt")
REGRESSION_PATH = Path("/output/regression.py")
DATA_PATH = "/workspace/train.csv"

# Verifier bands
PVALUE_THRESHOLD = 0.05
COEF_LOW = 0.30
COEF_HIGH = 0.60


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    # Check file exists
    if not REGRESSION_PATH.exists():
        write_reward(0, "missing /output/regression.py")
        return 0

    # Import the module
    spec = importlib.util.spec_from_file_location("regression", REGRESSION_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        write_reward(0, f"failed to import regression.py: {exc}")
        return 0

    # Check fit function exists
    if not hasattr(module, "fit"):
        write_reward(0, "regression.py is missing the fit() function")
        return 0

    # Call fit()
    try:
        result = module.fit(DATA_PATH)
    except Exception as exc:
        write_reward(0, f"fit() raised an exception: {exc}")
        return 0

    # Validate return type
    if not isinstance(result, dict):
        write_reward(0, f"fit() must return a dict, got {type(result).__name__}")
        return 0

    # Check required keys
    for key in ("X3_coef", "X3_pvalue"):
        if key not in result:
            write_reward(0, f"missing key '{key}' in fit() return value")
            return 0

    # Extract values
    try:
        x3_coef = float(result["X3_coef"])
        x3_pvalue = float(result["X3_pvalue"])
    except (TypeError, ValueError) as exc:
        write_reward(0, f"non-numeric value in fit() return: {exc}")
        return 0

    print(f"X3_coef={x3_coef:.6f}  X3_pvalue={x3_pvalue:.6f}", file=sys.stderr)

    # Check p-value
    if x3_pvalue >= PVALUE_THRESHOLD:
        write_reward(0, f"X3_pvalue={x3_pvalue:.4f} >= {PVALUE_THRESHOLD} (X3 not significant)")
        return 0

    # Check coefficient band
    if not (COEF_LOW <= x3_coef <= COEF_HIGH):
        write_reward(
            0,
            f"X3_coef={x3_coef:.4f} outside [{COEF_LOW}, {COEF_HIGH}] "
            f"(naive imputation attenuates the coefficient below {COEF_LOW})",
        )
        return 0

    write_reward(
        1,
        f"all checks passed: X3_coef={x3_coef:.4f} in [{COEF_LOW}, {COEF_HIGH}], "
        f"X3_pvalue={x3_pvalue:.6f} < {PVALUE_THRESHOLD}",
    )
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
