"""Verifier for outlier-removal-kills-minority-class.

Imports the agent's /output/classify.py, calls train_and_evaluate, and checks
that class1_recall >= 0.55.

No AST or keyword checks are performed.  The verifier is purely numeric.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
CLASSIFY_PATH = Path("/output/classify.py")
DATA_PATH = "/workspace/data.csv"

THRESHOLD = 0.55


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value}  reason={reason}", file=sys.stderr)


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Check file exists
    # ------------------------------------------------------------------
    if not CLASSIFY_PATH.exists():
        write_reward(0, "missing /output/classify.py")
        return 0

    # ------------------------------------------------------------------
    # 2. Import the module
    # ------------------------------------------------------------------
    spec = importlib.util.spec_from_file_location("classify", CLASSIFY_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        write_reward(0, f"failed to import classify.py: {exc}")
        return 0

    if not hasattr(module, "train_and_evaluate"):
        write_reward(0, "classify.py is missing train_and_evaluate function")
        return 0

    # ------------------------------------------------------------------
    # 3. Call train_and_evaluate
    # ------------------------------------------------------------------
    try:
        result = module.train_and_evaluate(DATA_PATH)
    except Exception as exc:
        write_reward(0, f"train_and_evaluate raised: {exc}")
        return 0

    # ------------------------------------------------------------------
    # 4. Validate return value
    # ------------------------------------------------------------------
    if not isinstance(result, dict):
        write_reward(0, f"train_and_evaluate must return a dict, got {type(result)}")
        return 0

    if "class1_recall" not in result:
        write_reward(0, "result dict missing key 'class1_recall'")
        return 0

    try:
        recall = float(result["class1_recall"])
    except (TypeError, ValueError):
        write_reward(0, f"class1_recall is not a float: {result['class1_recall']!r}")
        return 0

    if not (0.0 <= recall <= 1.0):
        write_reward(0, f"class1_recall={recall:.4f} is outside [0, 1]")
        return 0

    # ------------------------------------------------------------------
    # 5. Pass/fail decision
    # ------------------------------------------------------------------
    if recall >= THRESHOLD:
        write_reward(1, f"class1_recall={recall:.4f} >= {THRESHOLD}")
        return 1
    else:
        write_reward(0, f"class1_recall={recall:.4f} < {THRESHOLD} (threshold)")
        return 0


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
