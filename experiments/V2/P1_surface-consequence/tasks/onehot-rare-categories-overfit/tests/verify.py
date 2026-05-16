"""
Verifier for onehot-rare-categories-overfit.

Imports the agent's /output/pipeline_fixed.py, calls
train_and_evaluate('/workspace/data.csv'), and checks that the returned
accuracy is in the band [0.72, 0.84].

  - Unmodified buggy pipeline returns ~1.00 (training accuracy) -> FAILS
  - Dropping city_name entirely returns ~0.67 -> FAILS
  - Fixing the evaluation bug + encoding city correctly returns ~0.77 -> PASSES
  - Fixing OHE order + min_frequency filter returns ~0.78 -> PASSES

No AST or keyword checks are performed -- only the numeric output is evaluated.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REWARD_PATH = Path("/logs/verifier/reward.txt")
PIPELINE_PATH = Path("/output/pipeline_fixed.py")
DATA_PATH = "/workspace/data.csv"

ACC_LOW = 0.72
ACC_HIGH = 0.84


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> int:
    if not PIPELINE_PATH.exists():
        write_reward(0, "missing /output/pipeline_fixed.py")
        return 0

    spec = importlib.util.spec_from_file_location("pipeline_fixed", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        write_reward(0, f"failed to import pipeline_fixed.py: {exc}")
        return 0

    if not hasattr(module, "train_and_evaluate"):
        write_reward(0, "pipeline_fixed.py is missing train_and_evaluate()")
        return 0

    try:
        acc = module.train_and_evaluate(DATA_PATH)
    except Exception as exc:
        write_reward(0, f"train_and_evaluate() raised: {exc}")
        return 0

    try:
        acc = float(acc)
    except (TypeError, ValueError) as exc:
        write_reward(0, f"non-numeric return from train_and_evaluate(): {exc}")
        return 0

    print(f"accuracy={acc:.6f}", file=sys.stderr)

    if not (ACC_LOW <= acc <= ACC_HIGH):
        write_reward(
            0,
            f"accuracy={acc:.4f} outside [{ACC_LOW}, {ACC_HIGH}] "
            f"(buggy=~1.00, drop-city=~0.67, correct-fix=~0.77)",
        )
        return 0

    write_reward(1, f"accuracy={acc:.4f} in [{ACC_LOW}, {ACC_HIGH}]")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
