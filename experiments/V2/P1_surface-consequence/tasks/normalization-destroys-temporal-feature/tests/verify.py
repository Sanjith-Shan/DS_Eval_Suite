"""
Verifier for normalization-destroys-temporal-feature.

Imports the agent's /output/pipeline_fixed.py, calls
    train_and_evaluate(data_path: str) -> float
on /workspace/data.csv, and checks that the returned value is in [0.75, 0.86].

A pipeline that fixes the scaler bug but leaves the temporal feature unaddressed
yields ~0.746, which is BELOW the lower bound and fails.
A pipeline that also clips, quantile-transforms, or bins the temporal feature
yields ~0.79-0.80, which passes.

No keyword or AST checks are performed — only the numeric output is evaluated.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REWARD_PATH    = Path("/logs/verifier/reward.txt")
PIPELINE_PATH  = Path("/output/pipeline_fixed.py")
DATA_PATH      = "/workspace/data.csv"

ACC_LOW  = 0.75
ACC_HIGH = 0.86


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value}  reason={reason}", file=sys.stderr)


def main() -> int:
    # ── Check file exists ──────────────────────────────────────────────────────
    if not PIPELINE_PATH.exists():
        write_reward(0, "missing /output/pipeline_fixed.py")
        return 0

    # ── Import module ──────────────────────────────────────────────────────────
    spec = importlib.util.spec_from_file_location("pipeline_fixed", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        write_reward(0, f"failed to import pipeline_fixed.py: {exc}")
        return 0

    # ── Check function exists ──────────────────────────────────────────────────
    if not hasattr(module, "train_and_evaluate"):
        write_reward(0, "pipeline_fixed.py is missing train_and_evaluate()")
        return 0

    # ── Call function ──────────────────────────────────────────────────────────
    try:
        result = module.train_and_evaluate(DATA_PATH)
    except Exception as exc:
        write_reward(0, f"train_and_evaluate() raised: {exc}")
        return 0

    # ── Validate return value ──────────────────────────────────────────────────
    try:
        acc = float(result)
    except (TypeError, ValueError) as exc:
        write_reward(0, f"train_and_evaluate() returned non-numeric value: {exc}")
        return 0

    print(f"test_accuracy={acc:.6f}", file=sys.stderr)

    if not (ACC_LOW <= acc <= ACC_HIGH):
        write_reward(
            0,
            f"accuracy={acc:.4f} outside [{ACC_LOW}, {ACC_HIGH}]. "
            f"Fixing the scaler alone yields ~0.746 (temporal OOD extrapolation). "
            f"Also handle the out-of-distribution temporal feature.",
        )
        return 0

    write_reward(1, f"accuracy={acc:.4f} in [{ACC_LOW}, {ACC_HIGH}]")
    return 1


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
