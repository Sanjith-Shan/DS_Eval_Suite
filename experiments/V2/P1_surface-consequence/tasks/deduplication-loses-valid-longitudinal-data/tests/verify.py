"""
verify.py — Verifier for deduplication-loses-valid-longitudinal-data.

Imports the agent's /output/model.py, calls train_and_evaluate(data_path),
and checks that the returned dict contains "auc" >= 0.72.

No AST or keyword checks — the verifier is purely numeric.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REWARD_PATH = Path("/logs/verifier/reward.txt")
MODEL_PATH = Path("/output/model.py")
DATA_PATH = "/workspace/data.csv"


def write_reward(value: int, reason: str = "") -> None:
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(str(value))
    print(f"reward={value} reason={reason}", file=sys.stderr)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Check the output file exists                                     #
    # ------------------------------------------------------------------ #
    if not MODEL_PATH.exists():
        write_reward(0, "missing /output/model.py")
        return

    # ------------------------------------------------------------------ #
    # 2. Import the agent module                                          #
    # ------------------------------------------------------------------ #
    spec = importlib.util.spec_from_file_location("agent_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        write_reward(0, f"failed to import /output/model.py: {exc}")
        return

    # ------------------------------------------------------------------ #
    # 3. Check the required function exists                               #
    # ------------------------------------------------------------------ #
    if not hasattr(module, "train_and_evaluate"):
        write_reward(0, "/output/model.py is missing train_and_evaluate")
        return

    # ------------------------------------------------------------------ #
    # 4. Call the function                                                #
    # ------------------------------------------------------------------ #
    try:
        result = module.train_and_evaluate(DATA_PATH)
    except Exception as exc:
        write_reward(0, f"train_and_evaluate raised: {exc}")
        return

    # ------------------------------------------------------------------ #
    # 5. Validate return type                                             #
    # ------------------------------------------------------------------ #
    if not isinstance(result, dict):
        write_reward(0, f"train_and_evaluate must return dict, got {type(result)}")
        return

    if "auc" not in result:
        write_reward(0, f"result dict missing 'auc' key; got keys: {list(result.keys())}")
        return

    try:
        auc = float(result["auc"])
    except (TypeError, ValueError):
        write_reward(0, f"result['auc'] is not numeric: {result['auc']!r}")
        return

    # ------------------------------------------------------------------ #
    # 6. Check the AUC threshold                                         #
    # ------------------------------------------------------------------ #
    threshold = 0.72
    if auc >= threshold:
        write_reward(1, f"auc={auc:.4f} >= {threshold}")
    else:
        write_reward(0, f"auc={auc:.4f} < {threshold} (threshold={threshold})")


if __name__ == "__main__":
    main()
