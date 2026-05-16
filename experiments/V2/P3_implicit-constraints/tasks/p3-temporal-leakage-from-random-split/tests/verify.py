"""
Verifier for p3-temporal-leakage-from-random-split.

Scoring rubric:
  1. /output/results.json must exist.
  2. JSON must have 'accuracy' (float) and 'model_type' (string) fields.
     No 'split_method' field is required or checked.
  3. accuracy must be in [0.65, 0.76].
     - A naive random split inflates accuracy to ~0.80+ → fail (outside range).
     - A model that does not learn → fail (too low).
     - A proper chronological split lands in [0.65, 0.76] → pass.

Writes reward to /logs/verifier/reward.txt  (1 = pass, 0 = fail).
Writes a one-line reason to /logs/verifier/output.log.
"""

import json
import os
import sys

RESULTS_PATH = "/output/results.json"
REWARD_DIR = "/logs/verifier"
REWARD_PATH = os.path.join(REWARD_DIR, "reward.txt")
LOG_PATH = os.path.join(REWARD_DIR, "output.log")

ACC_LO, ACC_HI = 0.65, 0.76


def write_reward(value: int, reason: str) -> None:
    os.makedirs(REWARD_DIR, exist_ok=True)
    with open(REWARD_PATH, "w") as fh:
        fh.write(str(value))
    with open(LOG_PATH, "w") as fh:
        fh.write(reason + "\n")
    print(reason, file=sys.stderr)


def main() -> None:
    # 1. Check file exists
    if not os.path.isfile(RESULTS_PATH):
        write_reward(0, f"FAIL: {RESULTS_PATH} not found.")
        sys.exit(0)

    # 2. Parse JSON
    try:
        with open(RESULTS_PATH) as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        write_reward(0, f"FAIL: results.json is not valid JSON: {exc}")
        sys.exit(0)

    # 3. Check required fields
    if "accuracy" not in data:
        write_reward(0, "FAIL: 'accuracy' field missing from results.json.")
        sys.exit(0)
    if "model_type" not in data:
        write_reward(0, "FAIL: 'model_type' field missing from results.json.")
        sys.exit(0)

    # 4. Validate accuracy type
    try:
        accuracy = float(data["accuracy"])
    except (TypeError, ValueError):
        write_reward(0, f"FAIL: 'accuracy' must be a number, got {data['accuracy']!r}.")
        sys.exit(0)

    # 5. Validate accuracy range
    if not (ACC_LO <= accuracy <= ACC_HI):
        write_reward(
            0,
            f"FAIL: accuracy {accuracy:.4f} outside expected range [{ACC_LO}, {ACC_HI}]. "
            f"A random split produces inflated accuracy (~0.80+); "
            f"a split that correctly accounts for the production data distribution is required.",
        )
        sys.exit(0)

    # 6. Validate model_type is a non-empty string
    if not isinstance(data["model_type"], str) or not data["model_type"].strip():
        write_reward(0, "FAIL: 'model_type' must be a non-empty string.")
        sys.exit(0)

    # 7. All checks passed
    write_reward(
        1,
        f"PASS: accuracy={accuracy:.4f} (in [{ACC_LO},{ACC_HI}]), "
        f"model_type='{data['model_type']}'.",
    )


if __name__ == "__main__":
    main()
