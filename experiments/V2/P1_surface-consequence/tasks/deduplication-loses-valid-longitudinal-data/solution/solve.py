"""
solve.py — Reference solution for deduplication-loses-valid-longitudinal-data.

Copies model_fixed.py to /output/model.py so the verifier can import it.
Run via solve.sh inside the Docker environment.
"""

import os
import shutil
from pathlib import Path

SOLUTION_DIR = Path(__file__).parent
MODEL_FIXED_SRC = SOLUTION_DIR / "model_fixed.py"
OUTPUT_PATH = Path("/output/model.py")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODEL_FIXED_SRC, OUTPUT_PATH)
    print(f"Copied {MODEL_FIXED_SRC} -> {OUTPUT_PATH}")

    # Quick smoke-test: import and run
    import importlib.util
    spec = importlib.util.spec_from_file_location("model", OUTPUT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.train_and_evaluate("/workspace/data.csv")
    print(f"AUC = {result['auc']:.4f}")
    assert result["auc"] >= 0.72, f"Oracle AUC too low: {result['auc']:.4f}"
    print("Oracle self-check passed.")


if __name__ == "__main__":
    main()
