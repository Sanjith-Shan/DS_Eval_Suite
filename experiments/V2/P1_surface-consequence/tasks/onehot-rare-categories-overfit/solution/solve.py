"""Reference solution entry point: copies pipeline_fixed.py to /output/ and runs it."""

import os
import shutil
import sys
from pathlib import Path

SOLUTION_DIR = Path(__file__).parent
OUTPUT_DIR = Path("/output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
src = SOLUTION_DIR / "pipeline_fixed.py"
dst = OUTPUT_DIR / "pipeline_fixed.py"
shutil.copy(src, dst)
print(f"Copied {src} -> {dst}")

# Run the fixed pipeline to confirm it works.
import importlib.util

spec = importlib.util.spec_from_file_location("pipeline_fixed", dst)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
acc = module.train_and_evaluate("/workspace/data.csv")
print(f"TEST_ACCURACY={acc:.4f}")
