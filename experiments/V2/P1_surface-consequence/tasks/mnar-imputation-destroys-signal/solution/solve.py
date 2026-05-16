"""
Entry point for the oracle solution.
Calls regression_fixed.fit() on the workspace data and prints the results.
"""

import os
import sys

# Ensure the solution directory is in the path when run via solve.sh
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from regression_fixed import fit

DATA_PATH = "/workspace/train.csv"

if __name__ == "__main__":
    result = fit(DATA_PATH)
    print(f"X3_coef={result['X3_coef']:.6f}  X3_pvalue={result['X3_pvalue']:.6f}")
