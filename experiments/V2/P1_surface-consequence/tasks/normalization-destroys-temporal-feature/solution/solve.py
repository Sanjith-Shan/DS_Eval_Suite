"""
Entry point for the oracle solution.
Calls pipeline_fixed.train_and_evaluate() on the workspace data and prints the result.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pipeline_fixed import train_and_evaluate

DATA_PATH = "/workspace/data.csv"

if __name__ == "__main__":
    acc = train_and_evaluate(DATA_PATH)
    print(f"test_accuracy = {acc:.4f}")
