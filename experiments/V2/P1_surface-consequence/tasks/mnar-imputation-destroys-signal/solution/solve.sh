#!/bin/bash
set -eu
mkdir -p /output
DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$DIR/regression_fixed.py" /output/regression.py
python /output/regression.py
