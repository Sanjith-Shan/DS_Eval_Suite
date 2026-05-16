#!/bin/bash
set -eu
mkdir -p /output
DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$DIR/classify_fixed.py" /output/classify.py
python /output/classify.py
