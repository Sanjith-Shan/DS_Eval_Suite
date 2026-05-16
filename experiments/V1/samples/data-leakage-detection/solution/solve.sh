#!/bin/bash
set -eu
mkdir -p /output
DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$DIR/pipeline_fixed.py" /output/pipeline_fixed.py
python /output/pipeline_fixed.py
