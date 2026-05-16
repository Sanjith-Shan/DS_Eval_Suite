#!/bin/bash
mkdir -p /logs/verifier
cd /workspace
python /tests/verify.py > /logs/verifier/output.log 2>&1
