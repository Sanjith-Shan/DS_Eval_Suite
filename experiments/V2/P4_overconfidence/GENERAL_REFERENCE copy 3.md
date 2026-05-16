# General Reference -- Drop this into any pattern folder alongside CONTEXT.md

## What we're doing

We're building Harbor-format evaluation tasks for an Abundant.AI take-home assignment. Each task tests whether Google's gemini-3-flash-preview model can do a specific data science task. The goal is to find tasks the model fails at consistently (pass@3 < 30%).

This folder contains a CONTEXT.md that describes a specific failure pattern and lists 5-7 concrete tasks to build. Read CONTEXT.md first. This file covers everything else you need.

## Harbor Framework

Harbor is an open-source framework for running AI agents against evaluation tasks in sandboxed Docker containers.

### Task folder structure

Every task is a folder with these files:

```
task-name/
├── instruction.md              # Prompt given to the agent (2-3 paragraphs, clear, no hints)
├── task.toml                   # Task config
├── environment/
│   ├── Dockerfile              # Sets up the sandbox (python, libraries, data files)
│   └── (data files)            # CSV/JSON files the agent works with
├── tests/
│   ├── test.sh                 # Entry point for verifier
│   └── verify.py               # Actual verification logic
└── solution/
    ├── solve.sh                # Entry point for reference solution
    └── solve.py                # Reference solution script
```

### task.toml format

```toml
schema_version = "1.1"

[task]
name = "task-name"
description = "Short description"
authors = ["Sanjith"]
keywords = ["data-science"]

[agent]
timeout_sec = 600

[verifier]
timeout_sec = 120

[environment]
allow_internet = false
cpus = 2
memory_mb = 4096
storage_mb = 2048
```

### Critical rules

1. test.sh MUST write 1 (pass) or 0 (fail) to /logs/verifier/reward.txt
2. /tests is mounted ONLY when the verifier runs. The agent CANNOT see test files during execution.
3. NEVER put ground truth or reference answers in the Dockerfile or environment/ data files. Only in /tests.
4. instruction.md should be 2-3 paragraphs. Clear, self-contained, no hints toward the solution.
5. solution/solve.sh must produce a passing state when run in the environment. This proves solvability.

### test.sh pattern

```bash
#!/bin/bash
mkdir -p /logs/verifier
cd /workspace
python /tests/verify.py > /logs/verifier/output.log 2>&1
```

### verify.py pattern

```python
import json
import os

def verify():
    try:
        # Check agent output exists
        if not os.path.exists("/workspace/output/results.json"):
            return False

        with open("/workspace/output/results.json") as f:
            results = json.load(f)

        # Check specific conditions (customize per task)
        passed = True  # replace with actual checks

        return passed
    except Exception as e:
        print(f"Verification error: {e}")
        return False

reward = 1 if verify() else 0
os.makedirs("/logs/verifier", exist_ok=True)
with open("/logs/verifier/reward.txt", "w") as f:
    f.write(str(reward))
```

### solve.sh pattern

```bash
#!/bin/bash
cd /workspace
python /solution/solve.py
```

### Dockerfile pattern

```dockerfile
FROM python:3.11-slim

WORKDIR /workspace

RUN pip install --no-cache-dir pandas numpy scipy scikit-learn statsmodels

COPY data.csv /workspace/data.csv
```

Add more pip packages as needed (prophet, xgboost, etc). COPY all data files the agent needs.

### Verifier design rules

DO verify:
- Numeric output in a specific range (accuracy in [0.70, 0.85])
- Required JSON fields exist and are non-empty
- Re-running the agent's code and checking the return value
- Comparing output to hidden ground truth in /tests

DO NOT verify:
- Source code patterns or keywords (brittle, gameable)
- Keyword presence in explanations (model may phrase differently)
- Anything using LLM-as-judge (too noisy at n=3)

### CLI commands for testing

```bash
# Oracle test (must get reward 1)
harbor run -p <task-path> -a oracle

# Nop test (must get reward 0)
harbor run -p <task-path> -a nop

# Actual model test (run 3 times)
harbor run -p <task-path> -a gemini-cli -m google/gemini-3-flash-preview
```

Gemini API key: GEMINI_API_KEY=<REDACTED>

## Build workflow for each task

1. Write generate_data.py that creates synthetic data with a fixed random seed
2. Run generate_data.py locally to produce CSV/data files
3. Copy those data files into the task's environment/ directory
4. Write instruction.md, task.toml, Dockerfile, test.sh, verify.py, solve.sh, solve.py
5. Build and test locally:
   - `harbor run -p <task> -a oracle` (expect reward 1)
   - `harbor run -p <task> -a nop` (expect reward 0)
6. If oracle fails, fix solution or verifier
7. If nop passes, verifier is broken (it passes on inaction)

## Output format convention

Unless CONTEXT.md says otherwise, every task should instruct the agent to write results to /output/ as a JSON file. The verifier reads from /output/. This keeps things consistent.

Make sure the Dockerfile creates the /output directory:
```dockerfile
RUN mkdir -p /workspace/output
```

Or have the instruction tell the agent to create it.
