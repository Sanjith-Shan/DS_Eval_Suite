# Harbor Framework Reference

Harbor is an open-source framework for authoring evaluation tasks, running AI agents against them, and capturing trajectories. It's built by the Laude Institute (Stanford-affiliated) and used for agent benchmarks and RL training data generation.

## Install

```bash
# Install uv first if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install harbor
uv tool install harbor
```

Requires Docker to be running (all tasks execute in containers).

## Core Concept

A **task** is a self-contained problem with a verifiable outcome. An **agent** (an LLM like Gemini) receives the task instruction, acts inside a sandboxed Docker environment, and produces output. A **verifier** script then checks whether the agent succeeded (reward 1) or failed (reward 0).

## Task Folder Structure

```
my-task/
├── instruction.md              # The prompt given to the agent
├── task.toml                   # Task configuration
├── environment/
│   └── Dockerfile              # Sets up the sandboxed environment
├── tests/
│   ├── test.sh                 # Entry point for verifier
│   └── verify.py               # Verification logic (called by test.sh)
└── solution/
    └── solve.sh                # Reference solution proving the task is solvable
```

## task.toml Format

```toml
schema_version = "1.1"

[task]
name = "my-task"
description = "Short description of the task"
authors = ["Author Name"]
keywords = ["data-science", "statistics"]

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

## Key Rules

1. **test.sh must write reward to `/logs/verifier/reward.txt`** -- either `1` (pass) or `0` (fail). This is how Harbor reads the result.

2. **`/tests` is mounted ONLY when the verifier runs.** The agent cannot see test files during execution. So never rely on the agent reading anything from `/tests`.

3. **Never put reference answers in the Dockerfile.** Anything baked into the Docker image is readable by the agent. Put ground truth only in `/tests`.

4. **instruction.md should be clear and self-contained.** 2-3 paragraphs. No hints toward the solution. No formatting tricks.

5. **solution/solve.sh must produce a passing state.** This proves the task is solvable. It runs inside the same Docker environment.

## test.sh Pattern

```bash
#!/bin/bash
mkdir -p /logs/verifier
cd /workspace
python /tests/verify.py > /logs/verifier/output.log 2>&1
```

Where verify.py does the actual checking and writes the reward:

```python
import json

def verify():
    try:
        # ... check agent output ...
        passed = True  # or False
    except Exception as e:
        passed = False

    with open("/logs/verifier/reward.txt", "w") as f:
        f.write("1" if passed else "0")

verify()
```

## solution/solve.sh Pattern

```bash
#!/bin/bash
cd /workspace
python solve.py
```

Where solve.py is a reference solution script already present in the solution/ directory.

## Dockerfile Pattern

```dockerfile
FROM python:3.11-slim

WORKDIR /workspace

# Install dependencies
RUN pip install pandas numpy scipy scikit-learn statsmodels

# Copy data files into the workspace
COPY data.csv /workspace/data.csv

# Agent will execute here
```

## CLI Commands

```bash
# Run with Oracle agent (executes solve.sh -- expect reward 1)
harbor run -p path/to/task -a oracle

# Run with Nop agent (does nothing -- expect reward 0)
harbor run -p path/to/task -a nop

# Run with a real model
harbor run -p path/to/task -a gemini-cli -m google/gemini-3-flash-preview

# Quality check a task
harbor check path/to/task -r https://github.com/harbor-framework/terminal-bench-3/blob/main/rubrics/task-implementation.toml

# View job results
harbor view jobs
```

## Output Structure

After `harbor run`, results appear in `jobs/<job-id>/<trial>/`:
- `agent/trajectory.json` -- full ATIF-format execution trace
- `verifier/reward.txt` -- the reward (0 or 1)
- `result.json` -- summary

## Environment Variable

```bash
export GEMINI_API_KEY=<REDACTED>
```
