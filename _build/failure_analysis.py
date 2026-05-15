"""Pull failure summaries from Harbor trajectories for the report.

For each task, walks jobs/<task>-gemini/<trial>/agent/trajectory.json and
prints:
  - the verifier stdout (which usually contains 'reason=' from my verifiers),
  - the agent's final user-visible message,
  - the first agent error / non-zero exit observed during the trial.

This lets me write the failure-analysis section of the report from the
actual evidence rather than guessing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "jobs"


def first_line_match(text: str, needle: str) -> str | None:
    for line in text.splitlines():
        if needle in line:
            return line.strip()
    return None


def summarise_trial(trial_dir: Path) -> dict:
    reward_file = trial_dir / "verifier" / "reward.txt"
    reward = None
    if reward_file.exists():
        try:
            reward = int(reward_file.read_text().strip())
        except ValueError:
            reward = 0

    verifier_stdout = trial_dir / "verifier" / "test-stdout.txt"
    verifier_text = verifier_stdout.read_text() if verifier_stdout.exists() else ""
    # The verifier writes 'reward=... reason=...' to stderr in our tasks;
    # Harbor merges into test-stdout.txt only sometimes. Also check trial.log.
    trial_log = trial_dir / "trial.log"
    trial_text = trial_log.read_text() if trial_log.exists() else ""

    reason = first_line_match(verifier_text, "reason=") or first_line_match(trial_text, "reason=")

    final_msg = None
    traj = trial_dir / "agent" / "trajectory.json"
    if traj.exists():
        try:
            data = json.loads(traj.read_text())
            for step in reversed(data.get("steps", [])):
                if step.get("source") == "agent" and step.get("message"):
                    final_msg = step["message"]
                    break
        except Exception:
            pass

    exc = trial_dir / "exception.txt"
    exc_text = exc.read_text().strip() if exc.exists() else None

    return {
        "trial": trial_dir.name,
        "reward": reward,
        "reason": reason,
        "exception": exc_text,
        "final_msg_excerpt": (final_msg[:400] + "...") if final_msg and len(final_msg) > 400 else final_msg,
    }


def main() -> int:
    rows = []
    for job_dir in sorted(JOBS.iterdir()):
        if not job_dir.is_dir() or not job_dir.name.endswith("-gemini"):
            continue
        task = job_dir.name.removesuffix("-gemini")
        trials = []
        for trial in sorted(d for d in job_dir.iterdir() if d.is_dir()):
            trials.append(summarise_trial(trial))
        rows.append({"task": task, "trials": trials})

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
