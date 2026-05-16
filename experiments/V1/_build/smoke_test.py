"""Local smoke-test harness that mimics Harbor's filesystem layout.

For a given task, this script:
  1. Creates a tmpdir with /workspace, /output, /logs/verifier subdirs.
  2. Copies the task's environment/* into /workspace.
  3. Runs the oracle solution against /workspace, then runs the verifier.
  4. Repeats with a 'nop' agent that does nothing, ensuring reward=0.

We patch absolute paths ('/workspace', '/output', '/logs/verifier', '/tests')
to their tmp counterparts. This is a sanity check only — Harbor itself runs
the real container.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_with_patched_paths(script: Path, tmp: Path, extra_env: dict | None = None) -> int:
    """Execute a python script with '/workspace', '/output', '/logs', '/tests'
    rewritten to tmp paths via a tiny pyhook."""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "_build") + os.pathsep + env.get("PYTHONPATH", "")
    env["SMOKE_TMP"] = str(tmp)
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        [sys.executable, "-c", _BOOTSTRAP, str(script)],
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


_BOOTSTRAP = r"""
import builtins
import os
import sys
from pathlib import Path

ROOT = os.environ['SMOKE_TMP']
PREFIXES = ('/workspace', '/output', '/logs', '/tests')

real_open = builtins.open
def patched_open(file, *a, **kw):
    s = os.fspath(file)
    for p in PREFIXES:
        if s.startswith(p):
            s = ROOT + s
            break
    return real_open(s, *a, **kw)
builtins.open = patched_open

orig_mkdir = Path.mkdir
def _redirect(self):
    s = str(self)
    for p in PREFIXES:
        if s.startswith(p):
            return Path(ROOT + s)
    return self
def patched_mkdir(self, *a, **kw):
    return orig_mkdir(_redirect(self), *a, **kw)
Path.mkdir = patched_mkdir

orig_write_text = Path.write_text
def patched_write_text(self, *a, **kw):
    return orig_write_text(_redirect(self), *a, **kw)
Path.write_text = patched_write_text

orig_read_text = Path.read_text
def patched_read_text(self, *a, **kw):
    return orig_read_text(_redirect(self), *a, **kw)
Path.read_text = patched_read_text

orig_exists = Path.exists
def patched_exists(self):
    return orig_exists(_redirect(self))
Path.exists = patched_exists

import os as _os
real_makedirs = _os.makedirs
def patched_makedirs(path, *a, **kw):
    s = os.fspath(path)
    for p in PREFIXES:
        if s.startswith(p):
            s = ROOT + s
            break
    return real_makedirs(s, *a, **kw)
_os.makedirs = patched_makedirs

# Pandas / numpy honour the patched open() for paths passed as strings.
import pandas as _pd
orig_read_csv = _pd.read_csv
def patched_read_csv(filepath_or_buffer, *a, **kw):
    if isinstance(filepath_or_buffer, str):
        for p in PREFIXES:
            if filepath_or_buffer.startswith(p):
                filepath_or_buffer = ROOT + filepath_or_buffer
                break
    return orig_read_csv(filepath_or_buffer, *a, **kw)
_pd.read_csv = patched_read_csv

orig_to_csv = _pd.DataFrame.to_csv
def patched_to_csv(self, path_or_buf=None, *a, **kw):
    if path_or_buf is not None and not hasattr(path_or_buf, 'write'):
        s = os.fspath(path_or_buf)
        for p in PREFIXES:
            if s.startswith(p):
                s = ROOT + s
                path_or_buf = s
                break
    return orig_to_csv(self, path_or_buf, *a, **kw)
_pd.DataFrame.to_csv = patched_to_csv

# Patch importlib.util.spec_from_file_location for verifiers that import the
# agent's /output script. The 'data_path' arg passed to imported funcs needs
# patching too — handled via the patched_open / pandas read_csv hooks above.
import importlib.util as _ilu
orig_spec = _ilu.spec_from_file_location
def patched_spec(name, location, *a, **kw):
    s = os.fspath(location)
    for p in PREFIXES:
        if s.startswith(p):
            s = ROOT + s
            break
    return orig_spec(name, s, *a, **kw)
_ilu.spec_from_file_location = patched_spec

script_path = sys.argv[1]
with real_open(script_path) as f:
    code = compile(f.read(), script_path, 'exec')
exec(code, {'__name__': '__main__', '__file__': script_path})
"""


def smoke_task(task_dir: Path, expect_reward: int, run_oracle: bool) -> bool:
    print(f"\n=== {task_dir.name} (oracle={run_oracle}, expect={expect_reward}) ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        for sub in ("workspace", "output", "logs/verifier", "tests"):
            (tmp_p / sub).mkdir(parents=True, exist_ok=True)

        # Copy environment into /workspace.
        env_dir = task_dir / "environment"
        for item in env_dir.iterdir():
            if item.name == "Dockerfile":
                continue
            dest = tmp_p / "workspace" / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy(item, dest)

        # Copy tests so /tests/verify.py is reachable.
        for item in (task_dir / "tests").iterdir():
            shutil.copy(item, tmp_p / "tests" / item.name)

        if run_oracle:
            solve_py = task_dir / "solution" / "solve.py"
            rc = run_with_patched_paths(solve_py, tmp_p)
            if rc != 0:
                print(f"  oracle exited with code {rc}")
                return False

        # Run verifier
        verify_py = task_dir / "tests" / "verify.py"
        rc = run_with_patched_paths(verify_py, tmp_p)
        reward_file = tmp_p / "logs/verifier/reward.txt"
        if not reward_file.exists():
            print("  verifier did not write reward.txt")
            return False
        reward = int(reward_file.read_text().strip())
        print(f"  observed reward={reward}")
        return reward == expect_reward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", help="Task folder names (under samples/). Default: all.")
    args = ap.parse_args()

    samples = ROOT / "samples"
    if args.tasks:
        task_dirs = [samples / t for t in args.tasks]
    else:
        task_dirs = sorted(d for d in samples.iterdir() if d.is_dir())

    ok = True
    for d in task_dirs:
        if not (d / "solution" / "solve.py").exists():
            print(f"SKIP {d.name}: no solve.py yet")
            continue
        if not smoke_task(d, expect_reward=1, run_oracle=True):
            ok = False
            print(f"FAIL oracle {d.name}")
        if not smoke_task(d, expect_reward=0, run_oracle=False):
            ok = False
            print(f"FAIL nop    {d.name}")

    print("\nALL OK" if ok else "\nFAILURES DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
