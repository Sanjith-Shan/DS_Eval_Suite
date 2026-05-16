"""Generate the report's difficulty-profile plots from Harbor job results.

Reads jobs/<task>-gemini/result.json for each task, computes per-task pass@1
and pass@3, and renders:
  - report/figures/pass_rates.png  (bar chart, per-task pass@1 & pass@3)
  - report/figures/difficulty_curve.png (cumulative passed)
  - report/figures/pass_table.md  (markdown table)

Pass@1 := mean over trials of (reward == 1).
Pass@3 := 1 if any of the 3 trials returns reward 1, else 0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
FIG_DIR = ROOT / "report" / "figures"

ORDER = [
    "confounder-identification",
    "ab-test-early-stopping",
    "data-leakage-detection",
    "statistical-test-assumptions",
    "etl-timezone-schema-merge",
    "time-series-regime-change",
    "simpsons-paradox",
]
SHORT = {
    "confounder-identification": "confounder",
    "ab-test-early-stopping": "ab-test",
    "data-leakage-detection": "leakage",
    "statistical-test-assumptions": "test-assump",
    "etl-timezone-schema-merge": "etl",
    "time-series-regime-change": "time-series",
    "simpsons-paradox": "simpsons",
}


def trial_rewards(task: str) -> list[float]:
    """Collect per-attempt rewards from logs/<task>/trial*."""
    task_dir = LOGS / task
    if not task_dir.exists():
        return []
    rewards: list[float] = []
    for trial_dir in sorted(task_dir.glob("trial*")):
        rfile = trial_dir / "verifier" / "reward.txt"
        if rfile.exists():
            try:
                rewards.append(float(rfile.read_text().strip()))
            except ValueError:
                rewards.append(0.0)
        else:
            # Some failure modes (e.g. timeout) leave no reward; treat as 0.
            rewards.append(0.0)
    return rewards


def main() -> int:
    rows: list[tuple[str, list[float], float, int]] = []
    for t in ORDER:
        rewards = trial_rewards(t)
        pass1 = sum(rewards) / len(rewards) if rewards else 0.0
        pass3 = int(any(r >= 1.0 for r in rewards[:3])) if rewards else 0
        rows.append((t, rewards, pass1, pass3))

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: per-task bars ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = [SHORT[t] for t in ORDER]
    pass1s = [r[2] for r in rows]
    pass3s = [r[3] for r in rows]
    x = list(range(len(labels)))
    bw = 0.38
    ax.bar([i - bw/2 for i in x], pass1s, width=bw, label="pass@1", color="#4c78a8")
    ax.bar([i + bw/2 for i in x], pass3s, width=bw, label="pass@3", color="#e45756")
    ax.axhline(0.30, color="grey", linestyle="--", linewidth=1, label="30% target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pass rate")
    ax.set_title("gemini-3-flash-preview pass@1 vs pass@3 (3 trials/task)")
    ax.legend(loc="upper right", frameon=False)
    for i, (p1, p3) in enumerate(zip(pass1s, pass3s)):
        ax.text(i - bw/2, p1 + 0.02, f"{p1:.2f}", ha="center", fontsize=8)
        ax.text(i + bw/2, p3 + 0.02, f"{p3}", ha="center", fontsize=8)
    fig.tight_layout()
    out1 = FIG_DIR / "pass_rates.png"
    fig.savefig(out1, dpi=140)
    plt.close(fig)
    print(f"wrote {out1}")

    # --- Plot 2: difficulty curve (sorted ascending by pass@1) ---
    sorted_rows = sorted(rows, key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = [SHORT[r[0]] for r in sorted_rows]
    ys = [r[2] for r in sorted_rows]
    ax.plot(xs, ys, marker="o", color="#4c78a8", linewidth=2)
    ax.fill_between(xs, ys, alpha=0.15, color="#4c78a8")
    ax.axhline(0.30, color="grey", linestyle="--", linewidth=1, label="30% pass@3 target")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(xs, rotation=20, ha="right")
    ax.set_ylabel("pass@1")
    ax.set_title("Difficulty curve (tasks sorted by pass@1)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out2 = FIG_DIR / "difficulty_curve.png"
    fig.savefig(out2, dpi=140)
    plt.close(fig)
    print(f"wrote {out2}")

    # --- Markdown table ---
    lines = [
        "| Task | Trials | Rewards | pass@1 | pass@3 |",
        "|---|---|---|---|---|",
    ]
    agg_p1: list[float] = []
    agg_p3: list[int] = []
    for t, rewards, p1, p3 in rows:
        r_str = ", ".join(f"{int(r)}" for r in rewards) if rewards else "—"
        lines.append(f"| `{t}` | {len(rewards)} | [{r_str}] | {p1:.2f} | {p3} |")
        agg_p1.append(p1)
        agg_p3.append(p3)
    if agg_p1:
        avg_p1 = sum(agg_p1) / len(agg_p1)
        avg_p3 = sum(agg_p3) / len(agg_p3)
        lines.append(f"| **aggregate** | — | — | **{avg_p1:.3f}** | **{avg_p3:.3f}** |")
    table = "\n".join(lines)
    (FIG_DIR / "pass_table.md").write_text(table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
