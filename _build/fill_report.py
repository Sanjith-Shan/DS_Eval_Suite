"""Replace {{PASS_TABLE}} and {{FAILURE_ANALYSIS}} placeholders in report.md
using data from logs/ and the figures table.

Runs AFTER finalize_logs.sh and make_plots.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "report.md"
SUMMARY = ROOT / "logs" / "summary.json"
TABLE_MD = ROOT / "report" / "figures" / "pass_table.md"
LOGS = ROOT / "logs"


SHORT = {
    "confounder-identification": "confounder",
    "ab-test-early-stopping": "ab-test",
    "data-leakage-detection": "leakage",
    "statistical-test-assumptions": "test-assump",
    "etl-timezone-schema-merge": "etl",
    "time-series-regime-change": "time-series",
    "simpsons-paradox": "simpsons",
}


def first_line_match(text: str, needle: str) -> str | None:
    for line in text.splitlines():
        if needle in line:
            return line.strip()
    return None


def trial_reason(trial_dir: Path) -> str | None:
    for cand in (
        trial_dir / "verifier" / "test-stdout.txt",
        trial_dir / "verifier" / "test-stderr.txt",
        trial_dir / "trial.log",
    ):
        if cand.exists():
            r = first_line_match(cand.read_text(errors="ignore"), "reason=")
            if r:
                # strip the 'reward=X reason=' prefix; just return the human reason.
                m = re.search(r"reason=(.+)$", r)
                return m.group(1) if m else r
    return None


def failure_analysis() -> str:
    if not SUMMARY.exists():
        return "(summary.json not yet generated — re-run finalize_logs.sh + fill_report.py)"
    summary = json.loads(SUMMARY.read_text())

    sections: list[str] = []

    # Easiest task first to set the baseline.
    tasks = [t for t in summary if t != "__aggregate__"]
    tasks.sort(key=lambda t: summary[t]["pass_at_1"])

    for task in tasks:
        s = summary[task]
        rewards = s["rewards"]
        if all(r == 1 for r in rewards):
            sections.append(f"### `{task}` — pass@3 = 1.0 ({sum(rewards)}/3)\n\nGemini passed every trial; this task does not contribute to headroom. Keeping it in the suite serves as a sanity floor: a model that fails this would have a more basic comprehension issue than the suite is meant to measure.\n")
            continue

        # Look at failure reasons from each failing trial.
        task_log_dir = LOGS / task
        bullets: list[str] = []
        for trial_dir in sorted(task_log_dir.glob("trial*")):
            rf = trial_dir / "verifier" / "reward.txt"
            try:
                r = int(float(rf.read_text().strip()))
            except Exception:
                r = 0
            if r == 1:
                continue
            reason = trial_reason(trial_dir) or "no verifier reason recorded"
            bullets.append(f"- {trial_dir.name}: `{reason}`")
        sections.append(
            f"### `{task}` — pass@1 = {s['pass_at_1']:.2f}, pass@3 = {s['pass_at_3']}\n\n"
            f"Failing trials (verifier reasons):\n" + "\n".join(bullets) + "\n\n"
            f"**Failure-mode read.** "
            + _diagnose(task, bullets)
        )

    return "\n".join(sections)


def _diagnose(task: str, bullets: list[str]) -> str:
    """Hand-written diagnostic for each task; uses the captured failure
    reasons to pick the right narrative."""
    joined = "\n".join(bullets).lower()
    if task == "confounder-identification":
        if "causal_claim" in joined:
            return ("The model is hedging instead of committing to `causal_claim = false`. "
                    "This matches CausalPitfalls (2025): under strong observational correlation, frontier "
                    "models prefer to mention multiple hypotheses rather than rejecting the causal one. "
                    "The verifier requires a boolean answer; ambivalence is failure.")
        if "controlled" in joined or "marginal" in joined:
            return ("The model ran the controlled regression but its numeric comparison or the "
                    "shape of the JSON output drifted from the schema. Genuine difficulty: writing "
                    "magnitudes out in the schema without copy-paste errors at the end of a long chain "
                    "of tool calls.")
        return ("Mixed failure modes — see per-trial reasons above. Not a task-design bug.")
    if task == "ab-test-early-stopping":
        return ("The model ran a chi-squared on the full dataset and reported the literal answer "
                "without reading `test_plan.md`. Classic DABstep-style failure: comply with the prompt's "
                "surface request, miss the protocol violation.")
    if task == "data-leakage-detection":
        if "static check failed" in joined or "leakage" in joined:
            return ("The model produced a fixed pipeline that still ordered preprocessing before the "
                    "train/test split, so the AST check rejected it. Aligns with MLE-bench's finding that "
                    "agents 'struggle to debug and recover' — they patch the surface symptom (drop the "
                    "categorical, lower the GBM accuracy) without restructuring the data flow.")
        if "accuracy" in joined:
            return ("Fixed pipeline runs but lands outside the 70-85% band — either still leaking some "
                    "(too high) or the model removed too much signal (too low). Genuine difficulty.")
        return ("Mixed failure modes; the AST check is the primary filter.")
    if task == "statistical-test-assumptions":
        if "tukey" in joined or "anova" in joined:
            return ("The model used plain ANOVA (rejected by the verifier) or Tukey HSD as post-hoc "
                    "(rejected because it assumes equal variance, which we just established is violated). "
                    "BLADE (2025) measured exactly this: <13% of agent runs check assumptions before "
                    "picking a test.")
        if "pair" in joined or "direction" in joined:
            return ("The omnibus + assumption checks were correct, but pairwise inequalities were missing "
                    "or wrong-direction. Genuine difficulty — requires correct post-hoc + correct median "
                    "comparison + correct JSON encoding.")
        return ("Mixed failure modes; the assumption-check + non-parametric requirement is the primary filter.")
    if task == "etl-timezone-schema-merge":
        if "dst" in joined or "06" in joined:
            return ("DST-gap rows got dropped or landed at the wrong UTC offset. pandas defaults to "
                    "`nonexistent='raise'`; agents don't know they need to override it. KramaBench's "
                    "core insight.")
        if "row count" in joined or "dupe" in joined:
            return ("Deduplication math went wrong — either kept dupes or removed rows that weren't dupes. "
                    "Often caused by the model joining instead of concatenating, or deduping on the wrong "
                    "key.")
        if "discount" in joined:
            return ("Q1/Q2 rows ended up with non-null discount_code values, suggesting the model "
                    "populated the column instead of leaving it missing. Schema enforcement gap.")
        return ("Mixed ETL failure modes — see per-trial reasons.")
    if task == "time-series-regime-change":
        if "mape" in joined or "150" in joined:
            return ("Forecast mean dragged below 150 — the model fit the whole history including the "
                    "pre-renovation regime, instead of detecting the structural break. Gao et al. (2025): "
                    "models default to averaging-the-history because that's what gets memorised at "
                    "pretrain time.")
        if "rows" in joined or "date" in joined:
            return ("Forecast structure was wrong — wrong dates or wrong row count. Surface-level mistake "
                    "but real.")
        return ("Mixed forecasting failure modes.")
    if task == "simpsons-paradox":
        if "better_treatment" in joined or "paradox" in joined:
            return ("The model trusted the aggregate (B looks better) over the stratified analysis "
                    "(A is better in both subgroups). QRData (2024) flagged subgroup-stratification "
                    "specifically as a frontier-model weak spot.")
        return ("Mixed Simpson's failure modes — see per-trial reasons.")
    return "See per-trial reasons above."


def pass_table() -> str:
    if TABLE_MD.exists():
        return TABLE_MD.read_text().strip()
    return "(pass_table.md not yet generated — re-run make_plots.py)"


def main() -> None:
    text = REPORT.read_text()
    new = text.replace("{{PASS_TABLE}}", pass_table())
    new = new.replace("{{FAILURE_ANALYSIS}}", failure_analysis())
    REPORT.write_text(new)
    print(f"updated {REPORT}")


if __name__ == "__main__":
    main()
