"""
make_plots.py
-------------
Generate the difficulty profile plot: pass@3 for each of the 10 tasks
against gemini-3-flash-preview. All 10 tasks scored 0/3 in the final
evaluation, so all bars are at 0%. The horizontal dashed line marks
the 30% target threshold from the assignment spec.

Output: figures/pass_at_3.png
"""

import os
import matplotlib.pyplot as plt

TASKS = [
    ("autocorrelated-residuals",     "Autocorr.\nresiduals"),
    ("censored-survival",            "Censored\nsurvival"),
    ("clustered-parametric-test",    "Clustered\nparametric"),
    ("clustered-treatment",          "Clustered\ntreatment"),
    ("confounded-comparison",        "Confounded\ncomparison"),
    ("multicollinearity-after-log",  "Multicol.\nafter log"),
    ("multiple-comparisons",         "Multiple\ncomparisons"),
    ("regression-to-mean",           "Regression\nto mean"),
    ("spurious-regression",          "Spurious\nregression"),
    ("survivorship-bias-sample",     "Survivorship\nbias"),
]

PASS_AT_3 = [0.0] * len(TASKS)
THRESHOLD = 30.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pass_at_3.png")

    short_names = [t[1] for t in TASKS]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(TASKS)), PASS_AT_3, color="#3b6ea5", edgecolor="black", linewidth=0.5)
    ax.axhline(THRESHOLD, linestyle="--", color="red", linewidth=1.2,
               label=f"target threshold ({THRESHOLD:.0f}%)")

    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("pass@3 (%)")
    ax.set_title("Pass@3 per task against gemini-3-flash-preview")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
