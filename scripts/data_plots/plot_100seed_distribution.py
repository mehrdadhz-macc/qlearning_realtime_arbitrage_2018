"""Plot the average and interval of Reward 1 vs Reward 2 profit across many
paired trial seeds (Experiment 2: same seed used for both rewards in each
trial, so the comparison isn't contaminated by unrelated exploration luck).

Works for either TRAINING profit (train.py's own summary.json) or HELD-OUT
2017 profit (evaluate.py's held_out_2017_eval_summary.json) via --which --
each reward's full distribution of trial profits (one dot per seed) plus
the mean +/- 1 std interval; N is usually too large for a bar chart alone
to be an honest summary of the spread, so individual trials stay visible.

Usage:
    venv/bin/python3 scripts/data_plots/plot_100seed_distribution.py \
        --run-dir outputs/runs/20260810_154500_exp2_1mw_100seed_distribution --which training
    venv/bin/python3 scripts/data_plots/plot_100seed_distribution.py \
        --run-dir outputs/runs/20260810_154500_exp2_1mw_100seed_distribution --which held_out
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SOURCES = {
    "training": {
        "summary_file": "summary.json",
        "profit_key": "trial_final_pass_profits",
        "ylabel": "Training profit ($)",
        "out_prefix": "training",
        "title_suffix": "(training, 2016)",
    },
    "held_out": {
        "summary_file": "held_out_2017_eval_summary.json",
        "profit_key": "trial_final_profits",
        "ylabel": "Held-out 2017 profit ($)",
        "out_prefix": "held_out_2017",
        "title_suffix": "(held-out, 2017, frozen greedy)",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--which", choices=["training", "held_out"], default="training")
    args = parser.parse_args()

    cfg = SOURCES[args.which]
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / cfg["summary_file"]).read_text())
    r1 = np.array(summary["results"]["reward_1"][cfg["profit_key"]])
    r2 = np.array(summary["results"]["reward_2"][cfg["profit_key"]])
    n = len(r1)
    assert len(r2) == n

    d = r2 - r1
    paired_se = d.std(ddof=1) / np.sqrt(n)
    paired_t = d.mean() / paired_se if paired_se > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    rng = np.random.default_rng(0)
    groups = [("Reward 1", r1, "tab:red"), ("Reward 2", r2, "tab:blue")]
    for i, (label, vals, color) in enumerate(groups):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, alpha=0.4, s=28,
                   zorder=3, edgecolors="none")
        mean, std = vals.mean(), vals.std()
        ci95 = 1.96 * std / np.sqrt(n)
        ci_low, ci_high = mean - ci95, mean + ci95
        ax.errorbar([i], [mean], yerr=[std], fmt="o", color=color, ecolor=color,
                   elinewidth=2, capsize=6, markersize=9, zorder=4,
                   markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"${mean:,.2f} (mean)\nn={n}, std=${std:,.0f}\n95% CI: [${ci_low:,.0f}, ${ci_high:,.0f}]",
                   (i, mean), color=color, fontsize=8.5,
                   xytext=(14, 0), textcoords="offset points", va="center", fontweight="bold")

    ax.axhline(0, color="gray", linewidth=0.7, zorder=1)
    ax.set_xlim(-0.5, 1.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Reward 1", "Reward 2"])
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(f"Reward 1 vs Reward 2, {n} paired seeds {cfg['title_suffix']}", loc="left")
    ax.set_title(f"Dots = individual seeds; marker = mean +/- 1 std; paired t={paired_t:.2f}",
                fontsize=9, color="gray", loc="left", y=-0.13)

    fig.tight_layout()
    out_path = run_dir / f"{cfg['out_prefix']}_distribution_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    def summarize(vals):
        mean, std = float(vals.mean()), float(vals.std())
        ci95 = 1.96 * std / np.sqrt(n)
        return {"mean": mean, "std": std, "min": float(vals.min()), "max": float(vals.max()),
                "ci95_halfwidth": ci95, "ci95_low": mean - ci95, "ci95_high": mean + ci95}

    stats = {
        "n_trials": n,
        "reward_1": summarize(r1),
        "reward_2": summarize(r2),
        "paired_diff_mean": float(d.mean()),
        "paired_diff_std": float(d.std(ddof=1)),
        "paired_t_stat": float(paired_t),
        "reward_2_wins": int((d > 0).sum()),
    }
    stats_path = run_dir / f"{cfg['out_prefix']}_distribution_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Saved stats to {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
