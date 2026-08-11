"""Plot cumulative profit OVER TIME across all trials of a train.py
`--reward both --n-trials N` run (Experiment 2's 100-seed distribution),
not just each trial's final value.

Loads every trial_NN/history_{reward_kind}.npy (columns: t, price, action,
c, d, profit, cumulative_profit -- see train.py), stacks the
cumulative_profit column across trials, and plots the mean trajectory with
a 10th-90th percentile band per reward. A band, not N overlapping raw
lines, because 100 spaghetti lines on one axis stops being readable; the
band still shows the full spread at every hour, not just the endpoint.

Usage:
    venv/bin/python3 scripts/data_plots/plot_100seed_cumulative_profit.py \
        --run-dir outputs/runs/20260810_154500_exp2_1mw_100seed_distribution
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_price_series


def load_curves(run_dir, reward_kind):
    # trial_NN/ may live directly under run_dir, or nested under
    # run_dir/individual_trials/ if the run was reorganized for readability.
    trial_dirs = sorted(run_dir.glob("trial_*")) or sorted((run_dir / "individual_trials").glob("trial_*"))
    trial_dirs = sorted(trial_dirs, key=lambda p: int(re.search(r"\d+", p.name).group()))
    curves = []
    for td in trial_dirs:
        h = np.load(td / f"history_{reward_kind}.npy")
        curves.append(h[:, 6])  # cumulative_profit column
    return np.stack(curves)  # (n_trials, T)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", default="data/train/isone_rt_hourly_lmp_2016.csv",
                         help="Price series shown in the top panel (must match the hours these trials trained on)")
    parser.add_argument("--lower-pct", type=float, default=10.0)
    parser.add_argument("--upper-pct", type=float, default=90.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    colors = {"reward_1": "tab:red", "reward_2": "tab:blue"}
    _, prices = load_price_series(args.data)

    fig, (price_ax, ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[1, 2])
    price_ax.plot(np.arange(len(prices)), prices, color="tab:gray", linewidth=0.5)
    price_ax.set_ylabel("Price ($/MWh)")
    price_ax.set_title("Underlying price series", fontsize=10, color="dimgray", loc="left")
    band_stats = {}
    for reward_kind in ["reward_1", "reward_2"]:
        curves = load_curves(run_dir, reward_kind)
        n_trials, T = curves.shape
        mean = curves.mean(axis=0)
        lower = np.percentile(curves, args.lower_pct, axis=0)
        upper = np.percentile(curves, args.upper_pct, axis=0)

        x = np.arange(T)
        color = colors[reward_kind]
        ax.fill_between(x, lower, upper, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.5,
                label=f"{reward_kind.replace('_', ' ').title()} (mean, n={n_trials}): ${mean[-1]:,.2f}")
        ax.annotate(f"${mean[-1]:,.2f}", (T - 1, mean[-1]), color=color, fontsize=9,
                    fontweight="bold", xytext=(6, 0), textcoords="offset points", va="center")

        band_stats[reward_kind] = {
            "n_trials": n_trials, "final_mean": float(mean[-1]),
            f"final_p{args.lower_pct:g}": float(lower[-1]),
            f"final_p{args.upper_pct:g}": float(upper[-1]),
        }

    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("Cumulative profit ($)")
    ax.set_title(f"Reward 1 vs Reward 2, cumulative profit over time -- "
                 f"mean +/- [{args.lower_pct:g}th, {args.upper_pct:g}th] percentile band across trials")
    ax.legend(loc="upper left")
    ax.margins(x=0.09)  # room for the end-of-curve annotations
    fig.tight_layout()
    out_path = run_dir / "training_cumulative_profit_over_time_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    (run_dir / "training_cumulative_profit_over_time_stats.json").write_text(json.dumps(band_stats, indent=2))
    print(json.dumps(band_stats, indent=2))


if __name__ == "__main__":
    main()
