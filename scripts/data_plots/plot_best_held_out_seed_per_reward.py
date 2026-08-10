"""Train Reward 1 and Reward 2 each on ITS OWN best HELD-OUT seed (not best
training seed), and plot both training and held-out cumulative-profit
curves.

Companion to plot_best_seed_per_reward.py (which uses each reward's best
TRAINING seed). Here the selection criterion is flipped: which seed gives
each reward the highest profit on the held-out 2017 year, found by
cross-referencing Experiment 2's existing 100-trial paired sweep
(outputs/runs/20260810_154500_exp2_100seed_distribution/summary.json for
seeds + training profit, held_out_2017_eval_summary.json for held-out
profit -- trial index i matches the same seed in both files) rather than
re-running a new sweep. Reward 1's best-held-out trial used seed 345075200
($21,739.65 held-out, only $803.24 training); Reward 2's used seed
406886644 ($25,388.67 held-out, $4,149.10 training) -- both are mediocre
training performers, underscoring that training profit doesn't predict
held-out profit in either direction.

Usage:
    venv/bin/python3 scripts/data_plots/plot_best_held_out_seed_per_reward.py
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_loader import load_price_series
from train import run_training
from evaluate import greedy_rollout

BEST_HELD_OUT_SEED_REWARD_1 = 345075200  # highest Reward-1 held-out profit ($21,739.65) in Experiment 2's 100-trial sweep
BEST_HELD_OUT_SEED_REWARD_2 = 406886644  # highest Reward-2 held-out profit ($25,388.67) in Experiment 2's 100-trial sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train/isone_rt_hourly_lmp_2016.csv")
    parser.add_argument("--test-data", default="data/test/isone_rt_hourly_lmp_2017.csv")
    parser.add_argument("--seed-reward1", type=int, default=BEST_HELD_OUT_SEED_REWARD_1)
    parser.add_argument("--seed-reward2", type=int, default=BEST_HELD_OUT_SEED_REWARD_2)
    parser.add_argument("--capacity-mwh", type=float, default=8.0)
    parser.add_argument("--max-rate-mw", type=float, default=1.0)
    parser.add_argument("--n-price-bins", type=int, default=5)
    parser.add_argument("--price-bin-method", choices=["equal_width", "quantile"], default="quantile")
    parser.add_argument("--bin-calibration-hours", type=int, default=24 * 30)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.9)
    parser.add_argument("--smoothing", type=float, default=0.001)
    parser.add_argument("--efficiency-charge", type=float, default=1.0)
    parser.add_argument("--efficiency-discharge", type=float, default=1.0)
    parser.add_argument("--n-passes", type=int, default=1)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    _, prices = load_price_series(args.data)
    _, test_prices = load_price_series(args.test_data)

    run_dir = Path(args.out_dir) if args.out_dir else Path("outputs/runs") / (
        time.strftime("%Y%m%d_%H%M%S") + "_exp3_best_held_out_seed_per_reward")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_fig, train_ax = plt.subplots(figsize=(10, 5))
    eval_fig, eval_ax = plt.subplots(figsize=(10, 5))
    seeds = {"reward_1": args.seed_reward1, "reward_2": args.seed_reward2}
    colors = {"reward_1": "tab:red", "reward_2": "tab:blue"}
    results = {}

    for reward_kind in ["reward_1", "reward_2"]:
        seed = seeds[reward_kind]
        agent, history, cumulative_profit, final_pass_profit, edges = run_training(
            prices, reward_kind, args.capacity_mwh, args.max_rate_mw, args.n_price_bins,
            args.alpha, args.gamma, args.epsilon, args.smoothing, seed,
            args.price_bin_method, args.bin_calibration_hours,
            args.efficiency_charge, args.efficiency_discharge, False,
            args.n_passes,
        )
        print(f"{reward_kind} (seed={seed}): training profit = ${final_pass_profit:,.2f}")

        train_curve = np.array([h["cumulative_profit"] for h in history])
        np.save(run_dir / f"training_cumulative_profit_curve_{reward_kind}.npy", train_curve)
        np.save(run_dir / f"price_bin_edges_{reward_kind}.npy", edges)
        agent.save(run_dir / f"q_table_{reward_kind}.npy")

        held_out_curve = greedy_rollout(
            test_prices, agent.q, args.capacity_mwh, args.max_rate_mw, edges,
            args.efficiency_charge, args.efficiency_discharge)
        held_out_profit = float(held_out_curve[-1])
        np.save(run_dir / f"held_out_2017_cumulative_profit_curve_{reward_kind}.npy", held_out_curve)
        print(f"{reward_kind} (seed={seed}): held-out 2017 profit = ${held_out_profit:,.2f}")

        results[reward_kind] = {"seed": seed, "training_final_profit": float(final_pass_profit),
                                 "held_out_2017_final_profit": held_out_profit}

        train_ax.plot(np.arange(len(train_curve)), train_curve,
                      label=f"{reward_kind.replace('_', ' ').title()} (seed={seed})",
                      color=colors[reward_kind], linewidth=1)
        eval_ax.plot(np.arange(len(held_out_curve)), held_out_curve,
                     label=f"{reward_kind.replace('_', ' ').title()} (seed={seed})",
                     color=colors[reward_kind], linewidth=1)

    train_ax.set_xlabel("Time (hour)")
    train_ax.set_ylabel("Cumulative profit ($)")
    train_ax.set_title("TRAINING profit: Reward 1 vs Reward 2, each on its OWN best HELD-OUT seed (2016, single pass)")
    train_ax.axhline(0, color="gray", linewidth=0.7)
    train_ax.legend()
    train_fig.tight_layout()
    train_plot_path = run_dir / "training_best_held_out_seed_per_reward_plot.png"
    train_fig.savefig(train_plot_path, dpi=150)
    print(f"Saved plot to {train_plot_path}")

    eval_ax.set_xlabel("Time (hour)")
    eval_ax.set_ylabel("Cumulative profit ($)")
    eval_ax.set_title("HELD-OUT 2017 profit: Reward 1 vs Reward 2, each on its OWN best HELD-OUT seed")
    eval_ax.axhline(0, color="gray", linewidth=0.7)
    eval_ax.legend()
    eval_fig.tight_layout()
    eval_plot_path = run_dir / "held_out_2017_best_held_out_seed_per_reward_plot.png"
    eval_fig.savefig(eval_plot_path, dpi=150)
    print(f"Saved plot to {eval_plot_path}")

    params_text = f"""Experiment 3: best seed PER REWARD, selected by HELD-OUT profit
====================================================================

Each reward function is trained on the seed that gave IT the highest
HELD-OUT (2017) profit -- the opposite selection criterion from Experiment
1 (which picks each reward's best TRAINING seed). Best held-out seeds were
found by cross-referencing Experiment 2's existing 100-trial paired sweep
(../20260810_154500_exp2_100seed_distribution/summary.json for seeds +
training profit, held_out_2017_eval_summary.json for held-out profit;
trial index i is the same seed in both files) rather than running a new
sweep.

Parameters (shared, except seed)
---------------------------------
data:                  {args.data}
test_data:             {args.test_data}
capacity_mwh:          {args.capacity_mwh}
max_rate_mw:           {args.max_rate_mw}
n_price_bins (M):      {args.n_price_bins}
price_bin_method:      {args.price_bin_method}
bin_calibration_hours: {args.bin_calibration_hours}
alpha:                 {args.alpha}
gamma:                 {args.gamma}
epsilon:               {args.epsilon}
smoothing (eta):       {args.smoothing}
efficiency_charge:     {args.efficiency_charge}
efficiency_discharge:  {args.efficiency_discharge}
n_passes:              {args.n_passes}

Results
-------
Reward 1: seed={results['reward_1']['seed']}
  held-out profit (2017, frozen greedy): ${results['reward_1']['held_out_2017_final_profit']:,.2f}  (highest Reward-1 held-out profit found in the 100-trial sweep)
  training profit (2016, online):        ${results['reward_1']['training_final_profit']:,.2f}
Reward 2: seed={results['reward_2']['seed']}
  held-out profit (2017, frozen greedy): ${results['reward_2']['held_out_2017_final_profit']:,.2f}  (highest Reward-2 held-out profit found in the 100-trial sweep)
  training profit (2016, online):        ${results['reward_2']['training_final_profit']:,.2f}

See training_best_held_out_seed_per_reward_plot.png for the training-time
curves and held_out_2017_best_held_out_seed_per_reward_plot.png for the
held-out curves. This is a single seed per reward (not an averaged
distribution) -- see Experiment 2 (100-seed distribution) for the
statistical version of this comparison.
"""
    (run_dir / "params.txt").write_text(params_text)
    print(f"Saved parameters to {run_dir / 'params.txt'}")

    summary = {"data": args.data, "test_data": args.test_data, "args": vars(args), "results": results,
               "note": "each reward trained on its own independently-best HELD-OUT seed, "
                       "found by cross-referencing Experiment 2's 100-trial sweep "
                       "(../20260810_154500_exp2_100seed_distribution/)"}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Run artefacts written to {run_dir}")


if __name__ == "__main__":
    main()
