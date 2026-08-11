"""Train Reward 1 and Reward 2 each on ITS OWN best seed (not a shared one),
plot both TRAINING cumulative-profit curves together, then freeze each
learned Q-table and evaluate it greedily on the held-out 2017 year too.

Unlike plot_best_seed_reward_comparison.py (one seed used for both rewards,
so the two curves share the same exploration trajectory), this script picks
the highest-profit seed independently for each reward function, from the
same paired 50-trial sweep this project already ran at its best confirmed
single-pass config (eta=0.001, M=5, n_passes=1):
  - outputs/runs/sweeps/bins_p1/m5           (Reward 2, seed 42 -> 50 trial seeds)
  - outputs/runs/sweeps/bins_p1_reward1/m5   (Reward 1, same seed 42 -> same 50 trial seeds)
Reward 1's best trial used seed 1688060240 ($5,956.68); Reward 2's best
trial used seed 1455819991 ($7,414.94). Each reward is retrained here on
its own best seed, independently.

All output files are prefixed training_/held_out_2017_ so the two kinds of
profit (accumulated online during learning vs. a frozen greedy policy
replayed on a year the agent never trained on) can't be confused.

Usage:
    venv/bin/python3 scripts/data_plots/plot_best_seed_per_reward.py
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_loader import load_price_series
from train import run_training
from evaluate import greedy_rollout
from scripts.data_plots._plot_helpers import make_price_profit_figure

BEST_SEED_REWARD_1 = 1688060240  # highest Reward-1 profit ($5,956.68) in outputs/runs/sweeps/bins_p1_reward1/m5
BEST_SEED_REWARD_2 = 1455819991  # highest Reward-2 profit ($7,414.94) in outputs/runs/sweeps/bins_p1/m5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train/isone_rt_hourly_lmp_2016.csv")
    parser.add_argument("--test-data", default="data/test/isone_rt_hourly_lmp_2017.csv")
    parser.add_argument("--seed-reward1", type=int, default=BEST_SEED_REWARD_1)
    parser.add_argument("--seed-reward2", type=int, default=BEST_SEED_REWARD_2)
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

    rate_tag = f"{args.max_rate_mw:g}mw"
    run_dir = Path(args.out_dir) if args.out_dir else Path("outputs/runs") / (
        time.strftime("%Y%m%d_%H%M%S") + f"_exp1_{rate_tag}_best_seed_per_reward")
    run_dir.mkdir(parents=True, exist_ok=True)

    seeds = {"reward_1": args.seed_reward1, "reward_2": args.seed_reward2}
    colors = {"reward_1": "tab:red", "reward_2": "tab:blue"}
    results = {}
    train_curves, eval_curves = {}, {}

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

        label = f"{reward_kind.replace('_', ' ').title()} (seed={seed})"
        train_curves[label] = (reward_kind, train_curve)
        eval_curves[label] = (reward_kind, held_out_curve)

    train_fig = make_price_profit_figure(
        prices, train_curves, colors,
        "TRAINING profit: Reward 1 vs Reward 2, each on its OWN best seed (2016, single pass)")
    train_plot_path = run_dir / "training_best_seed_per_reward_plot.png"
    train_fig.savefig(train_plot_path, dpi=150)
    print(f"Saved plot to {train_plot_path}")

    eval_fig = make_price_profit_figure(
        test_prices, eval_curves, colors,
        "HELD-OUT 2017 profit: same Q-tables, frozen greedy policy")
    eval_plot_path = run_dir / "held_out_2017_best_seed_per_reward_plot.png"
    eval_fig.savefig(eval_plot_path, dpi=150)
    print(f"Saved plot to {eval_plot_path}")

    params_text = f"""Experiment 1: best seed PER REWARD (not a shared seed)
=========================================================

Each reward function is trained on the seed that gave IT the highest
TRAINING profit, independently -- unlike the earlier best-seed comparison,
these two curves do NOT share an exploration trajectory. Best seeds were
found from a paired 50-trial sweep at this project's best confirmed
single-pass config (eta=0.001, M=5, n_passes=1):
  outputs/runs/sweeps/bins_p1/m5           (Reward 2 sweep)
  outputs/runs/sweeps/bins_p1_reward1/m5   (Reward 1 sweep, same --seed 42
                                             so the 50 generated trial
                                             seeds are identical to Reward
                                             2's, but each reward's BEST
                                             individual seed differs)

Each trained Q-table is then ALSO frozen (epsilon=0, no further learning)
and replayed greedily on data/test/isone_rt_hourly_lmp_2017.csv, a year
the agent never trained on -- an additional check beyond what the paper
itself does (the paper never holds out a test year; see README).

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
  training profit (2016, online):        ${results['reward_1']['training_final_profit']:,.2f}  (highest Reward-1 profit found in the 50-trial sweep)
  held-out profit (2017, frozen greedy): ${results['reward_1']['held_out_2017_final_profit']:,.2f}
Reward 2: seed={results['reward_2']['seed']}
  training profit (2016, online):        ${results['reward_2']['training_final_profit']:,.2f}  (highest Reward-2 profit found in the 50-trial sweep)
  held-out profit (2017, frozen greedy): ${results['reward_2']['held_out_2017_final_profit']:,.2f}

See training_best_seed_per_reward_plot.png for the training-time curves and
held_out_2017_best_seed_per_reward_plot.png for the held-out curves. This is
a single seed per reward (not an averaged distribution), so held-out
generalization here is illustrative of these two specific Q-tables, not a
statistical claim -- see Experiment 2 (100-seed distribution) for that.
"""
    (run_dir / "params.txt").write_text(params_text)
    print(f"Saved parameters to {run_dir / 'params.txt'}")

    summary = {"data": args.data, "test_data": args.test_data, "args": vars(args), "results": results,
               "note": "each reward trained on its own independently-best TRAINING seed, "
                       "found in outputs/runs/sweeps/bins_p1/m5 (reward_2) and "
                       "outputs/runs/sweeps/bins_p1_reward1/m5 (reward_1); held-out 2017 "
                       "profit is that SAME seed's Q-table frozen and replayed greedily"}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Run artefacts written to {run_dir}")


if __name__ == "__main__":
    main()
