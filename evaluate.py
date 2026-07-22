"""Greedily evaluate a trained Q-table on a held-out price series.

The paper itself never holds out a test year -- Fig. 4's "results" ARE the
training run. Here we additionally freeze the learned Q-table (epsilon=0, no
further updates) and replay it on a year the agent never trained on, which is
a stronger and more standard check of whether the learned policy generalizes.

Usage:
    venv/bin/python3 evaluate.py --run outputs/runs/<timestamp> --data data/test/isone_rt_hourly_lmp_2017.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_price_series
from src.environment import StorageArbitrageEnv
from src.qlearning_agent import QLearningAgent


def latest_run_dir():
    runs = sorted(Path("outputs/runs").glob("*/"))
    if not runs:
        raise SystemExit("No runs found under outputs/runs/. Run train.py first.")
    return runs[-1]


def greedy_rollout(prices, q_table, capacity_mwh, max_rate_mw, price_bin_edges,
                    efficiency_charge=1.0, efficiency_discharge=1.0):
    env = StorageArbitrageEnv(prices, capacity_mwh=capacity_mwh, max_rate_mw=max_rate_mw,
                               price_bin_edges=price_bin_edges,
                               efficiency_charge=efficiency_charge, efficiency_discharge=efficiency_discharge)
    agent = QLearningAgent(env.n_price_bins, env.n_energy_bins, epsilon=0.0)
    agent.q = q_table

    state = env.reset()
    cumulative_profit = 0.0
    curve = []
    done = False
    while not done:
        action = agent.select_action(state, greedy=True)
        next_state, price, c, d, c_tilde, d_tilde, done = env.step(action)
        cumulative_profit += env.true_profit(price, c, d)  # AMP objective (Sec. II)
        curve.append(cumulative_profit)
        state = next_state
    return np.array(curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="Run directory (default: most recent under outputs/runs)")
    parser.add_argument("--data", default="data/test/isone_rt_hourly_lmp_2017.csv")
    parser.add_argument("--capacity-mwh", type=float, default=8.0)
    parser.add_argument("--max-rate-mw", type=float, default=1.0)
    parser.add_argument("--efficiency-charge", type=float, default=1.0,
                         help="Must match the value train.py was run with for a coherent comparison")
    parser.add_argument("--efficiency-discharge", type=float, default=1.0,
                         help="Must match the value train.py was run with for a coherent comparison")
    args = parser.parse_args()

    run_dir = Path(args.run) if args.run else latest_run_dir()
    _, prices = load_price_series(args.data)
    print(f"Evaluating run {run_dir} on {len(prices)} held-out hours from {args.data}")

    curves = {}
    for reward_kind in ("reward_1", "reward_2"):
        q_path = run_dir / f"q_table_{reward_kind}.npy"
        edges_path = run_dir / f"price_bin_edges_{reward_kind}.npy"
        if not q_path.exists() or not edges_path.exists():
            continue
        q_table = np.load(q_path)
        price_bin_edges = np.load(edges_path)
        # Reuse the TRAINING series' bin edges, not edges refit from this
        # held-out series -- see train.py's comment on why that matters.
        # Test-set prices outside the train range simply clip to the
        # nearest edge bin (StorageArbitrageEnv.price_bin), which is
        # expected, imperfect-but-safe behavior for out-of-distribution
        # prices rather than a crash.
        curve = greedy_rollout(prices, q_table, args.capacity_mwh, args.max_rate_mw, price_bin_edges,
                                args.efficiency_charge, args.efficiency_discharge)
        curves[reward_kind] = curve
        print(f"  {reward_kind}: held-out cumulative profit = ${curve[-1]:,.2f}")

    plt.figure(figsize=(9, 5))
    for reward_kind, curve in curves.items():
        plt.plot(curve, label=reward_kind)
    plt.xlabel("Time (hour)")
    plt.ylabel("Cumulative profit ($)")
    plt.title("Held-out evaluation (greedy policy, frozen Q-table)")
    plt.legend()
    plt.tight_layout()

    # Kept inside the run's own directory (alongside q_table_*.npy,
    # summary.json, etc.) rather than a separate top-level outputs/eval_plots/
    # -- everything about one run's performance lives in one place.
    out_path = run_dir / "eval_plot.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    (run_dir / "eval_summary.json").write_text(json.dumps(
        {"data": args.data, "results": {k: float(v[-1]) for k, v in curves.items()}}, indent=2))


if __name__ == "__main__":
    main()
