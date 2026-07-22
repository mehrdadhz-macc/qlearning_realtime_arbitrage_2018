"""Train the Q-learning arbitrage policy on one price series (paper Sec. IV-B/C).

The paper trains online: a single pass through the price series IS the
training run, and cumulative profit during that pass is what Fig. 4 plots.
We do the same here, then (going beyond the paper) also freeze the learned
Q-table and evaluate it greedily on a held-out year in evaluate.py -- the
paper only ever reports in-sample training-time profit, which conflates
learning and evaluation; a genuine held-out check is a stronger test.

Usage:
    venv/bin/python3 train.py --data data/train/isone_rt_hourly_lmp_2016.csv --reward both
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data_loader import load_price_series
from src.environment import StorageArbitrageEnv
from src.qlearning_agent import QLearningAgent
from src.rewards import MovingAveragePrice, reward_1, reward_2


def run_training(prices, reward_kind, capacity_mwh, max_rate_mw, n_price_bins,
                  alpha, gamma, epsilon, smoothing, seed):
    env = StorageArbitrageEnv(
        prices, capacity_mwh=capacity_mwh, max_rate_mw=max_rate_mw, n_price_bins=n_price_bins,
    )
    agent = QLearningAgent(env.n_price_bins, env.n_energy_bins, alpha=alpha, gamma=gamma,
                            epsilon=epsilon, seed=seed)
    avg_price = MovingAveragePrice(smoothing=smoothing)

    state = env.reset()
    cumulative_profit = 0.0
    history = []
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, price, c, d, c_tilde, d_tilde, done = env.step(action)

        p_bar = avg_price.update(price)
        if reward_kind == "reward_1":
            r = reward_1(price, c, d)
        else:
            r = reward_2(price, p_bar, c, d)

        agent.update(state, action, r, next_state)

        profit = -price * c + price * d
        cumulative_profit += profit
        history.append({"t": env.t - 1, "price": price, "action": action,
                         "c": c, "d": d, "profit": profit,
                         "cumulative_profit": cumulative_profit})

        state = next_state
        if done:
            break

    return agent, history, cumulative_profit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train/isone_rt_hourly_lmp_2016.csv")
    parser.add_argument("--reward", choices=["reward_1", "reward_2", "both"], default="both")
    parser.add_argument("--capacity-mwh", type=float, default=8.0)
    parser.add_argument("--max-rate-mw", type=float, default=1.0)
    parser.add_argument("--n-price-bins", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5, help="Q-learning rate (paper Algorithm 1)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Epsilon-greedy exploration prob")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Reward 2 moving-average eta (Eq. 6)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None, help="Override outputs/runs/<timestamp>")
    args = parser.parse_args()

    _, prices = load_price_series(args.data)
    print(f"Loaded {len(prices)} hourly prices from {args.data} "
          f"(min={prices.min():.2f}, max={prices.max():.2f}, mean={prices.mean():.2f} $/MWh)")

    run_dir = Path(args.out_dir) if args.out_dir else Path("outputs/runs") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    reward_kinds = ["reward_1", "reward_2"] if args.reward == "both" else [args.reward]
    summary = {"data": args.data, "n_hours": len(prices), "args": vars(args), "results": {}}

    for reward_kind in reward_kinds:
        print(f"\nTraining with {reward_kind} ...")
        agent, history, cumulative_profit = run_training(
            prices, reward_kind, args.capacity_mwh, args.max_rate_mw, args.n_price_bins,
            args.alpha, args.gamma, args.epsilon, args.smoothing, args.seed,
        )
        print(f"  {reward_kind}: cumulative training profit = ${cumulative_profit:,.2f}")

        agent.save(run_dir / f"q_table_{reward_kind}.npy")
        # Price bins are fit from THIS series' own min/max (env construction
        # in run_training). evaluate.py must reuse these exact edges on the
        # held-out series instead of re-fitting from the test set's own
        # range, or a raw price would map to a different bin index at eval
        # time than it did during training and the Q-table would be read
        # against the wrong state.
        env_for_edges = StorageArbitrageEnv(prices, capacity_mwh=args.capacity_mwh,
                                             max_rate_mw=args.max_rate_mw, n_price_bins=args.n_price_bins)
        np.save(run_dir / f"price_bin_edges_{reward_kind}.npy", env_for_edges.price_bin_edges)
        np.save(run_dir / f"history_{reward_kind}.npy",
                np.array([(h["t"], h["price"], h["action"], h["c"], h["d"],
                           h["profit"], h["cumulative_profit"]) for h in history]))
        summary["results"][reward_kind] = {"cumulative_profit": cumulative_profit}

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nRun artefacts written to {run_dir}")


if __name__ == "__main__":
    main()
