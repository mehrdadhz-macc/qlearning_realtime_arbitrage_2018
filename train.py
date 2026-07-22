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
                  alpha, gamma, epsilon, smoothing, seed, price_bin_method,
                  bin_calibration_hours, efficiency_charge, efficiency_discharge,
                  reward_efficiency_aware):
    # Fit price bins causally: only the first `bin_calibration_hours` of the
    # series are used to set edges, not the whole year (see
    # StorageArbitrageEnv's module docstring for why that matters).
    calibration_prices = prices[:min(bin_calibration_hours, len(prices))]
    env = StorageArbitrageEnv(
        prices, capacity_mwh=capacity_mwh, max_rate_mw=max_rate_mw, n_price_bins=n_price_bins,
        price_bin_method=price_bin_method, bin_fit_prices=calibration_prices,
        efficiency_charge=efficiency_charge, efficiency_discharge=efficiency_discharge,
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
            r = reward_1(price, c, d, eta_c=env.eta_c, eta_d=env.eta_d,
                         efficiency_aware=reward_efficiency_aware)
        else:
            r = reward_2(price, p_bar, c, d, eta_c=env.eta_c, eta_d=env.eta_d,
                          efficiency_aware=reward_efficiency_aware)

        agent.update(state, action, r, next_state)

        profit = env.true_profit(price, c, d)  # AMP objective (Sec. II), not the shaped reward
        cumulative_profit += profit
        history.append({"t": env.t - 1, "price": price, "action": action,
                         "c": c, "d": d, "profit": profit,
                         "cumulative_profit": cumulative_profit})

        state = next_state
        if done:
            break

    return agent, history, cumulative_profit, env.price_bin_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train/isone_rt_hourly_lmp_2016.csv")
    parser.add_argument("--reward", choices=["reward_1", "reward_2", "both"], default="both")
    parser.add_argument("--capacity-mwh", type=float, default=8.0)
    parser.add_argument("--max-rate-mw", type=float, default=1.0)
    parser.add_argument("--n-price-bins", type=int, default=10)
    parser.add_argument("--price-bin-method", choices=["equal_width", "quantile"], default="quantile",
                         help="equal_width = paper's literal 'M even price intervals'; quantile (default) "
                              "avoids wasting most bins on rarely-visited price spikes")
    parser.add_argument("--bin-calibration-hours", type=int, default=24 * 30,
                         help="Fit price bins from only the first N hours (causal), not the whole series")
    parser.add_argument("--alpha", type=float, default=0.5, help="Q-learning rate (paper Algorithm 1)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Epsilon-greedy exploration prob")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Reward 2 moving-average eta (Eq. 6)")
    parser.add_argument("--efficiency-charge", type=float, default=1.0, help="eta_c (Sec. II AMP objective)")
    parser.add_argument("--efficiency-discharge", type=float, default=1.0, help="eta_d (Sec. II AMP objective)")
    parser.add_argument("--reward-efficiency-aware", action="store_true",
                         help="Fold eta_c/eta_d into Reward 1/2 too (paper's literal Sec. III-C formulas omit them)")
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
        agent, history, cumulative_profit, price_bin_edges = run_training(
            prices, reward_kind, args.capacity_mwh, args.max_rate_mw, args.n_price_bins,
            args.alpha, args.gamma, args.epsilon, args.smoothing, args.seed,
            args.price_bin_method, args.bin_calibration_hours,
            args.efficiency_charge, args.efficiency_discharge, args.reward_efficiency_aware,
        )
        print(f"  {reward_kind}: cumulative training profit = ${cumulative_profit:,.2f}")

        agent.save(run_dir / f"q_table_{reward_kind}.npy")
        # evaluate.py must reuse these EXACT edges (fit causally from the
        # training series' calibration prefix) on the held-out series rather
        # than re-fitting from the test set's own range, or a raw price would
        # map to a different bin index at eval time than it did in training.
        np.save(run_dir / f"price_bin_edges_{reward_kind}.npy", price_bin_edges)
        np.save(run_dir / f"history_{reward_kind}.npy",
                np.array([(h["t"], h["price"], h["action"], h["c"], h["d"],
                           h["profit"], h["cumulative_profit"]) for h in history]))
        summary["results"][reward_kind] = {"cumulative_profit": cumulative_profit}

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nRun artefacts written to {run_dir}")


if __name__ == "__main__":
    main()
