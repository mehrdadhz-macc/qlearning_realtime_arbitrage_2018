"""Train the Q-learning arbitrage policy on one price series (paper Sec. IV-B/C).

The paper trains online: a single pass through the price series IS the
training run, and cumulative profit during that pass is what Fig. 4 plots.
We do the same here, then (going beyond the paper) also freeze the learned
Q-table and evaluate it greedily on a held-out year in evaluate.py -- the
paper only ever reports in-sample training-time profit, which conflates
learning and evaluation; a genuine held-out check is a stronger test.

Q-learning's epsilon-greedy exploration is random, and (as demonstrated in
this project's own README) that randomness swings results a lot from one
seed to the next -- a single run's profit number is one sample, not "the"
result. --n-trials runs several independent trials per reward kind, saves
every trial's Q-table separately under trial_NN/, and reports the mean and
std of the profit across trials -- the expected-value estimate this
project's results should actually be judged on, not any single trial's
number. --seed doesn't feed into any trial directly; it seeds an RNG that
GENERATES each trial's own (effectively random-looking) seed, so the same
--seed always reproduces the same set of trial seeds, but two adjacent
trials never differ by a suspiciously simple +1.

Usage:
    venv/bin/python3 train.py --data data/train/isone_rt_hourly_lmp_2016.csv --reward both --n-trials 10
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
                  reward_efficiency_aware, n_passes=1):
    # Fit price bins causally: only the first `bin_calibration_hours` of the
    # ORIGINAL (un-repeated) series are used to set edges (see
    # StorageArbitrageEnv's module docstring for why that matters) --
    # computed before n_passes repetition so more passes doesn't change
    # what "causal" means here.
    calibration_prices = prices[:min(bin_calibration_hours, len(prices))]

    # n_passes > 1: Algorithm 1 never states how many times the storage
    # sees the data (see this project's README's "Next step") -- one
    # linear pass is the literal "online" reading, but Q-learning over a
    # (price_bin, energy_bin) table with only ~91k possible (state,action)
    # visits in a single 2016 pass may not have converged by the time
    # profit is totaled. Repeating the series lets the SAME single
    # continuous episode (energy level and the moving average both carry
    # over across the repeat boundary, exactly like the ordinary
    # wrap-free single pass) run longer before judgment; the reported
    # number stays comparable to a true single year by taking only the
    # FINAL lap's profit, not the sum across every repeat.
    training_prices = np.tile(prices, n_passes) if n_passes > 1 else prices

    env = StorageArbitrageEnv(
        training_prices, capacity_mwh=capacity_mwh, max_rate_mw=max_rate_mw, n_price_bins=n_price_bins,
        price_bin_method=price_bin_method, bin_fit_prices=calibration_prices,
        efficiency_charge=efficiency_charge, efficiency_discharge=efficiency_discharge,
    )
    agent = QLearningAgent(env.n_price_bins, env.n_energy_bins, alpha=alpha, gamma=gamma,
                            epsilon=epsilon, seed=seed)
    avg_price = MovingAveragePrice(smoothing=smoothing)

    state = env.reset()
    cumulative_profit = 0.0
    final_pass_start_profit = None  # cumulative_profit at the start of the last lap
    final_pass_start_t = (n_passes - 1) * len(prices)
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
        if env.t - 1 == final_pass_start_t:
            final_pass_start_profit = cumulative_profit
        cumulative_profit += profit
        history.append({"t": env.t - 1, "price": price, "action": action,
                         "c": c, "d": d, "profit": profit,
                         "cumulative_profit": cumulative_profit})

        state = next_state
        if done:
            break

    final_pass_profit = cumulative_profit - (final_pass_start_profit or 0.0)
    return agent, history, cumulative_profit, final_pass_profit, env.price_bin_edges


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
    parser.add_argument("--seed", type=int, default=0,
                         help="Seeds the RNG that GENERATES each trial's own seed (not used directly "
                              "as a trial seed) -- same --seed always reproduces the same set of "
                              "trial seeds, but they aren't a predictable seed/seed+1/seed+2 sequence")
    parser.add_argument("--n-trials", type=int, default=1,
                         help="Independent trials per reward kind, each with its own seed; "
                              "results are reported as mean +/- std across trials")
    parser.add_argument("--n-passes", type=int, default=1,
                         help="Repeat the price series this many times before totaling profit (Algorithm 1 "
                              "never states how many passes -- 1 is the literal 'online' reading). Reported "
                              "profit is only the FINAL pass's, comparable across --n-passes values; the "
                              "earlier passes just give the Q-table more time to converge first.")
    parser.add_argument("--out-dir", default=None, help="Override outputs/runs/<timestamp>")
    args = parser.parse_args()

    _, prices = load_price_series(args.data)
    print(f"Loaded {len(prices)} hourly prices from {args.data} "
          f"(min={prices.min():.2f}, max={prices.max():.2f}, mean={prices.mean():.2f} $/MWh)")

    run_dir = Path(args.out_dir) if args.out_dir else Path("outputs/runs") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    reward_kinds = ["reward_1", "reward_2"] if args.reward == "both" else [args.reward]
    summary = {"data": args.data, "n_hours": len(prices), "args": vars(args), "results": {}}

    # --seed controls the RNG that GENERATES each trial's seed, rather than
    # serving as a trial seed itself -- generated once here (not per reward
    # kind) so reward_1's trial i and reward_2's trial i share the same
    # seed, a paired comparison that reduces variance attributable to pure
    # exploration-randomness luck when comparing the two reward functions.
    trial_seeds = np.random.default_rng(args.seed).integers(0, 2**31 - 1, size=args.n_trials).tolist()

    for reward_kind in reward_kinds:
        print(f"\nTraining {reward_kind} -- {args.n_trials} trial(s), seeds {trial_seeds}")
        trial_profits = []
        price_bin_edges = None

        for trial, seed in enumerate(trial_seeds):
            agent, history, cumulative_profit, final_pass_profit, edges = run_training(
                prices, reward_kind, args.capacity_mwh, args.max_rate_mw, args.n_price_bins,
                args.alpha, args.gamma, args.epsilon, args.smoothing, seed,
                args.price_bin_method, args.bin_calibration_hours,
                args.efficiency_charge, args.efficiency_discharge, args.reward_efficiency_aware,
                args.n_passes,
            )
            pass_text = f" (final pass of {args.n_passes}; {args.n_passes} total-passes profit=${cumulative_profit:,.2f})" if args.n_passes > 1 else ""
            print(f"  trial {trial} (seed={seed}): training profit = ${final_pass_profit:,.2f}{pass_text}")

            # Every trial's own model, kept separately -- these are genuinely
            # different Q-tables (different exploration trajectories), not
            # redundant copies of the same thing.
            trial_dir = run_dir / f"trial_{trial:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            agent.save(trial_dir / f"q_table_{reward_kind}.npy")
            np.save(trial_dir / f"history_{reward_kind}.npy",
                    np.array([(h["t"], h["price"], h["action"], h["c"], h["d"],
                               h["profit"], h["cumulative_profit"]) for h in history]))
            trial_profits.append(final_pass_profit)
            price_bin_edges = edges  # identical across trials (fit_price_bin_edges takes no RNG)

        mean_profit = float(np.mean(trial_profits))
        std_profit = float(np.std(trial_profits))
        print(f"  {reward_kind}: mean training profit over {args.n_trials} trial(s) = "
              f"${mean_profit:,.2f} (std ${std_profit:,.2f})")

        # Bin edges are a property of the price data + calibration window, not
        # of any one trial's seed, so they're saved once per reward kind at
        # the run level, not duplicated into every trial_NN/ directory.
        np.save(run_dir / f"price_bin_edges_{reward_kind}.npy", price_bin_edges)
        summary["results"][reward_kind] = {
            "n_trials": args.n_trials,
            "n_passes": args.n_passes,
            "seeds": trial_seeds,
            "trial_final_pass_profits": trial_profits,
            "mean_final_pass_profit": mean_profit,
            "std_final_pass_profit": std_profit,
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nRun artefacts written to {run_dir}")


if __name__ == "__main__":
    main()
