"""Run the OMG baseline (src/omg_baseline.py) on both the training year
(in-sample, paralleling train.py's own reporting) and the held-out test
year (paralleling evaluate.py) -- this project's answer to Wang & Zhang's
Sec. IV-C comparison ("our algorithm" [Reward 2] vs. Qin et al.'s online
modified greedy baseline).

Unlike Q-learning, OMG is deterministic given its (Gamma, W) parameters --
no epsilon-greedy exploration, no seeds, no --n-trials needed. One run is
the whole result.

Usage:
    venv/bin/python3 evaluate_omg_baseline.py \
        --train-data data/train/isone_rt_hourly_lmp_2016.csv \
        --test-data data/test/isone_rt_hourly_lmp_2017.csv \
        --max-rate-mw 1.0 2.0
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_price_series
from src.environment import StorageArbitrageEnv
from src.omg_baseline import fit_omg_parameters, select_omg_action


def run_omg_rollout(prices, params, capacity_mwh, max_rate_mw,
                     efficiency_charge, efficiency_discharge, e_min=0.0):
    env = StorageArbitrageEnv(prices, capacity_mwh=capacity_mwh, max_rate_mw=max_rate_mw, e_min=e_min,
                               efficiency_charge=efficiency_charge, efficiency_discharge=efficiency_discharge)
    env.reset()
    cumulative_profit = 0.0
    curve = np.empty(env.T, dtype=float)
    done = False
    while not done:
        price = env.prices[env.t]
        action = select_omg_action(env.energy, price, params, env.HOLD, env.CHARGE, env.DISCHARGE)
        _, price_taken, c, d, c_tilde, d_tilde, done = env.step(action)
        cumulative_profit += env.true_profit(price_taken, c, d)  # AMP objective (Sec. II)
        curve[env.t - 1] = cumulative_profit
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default="data/train/isone_rt_hourly_lmp_2016.csv")
    parser.add_argument("--test-data", default="data/test/isone_rt_hourly_lmp_2017.csv")
    parser.add_argument("--capacity-mwh", type=float, default=8.0)
    parser.add_argument("--max-rate-mw", type=float, nargs="+", default=[1.0, 2.0],
                         help="Paper Sec. IV-C reports both 1MW and 2MW cases")
    parser.add_argument("--bin-calibration-hours", type=int, default=24 * 30,
                         help="Causal prefix of --train-data used to estimate [p_min, p_max] "
                              "(same no-lookahead principle as train.py's price-bin fitting)")
    parser.add_argument("--efficiency-charge", type=float, default=1.0)
    parser.add_argument("--efficiency-discharge", type=float, default=1.0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    _, train_prices = load_price_series(args.train_data)
    _, test_prices = load_price_series(args.test_data)
    print(f"Train: {len(train_prices)} hours from {args.train_data}")
    print(f"Test:  {len(test_prices)} hours from {args.test_data}")

    calibration_prices = train_prices[:min(args.bin_calibration_hours, len(train_prices))]
    print(f"Calibrating [p_min, p_max] from the first {len(calibration_prices)} training hours "
          f"(causal, no lookahead)")

    run_dir = Path(args.out_dir) if args.out_dir else Path("outputs/runs") / (time.strftime("%Y%m%d_%H%M%S") + "_omg_baseline")
    run_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    fig, axes = plt.subplots(1, len(args.max_rate_mw), figsize=(7 * len(args.max_rate_mw), 5), squeeze=False)

    for i, max_rate_mw in enumerate(args.max_rate_mw):
        params = fit_omg_parameters(calibration_prices, e_min=0.0, e_max=args.capacity_mwh,
                                     c_max=max_rate_mw, d_max=max_rate_mw)
        print(f"\n{args.capacity_mwh}MWh / {max_rate_mw}MW: "
              f"p_min={params.p_min:.2f}, p_max={params.p_max:.2f}, "
              f"W={params.weight:.6g}, Gamma={params.gamma_shift:.6g}")

        train_curve = run_omg_rollout(train_prices, params, args.capacity_mwh, max_rate_mw,
                                       args.efficiency_charge, args.efficiency_discharge)
        test_curve = run_omg_rollout(test_prices, params, args.capacity_mwh, max_rate_mw,
                                      args.efficiency_charge, args.efficiency_discharge)
        print(f"  in-sample training profit (2016-style): ${train_curve[-1]:,.2f}")
        print(f"  held-out profit (2017-style):           ${test_curve[-1]:,.2f}")

        key = f"{max_rate_mw:g}mw"
        results[key] = {
            "max_rate_mw": max_rate_mw,
            "capacity_mwh": args.capacity_mwh,
            "p_min": params.p_min, "p_max": params.p_max,
            "weight": params.weight, "gamma_shift": params.gamma_shift,
            "train_profit": float(train_curve[-1]),
            "test_profit": float(test_curve[-1]),
        }

        ax = axes[0][i]
        ax.plot(np.arange(len(train_curve)), train_curve, label="OMG, in-sample (train year)")
        ax.plot(np.arange(len(test_curve)), test_curve, label="OMG, held-out (test year)")
        ax.set_xlabel("Time (hour)")
        ax.set_ylabel("Cumulative profit ($)")
        ax.set_title(f"{args.capacity_mwh}MWh-{max_rate_mw:g}MW battery")
        ax.legend()

    fig.tight_layout()
    out_plot = run_dir / "omg_baseline_plot.png"
    fig.savefig(out_plot, dpi=150)
    print(f"\nSaved plot to {out_plot}")

    (run_dir / "omg_baseline_summary.json").write_text(json.dumps({
        "train_data": args.train_data, "test_data": args.test_data,
        "bin_calibration_hours": args.bin_calibration_hours, "args": vars(args),
        "results": results,
    }, indent=2))
    print(f"Run artefacts written to {run_dir}")


if __name__ == "__main__":
    main()
