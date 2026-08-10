"""Experiment 4: run the OMG baseline (src/omg_baseline.py) on the 2017
held-out year and plot its cumulative profit -- separate from Experiments
1-3, which are all about the Q-learning agent's seed-to-seed variance. OMG
needs no seed (deterministic, one-time causal calibration from the first 30
days of 2016 training data, then a fixed threshold rule -- see the module
docstring in src/omg_baseline.py for why this isn't "training" in the ML
sense), so there's nothing to average over here: one run is the whole
result.

Reports both the paper's two battery configs (8 MWh, 1 MW and 2 MW
charge/discharge rate) on the same axes.

Usage:
    venv/bin/python3 scripts/data_plots/plot_omg_baseline_2017.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_loader import load_price_series
from src.omg_baseline import fit_omg_parameters
from evaluate_omg_baseline import run_omg_rollout
from scripts.data_plots._plot_helpers import make_price_profit_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", default="data/train/isone_rt_hourly_lmp_2016.csv",
                         help="Used only for causal calibration (first --bin-calibration-hours), not rolled out")
    parser.add_argument("--test-data", default="data/test/isone_rt_hourly_lmp_2017.csv")
    parser.add_argument("--capacity-mwh", type=float, default=8.0)
    parser.add_argument("--max-rate-mw", type=float, nargs="+", default=[1.0, 2.0],
                         help="Paper Sec. IV-C reports both 1MW and 2MW cases")
    parser.add_argument("--bin-calibration-hours", type=int, default=24 * 30)
    parser.add_argument("--efficiency-charge", type=float, default=1.0)
    parser.add_argument("--efficiency-discharge", type=float, default=1.0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    _, train_prices = load_price_series(args.train_data)
    _, test_prices = load_price_series(args.test_data)
    calibration_prices = train_prices[:min(args.bin_calibration_hours, len(train_prices))]

    run_dir = Path(args.out_dir) if args.out_dir else Path("outputs/runs") / (
        time.strftime("%Y%m%d_%H%M%S") + "_exp4_omg_baseline_2017")
    run_dir.mkdir(parents=True, exist_ok=True)

    colors = {"1mw": "tab:purple", "2mw": "tab:orange"}
    curves = {}
    results = {}

    for max_rate_mw in args.max_rate_mw:
        params = fit_omg_parameters(calibration_prices, e_min=0.0, e_max=args.capacity_mwh,
                                     c_max=max_rate_mw, d_max=max_rate_mw)
        curve = run_omg_rollout(test_prices, params, args.capacity_mwh, max_rate_mw,
                                 args.efficiency_charge, args.efficiency_discharge)
        final_profit = float(curve[-1])
        print(f"{max_rate_mw:g}MW: held-out 2017 profit = ${final_profit:,.2f} "
              f"(p_min={params.p_min:.2f}, p_max={params.p_max:.2f}, W={params.weight:.6g}, Gamma={params.gamma_shift:.6g})")

        key = f"{max_rate_mw:g}mw"
        np.save(run_dir / f"held_out_2017_cumulative_profit_curve_{key}.npy", curve)
        curves[f"{max_rate_mw:g} MW"] = (key, curve)
        results[key] = {
            "max_rate_mw": max_rate_mw, "capacity_mwh": args.capacity_mwh,
            "p_min": params.p_min, "p_max": params.p_max,
            "weight": params.weight, "gamma_shift": params.gamma_shift,
            "held_out_2017_final_profit": final_profit,
        }

    fig = make_price_profit_figure(
        test_prices, curves, colors,
        f"OMG baseline, held-out 2017 cumulative profit ({args.capacity_mwh:g} MWh battery)",
        )
    plot_path = run_dir / "held_out_2017_omg_baseline_plot.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")

    params_text = f"""Experiment 4: OMG baseline, held-out 2017

Runs the deterministic OMG baseline (src/omg_baseline.py, Qin et al. 2016,
Wang & Zhang Sec. IV-C's comparison) directly on the 2017 held-out year --
kept separate from Experiments 1-3, which are all about the Q-learning
agent's seed-to-seed variance. OMG has no seed: its only "fitting" step is
a one-time causal calibration of [p_min, p_max] from the first
{args.bin_calibration_hours} hours of {args.train_data} (not iterative
training -- see src/omg_baseline.py's module docstring), so a single run
is the complete, reproducible result.

Parameters
----------
train_data (calibration only): {args.train_data}
test_data (rolled out):        {args.test_data}
bin_calibration_hours:         {args.bin_calibration_hours}
capacity_mwh:                  {args.capacity_mwh}
max_rate_mw:                   {args.max_rate_mw}
efficiency_charge:             {args.efficiency_charge}
efficiency_discharge:          {args.efficiency_discharge}

Results
-------
"""
    for key, r in results.items():
        params_text += (f"{r['max_rate_mw']:g} MW: held-out 2017 profit = ${r['held_out_2017_final_profit']:,.2f}  "
                         f"(p_min={r['p_min']:.2f}, p_max={r['p_max']:.2f}, W={r['weight']:.6g}, Gamma={r['gamma_shift']:.6g})\n")
    params_text += "\nSee held_out_2017_omg_baseline_plot.png for the cumulative profit curves.\n"

    (run_dir / "params.txt").write_text(params_text)
    print(f"Saved parameters to {run_dir / 'params.txt'}")

    summary = {"train_data": args.train_data, "test_data": args.test_data, "args": vars(args), "results": results}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Run artefacts written to {run_dir}")


if __name__ == "__main__":
    main()
