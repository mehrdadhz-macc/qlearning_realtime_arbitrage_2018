# Experiment 6: 100-seed paired distribution, 2 MW rate (mirrors Experiment 2)

## Intuition

Same design as [Experiment 2](../20260810_154500_exp2_100seed_distribution/):
100 paired trials (same seed used for both rewards in each trial), same
hyperparameters (`--smoothing 0.001 --n-price-bins 5 --n-passes 1`), but
at the paper's other battery configuration -- 8 MWh, **2 MW** rate instead
of 1 MW.

## Finding 1: training still favors Reward 2, slightly less decisively than at 1 MW

| | mean | std | 95% CI |
|---|---|---|---|
| Reward 1 | $2,368.22 | $3,665.62 | -- |
| Reward 2 | $4,249.79 | $3,476.44 | -- |

Paired diff (Reward 2 - Reward 1): mean +$2,076.56, paired **t=18.07**,
Reward 2 wins 99/100 (vs. 100/100 at 1 MW, t=23.54 -- still overwhelming,
just not literally universal).

## Finding 2: held-out data flips DECISIVELY against Reward 2 at 2 MW -- not just non-significant, but significantly reversed

| | mean | std | 95% CI |
|---|---|---|---|
| Reward 1 | $11,554.37 | $8,791 | [$9,831, $13,277] |
| Reward 2 | $3,998.89 | $7,683 | [$2,493, $5,505] |

Paired diff (Reward 2 - Reward 1): mean **-$7,555.48**, paired **t=-7.09**
(highly significant), **Reward 1 wins 75/100**. This is a materially
different result from Experiment 2's 1 MW finding, where the held-out
difference was not statistically significant either way (t=1.26, 53/100).
At 2 MW, Reward 2 doesn't just lose its training-time edge on held-out
data -- Reward 1 actively and significantly outperforms it. The
`held_out_2017_distribution_plot.png` scatter also shows Reward 2's
distribution bunching up near $0 for a large cluster of seeds (consistent
with the "policy stops trading" failure mode seen in Experiment 5's best
Reward-2 training seed), while Reward 1's stays broadly spread and
positive.

## Files

Same layout as Experiment 2: `training_distribution_plot.png`,
`held_out_2017_distribution_plot.png`, `training_cumulative_profit_over_time_plot.png`,
`held_out_2017_eval_plot.png`, matching `*_stats.json` / `*_summary.json`
files, `params.txt`, and `individual_trials/trial_NN/` (Q-tables and
histories, not pushed to git -- see root README's `.gitignore` note).

## Reproduce

```bash
venv/bin/python3 train.py --data data/train/isone_rt_hourly_lmp_2016.csv --reward both \
  --smoothing 0.001 --n-passes 1 --n-price-bins 5 --max-rate-mw 2.0 --n-trials 100 --seed 42 \
  --out-dir outputs/runs/<new_timestamp>_exp6_2mw_100seed_distribution
venv/bin/python3 evaluate.py --run outputs/runs/<new_timestamp>_exp6_2mw_100seed_distribution \
  --data data/test/isone_rt_hourly_lmp_2017.csv --max-rate-mw 2.0
venv/bin/python3 scripts/data_plots/plot_100seed_distribution.py --run-dir outputs/runs/<new_timestamp>_exp6_2mw_100seed_distribution --which training
venv/bin/python3 scripts/data_plots/plot_100seed_distribution.py --run-dir outputs/runs/<new_timestamp>_exp6_2mw_100seed_distribution --which held_out
venv/bin/python3 scripts/data_plots/plot_100seed_cumulative_profit.py --run-dir outputs/runs/<new_timestamp>_exp6_2mw_100seed_distribution
```
