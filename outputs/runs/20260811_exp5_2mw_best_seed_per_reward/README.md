# Experiment 5: best seed per reward, 2 MW rate (mirrors Experiment 1)

## Intuition

Same question as [Experiment 1](../20260810_154929_exp1_best_seed_per_reward/),
at the paper's other battery configuration: 8 MWh capacity, **2 MW**
charge/discharge rate (vs. 1 MW everywhere else in this project). Each
reward trained on the seed that gave *it* the highest training profit,
found from a 50-trial sweep at the same hyperparameters used throughout
(`--smoothing 0.001 --n-price-bins 5 --n-passes 1`), just with
`--max-rate-mw 2.0`.

Interestingly, **the best-training seeds turned out identical to the 1 MW
case** (1688060240 for Reward 1, 1455819991 for Reward 2) -- these are
generally-favorable exploration trajectories for this price series,
independent of the battery's rate.

## Finding: the training-best-seed generalization gap is even more extreme at 2 MW

| | seed | training profit (2016) | held-out profit (2017) |
|---|---|---|---|
| Reward 1 | 1688060240 | $8,269.30 | **$10,155.82** |
| Reward 2 | 1455819991 | $11,050.14 | **-$281.10** |

Reward 2's best-training seed doesn't just underperform held-out here --
its held-out curve is **flat for the entire year** (see
`held_out_2017_best_seed_per_reward_plot.png`): the frozen policy
essentially stops trading almost immediately and never resumes, ending
slightly negative. Reward 1's same seed, meanwhile, climbs smoothly to
over $10,000. At 1 MW (Experiment 1) this same comparison was $13,318.85
vs. $544.39 -- a big gap, but Reward 2 was at least still profitable.
At 2 MW, Reward 2's training-best seed's generalization failure is total.

## Files

- `training_best_seed_per_reward_plot.png` / `held_out_2017_best_seed_per_reward_plot.png`
- `params.txt` -- exact hyperparameters and results

## Reproduce

```bash
venv/bin/python3 scripts/data_plots/plot_best_seed_per_reward.py \
  --max-rate-mw 2.0 --seed-reward1 1688060240 --seed-reward2 1455819991
```
(seeds found from `outputs/runs/sweeps_2mw/reward1_m5` and `reward2_m5`,
50-trial sweeps at `--max-rate-mw 2.0`, otherwise identical to the sweeps
behind Experiment 1)
