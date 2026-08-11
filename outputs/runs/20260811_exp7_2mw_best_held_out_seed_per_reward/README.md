# Experiment 7: best seed per reward by held-out profit, 2 MW rate (mirrors Experiment 3)

## Intuition

Mirror image of [Experiment 5](../20260811_exp5_2mw_best_seed_per_reward/),
same design as [Experiment 3](../20260810_162224_exp3_1mw_best_held_out_seed_per_reward/):
each reward's best-*held-out* seed, found by cross-referencing
[Experiment 6](../20260811_exp6_2mw_100seed_distribution/)'s 100-trial
sweep, at 8 MWh / **2 MW**.

## Finding: both best-held-out seeds are close training performers, and one is a training LOSER

| | seed | held-out profit (2017) | training profit (2016) |
|---|---|---|---|
| Reward 1 | 145857092 | **$30,629.72** | **-$3,399.76** |
| Reward 2 | 2077711328 | **$34,787.44** | $3,727.04 |

Reward 1's best-held-out seed actually *lost* money during training
(-$3,399.76) yet produced the best held-out result of all 100 Reward-1
trials at this config ($30,629.72) -- the starkest example yet in this
project of training and held-out profit being independent signals for a
single seed. Both curves climb steadily together for most of the year
(see the plot) and only diverge meaningfully after hour ~7,000, with
Reward 2 pulling slightly ahead by year-end.

## Files

- `training_best_held_out_seed_per_reward_plot.png` / `held_out_2017_best_held_out_seed_per_reward_plot.png`
- `params.txt`

## Reproduce

```bash
venv/bin/python3 scripts/data_plots/plot_best_held_out_seed_per_reward.py \
  --max-rate-mw 2.0 --seed-reward1 145857092 --seed-reward2 2077711328
```
(seeds found by cross-referencing Experiment 6's summary.json /
held_out_2017_eval_summary.json)
