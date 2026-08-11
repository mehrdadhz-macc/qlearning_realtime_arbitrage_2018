# Experiment 7: best seed per reward by held-out profit, 2 MW rate (mirrors Experiment 3)

## Intuition

Mirror image of [Experiment 5](../20260811_exp5_2mw_best_seed_per_reward/),
same design as [Experiment 3](../20260810_162224_exp3_1mw_best_held_out_seed_per_reward/):
each reward's best-*held-out* seed, found by cross-referencing
[Experiment 6](../20260811_exp6_2mw_100seed_distribution/)'s 100-trial
sweep, at 8 MWh / **2 MW**.

## Finding: both best-held-out seeds are close training performers, and one is a training LOSER -- by the raw metric only

| | seed | held-out profit (2017) | raw training profit (2016, online, epsilon=0.9) | greedy-after-training profit (2016, epsilon=0) |
|---|---|---|---|---|
| Reward 1 | 145857092 | **$30,629.72** | **-$3,399.76** | $28,065.44 |
| Reward 2 | 2077711328 | **$34,787.44** | $3,727.04 | $28,098.34 |

Reward 1's best-held-out seed *lost* money on the raw training metric
(-$3,399.76) yet produced the best held-out result of all 100 Reward-1
trials at this config ($30,629.72) -- the starkest example yet in this
project of *raw* training and held-out profit being independent signals
for a single seed. But this is fully resolved by the greedy-after-training
number: the SAME final Q-table, replayed greedily (epsilon=0) on the same
2016 data, is worth $28,065.44 -- almost exactly its held-out result, and
nothing like the raw -$3,399.76. Reward 2's seed tells the same story:
greedy $28,098.34 lines up with held-out $34,787.44, not raw training's
$3,727.04. Once exploration noise (epsilon=0.9 never decays -- Algorithm
1) is stripped out, "training" and "held-out" profit agree closely for
both seeds -- it was only ever the *raw online* number that looked
independent. Both curves climb steadily together for most of the year
(see the plot) and only diverge meaningfully after hour ~7,000, with
Reward 2 pulling slightly ahead by year-end.

## Files

- `training_best_held_out_seed_per_reward_plot.png` / `held_out_2017_best_held_out_seed_per_reward_plot.png` -- training legend shows both the raw online (epsilon=0.9) and greedy-after-training (epsilon=0) final profit
- `greedy_2016_cumulative_profit_curve_reward_*.npy` -- the greedy-after-training rollout curves
- `params.txt`

## Reproduce

```bash
venv/bin/python3 scripts/data_plots/plot_best_held_out_seed_per_reward.py \
  --max-rate-mw 2.0 --seed-reward1 145857092 --seed-reward2 2077711328
```
(seeds found by cross-referencing Experiment 6's summary.json /
held_out_2017_eval_summary.json)
