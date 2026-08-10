# Experiment 1: best seed per reward

## Intuition

Each reward function is trained on the single seed that gave *it* the
highest **training** profit, found independently for each reward from a
paired 50-trial sweep at this project's best confirmed single-pass
hyperparameters (`--smoothing 0.001 --n-price-bins 5 --n-passes 1`). Unlike
`plot_best_seed_reward_comparison.py`'s earlier run (one shared seed for
both rewards), the two Q-tables here do **not** share an exploration
trajectory. Question: does "a seed that trains well" say anything about
"a policy that generalizes well" on unseen data?

Each trained Q-table is then frozen (epsilon=0, no further learning) and
replayed greedily on the held-out 2017 year -- an additional check beyond
what the paper itself does (it never holds out a test year; see the root
README).

## Finding: no -- and it can flip entirely

| | seed | training profit (2016) | held-out profit (2017) |
|---|---|---|---|
| Reward 1 | 1688060240 | $5,956.68 (highest Reward-1 seed found) | **$13,318.85** |
| Reward 2 | 1455819991 | $7,414.94 (highest Reward-2 seed found) | **$544.39** |

Reward 1's best-training seed generalizes to a smooth, steady climb all
year on 2017. Reward 2's best-training seed -- the one that nearly matched
the paper's own Fig. 4 number ($6,900) during training -- barely breaks
even on 2017, staying flat near $0-800 the whole year. A seed picked purely
for training performance is not a reliable indicator of generalization,
and here the two rewards' best-training seeds land on opposite ends of the
held-out spectrum.

This is a single seed per reward, not an averaged distribution -- treat it
as illustrative of these two specific Q-tables, not a statistical claim.
See `../20260810_154500_exp2_100seed_distribution/` for the 100-seed
version of this same question, which reproduces the same flip with
statistical power (training: t=23.54, Reward 2 wins 100/100; held-out:
t=1.26, not significant, Reward 2 wins only 53/100).

## Files

- `training_best_seed_per_reward_plot.png` -- cumulative training profit (2016), both seeds
- `held_out_2017_best_seed_per_reward_plot.png` -- cumulative held-out profit (2017), same two Q-tables
- `params.txt` -- exact hyperparameters and results, and reproduction command context

## Reproduce

```bash
venv/bin/python3 scripts/data_plots/plot_best_seed_per_reward.py
```
(seeds 1688060240 / 1455819991 are this script's defaults, found from the
sweeps referenced in `params.txt`)
