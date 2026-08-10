# Experiment 3: best seed per reward, selected by HELD-OUT profit

## Intuition

[Experiment 1](../20260810_154929_exp1_best_seed_per_reward/) picks each
reward's best *training* seed and checks how it does held-out. This
experiment flips the selection criterion: pick each reward's best
*held-out* seed instead, found by cross-referencing
[Experiment 2](../20260810_154500_exp2_100seed_distribution/)'s existing
100-trial paired sweep (no new sweep needed -- just the argmax of
`held_out_2017_eval_summary.json`'s profits, matched back to
`summary.json`'s seed list by trial index). Question: does the training
curve of a held-out-best seed look any different from a typical or a
training-best one?

## Finding: the best held-out seeds are middling-to-poor training performers

| | seed | held-out profit (2017) | training profit (2016) |
|---|---|---|---|
| Reward 1 | 345075200 | **$21,739.65** (highest Reward-1 held-out found) | $803.24 |
| Reward 2 | 406886644 | **$25,388.67** (highest Reward-2 held-out found) | $4,149.10 |

Reward 1's best-held-out seed is a training laggard: **negative** for most
of 2016 (-$500 to -$1,500), rescued only by the Aug-11 price-spike jump,
ending at a modest $803.24 -- nowhere near that reward's training-best seed
($5,956.68, see Experiment 1). Reward 2's best-held-out seed does better in
training ($4,149.10) but still well short of its own training-best
($7,414.94). Combined with Experiment 1's opposite-direction result (best
training seeds generalizing anywhere from terribly to excellently), this
confirms training profit and held-out profit are close to independent
signals for a single seed -- selecting on one tells you almost nothing
about the other, in either direction.

Worth noting: unlike the training-time comparisons elsewhere in this
project (which regularly show one dominant jump at the Aug-2016 heat-wave
spike), the **held-out 2017 curves here are smooth, steady climbs with no
single dominant jump** -- 2017 has no outlier event on the scale of 2016's
$1,439 spike (max ~$700, see `outputs/data_plots/isone_rt_hourly_lmp_2017.png`),
so held-out profit here comes from many moderate opportunities rather than
one extreme one.

## Files

- `training_best_held_out_seed_per_reward_plot.png` -- cumulative training profit (2016), both seeds
- `held_out_2017_best_held_out_seed_per_reward_plot.png` -- cumulative held-out profit (2017), same two Q-tables
- `params.txt` -- exact hyperparameters and results

## Reproduce

```bash
venv/bin/python3 scripts/data_plots/plot_best_held_out_seed_per_reward.py
```
(seeds 345075200 / 406886644 are this script's defaults, found by
cross-referencing Experiment 2's sweep -- see `params.txt`)
