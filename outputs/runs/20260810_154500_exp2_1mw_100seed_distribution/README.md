# Experiment 2: 100-seed paired distribution

## Intuition

[Experiment 1](../20260810_154929_exp1_1mw_best_seed_per_reward/) is two single
trials -- illustrative, not statistical. This experiment trains both Reward
1 and Reward 2 on the SAME 100 seeds (paired design: trial *i*'s seed is
identical for both rewards, so the comparison isn't contaminated by
unrelated exploration luck), at this project's best confirmed single-pass
hyperparameters (`--smoothing 0.001 --n-price-bins 5 --n-passes 1`), to get
a reliable average and interval for both training and held-out profit.

## Finding 1: during training, Reward 2's edge is not just significant, it's universal -- but read the raw number with caution

| | mean | std | 95% CI | range |
|---|---|---|---|---|
| Reward 1 | $1,264.31 | $2,073.25 | +/-$406 | [-$2,574.77, $7,227.58] |
| Reward 2 | $3,208.36 | $2,173.64 | +/-$426 | [-$1,754.88, $8,029.65] |

Paired diff (Reward 2 - Reward 1): mean +$1,944.05, std $825.77, **paired
t=23.54**, and **Reward 2 wins all 100/100 paired seeds**. That's a much
cleaner result than the earlier 100-trial run at this project's original,
untuned hyperparameters (M=10, eta=0.1), which found Reward 2 winning only
83/100 trials with paired t=9.97 (see the root README). Tuning eta/M
doesn't just raise the mean, it makes Reward 2's training-time advantage
essentially deterministic.

**But this training-time number is dominated by exploration noise, not
policy quality.** Epsilon=0.9 never decays (Algorithm 1), so ~90% of every
trial's actions are random. Freezing each trial's final Q-table and
replaying it greedily on the SAME 2016 data it trained on (epsilon=0)
gives a much better read of what was actually learned: **Reward 1 mean
$5,997.41, Reward 2 mean $6,745.48** -- both far above their raw training
means, and much closer to each other (and to the held-out means in
Finding 2 below) than the raw training numbers suggested. See
`training_distribution_plot.png`'s "greedy after training" annotation and
`greedy_2016_eval_plot.png`.

## Finding 2: on held-out data, that advantage does not survive

| | mean | std | range |
|---|---|---|---|
| Reward 1 | $6,590.73 | $5,878.82 | [-$5,736.92, $21,739.65] |
| Reward 2 | $7,713.17 | $7,110.87 | [-$7,899.76, $25,388.67] |

Paired diff: mean +$1,122.43, std $8,913.93, **paired t=1.26** (not
significant, needs \|t\|>1.98), **Reward 2 wins only 53/100** -- barely
better than a coin flip. This reproduces
[Experiment 1](../20260810_154929_exp1_1mw_best_seed_per_reward/)'s flip at
full statistical power: whatever makes Reward 2 win so reliably during
training does not reliably carry over to an unseen year. Also consistent
with this project's original 100-trial-at-default-hyperparameters finding
(see root README): Reward 2's edge is real and large during training, but
doesn't reliably generalize.

## Files

- `training_distribution_plot.png` / `held_out_2017_distribution_plot.png` -- per-seed scatter, mean +/- 1 std, for training and held-out respectively (training annotation also shows the greedy-after-training mean)
- `training_distribution_stats.json` / `held_out_2017_distribution_stats.json` -- same numbers, machine-readable
- `training_cumulative_profit_over_time_plot.png` -- mean +/- [10th,90th] percentile band across all 100 trials, full 8,784-hour training curve, legend also shows the greedy-after-training mean
- `held_out_2017_eval_plot.png` / `held_out_2017_eval_summary.json` -- raw per-trial held-out results (produced by `evaluate.py`)
- `greedy_2016_eval_plot.png` / `greedy_2016_eval_summary.json` -- each trial's final Q-table replayed greedily (epsilon=0) on the SAME 2016 training data, isolating learned-policy quality from epsilon=0.9's exploration noise (`evaluate.py --data data/train/... --label greedy_2016`)
- `params.txt` -- exact hyperparameters and all results above in one place
- `individual_trials/trial_NN/` -- each trial's Q-tables and step-by-step history (**not pushed to git**, regenerable locally, see Reproduce below)

## Reproduce

```bash
venv/bin/python3 train.py --data data/train/isone_rt_hourly_lmp_2016.csv --reward both \
  --smoothing 0.001 --n-passes 1 --n-price-bins 5 --n-trials 100 --seed 42 \
  --out-dir outputs/runs/<new_timestamp>_exp2_1mw_100seed_distribution
venv/bin/python3 evaluate.py --run outputs/runs/<new_timestamp>_exp2_1mw_100seed_distribution \
  --data data/test/isone_rt_hourly_lmp_2017.csv
venv/bin/python3 evaluate.py --run outputs/runs/<new_timestamp>_exp2_1mw_100seed_distribution \
  --data data/train/isone_rt_hourly_lmp_2016.csv --label greedy_2016
venv/bin/python3 scripts/data_plots/plot_100seed_distribution.py --run-dir outputs/runs/<new_timestamp>_exp2_1mw_100seed_distribution --which training
venv/bin/python3 scripts/data_plots/plot_100seed_distribution.py --run-dir outputs/runs/<new_timestamp>_exp2_1mw_100seed_distribution --which held_out
venv/bin/python3 scripts/data_plots/plot_100seed_cumulative_profit.py --run-dir outputs/runs/<new_timestamp>_exp2_1mw_100seed_distribution
```
