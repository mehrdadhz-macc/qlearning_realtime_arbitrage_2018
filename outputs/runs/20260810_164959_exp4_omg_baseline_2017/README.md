# Experiment 4: OMG baseline, held-out 2017

## Intuition

Experiments 1-3 are all about the Q-learning agent's seed-to-seed
variance. The OMG (Online Modified Greedy) baseline -- Qin et al. 2016,
the reference Wang & Zhang's Sec. IV-C compares their RL policy against
-- is different in kind: it's a deterministic threshold rule, not a
learned model (see `src/omg_baseline.py`). Its only "fitting" step is a
one-time causal calibration of `[p_min, p_max]` from the first 30 days of
2016 training data, via a closed-form formula -- not iterative training.
So there's no seed, no variance, and no multi-trial distribution to run:
one pass through the held-out 2017 year is the complete, exactly
reproducible result. Kept in its own experiment directory rather than
mixed into Experiments 1-3, which exist specifically to characterize
randomness that this algorithm doesn't have.

## Result

| battery config | held-out 2017 profit |
|---|---|
| 8 MWh, 1 MW | **$10,841.02** |
| 8 MWh, 2 MW | **$8,753.44** |

Both configs climb in a step-wise pattern tracking the sparse price spikes
visible in the top panel of the plot -- most of the profit accumulates in
a handful of sharp jumps (e.g. around hour 6,400) rather than smoothly
across the year, since OMG's fixed threshold only pays off when the price
clears it by a meaningful margin. The 1 MW config outperforms the 2 MW
config here (more feasible headroom per MWh of capacity at the lower
rate) -- the same qualitative ordering as the paper's own baseline numbers
(Fig. 6: $5,845 at 1 MW vs. $4,603 at 2 MW, though on 2016 in-sample data,
not 2017 held-out).

For context (not this experiment's focus, but useful cross-reference): the
same OMG parameters evaluated in-sample on 2016 give $10,164.04 (1 MW) /
$7,550.96 (2 MW) -- see `evaluate_omg_baseline.py` and the root README's
OMG baseline section. OMG's held-out numbers here exceed our Q-learning
Reward 2's 100-trial training mean ($3,208.36, see Experiment 2), though
not necessarily a favorable, cherry-picked single Q-learning trial.

## Files

- `held_out_2017_omg_baseline_plot.png` -- price series + cumulative profit, both battery configs
- `held_out_2017_cumulative_profit_curve_1mw.npy` / `_2mw.npy` -- raw curves
- `params.txt` -- exact parameters (including the fitted Gamma/W per config) and results

## Reproduce

```bash
venv/bin/python3 scripts/data_plots/plot_omg_baseline_2017.py
```
