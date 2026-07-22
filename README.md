# Q-Learning Real-Time Arbitrage (ISO-NE)

Replication of Wang, H., & Zhang, B. (2018/2020). *Energy Storage Arbitrage
in Real-Time Markets via Reinforcement Learning*. IEEE PES General Meeting
2018; extended draft [arXiv:1711.03127](https://arxiv.org/abs/1711.03127).

The paper models a battery operating in a real-time market as an MDP over
(price bin, energy-level bin) states with three actions -- charge at max
rate, hold, discharge at max rate (Lemma 1: the optimal policy is always
bang-bang, so no partial charge/discharge actions are needed). It trains
tabular Q-learning online, directly on the price trajectory, and shows that a
naive instant-profit reward (**Reward 1**) fails to turn a profit on real
data, while a reward relative to a moving-average price (**Reward 2**) does.

## Data

Real ISO-NE hourly real-time LMP for the **Trading Hub** (`.H.INTERNAL_HUB`),
Jan 1 2016 - Dec 31 2017 (the paper's own range; its headline result, Fig. 4,
uses 2016 alone). Confirmed from the source file's own "Notes" sheet: *"'ISO
NE CA' tab contains values for the Trading Hub"* -- despite the sheet's name
suggesting a separate system-wide average, its LMP columns specifically are
the Hub price (the sheet's non-price columns, e.g. `System_Load`, genuinely
are system-wide).

Two caveats from that same Notes sheet, worth knowing before trusting the
numbers to the dollar:

- **RT_LMP's definition changed mid-series**: *"starting on March 1, 2017,
  this is the hourly average of the five-minute LMP in the hour."* Real-time
  prices actually clear every 5 minutes; the hourly figure is a derived
  average. The Notes sheet doesn't state the pre-March-2017 convention, so
  the 2016 training data and the Jan-Feb 2017 slice of the test data aren't
  guaranteed to be computed identically to the rest of 2017.
- **"Final" isn't literally permanent**: *"Hourly settlement values are
  subject to re-settlement by the ISO. Revised data may be posted at any
  time."* This is ISO-NE's official historical archive, but not an
  immutable number.
- DST transition hours aren't raw metered data: per the Notes sheet, the
  March "missing" hour and the November "duplicate" hour are each
  synthesized by averaging their two neighboring real hours.

```
venv/bin/pip install -r requirements.txt
venv/bin/python3 scripts/data_generation/download_isone_data.py
```

No account or credentials needed. The data comes from ISO-NE's public yearly
"SMD Hourly Data" archive files (`RT_LMP` column of the `ISO NE CA` sheet),
not the Web Services API:

    2016: https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls
    2017: https://www.iso-ne.com/static-assets/documents/2017/02/2017_smd_hourly.xlsx

**Why not the Web Services API** (what this script originally used, and what
requires a registered account): tested live against `/hourlylmp/rt/final/day/
{day}/location/{id}` -- authentication and Hub-location auto-detection both
worked (resolved to ID 4000, `.H.INTERNAL_HUB`, exactly as guessed), and it
does return real data for recent dates (verified back to mid-2018), but every
2016/2017 date tried came back with an empty `HourlyLmp` list. That
endpoint's historical retention doesn't reach the years this paper needs, so
this script instead uses ISO-NE's own bulk archive files for exactly that
situation.

The script caches each year's raw downloaded file under
`data/raw/isone_cache/`, writes the combined series to
`data/raw/isone_rt_hourly_lmp_2016_2017.csv`, and writes
`data/train/isone_rt_hourly_lmp_2016.csv` / `data/test/isone_rt_hourly_lmp_2017.csv`.
Verified: 8784 rows for 2016 (leap year), 8760 for 2017, zero missing
`RT_LMP` values in either (`Hr_End` always runs 1..24, see the DST note
above for why). `outputs/data_plots/` shows the same flat-baseline-with-
sparse-spikes pattern as paper Fig. 1 (confirmed -- see Known findings
below). Only 2016/2017 URLs are hardcoded; pass `--years`/`--url` to point
at a different year if ISO-NE publishes it in the same format.

## Project structure

```
scripts/
  data_generation/
    download_isone_data.py   # real ISO-NE data from the public SMD Hourly Data archive
  data_plots/
    plot_price_series.py     # Fig. 1 style raw + moving-average price plot

src/
  data_loader.py       # load a price CSV
  environment.py       # StorageArbitrageEnv: AMP dynamics (Sec. II) + Lemma 1 bang-bang actions
  rewards.py            # Reward 1 (instant profit) and Reward 2 (moving-average-relative), Sec. III-C
  qlearning_agent.py    # tabular Q-learning, Eq. 7 / Algorithm 1

train.py      # online Q-learning over one price series (paper's own training == evaluation)
evaluate.py   # freeze the learned Q-table, replay greedily on a held-out year (see below)

outputs/
  runs/<timestamp>/     # everything about ONE run lives here, nothing scattered elsewhere:
                        #   q_table_*.npy, price_bin_edges_*.npy, history_*.npy, summary.json
                        #   (written by train.py), eval_summary.json, eval_plot.png (evaluate.py)
  data_plots/           # raw price-series plots (plot_price_series.py) -- not tied to a run
```

## Train

```bash
venv/bin/python3 train.py --data data/train/isone_rt_hourly_lmp_2016.csv --reward both
```

Trains both Reward 1 and Reward 2 online over the 2016 series and prints
cumulative training profit for each (paper Fig. 4's headline comparison), now
using the true AMP-objective profit (`env.true_profit`, Sec. II) rather than
the shaped reward signal. Key flags: `--capacity-mwh` (default 8),
`--max-rate-mw` (default 1, paper also reports a 2 MW case), `--n-price-bins`
(default 10 -- **not specified numerically in the paper**),
`--price-bin-method` (`quantile` default, `equal_width` for the paper's more
literal "M even price intervals" reading -- see Deviations below),
`--bin-calibration-hours` (default 720 = 30 days, how much of the series is
used to fit price bins before training starts), `--alpha`/`--gamma`/`--epsilon`
(0.5/0.9/0.9 per Algorithm 1), `--smoothing` (Reward 2's moving-average eta,
Eq. 6 -- **also not given a numeric value in the paper**; default 0.1),
`--efficiency-charge`/`--efficiency-discharge` (eta_c/eta_d, default 1.0),
`--reward-efficiency-aware` (fold efficiencies into Reward 1/2 too, off by
default to match the paper's literal Sec. III-C formulas).

## Evaluate

```bash
venv/bin/python3 evaluate.py --data data/test/isone_rt_hourly_lmp_2017.csv
```

Freezes the Q-table trained above (epsilon=0, no further learning) and
replays it on 2017, a year the agent never saw during training. **This is a
stronger check than the paper itself runs** -- the paper's own Fig. 4 plots
cumulative profit *during* the single online training pass, which conflates
learning and evaluation. Reuses the exact price-bin edges fit during
training (saved as `price_bin_edges_*.npy`) rather than refitting bins from
the test set's own price range, since evaluating a trained Q-table against
differently-defined state bins would silently read the wrong table entries.
If you trained with non-default `--efficiency-charge`/`--efficiency-discharge`,
pass the same values here -- they aren't auto-loaded from the training run.

## Deviations / assumptions (where the paper is ambiguous)

- **Algorithm 1's hyperparameter line** is printed as "alpha=0.5, alpha=0.9,
  epsilon=0.9" -- read as alpha (learning rate) = 0.5, gamma (discount) =
  0.9, epsilon = 0.9, since alpha/gamma are the paper's own distinct symbols
  everywhere else.
- **Epsilon is fixed, not decayed** -- Algorithm 1 doesn't specify a decay
  schedule, so training explores with probability 0.9 throughout. `evaluate.py`
  always runs the frozen Q-table greedily (epsilon=0), so this doesn't leak
  into the reported evaluation policy.
- **Efficiencies (eta_c, eta_d)** appear in the paper's AMP objective (Sec.
  II) but are dropped from the literal Reward 1 / Reward 2 formulas (Sec.
  III-C). `src/rewards.py` implements the literal (no-efficiency) formulas
  by default (`--reward-efficiency-aware` to fold them in), but the
  *reported profit* (`env.true_profit`) always applies eta_c/eta_d, since
  that's the AMP objective itself, not a reward-shaping choice.
- **Price bin count (M) and Reward 2's smoothing constant (eta)** are used
  in the paper's derivation but no numeric values are given for the case
  study -- both are exposed as CLI flags with documented (not paper-sourced)
  defaults.
- **Price bin *method*** -- Sec. III-A's "M even price intervals from the
  lowest to the highest" reads most literally as equal-width bins fit over
  the whole series. We instead default to `--price-bin-method quantile`
  (equal-frequency bins) fit causally from only the first
  `--bin-calibration-hours` of the series (default 720h = 30 days). Two
  separate reasons: (1) fitting bin edges from the *entire* series (including
  months that haven't "happened" yet from the agent's perspective) contradicts
  the paper's own framing that the storage "does not have a priori knowledge
  of the prices" (Sec. III intro) -- fitting only a causal prefix avoids that;
  (2) 2016 ISO-NE prices are heavy-tailed (-$156 to +$1439, most hours in
  $10-60), so equal-width bins spent ~8 of 10 bins on rarely-visited spike
  territory and crushed the everyday range -- where most arbitrage
  opportunity lives -- into 1-2 bins. Switching to causal quantile bins
  measurably changed results (see Known findings). `--price-bin-method
  equal_width` is available for the more literal paper reading.
- **Fig. 1's caption** says "PJM Real-time price," but the body text and
  citation [19] both specify ISO-NE hourly real-time LMP -- treated as a
  caption typo, not a real data-source ambiguity.
- **Data node -- resolved.** Earlier drafts of this README flagged "Hub vs.
  system-wide average" as an open, unverified question, guessing the `ISO NE
  CA` sheet was a separate system-wide load-weighted average distinct from
  the Web Services API's Hub node (`.H.INTERNAL_HUB`, ID 4000). That guess
  was wrong: the source file's own Notes sheet explicitly states the `ISO NE
  CA` tab's LMP columns (including `RT_LMP`) **are** the Trading Hub values.
  No live API cross-check was actually needed -- the answer was in the
  downloaded file's documentation the whole time. See the Data section above
  for what's still worth caveating (the March 2017 methodology change and
  the resettlement note).
- Qin et al.'s online modified greedy baseline ([15] in the paper, used for
  the Sec. IV-C comparison) is not implemented here -- would need reading
  that paper directly rather than guessing its threshold rule from a
  secondary description.

## Known findings from this replication

**Methodology audit.** After the initial real-data run, the implementation
was checked line-by-line against the paper's Sections II-IV. The core RL
loop matched exactly (state/action timing, Lemma 1 bang-bang actions, Eq. 7
Q-update, Reward 1/2 formulas, epsilon-greedy). Four gaps were found and the
first three fixed:

1. **Dead efficiency plumbing (fixed).** `StorageArbitrageEnv` accepted
   `efficiency_charge`/`efficiency_discharge` but never used them, and
   `train.py`/`evaluate.py` never wired eta_c/eta_d into the reward
   functions. Now: `env.true_profit()` always applies them (AMP objective,
   Sec. II); `--reward-efficiency-aware` optionally folds them into Reward
   1/2 as well.
2. **False docstring claim (fixed).** `qlearning_agent.py` claimed an
   `--epsilon-decay` flag existed; it never did. Removed the claim (no decay
   flag was added, since the paper doesn't call for one).
3. **Price-bin look-ahead + poor resolution on heavy-tailed prices (fixed).**
   See "Price bin method" in Deviations above -- switched to causal
   quantile bins fit from a 30-day prefix instead of equal-width bins fit
   from the whole series.
4. **Data node ambiguity (resolved, not actually a gap).** Originally flagged
   as an open question (whether `ISO NE CA` matches the Hub series) that
   would need live API credentials to settle. It didn't -- the source
   file's own Notes sheet already documents `ISO NE CA`'s LMP columns as the
   Trading Hub. See "Data node" in Deviations above.

**Results, before vs. after the fixes** (same hyperparameters otherwise:
`--n-price-bins 10 --alpha 0.5 --gamma 0.9 --epsilon 0.9 --smoothing 0.1`):

| | Before (equal-width, look-ahead) | After (causal quantile bins) |
|---|---|---|
| Training profit, Reward 1 (2016) | -$777.70 | -$430.49 |
| Training profit, Reward 2 (2016) | -$558.13 | **+$412.68** |
| Held-out profit, Reward 1 (2017) | +$4,078.26 | +$6,168.60 |
| Held-out profit, Reward 2 (2017) | +$5,416.05 | **+$15,801.46** |

Fixing the price-bin look-ahead/resolution issue alone flipped Reward 2's
training-time result from a loss to a genuine profit (matching the paper's
qualitative claim that Reward 2, unlike Reward 1, is profitable *during*
online training), and widened the Reward 2-over-Reward 1 margin on held-out
2017 from 1.3x to 2.6x -- much closer to the paper's own emphatic gap
(4.8x-8.6x in its baseline comparison, Sec. IV-C) even though the paper's own
absolute figure (~$28k on 2016) still isn't matched, which is expected given
M, eta, and the training regime (number of passes) are never given numeric
values in the paper.

- The downloaded 2016 price series visually reproduces paper Fig. 1's shape
  closely: a flat ~$20-50/MWh baseline with sparse sharp spikes and one
  dramatic outlier (here, ~$1439/MWh), both around a similar relative
  position in the year. See `outputs/data_plots/isone_rt_hourly_lmp_2016.png`.
- 2016 real-time prices include negative values (min -$156.04/MWh) --
  expected for real-time LMP, but worth knowing since it means Reward 1's
  "charge = pay the spot price" can occasionally be a *reward*, not a cost,
  in a way Reward 2 doesn't specially account for.

**Next step if you want to chase the paper's exact magnitude**: sweep
`--n-price-bins`, `--smoothing`, and `--bin-calibration-hours`, and consider
whether more than one pass over 2016 (the paper's Algorithm 1 doesn't state
number of episodes/passes either) is needed before judging convergence --
right now `train.py` does exactly one linear pass through the series,
matching the paper's literal "online" framing but not necessarily its total
amount of learning.
