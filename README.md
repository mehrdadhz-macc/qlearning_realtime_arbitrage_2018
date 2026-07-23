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

**The dataset (~21MB: two source workbooks, the cleaned CSV, and the
train/test splits) is checked into this repo under `data/`** -- small enough
that requiring every clone to re-download and rebuild it seemed like
needless friction. `venv/bin/pip install -r requirements.txt` is all you
need to start training. The pipeline below is for reproducing that data
from scratch (e.g. if ISO-NE revises the archive, or you want a different
year), not a required setup step:

```
venv/bin/python3 scripts/data_generation/download_isone_data.py    # 1/3: fetch the original files
venv/bin/python3 scripts/data_generation/preprocess_isone_data.py  # 2/3: extract + clean into one series
venv/bin/python3 scripts/data_generation/split_train_test.py       # 3/3: split by year
```

No account or credentials needed -- these are public static files, not the
Web Services API (see "Why not the Web Services API" below). Three separate
stages, each leaving its own artifact on disk so you can inspect or redo any
one of them independently:

1. **`download_isone_data.py`** fetches ISO-NE's own yearly "SMD Hourly Data"
   workbooks **byte-for-byte, untouched**, into `data/raw/2016_smd_hourly.xls`
   / `data/raw/2017_smd_hourly.xlsx` (skips re-downloading if already
   present). Also writes `data/raw/DATA_SOURCE.txt` -- a plain-text guide to
   where this data comes from, the direct URLs, and how to find the same
   report by hand on iso-ne.com's own website.

        2016: https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls
        2017: https://www.iso-ne.com/static-assets/documents/2017/02/2017_smd_hourly.xlsx

2. **`preprocess_isone_data.py`** reads those raw workbooks, pulls the
   `ISO NE CA` sheet's `RT_LMP` column (the Trading Hub price -- see Data
   Source below), builds a proper hourly timestamp, combines both years, and
   validates the result (checks: no duplicate timestamps, no missing values,
   strictly hourly spacing, row counts match the expected 8784/8760 for
   leap/non-leap years) before writing
   `data/raw/isone_rt_hourly_lmp_2016_2017_clean.csv`. "Clean" means
   standardized columns and validated structure, not re-deriving ISO-NE's
   own RT_LMP figures -- the March 2017 convention change and DST-hour
   synthesis (see caveats above) are passed through as-is.

3. **`split_train_test.py`** splits that clean CSV by calendar year into
   `data/train/isone_rt_hourly_lmp_2016.csv` and
   `data/test/isone_rt_hourly_lmp_2017.csv`.

**Why not the Web Services API** (what stage 1 originally used, and what
requires a registered account): tested live against `/hourlylmp/rt/final/day/
{day}/location/{id}` -- authentication and Hub-location auto-detection both
worked (resolved to ID 4000, `.H.INTERNAL_HUB`, exactly as guessed), and it
does return real data for recent dates (verified back to mid-2018), but every
2016/2017 date tried came back with an empty `HourlyLmp` list. That
endpoint's historical retention doesn't reach the years this paper needs, so
stage 1 instead uses ISO-NE's own bulk archive files for exactly that
situation.

Only 2016/2017 URLs are hardcoded in stage 1; pass `--years`/`--url` to
point at a different year if ISO-NE publishes it in the same format.
`outputs/data_plots/` shows the same flat-baseline-with-sparse-spikes
pattern as paper Fig. 1 (confirmed -- see Known findings below).

## Project structure

```
scripts/
  data_generation/
    download_isone_data.py     # 1/3: fetch original workbooks -> data/raw/
    preprocess_isone_data.py   # 2/3: extract + clean -> data/raw/*_clean.csv
    split_train_test.py        # 3/3: split by year -> data/train/, data/test/
  data_plots/
    plot_price_series.py       # Fig. 1 style raw + moving-average price plot

data/
  raw/     original workbooks, DATA_SOURCE.txt, and the cleaned combined CSV
  train/   isone_rt_hourly_lmp_2016.csv
  test/    isone_rt_hourly_lmp_2017.csv

src/
  data_loader.py       # load a price CSV
  environment.py       # StorageArbitrageEnv: AMP dynamics (Sec. II) + Lemma 1 bang-bang actions
  rewards.py            # Reward 1 (instant profit) and Reward 2 (moving-average-relative), Sec. III-C
  qlearning_agent.py    # tabular Q-learning, Eq. 7 / Algorithm 1

train.py      # online Q-learning over one price series (paper's own training == evaluation)
evaluate.py   # freeze the learned Q-table, replay greedily on a held-out year (see below)

outputs/
  runs/<timestamp>/           # everything about ONE run lives here, nothing scattered elsewhere:
    price_bin_edges_*.npy     #   shared across all trials (see Train below), summary.json (train.py)
    summary.json              #   eval_summary.json, eval_plot.png (evaluate.py)
    trial_00/, trial_01/, ...
      q_table_*.npy           #   one independently-trained model per trial
      history_*.npy
  data_plots/                 # raw price-series plots (plot_price_series.py) -- not tied to a run
```

## Train

```bash
venv/bin/python3 train.py --data data/train/isone_rt_hourly_lmp_2016.csv --reward both --n-trials 10
```

Trains both Reward 1 and Reward 2 online over the 2016 series. Q-learning's
epsilon-greedy exploration is genuinely random -- a single training run is
one sample, and re-running with a different seed can swing the profit by
more than an order of magnitude (seen directly on this project's own data:
$412 to $5,946 to $3,919 across three seeds, same everything else). So
`--n-trials N` (default 1) runs N independent trials per reward kind, saves
every trial's Q-table separately under `outputs/runs/<timestamp>/trial_NN/`,
and reports **mean +/- std cumulative training profit across trials** -- the
expected-value estimate that should actually be judged, not any one trial's
number.

`--seed` does **not** feed into any trial directly, and is **not** a base
that trial i offsets by (`seed`, `seed+1`, `seed+2`, ...). Instead it seeds
an RNG that *generates* each trial's own seed
(`np.random.default_rng(seed).integers(...)`), so the same `--seed` always
reproduces the same set of trial seeds, but adjacent trials never differ by
a suspiciously simple +1, and reward_1's trial i / reward_2's trial i share
the same generated seed (a paired comparison that reduces noise when judging
which reward function is actually better). One consequence: `--seed 0` no
longer reproduces this project's earlier single-trial numbers verbatim --
trial 0's *generated* seed under `--seed 0` is some large pseudo-random
integer, not literally `0`. That's an intentional trade-off (a properly
generated seed sequence over a predictable arithmetic one), not a
regression -- rerun with `--n-trials` to get the properly-averaged result
instead of chasing one specific historical number.

Reported profit is always the true AMP-objective profit (`env.true_profit`,
Sec. II), not the shaped reward signal. Key flags: `--capacity-mwh` (default
8), `--max-rate-mw` (default 1, paper also reports a 2 MW case),
`--n-price-bins` (default 10 -- **not specified numerically in the paper**),
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

Freezes **every trial's** Q-table from the run above (epsilon=0, no further
learning), replays each on 2017 (a year the agent never saw during
training), and reports mean +/- std held-out profit across trials, plus
each trial's own number for transparency. **This is a stronger check than
the paper itself runs** -- the paper's own Fig. 4 plots cumulative profit
*during* the single online training pass, which conflates learning and
evaluation. Reuses the exact price-bin edges fit during training (saved
once per reward kind as `price_bin_edges_*.npy`, shared across all trials
since fitting them doesn't depend on the seed) rather than refitting bins
from the test set's own price range, since evaluating a trained Q-table
against differently-defined state bins would silently read the wrong table
entries. The saved plot shows each reward kind's mean curve with a shaded
+/-1 std band, not just a single line. If you trained with non-default
`--efficiency-charge`/`--efficiency-discharge`, pass the same values here --
they aren't auto-loaded from the training run.

Runs from before `--n-trials` was added (flat `q_table_*.npy` directly under
the run directory, no `trial_NN/` subdirectories) aren't compatible with the
current `evaluate.py` -- retrain to pick up the new layout.

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

(Both columns are single-trial numbers from literal `--seed 0`, from before
`--n-trials`/multi-trial seed generation existed -- kept here as a historical
record of what the bin-fitting fix changed, holding the seed fixed as the
one variable. They are **not** reproducible today with `--seed 0 --n-trials
1`, since `--seed` now seeds an RNG that generates trial seeds rather than
being used as a trial seed directly -- see Train below. Given the seed-to-
seed variance demonstrated throughout this README, treat this table as
illustrating the *direction* and rough *size* of the binning fix's effect,
not as two exact numbers to reproduce.)

Fixing the price-bin look-ahead/resolution issue alone flipped Reward 2's
training-time result from a loss to a genuine profit (matching the paper's
qualitative claim that Reward 2, unlike Reward 1, is profitable *during*
online training). The single-trial held-out margin (1.3x, then 2.6x after
the fix) turned out to be an artifact of small sample size -- see the
100-trial result immediately below, which supersedes it.

**100-trial result (the one to trust): Reward 2's advantage is real during
training, but does not hold up on held-out data.** Ran `--n-trials 100
--seed 42` (same hyperparameters as above) and compared Reward 1 vs. Reward
2 with a *paired* test -- each trial index uses the same generated seed for
both reward kinds, so the comparison isn't contaminated by unrelated
exploration luck:

| | Reward 1 | Reward 2 | paired diff (R2-R1) | paired t-stat | R2 wins |
|---|---|---|---|---|---|
| Training (2016) | mean $1,222.54, std $2,000.27 | mean $2,137.26, std $2,095.86 | mean +$914.72, std $912.81 | **t=9.97** (highly significant) | 83/100 |
| Held-out (2017) | mean $8,361.82, std $4,786.88 | mean $7,041.94, std $6,406.44 | mean -$1,319.88, std $8,225.41 | **t=-1.60** (not significant, needs \|t\|>1.98) | 47/100 |

During training, Reward 2's edge is large, consistent, and statistically
solid -- matches the paper. On the held-out year, the mean actually *flips*
in Reward 1's favor, the difference is statistically indistinguishable from
noise, and Reward 2 wins fewer than half the paired trials. Reward 2 also
has substantially higher held-out variance (std $6,406 vs. $4,787) -- some
trials generalize extremely well (+$21,952), others badly (-$10,558), a
spread wide enough to swamp any systematic advantage. A plausible reason
(not confirmed further here): Reward 2's moving-average-relative shaping may
make the learned policy more sensitive to exactly which (price, energy)
states got visited during that trial's particular exploration trajectory,
compared to Reward 1's plainer absolute-profit signal -- but this is a
hypothesis, not something this project verified mechanistically.

**Practical implication**: don't trust any single-trial (or even 5-trial)
comparison between the two rewards, including earlier numbers in this
README -- run `--n-trials` at a size like 100 before drawing a conclusion
about which reward function is actually better on held-out data.

The paper's own absolute figure (~$28k on 2016) still isn't matched by
either reward's training-time mean, which is expected given M, eta, and the
training regime (number of passes) are never given numeric values in the
paper.

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
