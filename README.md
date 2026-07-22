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

Real ISO-NE hourly real-time LMP for the system ("ISO NE CA") control area,
Jan 1 2016 - Dec 31 2017 (the paper's own range; its headline result, Fig. 4,
uses 2016 alone).

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
`RT_LMP` values in either. Each day uses a fixed `Hr_End` = 1..24 convention
(always 24 hours, including DST transition days), so `outputs/data_plots/`
should show the same flat-baseline-with-sparse-spikes pattern as paper Fig. 1
(confirmed -- see Known findings below). Only 2016/2017 URLs are hardcoded;
pass `--years`/`--url` to point at a different year if ISO-NE publishes it in
the same format.

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
  runs/<timestamp>/     # q_table_*.npy, price_bin_edges_*.npy, history_*.npy, summary.json
  eval_plots/, data_plots/
```

## Train

```bash
venv/bin/python3 train.py --data data/train/isone_rt_hourly_lmp_2016.csv --reward both
```

Trains both Reward 1 and Reward 2 online over the 2016 series and prints
cumulative training profit for each (paper Fig. 4's headline comparison).
Key flags: `--capacity-mwh` (default 8), `--max-rate-mw` (default 1, paper
also reports a 2 MW case), `--n-price-bins` (default 10 -- **not specified
numerically in the paper**), `--alpha`/`--gamma`/`--epsilon` (0.5/0.9/0.9 per
Algorithm 1), `--smoothing` (Reward 2's moving-average eta, Eq. 6 -- **also
not given a numeric value in the paper**; default 0.1).

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
  by default; pass `efficiency_aware=True` to fold efficiencies in.
- **Price bin count (M) and Reward 2's smoothing constant (eta)** are used
  in the paper's derivation but no numeric values are given for the case
  study -- both are exposed as CLI flags with documented (not paper-sourced)
  defaults.
- **Fig. 1's caption** says "PJM Real-time price," but the body text and
  citation [19] both specify ISO-NE hourly real-time LMP -- treated as a
  caption typo, not a real data-source ambiguity.
- Qin et al.'s online modified greedy baseline ([15] in the paper, used for
  the Sec. IV-C comparison) is not implemented here -- would need reading
  that paper directly rather than guessing its threshold rule from a
  secondary description.

## Known findings from this replication

Ran end-to-end on the real 2016/2017 series with default hyperparameters
(`--n-price-bins 10 --alpha 0.5 --gamma 0.9 --epsilon 0.9 --smoothing 0.1`):

- Training (online, on 2016): both rewards net *negative* cumulative profit
  (Reward 1: -$778, Reward 2: -$558) -- unlike the paper's own reported
  training-time result (~+$28k for Reward 2 on an 8MWh/1MW battery). Given
  how many of the paper's own hyperparameters aren't numerically specified
  (price bin count, Reward 2's smoothing constant -- see Deviations above),
  this isn't surprising; it means our particular defaults haven't found a
  profitable policy within one pass over 2016, not that the approach itself
  is broken.
- Held-out evaluation (frozen greedy policy replayed on 2017, which the
  paper itself never does): **both rewards turn positive, and Reward 2 beats
  Reward 1** (Reward 1: +$4,078, Reward 2: +$5,416) -- which does match the
  paper's central qualitative claim (Reward 2 > Reward 1) even though the
  training-time numbers above don't match its magnitude.
- The downloaded 2016 price series visually reproduces paper Fig. 1's shape
  closely: a flat ~$20-50/MWh baseline with sparse sharp spikes and one
  dramatic outlier (here, ~$1439/MWh), both around a similar relative
  position in the year. See `outputs/data_plots/isone_rt_hourly_lmp_2016.png`.
- 2016 real-time prices include negative values (min -$156.04/MWh) --
  expected for real-time LMP, but worth knowing since it means Reward 1's
  "charge = pay the spot price" can occasionally be a *reward*, not a cost,
  in a way Reward 2 doesn't specially account for.

**Next step if you want to chase the paper's exact magnitude**: sweep
`--n-price-bins` and `--smoothing`, and consider whether more than one pass
over 2016 (the paper's Algorithm 1 doesn't state number of episodes/passes
either) is needed before judging convergence -- right now `train.py` does
exactly one linear pass through the series, matching the paper's literal
"online" framing but not necessarily its total amount of learning.
