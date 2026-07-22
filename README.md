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

Real ISO-NE hourly real-time LMP at the system Hub, Jan 1 2016 - Dec 31 2017
(the paper's own range; its headline result, Fig. 4, uses 2016 alone).

```
venv/bin/pip install -r requirements.txt

export ISONE_WS_USER=you@example.com
export ISONE_WS_PASSWORD=your_password
venv/bin/python3 scripts/data_generation/download_isone_data.py
```

Credentials: free ISO Express account, then request Web Services access at
https://www.iso-ne.com/isoexpress/ -> "Web Services" (unlike PJM's Data
Miner 2, ISO-NE's public LMP data itself is free, but the API needs
registered Web Services credentials, not just a plain account login).

The script caches each day's raw response under `data/raw/isone_cache/`
(safe to re-run -- only missing days are re-fetched), writes the combined
series to `data/raw/isone_rt_hourly_lmp_2016_2017.csv`, and splits it by
calendar year into `data/train/isone_rt_hourly_lmp_2016.csv` and
`data/test/isone_rt_hourly_lmp_2017.csv`.

**I could not test this script against a live account** (no credentials
available while scaffolding this repo), so treat the Hub-location
auto-detection as best-effort: it queries `/locations/all.json` and looks for
a "HUB" entry, falling back to the commonly-cited Hub ID 4000 if that lookup
doesn't match ISO-NE's actual response shape. The script prints whatever
location it resolves -- **confirm that line says "HUB" before trusting the
downloaded prices**, and use `--inspect-only 20160104` to dump one raw day's
JSON if you need to adjust the parsing in `resolve_hub_location()` /
`parse_day()`.

## Project structure

```
scripts/
  data_generation/
    download_isone_data.py   # real ISO-NE data via the official Web Services API
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

## Status

Pipeline is smoke-tested end-to-end on synthetic data (no real ISO-NE
credentials were available while scaffolding this repo) -- training runs,
Reward 1/Reward 2 produce different behavior, evaluation runs on a held-out
series without crashing, and the price plot renders. It has **not** been run
against the real ISO-NE series yet; do that before trusting any profit
numbers as a genuine replication result.
