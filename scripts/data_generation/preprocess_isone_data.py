"""Stage 2/3: turn the raw ISO-NE workbooks into one clean hourly price series.

Reads the original .xls/.xlsx files data/raw/download_isone_data.py fetched,
extracts the "ISO NE CA" sheet's Date/Hr_End/RT_LMP columns (RT_LMP is the
Trading Hub price -- see data/raw/DATA_SOURCE.txt), builds a proper hourly
timestamp, combines both years, and validates the result before writing it
back out to data/raw/ as a single clean CSV. "Clean" here means: standard
column names (timestamp, lmp_total), chronologically sorted, no duplicate
timestamps, no missing values -- it does NOT mean re-deriving or correcting
ISO-NE's own RT_LMP figures (see DATA_SOURCE.txt's caveats about the March
2017 convention change and DST-hour synthesis, which are passed through
as-is, not "fixed").

Usage:
    venv/bin/python3 scripts/data_generation/preprocess_isone_data.py
"""

import argparse
from pathlib import Path

import pandas as pd

SHEET_NAME = "ISO NE CA"  # Trading Hub values, per the workbook's own Notes sheet
EXPECTED_HOURS_PER_YEAR = {2016: 8784, 2017: 8760}  # 2016 is a leap year


def load_raw_year(year, raw_dir):
    candidates = list(raw_dir.glob(f"{year}_smd_hourly.*"))
    if not candidates:
        raise SystemExit(f"No raw file found for {year} in {raw_dir}/ -- run download_isone_data.py first.")
    path = candidates[0]

    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=0)
    df["timestamp"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hr_End"].astype(int) - 1, unit="h")
    out = df[["timestamp", "RT_LMP"]].rename(columns={"RT_LMP": "lmp_total"})
    return out


def validate(df, years):
    problems = []

    n_dupes = df["timestamp"].duplicated().sum()
    if n_dupes:
        problems.append(f"{n_dupes} duplicate timestamps")

    n_missing = df["lmp_total"].isna().sum()
    if n_missing:
        problems.append(f"{n_missing} missing lmp_total values")

    for year in years:
        expected = EXPECTED_HOURS_PER_YEAR.get(year)
        actual = (df["timestamp"].dt.year == year).sum()
        if expected is not None and actual != expected:
            problems.append(f"{year}: expected {expected} hourly rows, got {actual}")

    gaps = df["timestamp"].diff().dropna()
    bad_gaps = gaps[gaps != pd.Timedelta(hours=1)]
    if len(bad_gaps):
        problems.append(f"{len(bad_gaps)} non-hourly gaps in the timestamp sequence")

    if problems:
        raise SystemExit("Validation failed:\n  - " + "\n  - ".join(problems))
    print(f"Validation OK: {len(df)} rows, no duplicates, no missing values, "
          f"strictly hourly, row counts match expected leap/non-leap year lengths.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2016, 2017])
    parser.add_argument("--raw-dir", default="data/raw", help="Where the raw workbooks live (default: data/raw)")
    parser.add_argument("--out", default=None, help="Override the output CSV path")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    frames = [load_raw_year(year, raw_dir) for year in args.years]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    validate(df, args.years)

    out_path = Path(args.out) if args.out else raw_dir / f"isone_rt_hourly_lmp_{min(args.years)}_{max(args.years)}_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote cleaned series ({len(df)} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}) to {out_path}")
    print("Next: scripts/data_generation/split_train_test.py")


if __name__ == "__main__":
    main()
