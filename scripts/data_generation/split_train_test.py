"""Stage 3/3: split the cleaned combined series into per-year train/test CSVs.

2016 -> data/train/ (the paper's own training year, and its headline
result -- Fig. 4 -- is reported on 2016 alone). 2017 -> data/test/ (a year
the agent never trains on, used by evaluate.py for a genuine held-out check
the paper itself never runs).

Usage:
    venv/bin/python3 scripts/data_generation/split_train_test.py
"""

import argparse
from pathlib import Path

import pandas as pd

YEAR_TO_SPLIT = {2016: "train", 2017: "test"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-csv", default="data/raw/isone_rt_hourly_lmp_2016_2017_clean.csv",
                         help="Output of preprocess_isone_data.py")
    parser.add_argument("--out-dir", default="data", help="Base output directory (default: data)")
    args = parser.parse_args()

    clean_path = Path(args.clean_csv)
    if not clean_path.exists():
        raise SystemExit(f"{clean_path} not found -- run preprocess_isone_data.py first.")

    df = pd.read_csv(clean_path, parse_dates=["timestamp"])
    out_base = Path(args.out_dir)

    for year, split in YEAR_TO_SPLIT.items():
        year_df = df[df["timestamp"].dt.year == year]
        if year_df.empty:
            print(f"  {year}: no rows in {clean_path}, skipping")
            continue
        split_path = out_base / split / f"isone_rt_hourly_lmp_{year}.csv"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        year_df.to_csv(split_path, index=False)
        print(f"  {year} ({split}): {len(year_df)} rows -> {split_path}")


if __name__ == "__main__":
    main()
