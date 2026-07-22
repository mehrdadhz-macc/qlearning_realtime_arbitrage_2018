"""Download real ISO-NE hourly real-time LMP data for 2016-2017 (paper's date range).

Paper (Wang & Zhang, 2018/2020, arXiv:1711.03127) uses hourly real-time
prices from ISO New England, Jan 1 2016 - Dec 31 2017. This script pulls the
"ISO NE CA" (system-wide control-area) sheet's RT_LMP column from ISO-NE's
public yearly "SMD Hourly Data" archive files -- no login required:

    2016: https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls
    2017: https://www.iso-ne.com/static-assets/documents/2017/02/2017_smd_hourly.xlsx

Why not the Web Services API (webservices.iso-ne.com), which is what the
project's docstrings originally pointed at: I tested it against a live ISO
Express account and its /hourlylmp/rt/final/day/{day}/location/{id} endpoint
returns real data back to roughly July 2018, but an EMPTY HourlyLmp list for
every date tried in 2016-2017 -- that endpoint's historical retention simply
doesn't reach the years this paper needs. The bulk yearly files are ISO-NE's
own historical archive for exactly this situation and cover 2016 (8784 rows,
leap year) and 2017 (8760 rows) with no missing RT_LMP values.

Each day in these files uses a fixed Hr_End = 1..24 convention (always
exactly 24 hours/day, including on DST transition days), so there's no
23/25-hour edge case to handle -- timestamp = Date + (Hr_End - 1) hours gives
a clean, gap-free hourly index.

Only years 2016 and 2017 have a known URL here (KNOWN_YEAR_URLS below); pass
--url to point at a different year's file ISO-NE publishes in the same
"SMD Hourly Data" format if you want to extend the range later.
"""

import argparse
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

KNOWN_YEAR_URLS = {
    2016: "https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls",
    2017: "https://www.iso-ne.com/static-assets/documents/2017/02/2017_smd_hourly.xlsx",
}

SHEET_NAME = "ISO NE CA"  # system-wide control-area sheet (as opposed to per-state sheets)


def download_year(year, url=None, cache_dir=Path("data/raw/isone_cache")):
    url = url or KNOWN_YEAR_URLS.get(year)
    if url is None:
        raise SystemExit(f"No known SMD Hourly Data URL for {year}; pass --url explicitly.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{year}_smd_hourly{Path(url).suffix}"

    if cache_file.exists():
        print(f"{year}: using cached {cache_file}")
        content = cache_file.read_bytes()
    else:
        print(f"{year}: downloading {url}")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        content = resp.content
        cache_file.write_bytes(content)

    df = pd.read_excel(BytesIO(content), sheet_name=SHEET_NAME, header=0)
    df["timestamp"] = pd.to_datetime(df["Date"]) + pd.to_timedelta(df["Hr_End"].astype(int) - 1, unit="h")
    out = df[["timestamp", "RT_LMP"]].rename(columns={"RT_LMP": "lmp_total"})
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2016, 2017])
    parser.add_argument("--url", default=None, help="Override the source URL (only meaningful with a single --years value)")
    parser.add_argument("--out-dir", default="data", help="Base output directory (default: data)")
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    year_to_split = {2016: "train", 2017: "test"}
    all_frames = []

    for year in args.years:
        url = args.url if (args.url and len(args.years) == 1) else None
        df = download_year(year, url=url)
        print(f"  {year}: {len(df)} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}, "
              f"mean ${df['lmp_total'].mean():.2f}/MWh")
        all_frames.append(df)

        split = year_to_split.get(year)
        if split:
            split_path = out_base / split / f"isone_rt_hourly_lmp_{year}.csv"
            split_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(split_path, index=False)
            print(f"    -> {split_path}")

    combined = pd.concat(all_frames, ignore_index=True).sort_values("timestamp")
    raw_path = out_base / "raw" / f"isone_rt_hourly_lmp_{min(args.years)}_{max(args.years)}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(raw_path, index=False)
    print(f"\nWrote combined series ({len(combined)} rows) to {raw_path}")


if __name__ == "__main__":
    main()
