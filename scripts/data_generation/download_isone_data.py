"""Stage 1/3: download the ORIGINAL, untouched ISO-NE files into data/raw/.

Paper (Wang & Zhang, 2018/2020, arXiv:1711.03127) uses hourly real-time
prices from ISO New England, Jan 1 2016 - Dec 31 2017. This script only
fetches ISO-NE's own yearly "SMD Hourly Data" archive files, byte-for-byte,
into data/raw/ -- it does NOT parse, clean, or reshape anything. That's
scripts/data_generation/preprocess_isone_data.py's job (stage 2/3); stage
3/3 is scripts/data_generation/split_train_test.py.

No account or credentials needed -- these are public static files:

    2016: https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls
    2017: https://www.iso-ne.com/static-assets/documents/2017/02/2017_smd_hourly.xlsx

Why not the Web Services API, which is what earlier versions of this script
used: tested live against a real ISO Express account -- authentication and
Hub-location lookup both worked, but /hourlylmp/rt/final/day's historical
retention doesn't reach back to 2016/2017 (verified empty for every date
tried in that range, real data returned from ~mid-2018 onward). ISO-NE's own
bulk yearly archive is the right tool for exactly this situation.

This script also (re)writes data/raw/DATA_SOURCE.txt, a plain-text guide to
where this data comes from and how to find it by hand on iso-ne.com, since
that context is easy to lose once only a CSV filename is left behind.
"""

import argparse
from pathlib import Path

import requests

KNOWN_YEAR_URLS = {
    2016: "https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls",
    2017: "https://www.iso-ne.com/static-assets/documents/2017/02/2017_smd_hourly.xlsx",
}

DATA_SOURCE_TXT = """\
ISO-NE HOURLY REAL-TIME LMP -- DATA SOURCE
===========================================

WHAT THIS IS
------------
Hourly Day-Ahead and Real-Time locational marginal price (LMP), demand,
weather, and regulation-market data for ISO New England, published by ISO-NE
as yearly "SMD Hourly Data" Excel workbooks. This project uses the RT_LMP
column of the "ISO NE CA" sheet, which -- per the workbook's OWN embedded
"Notes" sheet -- is explicitly the Trading Hub (.H.INTERNAL_HUB) price, not a
separate system-wide average, despite the "CA" (Control Area) name:

    "'ISO NE CA' tab contains values for the Trading Hub"

DIRECT DOWNLOAD URLS (what download_isone_data.py fetches)
------------------------------------------------------------
    2016: https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls
    2017: https://www.iso-ne.com/static-assets/documents/2017/02/2017_smd_hourly.xlsx

HOW TO FIND THIS FILE BY HAND ON THE WEBSITE
-----------------------------------------------
    iso-ne.com -> ISO Express -> Energy, Load, and Demand Reports -> Zonal Information

    Direct page: https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/zone-info

    Look for the "SMD Hourly Data" report template. No account/login is
    needed, but downloading through that web page's interface requires
    solving a CAPTCHA. The direct URLs above point at the same files without
    that CAPTCHA step (presumably intended for the report-viewer page's own
    use, but reachable directly).

CAVEATS (from the workbook's own "Notes" sheet -- read it yourself with
`pd.ExcelFile(path).parse("Notes", header=None)` for the full text)
------------------------------------------------------------------------
- RT_LMP's definition changed mid-series: "starting on March 1, 2017, this
  is the hourly average of the five-minute LMP in the hour." Real-time
  prices actually clear every 5 minutes; the hourly figure is a derived
  average. The pre-March-2017 convention isn't stated, so 2016 data and the
  Jan-Feb 2017 slice of 2017 aren't guaranteed to be computed identically to
  the rest of 2017.
- "Final" isn't permanent: "Hourly settlement values are subject to
  re-settlement by the ISO. Revised data may be posted at any time."
- DST transition hours are synthesized, not raw metered data: the March
  "missing" hour and the November "duplicate" hour are each built by
  averaging their two neighboring real hours.

PIPELINE (this project)
------------------------
    1. scripts/data_generation/download_isone_data.py     -> data/raw/{year}_smd_hourly.{ext} (this file's siblings)
    2. scripts/data_generation/preprocess_isone_data.py    -> data/raw/isone_rt_hourly_lmp_2016_2017_clean.csv
    3. scripts/data_generation/split_train_test.py         -> data/train/, data/test/
"""


def download_year(year, out_dir, url=None):
    url = url or KNOWN_YEAR_URLS.get(year)
    if url is None:
        raise SystemExit(f"No known SMD Hourly Data URL for {year}; pass --url explicitly.")

    out_path = out_dir / f"{year}_smd_hourly{Path(url).suffix}"
    if out_path.exists():
        print(f"{year}: already have {out_path}, skipping download")
        return out_path

    print(f"{year}: downloading {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"  -> {out_path} ({len(resp.content):,} bytes)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2016, 2017])
    parser.add_argument("--url", default=None, help="Override the source URL (only meaningful with a single --years value)")
    parser.add_argument("--out-dir", default="data/raw", help="Where the original files land (default: data/raw)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        url = args.url if (args.url and len(args.years) == 1) else None
        download_year(year, out_dir, url=url)

    (out_dir / "DATA_SOURCE.txt").write_text(DATA_SOURCE_TXT)
    print(f"\nWrote {out_dir / 'DATA_SOURCE.txt'}")
    print("Next: scripts/data_generation/preprocess_isone_data.py")


if __name__ == "__main__":
    main()
