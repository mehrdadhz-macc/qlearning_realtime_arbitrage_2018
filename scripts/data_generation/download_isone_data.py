"""Download real ISO-NE hourly real-time LMP data via the official Web Services API.

Paper (Wang & Zhang, 2018/2020, arXiv:1711.03127) uses hourly real-time prices
from ISO New England, Jan 1 2016 - Dec 31 2017, at the system Hub. This script
pulls that same series from https://webservices.iso-ne.com/api/v1.1.

Credentials: free ISO Express account, then request Web Services access at
https://www.iso-ne.com/isoexpress/ -> "Web Services". Provide them via env
vars (recommended, keeps them out of shell history) or CLI flags:

    export ISONE_WS_USER=you@example.com
    export ISONE_WS_PASSWORD=your_password
    venv/bin/python3 scripts/data_generation/download_isone_data.py

Each day's response is cached under data/raw/isone_cache/<YYYYMMDD>.json so a
re-run only fetches days that are missing (interrupted runs resume cheaply,
and we don't hammer ISO-NE's servers re-requesting data we already have).

NOTE on location auto-detection: this script queries /locations/all.json and
looks for the entry whose location type / name marks it as the system Hub. I
could not test this against a live account (no credentials available in this
environment), so the parsing is written defensively (recursive key search
rather than one hardcoded JSON path) and falls back to location ID 4000 -- a
value widely cited elsewhere as ISO-NE's Hub ID (".H.INTERNAL_HUB") -- if
auto-detection doesn't match the response shape. The script prints whatever
location name/ID it resolves to before downloading; verify that line says
"HUB" before trusting the output. Override with --location-id if it guesses
wrong. Use --inspect-only to dump one raw day's JSON and exit, which is the
fastest way to fix the parser if ISO-NE's schema differs from what's assumed
here.
"""

import argparse
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API_BASE = "https://webservices.iso-ne.com/api/v1.1"
FALLBACK_HUB_LOCATION_ID = 4000
FALLBACK_HUB_LOCATION_NAME = ".H.INTERNAL_HUB"


def _auth_from_env_or_args(args):
    user = args.user or os.environ.get("ISONE_WS_USER")
    password = args.password or os.environ.get("ISONE_WS_PASSWORD")
    if not user or not password:
        raise SystemExit(
            "Missing ISO-NE Web Services credentials.\n"
            "Register a free ISO Express account and request Web Services access at "
            "https://www.iso-ne.com/isoexpress/, then set:\n"
            "  export ISONE_WS_USER=you@example.com\n"
            "  export ISONE_WS_PASSWORD=your_password\n"
            "or pass --user / --password."
        )
    return (user, password)


def _find_values_by_key(obj, target_keys):
    """Recursively yield values for any of target_keys found anywhere in obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in target_keys:
                yield k, v
            yield from _find_values_by_key(v, target_keys)
    elif isinstance(obj, list):
        for item in obj:
            yield from _find_values_by_key(item, target_keys)


def _find_dicts_with_keys(obj, required_keys):
    """Recursively yield dicts that contain all of required_keys."""
    if isinstance(obj, dict):
        if required_keys.issubset(obj.keys()):
            yield obj
        for v in obj.values():
            yield from _find_dicts_with_keys(v, required_keys)
    elif isinstance(obj, list):
        for item in obj:
            yield from _find_dicts_with_keys(item, required_keys)


def resolve_hub_location(session, auth, override_id=None):
    if override_id is not None:
        print(f"Using --location-id override: {override_id}")
        return override_id

    url = f"{API_BASE}/locations/all.json"
    try:
        resp = session.get(url, auth=auth, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - best-effort discovery, see module docstring
        print(f"Location lookup failed ({exc}); falling back to hardcoded Hub ID "
              f"{FALLBACK_HUB_LOCATION_ID} ({FALLBACK_HUB_LOCATION_NAME}).")
        return FALLBACK_HUB_LOCATION_ID

    candidates = list(_find_dicts_with_keys(payload, {"LocationID"}))
    for cand in candidates:
        name = str(cand.get("LocationName", "")).upper()
        loc_type = str(cand.get("LocationType", "")).upper()
        if "HUB" in name or "HUB" in loc_type:
            print(f"Resolved Hub location: ID={cand['LocationID']} name={cand.get('LocationName')}")
            return cand["LocationID"]

    print(f"Could not find a HUB entry among {len(candidates)} locations returned; "
          f"falling back to hardcoded Hub ID {FALLBACK_HUB_LOCATION_ID} "
          f"({FALLBACK_HUB_LOCATION_NAME}). Run with --inspect-only to see the raw "
          f"locations response and adjust resolve_hub_location() if needed.")
    return FALLBACK_HUB_LOCATION_ID


def fetch_day(session, auth, day_str, location_id):
    url = f"{API_BASE}/hourlylmp/rt/final/day/{day_str}/location/{location_id}.json"
    resp = session.get(url, auth=auth, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_day(payload):
    """Extract (timestamp, lmp_total) pairs from one day's response, defensively."""
    records = list(_find_dicts_with_keys(payload, {"BeginDate", "LmpTotal"}))
    rows = []
    for rec in records:
        rows.append({"timestamp": rec["BeginDate"], "lmp_total": float(rec["LmpTotal"])})
    return rows


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2016-01-01", help="Start date, YYYY-MM-DD (default: 2016-01-01)")
    parser.add_argument("--end", default="2017-12-31", help="End date, YYYY-MM-DD (default: 2017-12-31)")
    parser.add_argument("--user", default=None, help="ISO-NE Web Services username (or set ISONE_WS_USER)")
    parser.add_argument("--password", default=None, help="ISO-NE Web Services password (or set ISONE_WS_PASSWORD)")
    parser.add_argument("--location-id", type=int, default=None, help="Skip auto-detection and use this location ID")
    parser.add_argument("--out-dir", default="data", help="Base output directory (default: data)")
    parser.add_argument("--request-delay", type=float, default=0.15, help="Seconds between requests (default: 0.15)")
    parser.add_argument("--inspect-only", metavar="YYYYMMDD", default=None,
                         help="Fetch and pretty-print one day's raw JSON, then exit (no CSV written)")
    args = parser.parse_args()

    auth = _auth_from_env_or_args(args)
    session = requests.Session()

    out_base = Path(args.out_dir)
    cache_dir = out_base / "raw" / "isone_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.inspect_only:
        location_id = resolve_hub_location(session, auth, args.location_id)
        payload = fetch_day(session, auth, args.inspect_only, location_id)
        print(json.dumps(payload, indent=2)[:5000])
        return

    location_id = resolve_hub_location(session, auth, args.location_id)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    days = list(daterange(start, end))

    all_rows = []
    n_fetched, n_cached, n_failed = 0, 0, 0
    for d in tqdm(days, desc="Downloading ISO-NE hourly RT LMP"):
        day_str = d.strftime("%Y%m%d")
        cache_file = cache_dir / f"{day_str}.json"

        if cache_file.exists():
            payload = json.loads(cache_file.read_text())
            n_cached += 1
        else:
            try:
                payload = fetch_day(session, auth, day_str, location_id)
            except requests.HTTPError as exc:
                print(f"\n  {day_str}: HTTP error {exc}, skipping")
                n_failed += 1
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"\n  {day_str}: {exc}, skipping")
                n_failed += 1
                continue
            cache_file.write_text(json.dumps(payload))
            n_fetched += 1
            time.sleep(args.request_delay)

        all_rows.extend(parse_day(payload))

    if not all_rows:
        raise SystemExit("No data parsed from any day's response. Run with --inspect-only "
                          "on a known-good date to check the response shape.")

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    raw_path = out_base / "raw" / f"isone_rt_hourly_lmp_{start.year}_{end.year}.csv"
    df.to_csv(raw_path, index=False)

    print(f"\nFetched {n_fetched} days new, {n_cached} from cache, {n_failed} failed.")
    print(f"Total rows: {len(df)}. Wrote combined raw series to {raw_path}")

    for year, split in ((2016, "train"), (2017, "test")):
        year_df = df[df["timestamp"].dt.year == year]
        if year_df.empty:
            continue
        split_path = out_base / split / f"isone_rt_hourly_lmp_{year}.csv"
        year_df.to_csv(split_path, index=False)
        print(f"  {split}: {len(year_df)} rows -> {split_path}")


if __name__ == "__main__":
    main()
