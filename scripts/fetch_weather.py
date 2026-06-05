"""Fetch weather time-series from NASA POWER and upload as CSV to backend.

Usage:
  python scripts/fetch_weather.py --start 2026-05-01 --end 2026-05-02 \
    --lat 51.509865 --lon -0.118092 --out out.csv --upload-url http://localhost:8000/simulation/data/upload-weather --token $TOKEN

Options:
  - start, end: ISO dates (YYYY-MM-DD)
  - lat, lon: location coordinates
  - variables: comma-separated NASA POWER parameters (default: ALLSKY_SFC_SW_DWN,T2M,WS10M,RH2M)
  - timezone: UTC by default

This script writes a CSV with a timestamp column and requested variables, and optionally posts it
to the backend upload endpoint using a Bearer token.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from typing import List, Optional

import requests

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def fetch_nasa_power(
    start: str,
    end: str,
    lat: float,
    lon: float,
    variables: List[str],
    timezone: str = "UTC",
) -> dict:
    params = {
        "start": start.replace("-", ""),
        "end": end.replace("-", ""),
        "latitude": lat,
        "longitude": lon,
        "community": "AG",
        "parameters": ",".join(variables),
        "format": "JSON",
        "time-standard": timezone,
    }
    resp = requests.get(NASA_POWER_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def json_to_csv(json_payload: dict, variables: List[str], out_path: str) -> int:
    # NASA POWER returns 'properties' -> 'parameter' mapping with date-keyed dicts
    params = json_payload.get("properties", {}).get("parameter", {})
    if not params:
        raise RuntimeError("Unexpected NASA POWER response structure")

    # date keys come from any parameter; pick first variable to enumerate dates
    first_var = variables[0]
    dates = sorted(params.get(first_var, {}).keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["timestamp"] + variables
        writer.writerow(header)
        for d in dates:
            row = [d]
            for v in variables:
                val = params.get(v, {}).get(d, None)
                row.append("" if val is None else val)
            writer.writerow(row)

    return len(dates)


def upload_file(upload_url: str, file_path: str, token: Optional[str] = None) -> dict:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with open(file_path, "rb") as fd:
        files = {"file": (os.path.basename(file_path), fd, "text/csv")}
        resp = requests.post(upload_url, files=files, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch NASA POWER and upload CSV")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--lat", required=True, type=float)
    parser.add_argument("--lon", required=True, type=float)
    parser.add_argument("--variables", default="ALLSKY_SFC_SW_DWN,T2M,WS10M,RH2M")
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--out", default="weather_out.csv")
    parser.add_argument("--upload-url", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args(argv)

    vars_list = [v.strip() for v in args.variables.split(",") if v.strip()]

    print(
        f"Fetching NASA POWER for {args.lat},{args.lon} {args.start}..{args.end} vars={vars_list}"
    )
    try:
        j = fetch_nasa_power(
            args.start, args.end, args.lat, args.lon, vars_list, args.timezone
        )
    except Exception as e:
        print("Fetch failed:", e, file=sys.stderr)
        return 2

    try:
        rows = json_to_csv(j, vars_list, args.out)
    except Exception as e:
        print("Failed to write CSV:", e, file=sys.stderr)
        return 3

    print(f"Wrote {rows} rows → {args.out}")

    if args.upload_url:
        try:
            print(f"Uploading to {args.upload_url}")
            res = upload_file(args.upload_url, args.out, token=args.token)
            print("Upload response:", res)
        except Exception as e:
            print("Upload failed:", e, file=sys.stderr)
            return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
