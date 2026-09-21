#!/usr/bin/env python3
"""
Fetch historical GP/TLE records for a NORAD catalog ID from Space-Track.org.

Requires free Space-Track credentials via environment variables:
  SPACETRACK_IDENTITY  (or SPACETRACK_USER)
  SPACETRACK_PASSWORD

Example:
  python scripts/fetch_spacetrack_gp_history.py \\
    --norad-id 43780 \\
    --epoch-start 2018-12-01 \\
    --epoch-end 2018-12-10 \\
    --output data/tle_archive/43780_2018-12-01_2018-12-10.tle
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
QUERY_TEMPLATE = (
    "https://www.space-track.org/basicspacedata/query/class/gp_history/"
    "NORAD_CAT_ID/{norad_id}/EPOCH/{epoch_range}/orderby/EPOCH asc/format/tle"
)


def parse_tle_epoch(tle1: str) -> datetime:
    parts = tle1.split()
    if len(parts) < 4:
        raise ValueError(f"Bad TLE line 1: {tle1!r}")
    yyddd = float(parts[3])
    yy = int(yyddd / 1000)
    year = 2000 + yy if yy < 57 else 1900 + yy
    from datetime import timedelta

    doy = yyddd % 1000
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)


def parse_tle_blocks(text: str) -> list[dict]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    records: list[dict] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 "):
            tle0 = ""
            tle1 = lines[i]
            if i + 1 >= len(lines) or not lines[i + 1].startswith("2 "):
                raise ValueError(f"Malformed TLE near line {i + 1}")
            tle2 = lines[i + 1]
            i += 2
        elif lines[i].startswith("0 ") and i + 2 < len(lines):
            tle0 = lines[i]
            tle1 = lines[i + 1]
            tle2 = lines[i + 2]
            i += 3
        else:
            i += 1
            continue
        if not tle1.startswith("1 "):
            continue
        epoch = parse_tle_epoch(tle1)
        records.append(
            {
                "tle0": tle0.strip(),
                "tle1": tle1.strip(),
                "tle2": tle2.strip(),
                "epoch_utc": epoch.isoformat(),
                "epoch_yyddd": tle1.split()[3],
            }
        )
    return records


def login(session: requests.Session, identity: str, password: str) -> None:
    resp = session.post(
        LOGIN_URL,
        data={"identity": identity, "password": password},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Space-Track login failed HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        return
    if isinstance(payload, dict) and payload.get("Login"):
        raise RuntimeError(f"Space-Track login rejected: {payload['Login']}")


def fetch_gp_history(
    norad_id: int,
    epoch_start: str,
    epoch_end: str,
    identity: str,
    password: str,
) -> str:
    session = requests.Session()
    login(session, identity, password)
    epoch_range = f"{epoch_start}--{epoch_end}"
    url = QUERY_TEMPLATE.format(norad_id=norad_id, epoch_range=epoch_range)
    resp = session.get(url, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"gp_history query failed HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.text


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Space-Track gp_history TLEs")
    ap.add_argument("--norad-id", type=int, default=43780)
    ap.add_argument("--epoch-start", default="2018-12-01")
    ap.add_argument("--epoch-end", default="2018-12-10")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/tle_archive/43780_2018-12-01_2018-12-10.tle"),
    )
    args = ap.parse_args()

    identity = os.environ.get("SPACETRACK_IDENTITY") or os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not identity or not password:
        print(
            "ERROR: Set SPACETRACK_IDENTITY (or SPACETRACK_USER) and SPACETRACK_PASSWORD.",
            file=sys.stderr,
        )
        sys.exit(1)

    text = fetch_gp_history(args.norad_id, args.epoch_start, args.epoch_end, identity, password)
    records = parse_tle_blocks(text)
    if not records:
        print(f"WARNING: gp_history returned 0 TLE records for NORAD {args.norad_id}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")

    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "source": "space-track.org gp_history",
                "norad_id": args.norad_id,
                "epoch_start": args.epoch_start,
                "epoch_end": args.epoch_end,
                "n_records": len(records),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} TLE(s) → {args.output}")
    print(f"Metadata → {meta_path}")


if __name__ == "__main__":
    main()
