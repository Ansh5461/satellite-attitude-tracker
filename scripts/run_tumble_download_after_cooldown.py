#!/usr/bin/env python3
"""Sleep until SatNOGS ban lifts, then download 2018 → 2019 → 2020 with long delay."""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

LOG = Path("data_move2_tracked/download_year_by_year.log")


def log_setup() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG)],
    )


def seconds_until_open() -> float:
    """Poll Retry-After without burning quota once open.

    On 429: return Retry-After.
    On 200: we already spent one request — return a cool-off (90s) so the
    real downloader does not immediately re-trip the throttle.
    """
    try:
        r = requests.get(
            "https://network.satnogs.org/api/observations/",
            params={
                "norad_cat_id": "43780",
                "status": "good",
                "start": "2018-11-01",
                "end": "2019-01-01",
                "format": "json",
            },
            timeout=30,
            headers={"User-Agent": "move2-cooldown-probe/1.0"},
        )
        if r.status_code == 429:
            return float(r.headers.get("Retry-After") or 120)
        # Success: cool off so listing starts with a fresh budget.
        return 90.0
    except Exception as exc:  # noqa: BLE001
        logging.warning("probe failed: %s", exc)
        return 120.0


def main() -> int:
    log_setup()
    # Hard floor: do not start before 00:05 UTC 9 Aug 2026 (post 23:59 ban).
    floor = datetime(2026, 8, 9, 0, 5, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if now < floor:
        wait = (floor - now).total_seconds()
        logging.info("Hard cooldown until %s (%.0fs)", floor.isoformat(), wait)
        time.sleep(wait)

    while True:
        rem = seconds_until_open()
        if rem <= 90.5:
            logging.info("Cooldown complete (rem=%.0fs) — launching downloads", rem)
            if rem > 0:
                time.sleep(rem)
            break
        sleep_for = min(rem, 3600)
        logging.info("Still throttled — sleeping %.0fs", sleep_for)
        time.sleep(sleep_for)

    cache_dir = Path("data_move2_tracked/api_cache/norad_43780_tracked_tumble_era_v1")
    cache_dir.mkdir(parents=True, exist_ok=True)
    windows = [
        ("2018", "2018-11-01", "2019-01-01"),
        ("2019", "2019-01-01", "2020-01-01"),
        ("2020", "2020-01-01", "2021-01-01"),
    ]
    for label, start, end in windows:
        cache = cache_dir / f"full_list_{label}.json"
        # Resume from cache if listing already finished for this year.
        refresh = not cache.is_file()
        cmd = [
            sys.executable,
            "scripts/download_move2_tumble_era.py",
            "--start",
            start,
            "--end",
            end,
            "--api-delay",
            "20.0",
            "--data-dir",
            "data_move2_tracked",
            "--reuse-dirs",
            "data",
            "--list-cache",
            str(cache),
        ]
        if refresh:
            cmd.append("--refresh-list")
        logging.info("=== %s (%s → %s) refresh_list=%s ===", label, start, end, refresh)
        rc = subprocess.call(cmd)
        logging.info("=== %s finished rc=%s ===", label, rc)
        # Extra gap between years
        time.sleep(120)
    logging.info("ALL WINDOWS DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
