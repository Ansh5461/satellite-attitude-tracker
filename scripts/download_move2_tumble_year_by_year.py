#!/usr/bin/env python3
"""Wait for SatNOGS throttle to clear, then download tumble-era year by year."""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

import requests

LOG_PATH = Path("data_move2_tracked/download_year_by_year.log")


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
    )
    return logging.getLogger("yby")


def throttle_remaining() -> float:
    try:
        r = requests.get(
            "https://network.satnogs.org/api/observations/",
            params={
                "norad_cat_id": 43780,
                "status": "good",
                "start": "2018-11-01",
                "end": "2019-01-01",
                "format": "json",
            },
            timeout=30,
            headers={"User-Agent": "move2-tumble-download/1.0"},
        )
        if r.status_code == 429:
            return float(r.headers.get("Retry-After") or 60)
        return 0.0
    except Exception:  # noqa: BLE001
        return 60.0


def main() -> int:
    log = setup_logging()
    while True:
        rem = throttle_remaining()
        if rem <= 0:
            log.info("API open — starting year downloads")
            break
        sleep_for = min(max(rem, 5), 3600)
        log.info("Throttled — sleeping %.0fs (Retry-After=%.0f)", sleep_for, rem)
        time.sleep(sleep_for)

    windows = [
        ("2018", "2018-11-01", "2019-01-01"),
        ("2019", "2019-01-01", "2020-01-01"),
        ("2020", "2020-01-01", "2021-01-01"),
    ]
    cache_dir = Path("data_move2_tracked/api_cache/norad_43780_tracked_tumble_era_v1")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for label, start, end in windows:
        cache = cache_dir / f"full_list_{label}.json"
        cmd = [
            sys.executable,
            "scripts/download_move2_tumble_era.py",
            "--start",
            start,
            "--end",
            end,
            "--api-delay",
            "12.0",
            "--data-dir",
            "data_move2_tracked",
            "--reuse-dirs",
            "data",
            "--list-cache",
            str(cache),
            "--refresh-list",
        ]
        log.info("=== Starting window %s (%s → %s) ===", label, start, end)
        rc = subprocess.call(cmd)
        log.info("=== Window %s finished rc=%s ===", label, rc)
        time.sleep(30)

    log.info("ALL WINDOWS DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
