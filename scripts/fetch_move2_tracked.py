#!/usr/bin/env python3
"""
Fresh SatNOGS download for genuine MOVE-II (NORAD 43780) observations.

Uses API filter norad_cat_id=43780 (tracked TLE object), not transmitter
association. Defaults target the early tumble era 2018-11-01 → 2021-01-01.

Quick start:

  python scripts/fetch_move2_tracked.py

Then:

  python scripts/run_move2_tracked_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

MOVEII_NORAD = 43780
LAUNCH_DATE = "2018-11-01"
# Exclusive upper bound for SatNOGS `end=` filter — covers through 2020-12-31.
TUMBLE_ERA_END = "2021-01-01"


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "scripts"))

    from fetch_and_vet_all_observations import main as fetch_main

    user_argv = list(argv if argv is not None else sys.argv[1:])

    # Build argv with defaults the user can override by passing the same flag.
    def take(flag: str, default: str) -> list[str]:
        if flag in user_argv:
            return []
        return [flag, default]

    argv_out = [
        *take("--norad-id", str(MOVEII_NORAD)),
        *take("--require-tracked-norad-id", str(MOVEII_NORAD)),
        *take("--status", "good"),
        *take("--start-after", LAUNCH_DATE),
        *take("--start-before", TUMBLE_ERA_END),
        *take("--order", "oldest"),
        *take("--window-days", "90"),
        *take("--reuse-dirs", "data"),
        *take("--data-dir", "data_move2_tracked"),
        *take("--filtered-dir", "data_move2_tracked/filtered_data"),
        *take(
            "--cache-dir",
            "data_move2_tracked/api_cache/norad_43780_tracked_tumble_era_v1",
        ),
        *user_argv,
    ]
    return fetch_main(argv_out)


if __name__ == "__main__":
    raise SystemExit(main())
