#!/usr/bin/env python3
"""
Build data/manifest_filtered.json from locally downloaded MOVE-II observations.

Reads meta/observation_*.json under data/move2_* folders — no network calls.

Applies the standard quality cuts:
  - drop passes with no local waterfall PNG
  - drop passes with missing station coordinates

Every surviving pass is tagged with lat_lng_precision (min decimal places in
lat/lng) and tle_age_hours (observation start minus TLE epoch). Coarse-coordinate
stations are kept — only tagged, not filtered out.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_move2_observations import (  # noqa: E402
    DECAY_DATE,
    DEFAULT_SAT_ID,
    DEFAULT_STATUS,
    LAUNCH_DATE,
    lat_lng_precision_score,
    observation_folder_name,
    setup_logging,
    tle_age_hours,
    tle_epoch_from_line1,
)


def has_coordinates(obs: dict) -> bool:
    lat = obs.get("station_lat")
    lng = obs.get("station_lng")
    return lat is not None and lng is not None and lat != "" and lng != ""


def has_waterfall(obs: dict, output_dir: Path) -> bool:
    folder = output_dir / observation_folder_name(obs)
    wf_path = folder / "raw" / f"waterfall_{obs['id']}.png"
    return wf_path.is_file() and wf_path.stat().st_size > 0


def manifest_entry(obs: dict, *, output_dir: Path) -> dict[str, Any]:
    epoch = tle_epoch_from_line1(obs.get("tle1") or "")
    age = tle_age_hours(obs)
    folder = output_dir / observation_folder_name(obs)
    return {
        "id": obs["id"],
        "start": obs.get("start"),
        "end": obs.get("end"),
        "ground_station": obs.get("ground_station"),
        "station_name": obs.get("station_name"),
        "station_lat": obs.get("station_lat"),
        "station_lng": obs.get("station_lng"),
        "station_alt": obs.get("station_alt"),
        "lat_lng_precision": lat_lng_precision_score(
            obs.get("station_lat"), obs.get("station_lng")
        ),
        "max_altitude": obs.get("max_altitude"),
        "tle_epoch": epoch.isoformat() if epoch else None,
        "tle_age_hours": round(age, 3) if age is not None else None,
        "transmitter_mode": obs.get("transmitter_mode"),
        "transmitter_uuid": obs.get("transmitter_uuid"),
        "observation_frequency": obs.get("observation_frequency"),
        "status": obs.get("status"),
        "waterfall_url": obs.get("waterfall"),
        "folder": str(folder),
    }


def build_manifest(
    observations: list[dict],
    *,
    output_dir: Path,
    sat_id: str,
    status: str,
) -> dict[str, Any]:
    dropped_waterfall: list[dict[str, Any]] = []
    dropped_no_coords: list[dict[str, Any]] = []
    included: list[dict[str, Any]] = []

    for obs in observations:
        entry = manifest_entry(obs, output_dir=output_dir)

        if not has_coordinates(obs):
            dropped_no_coords.append({**entry, "drop_reason": "no_coordinates"})
            continue

        if not has_waterfall(obs, output_dir):
            dropped_waterfall.append({**entry, "drop_reason": "missing_waterfall"})
            continue

        included.append(entry)

    precision_counts: dict[int, int] = {}
    for e in included:
        p = e["lat_lng_precision"]
        precision_counts[p] = precision_counts.get(p, 0) + 1

    starts = [o["start"] for o in observations if o.get("start")]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "local data/ move2_*/meta/observation_*.json",
        "sat_id": sat_id,
        "status": status,
        "date_range": {
            "launch": LAUNCH_DATE.isoformat(),
            "decay": DECAY_DATE.isoformat(),
            "data_start": min(starts) if starts else None,
            "data_end": max(starts) if starts else None,
        },
        "cuts": {
            "missing_waterfall": "drop passes with no local raw/waterfall_{id}.png",
            "no_coordinates": "drop passes with null/missing station_lat or station_lng",
            "coarse_coordinates": "kept — lat_lng_precision tags only, not filtered",
        },
        "summary": {
            "total_local": len(observations),
            "included": len(included),
            "dropped_missing_waterfall": len(dropped_waterfall),
            "dropped_no_coordinates": len(dropped_no_coords),
            "lat_lng_precision_counts": {
                str(k): v for k, v in sorted(precision_counts.items())
            },
        },
        "dropped": {
            "missing_waterfall": dropped_waterfall,
            "no_coordinates": dropped_no_coords,
        },
        "observations": included,
    }


def load_local_observations(output_dir: Path) -> list[dict]:
    obs_list: list[dict] = []
    for meta_path in sorted(output_dir.glob("move2_*/meta/observation_*.json")):
        with open(meta_path, encoding="utf-8") as fh:
            obs_list.append(json.load(fh))
    obs_list.sort(key=lambda o: o["start"])
    return obs_list


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build manifest_filtered.json from local data/ observations."
    )
    p.add_argument("--sat-id", default=DEFAULT_SAT_ID)
    p.add_argument("--status", default=DEFAULT_STATUS)
    p.add_argument("--output-dir", default="data")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    setup_logging(output_dir)

    observations = load_local_observations(output_dir)
    if not observations:
        logging.error("No local observations found under %s/move2_*/meta/", output_dir)
        return 1

    logging.info("Loaded %d observations from local meta/ JSON", len(observations))

    manifest = build_manifest(
        observations,
        output_dir=output_dir,
        sat_id=args.sat_id,
        status=args.status,
    )

    out_path = output_dir / "manifest_filtered.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    s = manifest["summary"]
    logging.info(
        "Wrote %s: %d included, %d dropped (waterfall=%d, no_coords=%d)",
        out_path,
        s["included"],
        s["total_local"] - s["included"],
        s["dropped_missing_waterfall"],
        s["dropped_no_coordinates"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
