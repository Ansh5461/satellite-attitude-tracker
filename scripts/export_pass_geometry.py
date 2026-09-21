#!/usr/bin/env python3
"""
Export real observation geometry for the two cross-validated 165.0 s MOVE-II passes:
  - Obs 349779  (Dec  4 2018, Binar-SN-01)
  - Obs 359214  (Dec 11 2018, KO2F-VHF-1)

For each pass the script produces a JSON file with:
  timestamps          : per-sample UTC strings (non-gap rows, same grid as MC aliasing)
  t_rel_s             : seconds from first valid sample (zero-shifted, matches mc_aliasing)
  station_lat         : ground station latitude (deg, WGS-84)
  station_lon         : ground station longitude (deg, WGS-84)
  station_alt_m       : ground station altitude (m, WGS-84)
  sat_gcrs_km         : 3×N satellite position in GCRS/ECI frame (km), from the same TLE
  los_unit_vector     : 3×N unit vector from station to satellite in GCRS frame
  amplitude           : path-loss-corrected amplitude (dB) at each valid timestamp

Frame notes
-----------
  GCRS (Geocentric Celestial Reference System, approximately J2000 ECI) is used
  throughout.  skyfield's EarthSatellite.at(t).position.km returns GCRS.
  The topocentric vector (satellite - station).at(t).position.km is also in GCRS
  and is used directly as the LOS vector before normalisation.

Timestamp grid
--------------
  Non-gap rows only (is_gap == 0), sorted ascending by t_rel_s, then zero-shifted —
  identical to how mc_aliasing_165s.py loads the data for its Lomb-Scargle runs.
  The absolute UTC is reconstructed by adding each relative time to obs_start.

Outputs
-------
  data/pass_geometry/pass_349779_geometry.json
  data/pass_geometry/pass_359214_geometry.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = PROJECT_ROOT / "data" / "amplitude_pipeline"
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "pass_geometry"

OBSERVATIONS = [
    {"obs_id": 349779, "date": "2018-12-04", "station": "Binar-SN-01"},
    {"obs_id": 359214, "date": "2018-12-11", "station": "KO2F-VHF-1"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_iso(s: str) -> datetime:
    """Parse an ISO-8601 UTC string (with or without trailing Z)."""
    s = s.rstrip("Z")
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc)


def load_meta(obs_id: int) -> dict:
    """
    Load the observation JSON from data/move2_<id>_<date>/meta/.
    Falls back to searching all move2_* folders if the exact name is unknown.
    """
    for root_name in ("data", "filtered_data"):
        root = PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for candidate in root.iterdir():
            if candidate.is_dir() and candidate.name.startswith(f"move2_{obs_id}_"):
                meta_path = candidate / "meta" / f"observation_{obs_id}.json"
                if meta_path.is_file():
                    with open(meta_path, encoding="utf-8") as fh:
                        return json.load(fh)
    raise FileNotFoundError(f"Meta JSON for obs {obs_id} not found under data/ or filtered_data/")


def load_npz(obs_id: int) -> np.lib.npyio.NpzFile:
    """Load the amplitude-pipeline NPZ for this observation."""
    path = PIPELINE_DIR / f"obs_{obs_id}" / f"amplitude_{obs_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"NPZ not found: {path}")
    return np.load(path)


def valid_indices(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    """
    Boolean mask of non-gap samples, sorted by t_rel_s ascending.
    Matches the filtering in mc_aliasing_165s.py::load_valid_timestamps().
    """
    t_all   = npz["t_rel_s"].astype(float)
    is_gap  = npz["is_gap"].astype(bool)
    valid   = ~is_gap
    # Sort indices by t_rel_s (the NPZ is stored in reverse time order)
    sort_order = np.argsort(t_all[valid])
    # Return the original NPZ indices corresponding to valid, sorted samples
    valid_idx = np.where(valid)[0][sort_order]
    return valid_idx


# ---------------------------------------------------------------------------
# SGP4 / geometry computation
# ---------------------------------------------------------------------------

def compute_geometry(
    meta: dict,
    timestamps: list[datetime],
) -> dict[str, np.ndarray]:
    """
    Run skyfield SGP4 propagation for the given UTC timestamps.

    Returns a dict with:
      sat_gcrs_km     : (N, 3)  satellite GCRS position (km)
      los_unit_vector : (N, 3)  unit vector station→satellite in GCRS
    """
    ts        = load.timescale()
    satellite = EarthSatellite(
        meta["tle1"], meta["tle2"], meta.get("tle0", ""), ts
    )
    station   = wgs84.latlon(
        latitude_degrees  = float(meta["station_lat"]),
        longitude_degrees = float(meta["station_lng"]),
        elevation_m       = float(meta.get("station_alt") or 0.0),
    )

    sky_times = ts.from_datetimes(timestamps)

    # Satellite GCRS position (geocentric, km)
    sat_gcrs_km = satellite.at(sky_times).position.km.T   # (N, 3)

    # Topocentric vector from station to satellite (GCRS, km)
    topo_km     = (satellite - station).at(sky_times).position.km.T  # (N, 3)

    # Normalise to unit LOS vector
    norms            = np.linalg.norm(topo_km, axis=1, keepdims=True)
    los_unit_vector  = topo_km / np.maximum(norms, 1e-12)

    return {
        "sat_gcrs_km"    : sat_gcrs_km,
        "los_unit_vector": los_unit_vector,
    }


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_pass(obs_id: int) -> Path:
    print(f"\n{'='*60}")
    print(f"Exporting obs {obs_id} ...")

    meta = load_meta(obs_id)
    npz  = load_npz(obs_id)

    obs_start = parse_iso(meta["start"])
    print(f"  obs start    : {obs_start.isoformat()}")
    print(f"  station      : {meta['station_name']}  "
          f"({meta['station_lat']:.4f}°, {meta['station_lng']:.4f}°, "
          f"{meta.get('station_alt', 0):.0f} m)")

    # ── Non-gap indices, sorted by t_rel_s ascending ────────────────────────
    valid_idx = valid_indices(npz)
    t_rel_all = npz["t_rel_s"].astype(float)
    amp_all   = npz["amplitude_pathloss_corrected"].astype(float)

    t_rel_valid = t_rel_all[valid_idx]          # seconds from obs_start, ascending
    amp_valid   = amp_all[valid_idx]

    N = len(valid_idx)
    print(f"  valid samples: {N} / {len(t_rel_all)} total  "
          f"({100 * N / len(t_rel_all):.1f}% non-gap)")

    # Zero-shift relative time (identical to mc_aliasing convention)
    t_zero = t_rel_valid[0]
    t_rel_zeroshifted = t_rel_valid - t_zero

    print(f"  t_rel span   : {t_rel_zeroshifted[0]:.3f} – {t_rel_zeroshifted[-1]:.3f} s  "
          f"(first absolute: t_rel={t_zero:.3f} s from obs_start)")

    # ── Reconstruct UTC timestamps (not zero-shifted — full UTC) ─────────────
    utc_timestamps = [
        obs_start + timedelta(seconds=float(t))
        for t in t_rel_valid
    ]
    utc_strings = [dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                   for dt in utc_timestamps]

    # ── SGP4 propagation ─────────────────────────────────────────────────────
    print("  Running SGP4 propagation …")
    geom = compute_geometry(meta, utc_timestamps)
    sat_gcrs_km     = geom["sat_gcrs_km"]       # (N, 3)
    los_unit_vector = geom["los_unit_vector"]   # (N, 3)

    # Sanity check: slant range from LOS magnitude before normalisation
    # (already normalised, so recompute from sat − station_gcrs if needed)
    slant_range_km = npz["range_km"].astype(float)[valid_idx]
    print(f"  slant range  : {slant_range_km.min():.1f} – {slant_range_km.max():.1f} km")

    # ── Build output dict ─────────────────────────────────────────────────────
    out = {
        "_meta": {
            "obs_id"          : obs_id,
            "station_name"    : meta["station_name"],
            "obs_start_utc"   : obs_start.isoformat(),
            "obs_end_utc"     : meta["end"],
            "tle0"            : meta.get("tle0", ""),
            "tle1"            : meta["tle1"],
            "tle2"            : meta["tle2"],
            "norad_cat_id"    : meta.get("norad_cat_id"),
            "n_valid_samples" : N,
            "n_total_rows"    : int(len(t_rel_all)),
            "frac_gap"        : float(1.0 - N / len(t_rel_all)),
            "frame"           : "GCRS (J2000 ECI, skyfield)",
            "amplitude_formula": "amplitude_pathloss_corrected = "
                                 "amplitude_interp + 20·log10(range_km / 1000 km) [dB]",
            "timestamp_note"  : (
                "UTC timestamps reconstructed as obs_start + t_rel_s. "
                "t_rel_s is zero-shifted so t_rel_s[0] == 0.0, "
                "matching the mc_aliasing_165s.py convention."
            ),
        },
        "station_lat"    : float(meta["station_lat"]),
        "station_lon"    : float(meta["station_lng"]),
        "station_alt_m"  : float(meta.get("station_alt") or 0.0),
        "timestamps"     : utc_strings,
        "t_rel_s"        : t_rel_zeroshifted.tolist(),
        "sat_gcrs_km"    : sat_gcrs_km.tolist(),
        "los_unit_vector": los_unit_vector.tolist(),
        "amplitude"      : amp_valid.tolist(),
    }

    # ── Write JSON ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"pass_{obs_id}_geometry.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"  → {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("MOVE-II pass geometry export")
    print(f"Pipeline dir : {PIPELINE_DIR}")
    print(f"Output dir   : {OUTPUT_DIR}")

    written: list[Path] = []
    for obs in OBSERVATIONS:
        path = export_pass(obs["obs_id"])
        written.append(path)

    # Quick cross-check: confirm non-gap sample counts match mc_aliasing docstring
    print("\n" + "=" * 60)
    print("Cross-check (expected from mc_aliasing_165s.py docstring):")
    print("  obs 349779: 814 valid pts, 47% gap")
    print("  obs 359214: 1203 valid pts, 22% gap")
    for path in written:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        m  = d["_meta"]
        print(f"  obs {m['obs_id']}: {m['n_valid_samples']} valid pts, "
              f"{m['frac_gap']*100:.0f}% gap")

    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
