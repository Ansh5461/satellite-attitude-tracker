#!/usr/bin/env python3
"""
Re-run MOVE-II validation chain with archival NORAD 43780 TLEs.

Steps:
  1. Load TLE archive (Space-Track gp_history file or JSON metadata)
  2. Epoch-match TLE per affected pass; report epoch gaps (hours)
  3. Doppler re-fit (carrier_doppler_check.predict_doppler) for 4 passes
  4. Recompute path-loss-corrected amplitudes with corrected range_km
  5. Lomb-Scargle on 6 confirmed passes
  6. Monte Carlo aliasing test (349779 vs 359214) at 120k trials

Usage:
  # After fetching TLEs:
  python scripts/archival_tle_validation_chain.py \\
    --tle-archive data/tle_archive/43780_2018-12-01_2018-12-10.json

  # Attempt Space-Track fetch first (needs env credentials):
  python scripts/archival_tle_validation_chain.py --fetch-spacetrack
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from carrier_doppler_check import (  # noqa: E402
    DEFAULT_EXCLUDE_FREQ_ABOVE_KHZ,
    DEFAULT_EXCLUDE_FREQ_BELOW_KHZ,
    DEFAULT_MIN_CONFIDENCE_PERCENTILE,
    extract_carrier_track_waterfall,
    fit_doppler,
    find_obs_folder,
    load_meta,
    parse_iso,
    predict_doppler,
)
from run_amplitude_pipeline import (  # noqa: E402
    RANGE_REF_KM,
    compute_slant_range_km,
    path_loss_correct,
)
from run_full_waterfall_sgp4_validation import classify_result  # noqa: E402
from lomb_scargle_tumble import (  # noqa: E402
    analyse_obs,
    find_npz,
    run_ls,
    _flag_cw_alias,
)

AFFECTED_IDS = [349561, 349779, 351032, 354023]
LS_SIX_IDS = [349561, 349779, 351032, 354023, 359214, 360071]
PROJECT_ROOT = _SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "archival_tle_validation"

# Original wrong-catalog embedded fits (43759/43761/43775 TLEs in tle1)
ORIGINAL_DOPPLER = {
    349561: {"B": 0.9961, "R2": 0.9886, "n": 318, "embedded_cat": 43761},
    349779: {"B": 0.9875, "R2": 0.9359, "n": 316, "embedded_cat": 43759},
    351032: {"B": 0.9986, "R2": 0.9999, "n": 335, "embedded_cat": 43759},
    354023: {"B": 0.8882, "R2": 0.9960, "n": 314, "embedded_cat": 43775},
}

# Baseline Lomb-Scargle peaks (deg=1, path-loss corrected, original TLE range)
BASELINE_LS_PEAKS = {
    349561: 51.5923,
    349779: 165.0008,
    351032: 215.809,
    354023: 243.6344,
    359214: 165.0056,
    360071: 69.1563,
}


def parse_tle_epoch(tle1: str) -> datetime:
    parts = tle1.split()
    yyddd = float(parts[3])
    yy = int(yyddd / 1000)
    year = 2000 + yy if yy < 57 else 1900 + yy
    doy = yyddd % 1000
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)


def load_tle_archive(path: Path) -> list[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "records" in data:
            return data["records"]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unrecognised JSON archive format: {path}")
    text = path.read_text(encoding="utf-8")
    from fetch_spacetrack_gp_history import parse_tle_blocks

    return parse_tle_blocks(text)


def select_nearest_tle(records: list[dict], obs_start: datetime) -> tuple[dict, float]:
    best = None
    best_gap_h = float("inf")
    for rec in records:
        ep = parse_tle_epoch(rec["tle1"])
        gap_h = abs((obs_start - ep).total_seconds()) / 3600.0
        if gap_h < best_gap_h:
            best_gap_h = gap_h
            best = rec
    if best is None:
        raise RuntimeError("No TLE records in archive")
    return best, best_gap_h


def load_carrier_track(obs_id: int) -> tuple[dict, dict]:
    obs_folder = find_obs_folder(obs_id, PROJECT_ROOT)
    meta = load_meta(obs_folder, obs_id)
    obs_start = parse_iso(meta["start"])
    obs_end = parse_iso(meta["end"])
    duration_s = (obs_end - obs_start).total_seconds()
    f0_hz = float(meta["observation_frequency"])
    wf_path = next((obs_folder / "raw").glob("waterfall_*.png"))
    track = extract_carrier_track_waterfall(
        wf_path,
        obs_start,
        duration_s,
        f0_hz=f0_hz,
        exclude_freq_below_khz=DEFAULT_EXCLUDE_FREQ_BELOW_KHZ,
        exclude_freq_above_khz=DEFAULT_EXCLUDE_FREQ_ABOVE_KHZ,
        min_confidence_percentile=DEFAULT_MIN_CONFIDENCE_PERCENTILE,
    )
    return meta, track


def doppler_refit(meta: dict, track: dict, tle_rec: dict) -> dict[str, Any]:
    m = deepcopy(meta)
    m["tle0"] = tle_rec.get("tle0") or "OBJECT Y"
    m["tle1"] = tle_rec["tle1"]
    m["tle2"] = tle_rec["tle2"]

    strong_idx = np.where(track["strong_mask"])[0]
    strong_freq = track["peak_freq_hz"][strong_idx]
    pred = predict_doppler(m, track["timestamps"], track["strong_mask"])
    fit = fit_doppler(strong_freq, pred)
    B, R2 = float(fit["B"]), float(fit["r2"])
    cls = classify_result(B, R2, m.get("station_name", ""), unknown_layout=False)

    obs_id = int(meta["id"])
    frac_gap = 0.0
    meta_json = PROJECT_ROOT / "data" / "amplitude_pipeline" / f"obs_{obs_id}" / f"amplitude_{obs_id}_meta.json"
    if meta_json.is_file():
        frac_gap = float(json.loads(meta_json.read_text())["gap_stats"]["frac_gap"])

    amp_q = R2 * (1.0 - frac_gap)
    return {
        "obs_id": obs_id,
        "B": B,
        "R2": R2,
        "n": int(len(strong_idx)),
        "classification": cls,
        "amplitude_quality": amp_q,
        "frac_gap": frac_gap,
        "tle1": m["tle1"],
        "tle2": m["tle2"],
        "meta_corrected": m,
    }


def recompute_amplitudes(obs_id: int, meta_corrected: dict, track: dict) -> dict[str, Any]:
    npz_path = find_npz(PROJECT_ROOT / "data" / "amplitude_pipeline", obs_id)
    if npz_path is None:
        raise FileNotFoundError(f"No amplitude NPZ for obs {obs_id}")
    z = np.load(npz_path)
    t_rel = z["t_rel_s"].astype(float)
    amp_interp = z["amplitude_interp"].astype(float)
    is_gap = z["is_gap"].astype(bool)
    obs_start = parse_iso(meta_corrected["start"])
    timestamps = [obs_start + timedelta(seconds=float(t)) for t in t_rel]

    range_old = z["range_km"].astype(float) if "range_km" in z else None
    range_new = compute_slant_range_km(meta_corrected, timestamps)
    amp_corr_new = path_loss_correct(amp_interp, range_new, RANGE_REF_KM)

    if range_old is not None:
        amp_corr_old = z["amplitude_pathloss_corrected"].astype(float)
        delta = amp_corr_new - amp_corr_old
        valid = ~is_gap
        stats = {
            "mean_delta_db": float(np.mean(delta[valid])),
            "max_abs_delta_db": float(np.max(np.abs(delta[valid]))),
            "mean_range_delta_km": float(np.mean(range_new[valid] - range_old[valid])),
            "max_abs_range_delta_km": float(np.max(np.abs(range_new[valid] - range_old[valid]))),
        }
    else:
        stats = {}

    out_npz = OUTPUT_DIR / "amplitude_pipeline" / f"obs_{obs_id}" / f"amplitude_{obs_id}.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    save = {k: z[k] for k in z.files}
    save["range_km"] = range_new
    save["amplitude_pathloss_corrected"] = amp_corr_new
    save["amplitude_pathloss_corrected_masked"] = np.where(is_gap, np.nan, amp_corr_new)
    np.savez_compressed(out_npz, **save)
    return {"npz_path": str(out_npz), **stats}


def run_lomb_scargle_six() -> list[dict]:
    results = []
    pipe = OUTPUT_DIR / "amplitude_pipeline"
    for obs_id in LS_SIX_IDS:
        npz = pipe / f"obs_{obs_id}" / f"amplitude_{obs_id}.npz"
        if not npz.is_file():
            npz = find_npz(PROJECT_ROOT / "data" / "amplitude_pipeline", obs_id)
        if npz is None:
            continue
        z = np.load(npz)
        t = z["t_rel_s"].astype(float)
        y = z["amplitude_pathloss_corrected"].astype(float)
        is_gap = z["is_gap"].astype(bool)
        valid = ~is_gap
        t_v, y_v = t[valid], y[valid]
        if len(t_v) < 20:
            continue
        ls = run_ls(t_v, y_v, detrend_deg=1)
        peak_p = float(ls["top3"][0]["period_s"])
        results.append(
            {
                "obs_id": obs_id,
                "peak_period_s": peak_p,
                "baseline_peak_s": BASELINE_LS_PEAKS.get(obs_id),
                "delta_vs_baseline_s": peak_p - BASELINE_LS_PEAKS.get(obs_id, peak_p),
                "cw_alias": _flag_cw_alias(peak_p),
                "used_corrected_npz": obs_id in AFFECTED_IDS,
            }
        )
    return results


def run_mc_120k() -> dict[str, Any]:
    """Run MC aliasing at 120k trials (timestamps from NPZ unchanged by TLE correction)."""
    import shutil

    mc_out = OUTPUT_DIR / "mc_aliasing"
    mc_out.mkdir(parents=True, exist_ok=True)
    mc_src = (_SCRIPT_DIR / "mc_aliasing_165s.py").read_text(encoding="utf-8")
    patched = mc_src.replace("N_TRIALS    = 10_000", "N_TRIALS    = 120_000")
    patched_path = OUTPUT_DIR / "mc_aliasing_165s_120k.py"
    patched_path.write_text(patched, encoding="utf-8")
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, str(patched_path)],
        check=True,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(_SCRIPT_DIR)},
    )
    elapsed = time.perf_counter() - t0

    results_csv = PROJECT_ROOT / "data" / "mc_aliasing" / "coincidence_table.csv"
    near165_hits = None
    cells_with = 0
    if results_csv.is_file():
        shutil.copy2(results_csv, mc_out / "coincidence_table.csv")
        with open(results_csv, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        near165 = [r for r in rows if int(r.get("n_near165", 0)) > 0]
        near165_hits = sum(int(r.get("n_near165", 0)) for r in rows)
        cells_with = len(near165)

    return {
        "n_trials": 120_000,
        "output_dir": str(mc_out),
        "elapsed_s": elapsed,
        "near165_total_hits": near165_hits,
        "cells_with_near165": cells_with,
    }


def try_fetch_spacetrack() -> Path | None:
    identity = os.environ.get("SPACETRACK_IDENTITY") or os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not identity or not password:
        return None
    out = PROJECT_ROOT / "data" / "tle_archive" / "43780_2018-12-01_2018-12-10.tle"
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "fetch_spacetrack_gp_history.py"),
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out.with_suffix(".json")


def build_fallback_archive() -> Path:
    """Collect unique 43780 TLEs from local obs where tle1 catalog is 43780."""
    records: dict[str, dict] = {}
    for p in PROJECT_ROOT.glob("data/move2_*/meta/observation_*.json"):
        d = json.loads(p.read_text())
        tle1 = d.get("tle1", "")
        if "43780U" not in tle1:
            continue
        ep = tle1.split()[3]
        if ep not in records:
            records[ep] = {
                "tle0": d.get("tle0", ""),
                "tle1": tle1,
                "tle2": d.get("tle2", ""),
                "epoch_utc": parse_tle_epoch(tle1).isoformat(),
                "epoch_yyddd": ep,
                "provenance_obs_id": d["id"],
            }
    out = PROJECT_ROOT / "data" / "tle_archive" / "43780_fallback_correct_norad_obs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "fallback: unique 43780 TLEs from local SatNOGS obs with correct tle1 catalog (NOT gp_history)",
        "note": "Space-Track credentials unavailable; earliest epoch may be Dec 8 2018",
        "n_records": len(records),
        "records": list(records.values()),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Archival 43780 TLE validation chain")
    ap.add_argument("--tle-archive", type=Path, help="TLE archive .json or .tle from Space-Track")
    ap.add_argument("--fetch-spacetrack", action="store_true")
    ap.add_argument("--skip-mc", action="store_true", help="Skip 120k MC (slow)")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    archive_path = args.tle_archive
    if args.fetch_spacetrack:
        fetched = try_fetch_spacetrack()
        if fetched:
            archive_path = fetched
        else:
            print("Space-Track fetch skipped: no credentials in environment.")

    if archive_path is None:
        archive_path = PROJECT_ROOT / "data" / "tle_archive" / "43780_2018-12-01_2018-12-10.json"
        if not archive_path.is_file():
            print(f"No archive at {archive_path}; building fallback from correct-NORAD local TLEs.")
            archive_path = build_fallback_archive()

    meta_doc = json.loads(archive_path.read_text()) if archive_path.suffix == ".json" else {}
    records = load_tle_archive(archive_path)
    print(f"\n=== Step 1: TLE archive ===")
    print(f"  path: {archive_path}")
    print(f"  source: {meta_doc.get('source', 'tle file')}")
    print(f"  n_records: {len(records)}")
    for rec in sorted(records, key=lambda r: r["epoch_yyddd"] if "epoch_yyddd" in r else r["tle1"].split()[3]):
        ep = rec.get("epoch_yyddd") or rec["tle1"].split()[3]
        print(f"    epoch {ep}  ({rec.get('epoch_utc', parse_tle_epoch(rec['tle1']).isoformat())})")

    print(f"\n=== Step 2: Epoch-matched TLE selection ===")
    epoch_report = []
    tle_by_obs: dict[int, dict] = {}
    for obs_id in AFFECTED_IDS:
        meta = load_meta(find_obs_folder(obs_id, PROJECT_ROOT), obs_id)
        obs_start = parse_iso(meta["start"])
        tle_rec, gap_h = select_nearest_tle(records, obs_start)
        tle_by_obs[obs_id] = tle_rec
        row = {
            "obs_id": obs_id,
            "obs_start_utc": obs_start.isoformat(),
            "matched_epoch": tle_rec["tle1"].split()[3],
            "epoch_gap_hours": round(gap_h, 2),
            "tle1": tle_rec["tle1"],
        }
        epoch_report.append(row)
        print(
            f"  obs {obs_id}: gap={gap_h:.2f} h  "
            f"(epoch {row['matched_epoch']}, obs_start {obs_start.strftime('%Y-%m-%d %H:%M:%S')} UTC)"
        )

    print(f"\n=== Step 3: Doppler re-fit (43780 archival TLE) ===")
    doppler_results = []
    tracks: dict[int, dict] = {}
    metas: dict[int, dict] = {}
    for obs_id in AFFECTED_IDS:
        meta, track = load_carrier_track(obs_id)
        metas[obs_id] = meta
        tracks[obs_id] = track
        orig = ORIGINAL_DOPPLER[obs_id]
        corr = doppler_refit(meta, track, tle_by_obs[obs_id])
        doppler_results.append(corr)
        print(f"\n  --- obs {obs_id} ---")
        print(
            f"  original (embedded cat {orig['embedded_cat']}): "
            f"B={orig['B']:.4f}  R²={orig['R2']:.4f}  n={orig['n']}"
        )
        print(
            f"  corrected  (43780 epoch {corr['tle1'].split()[3]}): "
            f"B={corr['B']:.4f}  R²={corr['R2']:.4f}  n={corr['n']}  "
            f"class={corr['classification']}  amp_q={corr['amplitude_quality']:.6f}"
        )

    print(f"\n=== Step 4: Path-loss-corrected amplitudes (corrected range_km) ===")
    amp_stats = []
    for corr in doppler_results:
        obs_id = corr["obs_id"]
        st = recompute_amplitudes(obs_id, corr["meta_corrected"], tracks[obs_id])
        amp_stats.append({"obs_id": obs_id, **st})
        print(
            f"  obs {obs_id}: mean Δamp={st.get('mean_delta_db', 0):+.4f} dB  "
            f"max |Δamp|={st.get('max_abs_delta_db', 0):.4f} dB  "
            f"mean Δrange={st.get('mean_range_delta_km', 0):+.2f} km"
        )

    print(f"\n=== Step 5: Lomb-Scargle (6 passes) ===")
    ls_results = run_lomb_scargle_six()
    for r in ls_results:
        base = r["baseline_peak_s"]
        delta = r["delta_vs_baseline_s"]
        tag = " [corrected amp]" if r["used_corrected_npz"] else ""
        print(
            f"  obs {r['obs_id']}: peak={r['peak_period_s']:.4f} s  "
            f"(baseline {base:.4f}, Δ={delta:+.4f} s){tag}  {r['cw_alias']}"
        )
    p349 = next((r for r in ls_results if r["obs_id"] == 349779), None)
    p359 = next((r for r in ls_results if r["obs_id"] == 359214), None)
    if p349 and p359:
        agree = abs(p349["peak_period_s"] - p359["peak_period_s"])
        print(f"\n  165 s cross-check: |349779 − 359214| = {agree:.4f} s")

    mc_summary = None
    if not args.skip_mc:
        print(f"\n=== Step 6: Monte Carlo aliasing (120,000 trials) ===")
        mc_summary = run_mc_120k()
        print(f"  near165 total hits (all cells): {mc_summary['near165_total_hits']}")
        print(f"  elapsed: {mc_summary['elapsed_s']:.0f} s")

    summary = {
        "tle_archive": str(archive_path),
        "tle_source": meta_doc.get("source"),
        "epoch_matching": epoch_report,
        "doppler_refit": [
            {k: v for k, v in d.items() if k != "meta_corrected"} for d in doppler_results
        ],
        "amplitude_recompute": amp_stats,
        "lomb_scargle": ls_results,
        "mc_aliasing": mc_summary,
    }
    out_json = OUTPUT_DIR / "validation_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
