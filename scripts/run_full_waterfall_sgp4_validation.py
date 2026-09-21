#!/usr/bin/env python3
"""
Batch waterfall + SGP4 Doppler validation for all MOVE-II SatNOGS observations.

Runs each observation through the corrected waterfall pipeline (OCR axis calibration,
spectral-inversion sign correction, column-median birdie subtraction, UTC/elapsed-seconds
Y-axis detection) validated on obs 359214, 14650494, and 14662445.

Usage:
    python run_full_waterfall_sgp4_validation.py              # full ~423 obs in data/
    python run_full_waterfall_sgp4_validation.py --golden-only  # 32 old golden candidates
    python run_full_waterfall_sgp4_validation.py --limit 5      # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carrier_doppler_check import (  # noqa: E402
    DEFAULT_EXCLUDE_FREQ_ABOVE_KHZ,
    DEFAULT_EXCLUDE_FREQ_BELOW_KHZ,
    DEFAULT_MIN_CONFIDENCE_PERCENTILE,
    _check_tesseract,
    _hhmm_to_elapsed,
    _ocr_one_tick,
    detect_spine_box,
    detect_ticks_x,
    detect_ticks_y,
    extract_carrier_track_waterfall,
    find_obs_folder,
    fit_doppler,
    load_meta,
    parse_iso,
    predict_doppler,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_RFI_STATIONS = frozenset({"VK5QI-2M", "VK5QI"})

# KO2F-VHF-1 / gr-satnogs standard waterfall: ~77 Hz/px (0.077 kHz/px)
KHZ_PER_PX_MIN = 0.04
KHZ_PER_PX_MAX = 0.15

# Implied plot duration from Y calibration must be within this fraction of meta duration
Y_DURATION_RATIO_MIN = 0.40
Y_DURATION_RATIO_MAX = 2.50

# |B| above this after fit is treated as a layout/calibration failure symptom
ABS_B_LAYOUT_SYMPTOM = 3.0

RESULT_FIELDS = [
    "obs_id",
    "station",
    "B",
    "R2",
    "n_samples",
    "bandwidth_khz_used",
    "classification",
    "old_golden",
    "golden_survives",
    "error",
]

OUTPUT_SUBDIR = "waterfall_sgp4_validation"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ObsResult:
    obs_id: int
    station: str = ""
    B: Optional[float] = None
    R2: Optional[float] = None
    n_samples: int = 0
    bandwidth_khz_used: Optional[float] = None
    classification: str = "processing_error"
    old_golden: bool = False
    golden_survives: bool = False
    error: str = ""
    ocr_debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Golden-candidate loader
# ---------------------------------------------------------------------------


def load_old_golden_ids(vetting_csv: Path) -> set[int]:
    """Return obs_ids flagged usable_for_doppler in the audio-based vetting report."""
    if not vetting_csv.is_file():
        logging.warning("Vetting report not found: %s", vetting_csv)
        return set()
    golden: set[int] = set()
    with open(vetting_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("usable_for_doppler") == "True":
                golden.add(int(row["obs_id"]))
    return golden


# ---------------------------------------------------------------------------
# Observation discovery
# ---------------------------------------------------------------------------


def discover_obs_ids(
    data_dir: Path,
    *,
    require_norad_id: int | None = None,
) -> list[int]:
    """Return sorted obs IDs from move2_{id}_{date} folders under data_dir.

    If require_norad_id is set, keep only folders whose observation JSON
    norad_cat_id matches (genuine tracked-object filter).
    """
    ids: list[int] = []
    for folder in data_dir.iterdir():
        if not folder.is_dir() or not folder.name.startswith("move2_"):
            continue
        parts = folder.name.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        obs_id = int(parts[1])
        if require_norad_id is not None:
            meta_path = folder / "meta" / f"observation_{obs_id}.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("norad_cat_id") != require_norad_id:
                continue
        ids.append(obs_id)
    return sorted(set(ids))


# ---------------------------------------------------------------------------
# Layout sanity checks
# ---------------------------------------------------------------------------


def dump_ocr_debug(waterfall_path: Path) -> dict[str, Any]:
    """Re-run spine/tick detection + OCR and return raw values for inspection."""
    from PIL import Image

    img_rgb = np.array(Image.open(waterfall_path).convert("RGB"))
    spine = detect_spine_box(img_rgb)
    top, bottom, left, right = spine
    x_ticks_px = detect_ticks_x(img_rgb, spine)
    y_ticks_px = detect_ticks_y(img_rgb, spine)

    x_ocr = [_ocr_one_tick(img_rgb, spine, col, "x") for col in x_ticks_px]
    y_ocr_raw = [_ocr_one_tick(img_rgb, spine, row, "y") for row in y_ticks_px]
    y_ocr_elapsed = _hhmm_to_elapsed(y_ocr_raw)

    return {
        "waterfall": str(waterfall_path),
        "spine": {"top": top, "bottom": bottom, "left": left, "right": right},
        "plot_width_px": right - left,
        "plot_height_px": bottom - top,
        "x_ticks_px": x_ticks_px,
        "x_ocr_raw": x_ocr,
        "y_ticks_px": y_ticks_px,
        "y_ocr_raw": y_ocr_raw,
        "y_ocr_elapsed_s": y_ocr_elapsed,
    }


def check_axis_calibration(
    track: dict,
    duration_s: float,
    *,
    abs_b: Optional[float] = None,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Sanity-check OCR-derived axis calibration before trusting the SGP4 fit.

    Returns (ok, reason, debug_info).
    """
    debug: dict[str, Any] = {}
    x_slope = track.get("x_slope_khz_per_px")
    y_slope = track.get("y_slope_s_per_px")
    n_rows = len(track.get("t_rel_s", []))

    if x_slope is None or y_slope is None:
        return False, "missing calibration slopes", debug

    abs_x_slope = abs(float(x_slope))
    debug["x_slope_khz_per_px"] = float(x_slope)
    debug["y_slope_s_per_px"] = float(y_slope)
    debug["n_plot_rows"] = n_rows

    if not (KHZ_PER_PX_MIN <= abs_x_slope <= KHZ_PER_PX_MAX):
        debug["x_slope_out_of_range"] = True
        return (
            False,
            f"x_slope {abs_x_slope:.5f} kHz/px outside "
            f"[{KHZ_PER_PX_MIN}, {KHZ_PER_PX_MAX}]",
            debug,
        )

    implied_duration_s = abs(float(y_slope)) * n_rows
    debug["implied_duration_s"] = implied_duration_s
    debug["meta_duration_s"] = duration_s

    if duration_s > 0:
        ratio = implied_duration_s / duration_s
        debug["y_duration_ratio"] = ratio
        if not (Y_DURATION_RATIO_MIN <= ratio <= Y_DURATION_RATIO_MAX):
            return (
                False,
                f"Y-axis duration ratio {ratio:.3f} outside "
                f"[{Y_DURATION_RATIO_MIN}, {Y_DURATION_RATIO_MAX}] "
                f"(implied {implied_duration_s:.0f}s vs meta {duration_s:.0f}s)",
                debug,
            )

    if abs_b is not None and abs(abs_b) > ABS_B_LAYOUT_SYMPTOM:
        debug["abs_B"] = abs_b
        return (
            False,
            f"|B|={abs(abs_b):.3f} > {ABS_B_LAYOUT_SYMPTOM} (layout/calibration symptom)",
            debug,
        )

    return True, "", debug


def compute_bandwidth_khz(track: dict) -> Optional[float]:
    """Estimate total frequency span (kHz) from X calibration and plot width."""
    x_slope = track.get("x_slope_khz_per_px")
    n_rows = len(track.get("t_rel_s", []))
    if x_slope is None or n_rows == 0:
        return None
    # Approximate plot width from row count (waterfall plot is typically wider than tall,
    # but we don't store width — use n_rows as proxy only if x intercept span unavailable).
    # Better: derive from intercept span using stored calibration.
    # bandwidth ≈ |slope| × span_px; span_px ≈ 2 × |intercept| / |slope| when ticks
    # span ±intercept kHz from centre.  Fallback: |slope| × n_rows (rough).
    x_intercept = track.get("x_intercept_khz")
    if x_intercept is not None and abs(x_slope) > 1e-9:
        # Typical SatNOGS waterfall: ticks span roughly ±intercept from edges
        return abs(float(x_intercept)) * 2.0
    return abs(float(x_slope)) * n_rows


def classify_result(
    B: float,
    R2: float,
    station: str,
    *,
    unknown_layout: bool,
) -> str:
    if unknown_layout:
        return "unknown_layout"
    station_key = station.strip()
    if station_key in KNOWN_RFI_STATIONS or any(
        station_key.startswith(s) for s in KNOWN_RFI_STATIONS
    ):
        return "station_rfi_suspect"
    if abs(B) < 0.1:
        return "fully_afc_corrected"
    if abs(B - 1.0) < 0.15 and R2 > 0.85:
        return "raw_uncorrected"
    return "partially_corrected"


# ---------------------------------------------------------------------------
# Per-observation processing
# ---------------------------------------------------------------------------


def process_one_observation(
    obs_id: int,
    project_root: Path,
    golden_ids: set[int],
    unknown_layout_dir: Path,
    *,
    exclude_freq_below_khz: float,
    exclude_freq_above_khz: float,
    min_confidence: float,
) -> ObsResult:
    result = ObsResult(
        obs_id=obs_id,
        old_golden=obs_id in golden_ids,
    )

    obs_folder = find_obs_folder(obs_id, project_root)
    meta = load_meta(obs_folder, obs_id)
    result.station = meta.get("station_name") or "unknown"

    obs_start = parse_iso(meta["start"])
    obs_end = parse_iso(meta["end"])
    duration_s = (obs_end - obs_start).total_seconds()
    f0_hz = float(meta["observation_frequency"])

    raw_dir = obs_folder / "raw"
    wf_files = list(raw_dir.glob("waterfall_*.png"))
    if not wf_files:
        raise FileNotFoundError(f"No waterfall_*.png in {raw_dir}")
    wf_path = wf_files[0]

    track = extract_carrier_track_waterfall(
        wf_path,
        obs_start,
        duration_s,
        f0_hz=f0_hz,
        exclude_freq_below_khz=exclude_freq_below_khz,
        exclude_freq_above_khz=exclude_freq_above_khz,
        min_confidence_percentile=min_confidence,
    )

    result.bandwidth_khz_used = compute_bandwidth_khz(track)

    # Pre-fit layout check (X/Y calibration sanity)
    cal_ok, cal_reason, cal_debug = check_axis_calibration(track, duration_s)
    unknown_layout = not cal_ok

    n_strong = int(track["strong_mask"].sum())
    result.n_samples = n_strong

    if n_strong < 2:
        raise ValueError(f"Only {n_strong} strong rows — insufficient for SGP4 fit")

    predicted_shift = predict_doppler(meta, track["timestamps"], track["strong_mask"])
    strong_idx = np.where(track["strong_mask"])[0]
    strong_freq = track["peak_freq_hz"][strong_idx]

    fit = fit_doppler(strong_freq, predicted_shift)
    B = float(fit["B"])
    R2 = float(fit["r2"])
    result.B = B
    result.R2 = R2

    # Post-fit |B| symptom check (catches cases that pass pre-check but still bad)
    if cal_ok:
        post_ok, post_reason, post_debug = check_axis_calibration(
            track, duration_s, abs_b=B
        )
        if not post_ok:
            unknown_layout = True
            cal_reason = post_reason
            cal_debug.update(post_debug)

    if unknown_layout:
        ocr_debug = dump_ocr_debug(wf_path)
        ocr_debug["cal_check_reason"] = cal_reason
        ocr_debug["cal_check_debug"] = cal_debug
        ocr_debug["fit_B"] = B
        ocr_debug["fit_R2"] = R2
        result.ocr_debug = ocr_debug

        unknown_layout_dir.mkdir(parents=True, exist_ok=True)
        dump_path = unknown_layout_dir / f"{obs_id}_ocr.json"
        with open(dump_path, "w", encoding="utf-8") as fh:
            json.dump(ocr_debug, fh, indent=2)

    result.classification = classify_result(
        B, R2, result.station, unknown_layout=unknown_layout
    )
    result.golden_survives = result.old_golden and result.classification == "raw_uncorrected"
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_results_csv(path: Path, results: list[ObsResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "obs_id": r.obs_id,
                    "station": r.station,
                    "B": "" if r.B is None else round(r.B, 4),
                    "R2": "" if r.R2 is None else round(r.R2, 4),
                    "n_samples": r.n_samples,
                    "bandwidth_khz_used": (
                        "" if r.bandwidth_khz_used is None
                        else round(r.bandwidth_khz_used, 2)
                    ),
                    "classification": r.classification,
                    "old_golden": r.old_golden,
                    "golden_survives": r.golden_survives,
                    "error": r.error,
                }
            )


def print_summary(results: list[ObsResult], elapsed_s: float, output_dir: Path) -> None:
    from collections import Counter

    counts = Counter(r.classification for r in results)
    errors = [r for r in results if r.error]
    raw_uncorrected = [
        r for r in results
        if r.classification == "raw_uncorrected" and r.R2 is not None
    ]
    raw_uncorrected.sort(key=lambda r: r.R2 or 0.0, reverse=True)

    old_golden = [r for r in results if r.old_golden]
    survived = [r for r in old_golden if r.golden_survives]
    downgraded = [r for r in old_golden if not r.golden_survives and not r.error]

    print("\n" + "=" * 72)
    print("WATERFALL + SGP4 VALIDATION SUMMARY")
    print("=" * 72)
    print(f"  Processed          : {len(results)} observations")
    print(f"  Elapsed            : {elapsed_s:.1f}s ({elapsed_s / max(len(results), 1):.1f}s/obs)")
    print(f"  Output directory   : {output_dir}")
    print()
    print("  Classification breakdown:")
    for cls in sorted(counts):
        print(f"    {cls:28s}: {counts[cls]}")
    if errors:
        print(f"    {'(with error message)':28s}: {len(errors)}")
    print()

    print("  Top-10 raw_uncorrected (by R²) — Doppler-usable golden candidates:")
    print(f"  {'obs_id':>10}  {'station':<30}  {'B':>8}  {'R²':>8}  {'n':>5}")
    print("  " + "-" * 68)
    for r in raw_uncorrected[:10]:
        print(
            f"  {r.obs_id:>10}  {r.station[:30]:<30}  "
            f"{r.B:>8.4f}  {r.R2:>8.4f}  {r.n_samples:>5}"
        )
    if not raw_uncorrected:
        print("  (none)")
    print()

    print("  Old golden cross-reference (audio-based usable_for_doppler):")
    print(f"    Old golden count   : {len(old_golden)}")
    print(f"    Survive (raw_uncorrected): {len(survived)}")
    print(f"    Downgraded       : {len(downgraded)}")
    if downgraded:
        print("    Downgraded obs_ids:")
        for r in sorted(downgraded, key=lambda x: x.obs_id):
            cls = r.classification
            b_str = f"B={r.B:.3f}" if r.B is not None else "B=N/A"
            r2_str = f"R²={r.R2:.3f}" if r.R2 is not None else "R²=N/A"
            print(f"      {r.obs_id:>10}  {r.station[:28]:<28}  → {cls}  ({b_str}, {r2_str})")
    if survived:
        print("    Surviving obs_ids:")
        for r in sorted(survived, key=lambda x: x.R2 or 0.0, reverse=True):
            print(
                f"      {r.obs_id:>10}  {r.station[:28]:<28}  "
                f"B={r.B:.4f}  R²={r.R2:.4f}"
            )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch waterfall + SGP4 Doppler validation for MOVE-II observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Project root (default: parent of scripts/).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing move2_* folders and validation output "
            "(default: <data-root>/data)."
        ),
    )
    p.add_argument(
        "--require-norad-id",
        type=int,
        default=None,
        metavar="NORAD",
        help="Only validate observations whose norad_cat_id matches (e.g. 43780).",
    )
    p.add_argument(
        "--golden-only",
        action="store_true",
        help="Process only the 32 obs flagged usable_for_doppler in vetting_report_full.csv.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N observations (for smoke tests).",
    )
    p.add_argument(
        "--exclude-freq-below-khz",
        type=float,
        default=DEFAULT_EXCLUDE_FREQ_BELOW_KHZ,
    )
    p.add_argument(
        "--exclude-freq-above-khz",
        type=float,
        default=DEFAULT_EXCLUDE_FREQ_ABOVE_KHZ,
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE_PERCENTILE,
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    args = build_parser().parse_args()
    project_root = args.data_root or Path(__file__).resolve().parent.parent
    data_dir = (args.data_dir or (project_root / "data")).resolve()
    output_dir = data_dir / OUTPUT_SUBDIR
    unknown_layout_dir = output_dir / "unknown_layout"
    results_csv = output_dir / "results.csv"
    failures_log = output_dir / "failures.log"

    _check_tesseract()

    golden_ids = load_old_golden_ids(data_dir / "vetting_report_full.csv")

    if args.golden_only:
        obs_ids = sorted(golden_ids)
        logging.info("Golden-only mode: %d observations", len(obs_ids))
    else:
        obs_ids = discover_obs_ids(data_dir, require_norad_id=args.require_norad_id)
        logging.info(
            "Full dataset: %d observations in %s (require_norad=%s)",
            len(obs_ids),
            data_dir,
            args.require_norad_id,
        )

    if args.limit is not None:
        obs_ids = obs_ids[: args.limit]
        logging.info("Limited to first %d observations", len(obs_ids))

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ObsResult] = []
    t0 = time.monotonic()

    with open(failures_log, "w", encoding="utf-8") as fail_fh:
        fail_fh.write(f"# Waterfall SGP4 validation failures — {datetime.now(timezone.utc).isoformat()}\n\n")

        for i, obs_id in enumerate(obs_ids, 1):
            logging.info("[%d/%d] Processing obs %d ...", i, len(obs_ids), obs_id)
            try:
                result = process_one_observation(
                    obs_id,
                    project_root,
                    golden_ids,
                    unknown_layout_dir,
                    exclude_freq_below_khz=args.exclude_freq_below_khz,
                    exclude_freq_above_khz=args.exclude_freq_above_khz,
                    min_confidence=args.min_confidence,
                )
                logging.info(
                    "  obs %d: %s  B=%.4f  R²=%.4f  n=%d",
                    obs_id,
                    result.classification,
                    result.B or 0.0,
                    result.R2 or 0.0,
                    result.n_samples,
                )
            except Exception as exc:  # noqa: BLE001 — per-obs isolation
                tb = traceback.format_exc()
                err_msg = f"{type(exc).__name__}: {exc}"
                logging.error("  obs %d FAILED: %s", obs_id, err_msg)
                fail_fh.write(f"obs {obs_id}: {err_msg}\n{tb}\n{'-' * 60}\n")
                fail_fh.flush()
                result = ObsResult(
                    obs_id=obs_id,
                    old_golden=obs_id in golden_ids,
                    classification="processing_error",
                    error=err_msg,
                )
                try:
                    result.station = load_meta(
                        find_obs_folder(obs_id, project_root), obs_id
                    ).get("station_name") or "unknown"
                except Exception:
                    pass
            results.append(result)

    elapsed = time.monotonic() - t0
    write_results_csv(results_csv, results)
    print_summary(results, elapsed, output_dir)
    logging.info("Results written to %s", results_csv)


if __name__ == "__main__":
    main()
