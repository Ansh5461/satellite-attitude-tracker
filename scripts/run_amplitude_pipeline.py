#!/usr/bin/env python3
"""
MOVE-II tumble pipeline — Stage 2: amplitude envelope extraction.

For the 32 raw_uncorrected observations from waterfall+SGP4 validation:

  1. Leaderboard: rank by Doppler R², join audio SNR / fraction-strong from
     fetch_and_vet_all_observations.py, flag Doppler-clean but amplitude-risky
     observations.
  2. Amplitude envelope along the existing waterfall carrier track
     (peak_power from extract_carrier_track_waterfall — no re-tracking).
  3. CW keying-gap detection / mask / interpolate.
  4. Free-space path-loss correction using SGP4 slant range.
  5. Per-obs CSV + npz + diagnostic plot under data/amplitude_pipeline/.

Usage:
    python run_amplitude_pipeline.py
    python run_amplitude_pipeline.py --limit 2          # smoke test
    python run_amplitude_pipeline.py --force-extract    # ignore cached tracks
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carrier_doppler_check import (  # noqa: E402
    DEFAULT_EXCLUDE_FREQ_ABOVE_KHZ,
    DEFAULT_EXCLUDE_FREQ_BELOW_KHZ,
    DEFAULT_MIN_CONFIDENCE_PERCENTILE,
    _check_tesseract,
    extract_carrier_track_waterfall,
    find_obs_folder,
    load_meta,
    parse_iso,
    save_carrier_track_csv,
)

# Vetting thresholds from fetch_and_vet_all_observations.py
DEFAULT_MIN_SNR = 15.0
DEFAULT_MIN_FRAC_STRONG = 0.3

# CW gap: peak SNR (peak − row median, viridis-index units) below this → keyed off
DEFAULT_GAP_MIN_PEAK_SNR = 40.0
# Also treat large frequency jumps off the smoothed track as gaps (birdie latch)
DEFAULT_GAP_FREQ_DEV_HZ = 2500.0
DEFAULT_FREQ_SMOOTH_WIN = 21

OUTPUT_SUBDIR = "amplitude_pipeline"
RANGE_REF_KM = 1000.0  # reference range for 20·log10(R/Rref) term


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardRow:
    obs_id: int
    station: str
    B: float
    R2: float
    n_samples: int
    snr_db: Optional[float] = None
    frac_time_strong: Optional[float] = None
    usable_for_amplitude: Optional[bool] = None
    amplitude_risk: bool = False
    risk_reasons: list[str] = field(default_factory=list)


@dataclass
class GapStats:
    obs_id: int
    n_rows: int
    n_gaps: int
    n_gap_rows: int
    frac_gap: float
    total_gap_duration_s: float
    longest_gap_s: float
    median_gap_s: float
    gap_min_peak_snr: float
    dt_s: float


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_raw_uncorrected(results_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(results_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("classification") != "raw_uncorrected":
                continue
            if not r.get("R2"):
                continue
            rows.append(
                {
                    "obs_id": int(r["obs_id"]),
                    "station": r.get("station") or "",
                    "B": float(r["B"]),
                    "R2": float(r["R2"]),
                    "n_samples": int(r["n_samples"] or 0),
                }
            )
    rows.sort(key=lambda x: x["R2"], reverse=True)
    return rows


def _parse_optional_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_optional_bool(val: Any) -> Optional[bool]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "none", "null"):
        return None
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def load_vetting_by_obs(data_dir: Path) -> dict[int, dict[str, Any]]:
    """
    Merge vetting_report_full.csv with vetting_report.csv (fallback for empty SNR).
    """
    merged: dict[int, dict[str, Any]] = {}
    for name in ("vetting_report_full.csv", "vetting_report.csv"):
        path = data_dir / name
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    oid = int(r["obs_id"])
                except (KeyError, ValueError):
                    continue
                cur = merged.setdefault(oid, {})
                for key in ("snr_db", "frac_time_strong", "usable_for_amplitude", "station_name"):
                    incoming = r.get(key)
                    if incoming is None or str(incoming).strip() == "":
                        continue
                    # Prefer first non-empty (full report is loaded first)
                    if key not in cur or cur[key] is None or str(cur[key]).strip() == "":
                        cur[key] = incoming
    return merged


def build_leaderboard(
    raw_rows: list[dict[str, Any]],
    vetting: dict[int, dict[str, Any]],
    *,
    min_snr: float,
    min_frac: float,
) -> list[LeaderboardRow]:
    out: list[LeaderboardRow] = []
    for r in raw_rows:
        v = vetting.get(r["obs_id"], {})
        snr = _parse_optional_float(v.get("snr_db"))
        frac = _parse_optional_float(v.get("frac_time_strong"))
        usable_amp = _parse_optional_bool(v.get("usable_for_amplitude"))
        reasons: list[str] = []
        if snr is None:
            reasons.append("snr_missing")
        elif snr < min_snr:
            reasons.append(f"snr<{min_snr:g}")
        if frac is None:
            reasons.append("frac_missing")
        elif frac < min_frac:
            reasons.append(f"frac<{min_frac:g}")
        out.append(
            LeaderboardRow(
                obs_id=r["obs_id"],
                station=r["station"],
                B=r["B"],
                R2=r["R2"],
                n_samples=r["n_samples"],
                snr_db=snr,
                frac_time_strong=frac,
                usable_for_amplitude=usable_amp,
                amplitude_risk=bool(reasons),
                risk_reasons=reasons,
            )
        )
    return out


def print_leaderboard(rows: list[LeaderboardRow], *, min_snr: float, min_frac: float) -> None:
    print("\n" + "=" * 100)
    print("AMPLITUDE PIPELINE — RAW_UNCORRECTED LEADERBOARD (sorted by R² ↓)")
    print("=" * 100)
    print(
        f"  Amplitude-risk flags use vetting thresholds: "
        f"SNR < {min_snr:g} dB  OR  frac_time_strong < {min_frac:g}"
    )
    print(
        f"  {'rank':>4}  {'obs_id':>10}  {'station':<28}  {'B':>7}  {'R²':>7}  "
        f"{'SNR':>7}  {'frac':>6}  {'risk':<6}  notes"
    )
    print("  " + "-" * 96)
    n_risk = 0
    for i, r in enumerate(rows, 1):
        snr_s = f"{r.snr_db:7.2f}" if r.snr_db is not None else f"{'—':>7}"
        frac_s = f"{r.frac_time_strong:6.3f}" if r.frac_time_strong is not None else f"{'—':>6}"
        risk_s = "YES" if r.amplitude_risk else "no"
        if r.amplitude_risk:
            n_risk += 1
        notes = ",".join(r.risk_reasons) if r.risk_reasons else ""
        print(
            f"  {i:>4}  {r.obs_id:>10}  {r.station[:28]:<28}  "
            f"{r.B:>7.4f}  {r.R2:>7.4f}  {snr_s}  {frac_s}  {risk_s:<6}  {notes}"
        )
    print()
    print(
        f"  Total: {len(rows)}  |  amplitude-risk (Doppler-clean, SNR/frac weak): {n_risk}"
    )
    print("=" * 100 + "\n")


def write_leaderboard_csv(path: Path, rows: list[LeaderboardRow]) -> None:
    fields = [
        "rank",
        "obs_id",
        "station",
        "B",
        "R2",
        "n_samples",
        "snr_db",
        "frac_time_strong",
        "usable_for_amplitude",
        "amplitude_risk",
        "risk_reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow(
                {
                    "rank": i,
                    "obs_id": r.obs_id,
                    "station": r.station,
                    "B": round(r.B, 4),
                    "R2": round(r.R2, 4),
                    "n_samples": r.n_samples,
                    "snr_db": "" if r.snr_db is None else round(r.snr_db, 2),
                    "frac_time_strong": (
                        "" if r.frac_time_strong is None else round(r.frac_time_strong, 4)
                    ),
                    "usable_for_amplitude": (
                        "" if r.usable_for_amplitude is None else r.usable_for_amplitude
                    ),
                    "amplitude_risk": r.amplitude_risk,
                    "risk_reasons": ";".join(r.risk_reasons),
                }
            )


# ---------------------------------------------------------------------------
# Carrier track load / extract
# ---------------------------------------------------------------------------


def _load_track_csv(path: Path) -> dict:
    timestamps: list[datetime] = []
    t_rel: list[float] = []
    peak_freq: list[float] = []
    peak_power: list[float] = []
    strong: list[bool] = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            timestamps.append(parse_iso(r["timestamp_utc"]))
            t_rel.append(float(r["t_rel_s"]))
            peak_freq.append(float(r["peak_freq_hz"]))
            peak_power.append(float(r["peak_power_db_relative"]))
            strong.append(bool(int(r["strong"])))
    return {
        "source": "waterfall_csv",
        "timestamps": timestamps,
        "t_rel_s": np.asarray(t_rel, dtype=float),
        "peak_freq_hz": np.asarray(peak_freq, dtype=float),
        "peak_power_db": np.asarray(peak_power, dtype=float),
        "strong_mask": np.asarray(strong, dtype=bool),
        "noise_floor_db": float(np.median(peak_power)) if peak_power else float("nan"),
    }


def get_carrier_track(
    obs_id: int,
    project_root: Path,
    *,
    force_extract: bool,
    exclude_freq_below_khz: float,
    exclude_freq_above_khz: float,
    min_confidence: float,
) -> tuple[dict, dict, Path]:
    """
    Return (track, meta, obs_folder).

    Reuses derived/carrier_track_{id}.csv when present (same per-row argmax
    output from extract_carrier_track_waterfall). Otherwise runs the extractor
    once and caches the CSV.
    """
    obs_folder = find_obs_folder(obs_id, project_root)
    meta = load_meta(obs_folder, obs_id)
    derived = obs_folder / "derived"
    track_csv = derived / f"carrier_track_{obs_id}.csv"

    if track_csv.is_file() and not force_extract:
        logging.info("  Loading cached carrier track: %s", track_csv)
        return _load_track_csv(track_csv), meta, obs_folder

    obs_start = parse_iso(meta["start"])
    obs_end = parse_iso(meta["end"])
    duration_s = (obs_end - obs_start).total_seconds()
    f0_hz = float(meta["observation_frequency"])
    raw_dir = obs_folder / "raw"
    wf_files = list(raw_dir.glob("waterfall_*.png"))
    if not wf_files:
        raise FileNotFoundError(f"No waterfall_*.png in {raw_dir}")

    logging.info("  Extracting carrier track via extract_carrier_track_waterfall ...")
    track = extract_carrier_track_waterfall(
        wf_files[0],
        obs_start,
        duration_s,
        f0_hz=f0_hz,
        exclude_freq_below_khz=exclude_freq_below_khz,
        exclude_freq_above_khz=exclude_freq_above_khz,
        min_confidence_percentile=min_confidence,
    )
    save_carrier_track_csv(track, derived, obs_id)
    logging.info("  Cached carrier track → %s", track_csv)
    return track, meta, obs_folder


# ---------------------------------------------------------------------------
# Gap detection + path loss
# ---------------------------------------------------------------------------


def _moving_median(x: np.ndarray, win: int) -> np.ndarray:
    if win < 3 or len(x) < 3:
        return x.copy()
    win = int(win) | 1  # odd
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i : i + win])
    return out


def detect_keying_gaps(
    track: dict,
    *,
    gap_min_peak_snr: float,
    gap_freq_dev_hz: float,
    freq_smooth_win: int,
) -> tuple[np.ndarray, GapStats, int]:
    """
    Detect CW keying gaps.

    A row is a gap when:
      (1) peak SNR (peak − row median) < gap_min_peak_snr, OR
      (2) |peak_freq − smoothed track| > gap_freq_dev_hz
          (argmax latched onto a birdie while the carrier is off).

    Returns (is_gap, stats, obs_id_placeholder) — caller fills obs_id.
    """
    amp = np.asarray(track["peak_power_db"], dtype=float)
    freq = np.asarray(track["peak_freq_hz"], dtype=float)
    t_rel = np.asarray(track["t_rel_s"], dtype=float)
    n = len(amp)

    low_snr = amp < gap_min_peak_snr

    # Frequency continuity: smooth only high-SNR samples for a reference track
    freq_ref = freq.copy()
    if (~low_snr).sum() >= 5:
        # Fill low-SNR with NaN then interpolate for smoothing seed
        f = freq.astype(float).copy()
        f[low_snr] = np.nan
        idx = np.arange(n)
        good = np.isfinite(f)
        if good.sum() >= 2:
            f[~good] = np.interp(idx[~good], idx[good], f[good])
        freq_ref = _moving_median(f, freq_smooth_win)
    else:
        freq_ref = _moving_median(freq, freq_smooth_win)

    freq_jump = np.abs(freq - freq_ref) > gap_freq_dev_hz
    is_gap = low_snr | freq_jump

    if n >= 2:
        dt = float(np.median(np.abs(np.diff(t_rel))))
    else:
        dt = 0.0

    # Contiguous gap runs
    gaps: list[tuple[int, int]] = []
    in_gap = False
    start = 0
    for i, g in enumerate(is_gap):
        if g and not in_gap:
            start = i
            in_gap = True
        elif not g and in_gap:
            gaps.append((start, i - 1))
            in_gap = False
    if in_gap:
        gaps.append((start, n - 1))

    lengths_s = [((e - s + 1) * dt) for s, e in gaps] if gaps else []
    stats = GapStats(
        obs_id=0,
        n_rows=n,
        n_gaps=len(gaps),
        n_gap_rows=int(is_gap.sum()),
        frac_gap=float(is_gap.mean()) if n else 0.0,
        total_gap_duration_s=float(sum(lengths_s)),
        longest_gap_s=float(max(lengths_s) if lengths_s else 0.0),
        median_gap_s=float(np.median(lengths_s) if lengths_s else 0.0),
        gap_min_peak_snr=gap_min_peak_snr,
        dt_s=dt,
    )
    return is_gap, stats, 0


def interpolate_amplitude(amp: np.ndarray, is_gap: np.ndarray) -> np.ndarray:
    """Linear-interpolate amplitude across keying gaps; leave ends held."""
    out = amp.astype(float).copy()
    if not is_gap.any() or is_gap.all():
        return out
    idx = np.arange(len(amp))
    good = ~is_gap
    out[is_gap] = np.interp(idx[is_gap], idx[good], amp[good])
    return out


def compute_slant_range_km(meta: dict, timestamps: list[datetime]) -> np.ndarray:
    """SGP4 / skyfield slant range (km) at every timestamp — same geometry as Doppler."""
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale()
    satellite = EarthSatellite(meta["tle1"], meta["tle2"], meta.get("tle0", ""), ts)
    station = wgs84.latlon(
        latitude_degrees=float(meta["station_lat"]),
        longitude_degrees=float(meta["station_lng"]),
        elevation_m=float(meta.get("station_alt") or 0.0),
    )
    sky_times = ts.from_datetimes(list(timestamps))
    topo = (satellite - station).at(sky_times)
    pos_km = topo.position.km
    return np.linalg.norm(pos_km, axis=0).astype(float)


def path_loss_correct(amp_db: np.ndarray, range_km: np.ndarray, range_ref_km: float = RANGE_REF_KM) -> np.ndarray:
    """
    Remove free-space geometric fading: A_corr = A + 20·log10(R / R_ref).

    Adding the path-loss term boosts far-range samples so residual variation
    is dominated by antenna-gain pattern / tumble fading.
    """
    r = np.maximum(range_km.astype(float), 1e-6)
    return amp_db.astype(float) + 20.0 * np.log10(r / range_ref_km)


# ---------------------------------------------------------------------------
# Output writers + plots
# ---------------------------------------------------------------------------


def write_amplitude_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "timestamp_utc",
        "t_rel_s",
        "peak_freq_hz",
        "amplitude_raw",
        "is_gap",
        "amplitude_interp",
        "range_km",
        "fspl_corr_db",
        "amplitude_pathloss_corrected",
        "amplitude_pathloss_corrected_masked",
        "strong_track",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_gap_stats_csv(path: Path, stats_list: list[GapStats]) -> None:
    fields = [
        "obs_id",
        "n_rows",
        "n_gaps",
        "n_gap_rows",
        "frac_gap",
        "total_gap_duration_s",
        "longest_gap_s",
        "median_gap_s",
        "gap_min_peak_snr",
        "dt_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s in stats_list:
            w.writerow(
                {
                    "obs_id": s.obs_id,
                    "n_rows": s.n_rows,
                    "n_gaps": s.n_gaps,
                    "n_gap_rows": s.n_gap_rows,
                    "frac_gap": round(s.frac_gap, 4),
                    "total_gap_duration_s": round(s.total_gap_duration_s, 3),
                    "longest_gap_s": round(s.longest_gap_s, 3),
                    "median_gap_s": round(s.median_gap_s, 3),
                    "gap_min_peak_snr": s.gap_min_peak_snr,
                    "dt_s": round(s.dt_s, 4),
                }
            )


def plot_amplitude_diagnostic(
    path: Path,
    *,
    obs_id: int,
    station: str,
    t_rel: np.ndarray,
    amp_raw: np.ndarray,
    amp_corr: np.ndarray,
    is_gap: np.ndarray,
    range_km: np.ndarray,
    gap_stats: GapStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    ax0, ax1, ax2 = axes

    # Gap overlay helper
    def shade_gaps(ax: Any) -> None:
        if not is_gap.any():
            return
        in_gap = False
        start_t = 0.0
        for i, g in enumerate(is_gap):
            if g and not in_gap:
                start_t = float(t_rel[i])
                in_gap = True
            elif not g and in_gap:
                ax.axvspan(start_t, float(t_rel[i - 1]), color="0.75", alpha=0.45, lw=0)
                in_gap = False
        if in_gap:
            ax.axvspan(start_t, float(t_rel[-1]), color="0.75", alpha=0.45, lw=0)

    shade_gaps(ax0)
    ax0.plot(t_rel, amp_raw, color="#1f4e79", lw=0.8, label="raw amplitude (track peak SNR)")
    amp_masked = amp_raw.astype(float).copy()
    amp_masked[is_gap] = np.nan
    ax0.plot(t_rel, amp_masked, color="#c45c26", lw=1.0, label="carrier-on (gaps masked)")
    ax0.set_ylabel("Amplitude (rel. dB)")
    ax0.set_title(
        f"obs {obs_id}  ({station})  —  raw amplitude + CW keying gaps  "
        f"[n_gaps={gap_stats.n_gaps}, total={gap_stats.total_gap_duration_s:.1f}s, "
        f"longest={gap_stats.longest_gap_s:.1f}s]"
    )
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    shade_gaps(ax1)
    ax1.plot(t_rel, amp_corr, color="#2e7d32", lw=0.9, label="path-loss corrected (interp through gaps)")
    corr_masked = amp_corr.astype(float).copy()
    corr_masked[is_gap] = np.nan
    ax1.plot(t_rel, corr_masked, color="#6a1b9a", lw=1.1, label="path-loss corrected (gaps masked)")
    ax1.set_ylabel("Amplitude + 20·log₁₀(R/Rref)")
    ax1.set_title(f"Range-corrected amplitude  (Rref = {RANGE_REF_KM:.0f} km)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_rel, range_km, color="#37474f", lw=1.0)
    ax2.set_ylabel("Slant range (km)")
    ax2.set_xlabel("t_rel (s)")
    ax2.set_title("SGP4 slant range (same geometry as Doppler fit)")
    ax2.grid(True, alpha=0.3)

    # Gap mask as binary strip on ax0 twin is enough; also mark on range panel lightly
    shade_gaps(ax2)

    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-observation processing
# ---------------------------------------------------------------------------


def process_one(
    lb: LeaderboardRow,
    project_root: Path,
    out_dir: Path,
    *,
    force_extract: bool,
    exclude_freq_below_khz: float,
    exclude_freq_above_khz: float,
    min_confidence: float,
    gap_min_peak_snr: float,
    gap_freq_dev_hz: float,
    freq_smooth_win: int,
) -> GapStats:
    track, meta, _obs_folder = get_carrier_track(
        lb.obs_id,
        project_root,
        force_extract=force_extract,
        exclude_freq_below_khz=exclude_freq_below_khz,
        exclude_freq_above_khz=exclude_freq_above_khz,
        min_confidence=min_confidence,
    )

    amp_raw = np.asarray(track["peak_power_db"], dtype=float)
    t_rel = np.asarray(track["t_rel_s"], dtype=float)
    freq = np.asarray(track["peak_freq_hz"], dtype=float)
    strong = np.asarray(track["strong_mask"], dtype=bool)
    timestamps = track["timestamps"]

    is_gap, gap_stats, _ = detect_keying_gaps(
        track,
        gap_min_peak_snr=gap_min_peak_snr,
        gap_freq_dev_hz=gap_freq_dev_hz,
        freq_smooth_win=freq_smooth_win,
    )
    gap_stats.obs_id = lb.obs_id

    amp_interp = interpolate_amplitude(amp_raw, is_gap)
    range_km = compute_slant_range_km(meta, timestamps)
    fspl_corr = 20.0 * np.log10(np.maximum(range_km, 1e-6) / RANGE_REF_KM)
    amp_corr = path_loss_correct(amp_interp, range_km)
    amp_corr_masked = amp_corr.copy()
    amp_corr_masked[is_gap] = np.nan

    obs_out = out_dir / f"obs_{lb.obs_id}"
    obs_out.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict[str, Any]] = []
    for i in range(len(amp_raw)):
        ts = timestamps[i]
        csv_rows.append(
            {
                "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "t_rel_s": round(float(t_rel[i]), 4),
                "peak_freq_hz": round(float(freq[i]), 4),
                "amplitude_raw": round(float(amp_raw[i]), 4),
                "is_gap": int(bool(is_gap[i])),
                "amplitude_interp": round(float(amp_interp[i]), 4),
                "range_km": round(float(range_km[i]), 4),
                "fspl_corr_db": round(float(fspl_corr[i]), 4),
                "amplitude_pathloss_corrected": round(float(amp_corr[i]), 4),
                "amplitude_pathloss_corrected_masked": (
                    "" if is_gap[i] else round(float(amp_corr[i]), 4)
                ),
                "strong_track": int(bool(strong[i])),
            }
        )

    write_amplitude_csv(obs_out / f"amplitude_{lb.obs_id}.csv", csv_rows)
    np.savez_compressed(
        obs_out / f"amplitude_{lb.obs_id}.npz",
        t_rel_s=t_rel,
        peak_freq_hz=freq,
        amplitude_raw=amp_raw,
        is_gap=is_gap.astype(np.uint8),
        amplitude_interp=amp_interp,
        range_km=range_km,
        fspl_corr_db=fspl_corr,
        amplitude_pathloss_corrected=amp_corr,
        strong_track=strong.astype(np.uint8),
        gap_min_peak_snr=np.array([gap_min_peak_snr]),
        range_ref_km=np.array([RANGE_REF_KM]),
        obs_id=np.array([lb.obs_id]),
        R2=np.array([lb.R2]),
        B=np.array([lb.B]),
    )

    plot_amplitude_diagnostic(
        obs_out / f"amplitude_{lb.obs_id}.png",
        obs_id=lb.obs_id,
        station=lb.station,
        t_rel=t_rel,
        amp_raw=amp_raw,
        amp_corr=amp_corr,
        is_gap=is_gap,
        range_km=range_km,
        gap_stats=gap_stats,
    )

    # Sidecar JSON with leaderboard + gap summary
    with open(obs_out / f"amplitude_{lb.obs_id}_meta.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "obs_id": lb.obs_id,
                "station": lb.station,
                "B": lb.B,
                "R2": lb.R2,
                "snr_db": lb.snr_db,
                "frac_time_strong": lb.frac_time_strong,
                "amplitude_risk": lb.amplitude_risk,
                "risk_reasons": lb.risk_reasons,
                "gap_stats": {
                    "n_gaps": gap_stats.n_gaps,
                    "n_gap_rows": gap_stats.n_gap_rows,
                    "frac_gap": gap_stats.frac_gap,
                    "total_gap_duration_s": gap_stats.total_gap_duration_s,
                    "longest_gap_s": gap_stats.longest_gap_s,
                    "median_gap_s": gap_stats.median_gap_s,
                    "dt_s": gap_stats.dt_s,
                    "gap_min_peak_snr": gap_stats.gap_min_peak_snr,
                },
                "range_ref_km": RANGE_REF_KM,
                "path_loss_formula": "A_corr = A_interp + 20*log10(range_km / range_ref_km)",
            },
            fh,
            indent=2,
        )

    return gap_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MOVE-II amplitude envelope pipeline (post Doppler validation).",
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
        help="Directory with move2_* folders + vetting CSV (default: <data-root>/data).",
    )
    p.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Path to waterfall_sgp4_validation/results.csv",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N observations.")
    p.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-run extract_carrier_track_waterfall even if a cached track CSV exists.",
    )
    p.add_argument("--min-snr", type=float, default=DEFAULT_MIN_SNR)
    p.add_argument("--min-frac-strong", type=float, default=DEFAULT_MIN_FRAC_STRONG)
    p.add_argument("--gap-min-peak-snr", type=float, default=DEFAULT_GAP_MIN_PEAK_SNR)
    p.add_argument("--gap-freq-dev-hz", type=float, default=DEFAULT_GAP_FREQ_DEV_HZ)
    p.add_argument("--freq-smooth-win", type=int, default=DEFAULT_FREQ_SMOOTH_WIN)
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
    results_csv = args.results_csv or (data_dir / "waterfall_sgp4_validation" / "results.csv")
    out_dir = data_dir / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_csv.is_file():
        raise SystemExit(f"results.csv not found: {results_csv}")

    raw_rows = load_raw_uncorrected(results_csv)
    if not raw_rows:
        raise SystemExit(f"No raw_uncorrected rows in {results_csv}")

    vetting = load_vetting_by_obs(data_dir)
    leaderboard = build_leaderboard(
        raw_rows,
        vetting,
        min_snr=args.min_snr,
        min_frac=args.min_frac_strong,
    )
    print_leaderboard(leaderboard, min_snr=args.min_snr, min_frac=args.min_frac_strong)
    write_leaderboard_csv(out_dir / "leaderboard.csv", leaderboard)

    if args.limit is not None:
        leaderboard = leaderboard[: args.limit]
        logging.info("Limited to first %d observations (by R²)", len(leaderboard))

    # Tesseract only needed when we must extract (no cache)
    need_extract = args.force_extract or any(
        not (find_obs_folder(r.obs_id, project_root) / "derived" / f"carrier_track_{r.obs_id}.csv").is_file()
        for r in leaderboard
    )
    if need_extract:
        _check_tesseract()

    gap_stats_list: list[GapStats] = []
    failures: list[tuple[int, str]] = []
    t0 = time.monotonic()

    for i, lb in enumerate(leaderboard, 1):
        logging.info("[%d/%d] Amplitude pipeline obs %d (R²=%.4f) ...", i, len(leaderboard), lb.obs_id, lb.R2)
        try:
            stats = process_one(
                lb,
                project_root,
                out_dir,
                force_extract=args.force_extract,
                exclude_freq_below_khz=args.exclude_freq_below_khz,
                exclude_freq_above_khz=args.exclude_freq_above_khz,
                min_confidence=args.min_confidence,
                gap_min_peak_snr=args.gap_min_peak_snr,
                gap_freq_dev_hz=args.gap_freq_dev_hz,
                freq_smooth_win=args.freq_smooth_win,
            )
            gap_stats_list.append(stats)
            logging.info(
                "  gaps=%d  total=%.1fs  longest=%.1fs  frac_gap=%.3f",
                stats.n_gaps,
                stats.total_gap_duration_s,
                stats.longest_gap_s,
                stats.frac_gap,
            )
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            logging.error("  FAILED obs %d: %s", lb.obs_id, err)
            failures.append((lb.obs_id, err))
            with open(out_dir / "failures.log", "a", encoding="utf-8") as fh:
                fh.write(f"obs {lb.obs_id}: {err}\n{traceback.format_exc()}\n{'-' * 60}\n")

    write_gap_stats_csv(out_dir / "gap_stats.csv", gap_stats_list)

    elapsed = time.monotonic() - t0
    print("\n" + "=" * 72)
    print("AMPLITUDE PIPELINE SUMMARY")
    print("=" * 72)
    print(f"  Processed     : {len(gap_stats_list)} / {len(leaderboard)}")
    print(f"  Failures      : {len(failures)}")
    print(f"  Elapsed       : {elapsed:.1f}s")
    print(f"  Output        : {out_dir}")
    if gap_stats_list:
        print()
        print(
            f"  {'obs_id':>10}  {'n_gaps':>7}  {'total_s':>8}  {'longest':>8}  "
            f"{'frac_gap':>8}  {'dt_s':>6}"
        )
        print("  " + "-" * 58)
        for s in gap_stats_list:
            print(
                f"  {s.obs_id:>10}  {s.n_gaps:>7}  {s.total_gap_duration_s:>8.1f}  "
                f"{s.longest_gap_s:>8.1f}  {s.frac_gap:>8.3f}  {s.dt_s:>6.3f}"
            )
    if failures:
        print()
        print("  Failures:")
        for oid, err in failures:
            print(f"    {oid}: {err}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
