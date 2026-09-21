#!/usr/bin/env python3
"""
Single-observation Doppler validation tool for MOVE-II SatNOGS recordings.

Usage:
    python carrier_doppler_check.py --obs-id 355464                  # waterfall (default)
    python carrier_doppler_check.py --obs-id 355464 --source audio   # audio / AFC sanity check

Source rationale
----------------
  --source waterfall  (default)
      The SatNOGS waterfall PNG is generated from the raw wideband IQ capture
      UPSTREAM of gr-satnogs' TLE-based Doppler correction.  The carrier track
      here preserves the full Doppler sweep and is the right input for orbital
      validation.

  --source audio
      The demodulated .ogg audio is produced AFTER gr-satnogs applies AFC /
      TLE-based frequency correction.  The carrier is nearly flat (B ≈ 0) and
      is only useful as a sanity-check that correction really happened, or to
      measure the residual AFC error.

Steps (both sources):
  1. Extract carrier track (per-frame frequency + confidence)
  2. Predict Doppler shift from TLE + station position (skyfield SGP4)
  3. Fit observed_freq = A + B × predicted_shift  (linear least squares)
  4. Two-panel plot saved to derived/
  5. Print summary block
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT_KM_S = 299_792.458

# Audio spectrogram — must match fetch_and_vet_all_observations.py
NPERSEG = 8192
NOVERLAP_FRAC = 0.75
BAND_HZ = 3000.0
DEFAULT_STRONG_THRESHOLD_DB = 10.0

# Waterfall defaults
DEFAULT_EXCLUDE_FREQ_BELOW_KHZ = -18.0   # skip low-freq birdie/RFI (tune per station)
DEFAULT_MIN_CONFIDENCE_PERCENTILE = 80.0 # keep top-20% confidence rows
DEFAULT_EXCLUDE_FREQ_ABOVE_KHZ = 20.0   # skip right-edge artifacts / high-freq RFI
SPINE_DARK_THRESHOLD = 60                # max(R,G,B) < this  →  "near-black"
SPINE_DARK_FRAC = 0.5                    # fraction of row/col pixels that must be dark
TICK_SCAN_BAND_PX = 8                    # pixels below/left of spine to scan for ticks
TICK_CLUSTER_GAP_PX = 5                  # max gap (px) within one tick cluster

# OCR label-crop geometry
OCR_X_LABEL_PAD_PX = 35   # horizontal half-width of crop centred on each x-tick column
OCR_X_LABEL_H_PX = 28     # height of the crop region below the x-axis tick scan band
OCR_Y_LABEL_HALF_PX = 14  # vertical half-height of crop centred on each y-tick row
# Upscale factors tried in order (empirically chosen to maximise read success):
# scale=3 is the most reliable overall; scale=4 fixes some cases that 3 misses;
# scale=2 rescues ticks where 3/4 add rendering artefacts; scale=5 as last resort.
OCR_SCALES = (3, 4, 2, 5)

SEARCH_ROOTS = ["data", "filtered_data"]


# ---------------------------------------------------------------------------
# Folder / meta helpers
# ---------------------------------------------------------------------------


def find_obs_folder(obs_id: int, project_root: Path) -> Path:
    for root_name in SEARCH_ROOTS:
        root = project_root / root_name
        if not root.is_dir():
            continue
        for candidate in root.iterdir():
            if candidate.is_dir() and candidate.name.startswith(f"move2_{obs_id}_"):
                return candidate
    raise FileNotFoundError(
        f"No folder matching move2_{obs_id}_* found under "
        + ", ".join(SEARCH_ROOTS)
    )


def load_meta(obs_folder: Path, obs_id: int) -> dict:
    meta_path = obs_folder / "meta" / f"observation_{obs_id}.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_iso(s: str) -> datetime:
    s = s.rstrip("Z")
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Step 1a — Audio carrier track extraction
# ---------------------------------------------------------------------------


def extract_carrier_track_audio(
    ogg_path: Path,
    obs_start: datetime,
    *,
    strong_threshold_db: float,
) -> dict:
    """
    Compute per-frame carrier frequency and power via short-time spectrogram.
    Same parameters as fetch_and_vet_all_observations.py::analyze_audio().

    NOTE: gr-satnogs applies TLE-based Doppler correction before saving audio.
    Use --source waterfall for actual Doppler validation; use this path only
    to verify AFC behaviour or measure correction residuals.
    """
    data, sr = sf.read(str(ogg_path))
    if data.ndim > 1:
        data = data[:, 0]

    noverlap = int(NPERSEG * NOVERLAP_FRAC)
    f, t_rel, Sxx = spectrogram(
        data, fs=sr, nperseg=NPERSEG, noverlap=noverlap, window="hann"
    )
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)

    band_mask = f < BAND_HZ
    band = Sxx_db[band_mask]
    f_band = f[band_mask]

    peak_idx = band.argmax(axis=0)
    peak_freq_hz = f_band[peak_idx]
    peak_power_db = band[peak_idx, np.arange(band.shape[1])]

    noise_floor_db = float(np.median(peak_power_db))
    strong_mask = peak_power_db > noise_floor_db + strong_threshold_db

    start_epoch = obs_start.timestamp()
    timestamps = [
        datetime.fromtimestamp(start_epoch + float(t), tz=timezone.utc)
        for t in t_rel
    ]

    return {
        "source": "audio",
        "timestamps": timestamps,
        "t_rel_s": t_rel,
        "peak_freq_hz": peak_freq_hz,
        "peak_power_db": peak_power_db,
        "strong_mask": strong_mask,
        "noise_floor_db": noise_floor_db,
    }


# ---------------------------------------------------------------------------
# Step 1b — Waterfall carrier track extraction
# ---------------------------------------------------------------------------


def _cluster_centers(positions: np.ndarray, min_gap: int = TICK_CLUSTER_GAP_PX) -> list[int]:
    """Group sorted pixel positions into clusters; return each cluster's center."""
    if len(positions) == 0:
        return []
    positions = np.sort(positions)
    clusters: list[int] = []
    current: list[int] = [int(positions[0])]
    for p in positions[1:]:
        if p - current[-1] <= min_gap:
            current.append(int(p))
        else:
            clusters.append(int(round(float(np.mean(current)))))
            current = [int(p)]
    clusters.append(int(round(float(np.mean(current)))))
    return clusters


def detect_spine_box(img_rgb: np.ndarray) -> tuple[int, int, int, int]:
    """
    Locate the plot-area bounding box by finding the outermost rows and columns
    where more than SPINE_DARK_FRAC of pixels are near-black (max channel < 60).

    Returns (top, bottom, left, right) — inclusive pixel indices.
    """
    dark = img_rgb.max(axis=2) < SPINE_DARK_THRESHOLD  # (H, W) bool
    row_dark_frac = dark.mean(axis=1)
    col_dark_frac = dark.mean(axis=0)

    dark_rows = np.where(row_dark_frac > SPINE_DARK_FRAC)[0]
    dark_cols = np.where(col_dark_frac > SPINE_DARK_FRAC)[0]

    if len(dark_rows) < 2 or len(dark_cols) < 2:
        raise ValueError(
            f"Could not detect spine box: only {len(dark_rows)} dark rows and "
            f"{len(dark_cols)} dark cols (threshold: >{SPINE_DARK_FRAC*100:.0f}% "
            f"pixels with max channel < {SPINE_DARK_THRESHOLD})"
        )

    top = int(dark_rows.min())
    bottom = int(dark_rows.max())
    # Use only the leftmost dark-col group for left spine; rightmost for right
    # (extra dark cols from colorbars etc. are excluded by taking the min/max
    # of the first and last contiguous cluster)
    col_clusters = _cluster_centers(dark_cols, min_gap=20)
    left = col_clusters[0]
    right = col_clusters[-1]

    return top, bottom, left, right


def detect_ticks_x(
    img_rgb: np.ndarray,
    spine: tuple[int, int, int, int],
) -> list[int]:
    """
    Find X-axis tick pixel positions by scanning a narrow band just below the
    bottom spine row.  Returns a list of absolute column pixel centers.
    """
    top, bottom, left, right = spine
    scan_h = min(TICK_SCAN_BAND_PX, img_rgb.shape[0] - bottom - 1)
    if scan_h <= 0:
        return []
    band = img_rgb[bottom + 1 : bottom + 1 + scan_h, left : right + 1]
    dark_cols_rel = np.where((band.max(axis=2) < SPINE_DARK_THRESHOLD).any(axis=0))[0]
    clusters = _cluster_centers(dark_cols_rel)
    return [c + left for c in clusters]


def detect_ticks_y(
    img_rgb: np.ndarray,
    spine: tuple[int, int, int, int],
) -> list[int]:
    """
    Find Y-axis tick pixel positions by scanning a narrow band just left of the
    left spine column.  Returns a list of absolute row pixel centers.
    """
    top, bottom, left, right = spine
    scan_w = min(TICK_SCAN_BAND_PX, left)
    if scan_w <= 0:
        return []
    band = img_rgb[top : bottom + 1, left - scan_w : left]
    dark_rows_rel = np.where((band.max(axis=2) < SPINE_DARK_THRESHOLD).any(axis=1))[0]
    clusters = _cluster_centers(dark_rows_rel)
    return [c + top for c in clusters]


def _hhmm_to_elapsed(ocr_values: list[Optional[float]]) -> Optional[list[Optional[float]]]:
    """
    Detect whether Y-axis OCR values are in HHMM format (e.g. "09:52" read by
    OCR as 952.0 after the colon is stripped by the character whitelist) and, if
    so, convert them to elapsed-seconds values relative to the earliest tick.

    Two OCR artefact patterns are handled:
      • Integer form:  "09:52" → OCR drops colon → "952" → 952.0
      • Decimal form:  "09:57" → OCR treats colon as decimal → "9.57" → 9.57

    Detection criterion: after normalisation to HHMM integers all non-None values
    must be in [0, 2359], the maximum consecutive difference (in minutes) must be
    ≤ 10, and the total span must be in (1, 180] minutes.

    Returns a new list with elapsed-seconds floats if HHMM format is detected,
    or None if the values look like elapsed seconds already.
    """
    valid = [v for v in ocr_values if v is not None]
    if not valid:
        return None

    def normalise_hhmm(v: float) -> Optional[int]:
        """Convert an OCR'd float to a plain HHMM integer, or return None."""
        # Decimal form: e.g. 9.57 means "9:57" (H.MM), valid if fractional part
        # is in [0.00, 0.59] and integer part is a valid hour [0, 23].
        h_int = int(v)
        frac = round(v - h_int, 4)   # round to avoid float noise
        if frac != 0.0 and 0.0 <= frac < 0.60:
            m = round(frac * 100)
            if 0 <= h_int <= 23 and 0 <= m <= 59:
                return h_int * 100 + m
            return None
        # Integer form: must be a whole number
        if v != int(v):
            return None
        iv = int(v)
        if iv < 0 or iv > 2359:
            return None
        return iv

    def to_minutes(hhmm: int) -> int:
        h, m = hhmm // 100, hhmm % 100
        if m >= 60:
            return -1
        return h * 60 + m

    # Normalise all non-None values
    norm: list[Optional[int]] = []
    for v in ocr_values:
        norm.append(None if v is None else normalise_hhmm(v))

    valid_norm = [n for n in norm if n is not None]
    if not valid_norm or any(n is None and v is not None for n, v in zip(norm, ocr_values)):
        # If any non-None value failed normalisation it's not HHMM format
        return None

    minutes = [to_minutes(n) for n in valid_norm]
    if any(m < 0 for m in minutes):
        return None

    # Outlier removal: if one normalised value breaks the monotone sequence by
    # more than 2 minutes (e.g. a misread "0" where "705" was expected), mark
    # it None — identical to the general "allow 1 OCR failure per axis" rule.
    def _passes_hhmm(mins: list[int]) -> bool:
        if len(mins) < 2:
            return False
        # Raw absolute minute differences — no wrapping.  Genuine HHMM tick
        # sequences advance one minute per tick (occasionally two), so max diff
        # must be small (≤ 2).  Elapsed-seconds values misinterpreted as HHMM
        # (e.g. [300,250,200] → minutes [180,170,120]) give diffs of 50 or 60
        # and are correctly rejected by this stricter bound.
        diffs = [abs(mins[i+1] - mins[i]) for i in range(len(mins)-1)]
        if max(diffs) > 2:
            return False
        span = max(mins) - min(mins)
        if span < 0:
            span += 24 * 60
        return 1 <= span <= 180

    if not _passes_hhmm(minutes):
        # Try removing each value in turn and see if the rest pass
        found_outlier = False
        for excl in range(len(minutes)):
            trial = [m for i, m in enumerate(minutes) if i != excl]
            if _passes_hhmm(trial):
                # Mark the outlier position as None
                # Identify which index in norm corresponds to this position
                non_none_idx = [i for i, n in enumerate(norm) if n is not None]
                norm[non_none_idx[excl]] = None
                minutes = trial
                found_outlier = True
                break
        if not found_outlier:
            return None

    min_min = min(minutes)

    # Convert: elapsed_s[i] = (minutes[i] - min_minutes) * 60
    result: list[Optional[float]] = []
    min_idx = 0
    for n in norm:
        if n is None:
            result.append(None)
        else:
            m = to_minutes(n)
            result.append(float((m - min_min) * 60))
    return result


def _check_tesseract() -> None:
    """Exit with a clear message if the tesseract-ocr binary is not on PATH."""
    import shutil
    if shutil.which("tesseract") is None:
        sys.exit(
            "ERROR: tesseract-ocr is not installed or not found on PATH.\n"
            "Install the system package first:\n"
            "  Ubuntu/Debian  : sudo apt install tesseract-ocr\n"
            "  Fedora/RHEL    : sudo dnf install tesseract\n"
            "  macOS (Homebrew): brew install tesseract\n"
            "  Windows        : https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Then also install the Python wrapper (if not already done):\n"
            "  pip install pytesseract\n"
            "and re-run."
        )


def _ocr_one_tick(
    img_rgb: np.ndarray,
    spine: tuple[int, int, int, int],
    tick_px: int,
    axis: str,
) -> Optional[float]:
    """
    OCR the numeric label adjacent to a detected tick mark and return it as a float.

    axis='x': crops a region directly below the bottom spine edge,
              horizontally centred on ``tick_px`` (an absolute column index).
    axis='y': crops a region directly to the left of the left spine edge,
              vertically centred on ``tick_px`` (an absolute row index).

    The crop is inverted (255-pixel) before OCR because tesseract prefers dark
    text on a light background.  Multiple upscale factors are tried in sequence
    (``OCR_SCALES``) and the first non-None result is returned.  Within each
    scale, when the OCR text contains more than one number (e.g. "2 400" due to
    a rendering artefact), the *longest* numeric token is used so that short
    spurious digits don't shadow the real value.

    Returns a parsed float, or None if no scale yields a usable number.
    """
    import re
    try:
        import pytesseract
        from PIL import Image as PILImage
    except ImportError:
        sys.exit(
            "ERROR: pytesseract is not installed.\n"
            "Install it with:  pip install pytesseract\n"
            "(The tesseract-ocr system package is also required — see --help for details.)"
        )

    top, bottom, left, right = spine
    H, W = img_rgb.shape[:2]
    OCR_CFG = "--psm 7 -c tessedit_char_whitelist=0123456789.-"

    if axis == "x":
        x0 = max(0, tick_px - OCR_X_LABEL_PAD_PX)
        x1 = min(W, tick_px + OCR_X_LABEL_PAD_PX)
        y0 = bottom + TICK_SCAN_BAND_PX + 2
        y1 = min(H, y0 + OCR_X_LABEL_H_PX)
    else:  # y-axis: label is to the left of the spine
        x0 = 0
        x1 = max(0, left - TICK_SCAN_BAND_PX - 2)
        y0 = max(0, tick_px - OCR_Y_LABEL_HALF_PX)
        y1 = min(H, tick_px + OCR_Y_LABEL_HALF_PX)

    if x1 <= x0 or y1 <= y0:
        return None

    crop_inv = 255 - img_rgb[y0:y1, x0:x1]
    ph, pw = crop_inv.shape[:2]

    for scale in OCR_SCALES:
        pil_img = PILImage.fromarray(crop_inv).resize(
            (pw * scale, ph * scale), PILImage.LANCZOS
        )
        try:
            text = pytesseract.image_to_string(pil_img, config=OCR_CFG).strip()
        except Exception:
            continue

        nums = re.findall(r"-?\d+\.?\d*", text)
        if nums:
            # When multiple tokens appear (e.g. "2 400"), use the longest one.
            best = max(nums, key=len)
            try:
                return float(best)
            except ValueError:
                pass

    return None


def build_viridis_lut() -> np.ndarray:
    """Return a (256, 3) uint8 array sampling the viridis colormap at 256 levels."""
    import matplotlib
    cmap = matplotlib.colormaps["viridis"].resampled(256)
    lut = np.array(
        [[int(r * 255), int(g * 255), int(b * 255)]
         for r, g, b, _ in (cmap(i / 255.0) for i in range(256))],
        dtype=np.uint8,
    )
    return lut


def map_pixels_to_power(plot_img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Map each plot-area pixel's RGB to its nearest viridis index (0 = darkest /
    lowest power, 255 = brightest / highest power) via KD-tree nearest-neighbour
    search, processed in 20 000-pixel chunks to bound peak memory usage.

    Non-viridis pixels (white text labels, black spine borders, grey annotations)
    are detected by their reconstruction error against the nearest LUT entry and
    are zeroed out so they cannot corrupt the per-row carrier argmax.
    Reconstruction-error threshold is 50 RGB units:
      • Pure white [255,255,255] → nearest viridis [253,231,37] → error ≈ 219  → masked
      • Black spine [0,0,0]     → nearest viridis [68,1,84]    → error ≈ 108  → masked
      • True viridis background [68,1,84]                      → error   ≈ 0  → kept

    Returns an (H, W) uint8 array of power indices (0 for non-viridis pixels).
    """
    H, W = plot_img.shape[:2]
    flat = plot_img.reshape(-1, 3).astype(np.float32)
    lut_f = lut.astype(np.float32)
    tree = KDTree(lut_f)

    CHUNK = 20_000
    REC_ERROR_THRESHOLD = 50.0  # RGB-space Euclidean distance; see docstring

    power_flat = np.zeros(len(flat), dtype=np.uint8)
    for start in range(0, len(flat), CHUNK):
        end = min(start + CHUNK, len(flat))
        chunk = flat[start:end]
        dist, idx = tree.query(chunk)
        # Zero out pixels whose colour cannot be explained by any viridis entry
        valid = dist < REC_ERROR_THRESHOLD
        power_flat[start:end][valid] = idx[valid].astype(np.uint8)

    return power_flat.reshape(H, W)


def extract_carrier_track_waterfall(
    waterfall_path: Path,
    obs_start: datetime,
    duration_s: float,
    *,
    f0_hz: float,
    exclude_freq_below_khz: float,
    exclude_freq_above_khz: float,
    min_confidence_percentile: float,
) -> dict:
    """
    Extract per-row carrier frequency from the raw SatNOGS waterfall PNG.

    The waterfall is generated upstream of gr-satnogs' Doppler correction, so the
    carrier track here carries the full Doppler sweep and is suitable for orbital
    validation (B ≈ 1 expected for an uncorrected pass).

    Pipeline
    --------
    1. Auto-detect spine box and tick marks.
    2. OCR each tick label (pytesseract) to get the actual kHz / seconds values
       written on the image — no hardcoded bandwidth assumption.
    3. Build pixel→kHz (X) and pixel→seconds (Y) linear fits from the OCR values.
       If more than 1 tick per axis fails OCR, raise ValueError so the caller can
       skip the observation rather than falling back to a guessed scale.
    4. Map pixel RGB → viridis power index via a pre-built 256-entry KD-tree LUT.
    5. Per-row vectorised argmax after excluding the configurable low-frequency band.
    6. Confidence = peak_power − row_median; strong mask = above percentile threshold.

    Returns the same dict structure as extract_carrier_track_audio() so that
    steps 2–4 (Doppler prediction, linear fit, plotting) work unchanged.
    """
    from PIL import Image

    _check_tesseract()

    img_rgb = np.array(Image.open(waterfall_path).convert("RGB"))

    # ------------------------------------------------------------------
    # Spine detection
    # ------------------------------------------------------------------
    spine = detect_spine_box(img_rgb)
    top, bottom, left, right = spine
    logging.info(
        "  Spine: top=%d, bottom=%d, left=%d, right=%d  (%d×%d plot px)",
        top, bottom, left, right, bottom - top, right - left,
    )

    # ------------------------------------------------------------------
    # Tick detection
    # ------------------------------------------------------------------
    x_ticks_px = detect_ticks_x(img_rgb, spine)
    y_ticks_px = detect_ticks_y(img_rgb, spine)

    n_x = len(x_ticks_px)
    n_y = len(y_ticks_px)
    logging.info("  X ticks: %d at cols %s", n_x, x_ticks_px)
    logging.info("  Y ticks: %d at rows %s", n_y, y_ticks_px)

    if n_x < 3:
        raise ValueError(
            f"Only {n_x} X-tick cluster(s) detected (need ≥ 3). "
            "The spine detection may be wrong — check the waterfall image."
        )
    if n_y < 3:
        raise ValueError(
            f"Only {n_y} Y-tick cluster(s) detected (need ≥ 3). "
            "Check spine detection output."
        )

    # ------------------------------------------------------------------
    # OCR: read actual kHz / seconds values from the tick labels.
    # Multi-scale approach with longest-token heuristic (see _ocr_one_tick).
    # ------------------------------------------------------------------
    logging.info("  Running OCR on tick labels ...")

    x_ocr: list[Optional[float]] = [
        _ocr_one_tick(img_rgb, spine, col, "x") for col in x_ticks_px
    ]
    y_ocr: list[Optional[float]] = [
        _ocr_one_tick(img_rgb, spine, row, "y") for row in y_ticks_px
    ]

    x_fail = [i for i, v in enumerate(x_ocr) if v is None]
    y_fail = [i for i, v in enumerate(y_ocr) if v is None]

    logging.info(
        "  X OCR: %s  (failed indices: %s)",
        [f"{v:.1f}" if v is not None else "None" for v in x_ocr],
        x_fail,
    )
    logging.info(
        "  Y OCR: %s  (failed indices: %s)",
        [f"{v:.1f}" if v is not None else "None" for v in y_ocr],
        y_fail,
    )

    if len(x_fail) > 1:
        raise ValueError(
            f"OCR failed on {len(x_fail)} of {n_x} X-axis tick labels "
            f"(failed indices: {x_fail}). "
            "Cannot build a reliable pixel→frequency calibration — skipping observation."
        )
    if len(y_fail) > 1:
        raise ValueError(
            f"OCR failed on {len(y_fail)} of {n_y} Y-axis tick labels "
            f"(failed indices: {y_fail}). "
            "Cannot build a reliable pixel→time calibration — skipping observation."
        )

    # Build calibration arrays using only the successfully OCR'd ticks.
    x_good_px = np.array([x_ticks_px[i] for i in range(n_x) if x_ocr[i] is not None], dtype=float)
    x_good_khz = np.array([x_ocr[i] for i in range(n_x) if x_ocr[i] is not None], dtype=float)

    # Spectral-inversion correction: gr-satnogs waterfalls from certain SDR
    # configurations (notably RTL-SDR with swapped I/Q) display a mirrored
    # spectrum.  The axis labels show e.g. "−20 kHz" on the left, but the
    # actual RF at that column is f0 + 20 kHz.  Negating the OCR'd kHz values
    # before the polyfit corrects the mapping, so that left-column pixels
    # correctly map to frequencies above f0 and B → +1 for uncorrected Doppler.
    # Evidence: without this correction B = −1 for obs 359214 (known-good
    # reference), while with it B = +0.99.
    x_good_khz = -x_good_khz

    # Y-axis format detection: newer gr-satnogs waterfalls label the left Y axis
    # with UTC times in H:MM format (e.g. "09:52"), which OCR reads as integers
    # like 952.  _hhmm_to_elapsed detects this pattern and converts to elapsed
    # seconds relative to the earliest tick so that the downstream calibration
    # is unaffected by the axis label style.
    y_ocr_converted = _hhmm_to_elapsed(y_ocr)
    if y_ocr_converted is not None:
        logging.info(
            "  Y OCR: UTC-time format (HHMM) detected — converting to elapsed seconds."
        )
        y_ocr = y_ocr_converted
        # Re-check failures after conversion (should be unchanged, but update log)
        y_fail = [i for i, v in enumerate(y_ocr) if v is None]
        logging.info(
            "  Y OCR (elapsed s): %s  (failed indices: %s)",
            [f"{v:.1f}" if v is not None else "None" for v in y_ocr],
            y_fail,
        )
        if len(y_fail) > 1:
            raise ValueError(
                f"After HHMM conversion, OCR failed on {len(y_fail)} of {n_y} "
                f"Y-axis tick labels. Cannot build reliable pixel→time calibration."
            )

    y_good_rel = np.array(
        [float(y_ticks_px[i]) - top for i in range(n_y) if y_ocr[i] is not None], dtype=float
    )
    y_good_s = np.array([y_ocr[i] for i in range(n_y) if y_ocr[i] is not None], dtype=float)

    # ------------------------------------------------------------------
    # X calibration: absolute column pixel → frequency offset (kHz from centre)
    # Sign is determined entirely by the OCR'd label values: if left ticks carry
    # negative kHz and right ticks carry positive kHz the slope is positive, and
    # vice versa.  No bandwidth assumption is needed.
    # ------------------------------------------------------------------
    x_slope, x_intercept = np.polyfit(x_good_px, x_good_khz, 1)
    logging.info(
        "  X cal (OCR): slope=%.5f kHz/px  intercept=%.3f kHz  "
        "(from %d/%d ticks)",
        x_slope, x_intercept, len(x_good_px), n_x,
    )

    # ------------------------------------------------------------------
    # Y calibration: plot-relative row → elapsed seconds since obs start.
    # The OCR values are elapsed seconds as printed on the axis; the fit slope
    # is negative when time increases upward (newer rows at the top) and positive
    # when time increases downward (conventional layout).  Both cases work without
    # any manual flip because the OCR values already encode the direction.
    # ------------------------------------------------------------------
    y_slope, y_intercept = np.polyfit(y_good_rel, y_good_s, 1)
    logging.info(
        "  Y cal (OCR): slope=%.5f s/px  intercept=%.3f s  "
        "(from %d/%d ticks)",
        y_slope, y_intercept, len(y_good_rel), n_y,
    )

    # ------------------------------------------------------------------
    # Viridis LUT and power mapping
    # ------------------------------------------------------------------
    logging.info("  Building viridis LUT and mapping pixel power ...")
    lut = build_viridis_lut()
    plot_img = img_rgb[top : bottom + 1, left : right + 1]  # (H_plot, W_plot, 3)
    power = map_pixels_to_power(plot_img, lut)               # (H_plot, W_plot) uint8

    n_rows, n_cols = power.shape

    # ------------------------------------------------------------------
    # Column → absolute frequency
    # ------------------------------------------------------------------
    col_abs = np.arange(n_cols, dtype=float) + left
    col_freq_khz = x_slope * col_abs + x_intercept
    col_freq_hz = f0_hz + col_freq_khz * 1000.0

    # ------------------------------------------------------------------
    # Per-row vectorised carrier extraction
    # ------------------------------------------------------------------
    exclude_mask = (
        (col_freq_khz < exclude_freq_below_khz) | (col_freq_khz > exclude_freq_above_khz)
    )  # (n_cols,) bool
    n_excluded = int(exclude_mask.sum())
    n_kept = n_cols - n_excluded
    if n_kept < 10:
        logging.warning(
            "  Only %d columns remain after exclusion (excluded %d/%d). "
            "Adjust --exclude-freq-below-khz / --exclude-freq-above-khz.",
            n_kept, n_excluded, n_cols,
        )

    power_f = power.astype(np.float32)
    power_f[:, exclude_mask] = -np.inf  # zero-out excluded band in-place

    # Column-median subtraction: for each frequency column, subtract its
    # median power (computed across all rows, before exclusion masking).
    # Signals that are present in ≥50% of rows (e.g. CW birdies, DC spike)
    # subtract away to near-zero; the satellite Doppler track — which occupies
    # each frequency column only briefly as it sweeps across — stays well
    # above the column baseline and becomes the dominant peak per row.
    # The -inf entries in excluded columns remain -inf after subtraction
    # (finite subtracted from -inf is still -inf), so no re-masking is needed.
    col_median = np.median(power.astype(float), axis=0)         # (n_cols,)
    power_f -= col_median[np.newaxis, :]

    peak_col_idx = power_f.argmax(axis=1)                       # (n_rows,)
    peak_power = power[np.arange(n_rows), peak_col_idx].astype(float)
    peak_freq_hz = col_freq_hz[peak_col_idx]                    # (n_rows,)

    # ------------------------------------------------------------------
    # Confidence score and strong mask
    # ------------------------------------------------------------------
    # Per-row background: median power over ALL columns (not just the kept band),
    # so the confidence reflects signal above the local noise level.
    row_median = np.median(power.astype(float), axis=1)         # (n_rows,)
    confidence = peak_power - row_median                        # (n_rows,)

    threshold = float(np.percentile(confidence, min_confidence_percentile))
    strong_mask = confidence >= threshold

    # ------------------------------------------------------------------
    # Time axis
    # ------------------------------------------------------------------
    row_indices = np.arange(n_rows, dtype=float)
    t_rel_all = y_slope * row_indices + y_intercept             # (n_rows,) seconds

    start_epoch = obs_start.timestamp()
    timestamps = [
        datetime.fromtimestamp(start_epoch + float(t), tz=timezone.utc)
        for t in t_rel_all
    ]

    return {
        "source": "waterfall",
        "timestamps": timestamps,
        "t_rel_s": t_rel_all,
        "peak_freq_hz": peak_freq_hz,
        "peak_power_db": confidence,
        "strong_mask": strong_mask,
        "noise_floor_db": float(np.median(confidence)),
        "n_x_ticks": n_x,
        "n_y_ticks": n_y,
        "x_slope_khz_per_px": float(x_slope),
        "x_intercept_khz": float(x_intercept),
        "y_slope_s_per_px": float(y_slope),
        "y_intercept_s": float(y_intercept),
    }


# ---------------------------------------------------------------------------
# Step 1 dispatcher + CSV save
# ---------------------------------------------------------------------------


def save_carrier_track_csv(track: dict, derived_dir: Path, obs_id: int) -> Path:
    """Write per-frame carrier CSV.  Source info is saved to a companion JSON."""
    out_path = derived_dir / f"carrier_track_{obs_id}.csv"
    derived_dir.mkdir(parents=True, exist_ok=True)

    fields = ["timestamp_utc", "t_rel_s", "peak_freq_hz", "peak_power_db_relative", "strong"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ts, t_rel, pf, pp, strong in zip(
            track["timestamps"],
            track["t_rel_s"],
            track["peak_freq_hz"],
            track["peak_power_db"],
            track["strong_mask"],
        ):
            writer.writerow(
                {
                    "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    "t_rel_s": round(float(t_rel), 4),
                    "peak_freq_hz": round(float(pf), 4),
                    "peak_power_db_relative": round(float(pp), 4),
                    "strong": int(strong),
                }
            )

    # Companion metadata (source, calibration params)
    meta_path = derived_dir / f"carrier_track_{obs_id}_meta.json"
    meta_out: dict = {"source": track.get("source", "unknown"), "obs_id": obs_id}
    for key in ("n_x_ticks", "n_y_ticks", "x_slope_khz_per_px",
                "x_intercept_khz", "y_slope_s_per_px", "y_intercept_s"):
        if key in track:
            meta_out[key] = track[key]
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_out, fh, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Step 2 — Doppler prediction via skyfield
# ---------------------------------------------------------------------------


def predict_doppler(
    meta: dict,
    timestamps: list[datetime],
    strong_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute predicted Doppler shift (Hz) for each strong-frame timestamp.

    Sign convention
    ---------------
    range_rate > 0  → satellite receding  → frequency shift is negative
    range_rate < 0  → satellite approaching → frequency shift is positive

        predicted_shift = -f0 * range_rate_km_s / c_km_s

    Thus the shift is positive (blue-shifted) on approach and negative
    (red-shifted) on recession — matching physical expectation.
    """
    from skyfield.api import EarthSatellite, load, wgs84

    ts = load.timescale()
    satellite = EarthSatellite(
        meta["tle1"], meta["tle2"], meta.get("tle0", ""), ts
    )
    station = wgs84.latlon(
        latitude_degrees=float(meta["station_lat"]),
        longitude_degrees=float(meta["station_lng"]),
        elevation_m=float(meta.get("station_alt") or 0.0),
    )
    f0 = float(meta["observation_frequency"])

    strong_idx = np.where(strong_mask)[0]
    strong_ts = [timestamps[i] for i in strong_idx]
    sky_times = ts.from_datetimes(strong_ts)

    diff = satellite - station
    topo = diff.at(sky_times)
    pos_km = topo.position.km          # (3, n)
    vel_km_s = topo.velocity.km_per_s  # (3, n)

    range_km = np.linalg.norm(pos_km, axis=0)
    range_rate_km_s = np.einsum("ij,ij->j", vel_km_s, pos_km) / range_km

    return (-f0 * range_rate_km_s / SPEED_OF_LIGHT_KM_S).astype(float)


# ---------------------------------------------------------------------------
# Step 3 — Linear fit
# ---------------------------------------------------------------------------


def fit_doppler(
    observed_freq: np.ndarray,
    predicted_shift: np.ndarray,
) -> dict:
    """
    Fit:  observed_freq = A + B × predicted_shift

    For the waterfall source, observed_freq is absolute RF frequency (Hz), so
    A ≈ f0 (centre frequency) and B ≈ 1 for uncorrected Doppler.
    For the audio source, observed_freq is baseband (Hz) and B ≈ 0 for AFC-corrected.
    """
    coeffs = np.polyfit(predicted_shift, observed_freq, deg=1)
    B, A = float(coeffs[0]), float(coeffs[1])

    fitted = A + B * predicted_shift
    residuals = observed_freq - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((observed_freq - observed_freq.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    absB = abs(B)
    if absB < 0.15:
        interpretation = "fully AFC-corrected — no usable Doppler information"
    elif absB > 0.7:
        cal_note = ""
        if not (0.75 <= absB <= 1.35):
            sign_note = " (check X-axis sign convention)" if B < 0 else ""
            cal_note = (
                f"  [calibration note: |B|={absB:.3f} ≠ 1; "
                f"check OCR tick values and Y-axis time direction{sign_note}]"
            )
        interpretation = (
            f"raw/uncorrected Doppler — track shape is physically validated{cal_note}"
        )
    else:
        interpretation = (
            f"partially corrected (B={B:.3f}) — some Doppler remains in track"
        )

    return {
        "A": A,
        "B": B,
        "abs_B": abs(B),
        "r2": r2,
        "rms_hz": rms,
        "interpretation": interpretation,
        "residuals": residuals,
        "fitted": fitted,
    }


# ---------------------------------------------------------------------------
# Step 4 — Plot
# ---------------------------------------------------------------------------


def make_plot(
    track: dict,
    strong_freq: np.ndarray,
    predicted_shift: np.ndarray,
    fit: dict,
    obs_id: int,
    station_name: str,
    derived_dir: Path,
    f0_hz: float,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    source = track.get("source", "audio")
    strong_idx = np.where(track["strong_mask"])[0]
    strong_ts = [track["timestamps"][i] for i in strong_idx]

    # For waterfall, plot offset from centre frequency for legibility
    if source == "waterfall":
        plot_freq = strong_freq - f0_hz
        fitted_curve = fit["A"] - f0_hz + fit["B"] * predicted_shift
        ylabel = "Carrier offset from f₀ (Hz)"
        track_label = "Waterfall carrier (strong rows)"
    else:
        plot_freq = strong_freq
        fitted_curve = fit["A"] + fit["B"] * predicted_shift
        ylabel = "Audio frequency (Hz)"
        track_label = "Audio carrier (strong frames)"

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        f"Doppler Validation — obs {obs_id}  ({station_name})  [{source}]",
        fontsize=13,
        fontweight="bold",
    )

    ax0 = axes[0]
    ax0.plot(strong_ts, plot_freq, color="steelblue", lw=1.0, label=track_label)
    ax0.set_ylabel(ylabel)
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, alpha=0.35)

    ax1 = axes[1]
    ax1.plot(strong_ts, plot_freq, color="steelblue", lw=1.0, label="Observed carrier")
    ax1.plot(
        strong_ts,
        fitted_curve,
        color="tomato",
        lw=1.5,
        linestyle="--",
        label=f"Fitted model  A={fit['A'] - (f0_hz if source == 'waterfall' else 0):.1f} Hz,  B={fit['B']:.4f}",
    )
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel("Time (UTC)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.35)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30)

    ann = (
        f"R² = {fit['r2']:.4f}   RMS = {fit['rms_hz']:.1f} Hz\n"
        f"{fit['interpretation']}"
    )
    ax1.text(
        0.02, 0.05, ann,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    out_path = derived_dir / f"doppler_validation_{obs_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# TLE epoch helpers
# ---------------------------------------------------------------------------


def _tle_epoch_from_line1(line1: str) -> Optional[datetime]:
    try:
        from datetime import timedelta
        epoch_str = line1[18:32].strip()
        year2 = int(epoch_str[:2])
        year = 2000 + year2 if year2 < 57 else 1900 + year2
        day_frac = float(epoch_str[2:])
        day = int(day_frac)
        frac = day_frac - day
        base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1)
        return base + timedelta(days=frac)
    except Exception:
        return None


def tle_epoch_delta_days(meta: dict) -> Optional[float]:
    epoch = _tle_epoch_from_line1(meta.get("tle1") or "")
    if epoch is None or not meta.get("start"):
        return None
    return abs((parse_iso(meta["start"]) - epoch).total_seconds()) / 86400.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-observation Doppler validation for MOVE-II SatNOGS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--obs-id", type=int, required=True, help="SatNOGS observation ID")
    p.add_argument(
        "--source",
        choices=["waterfall", "audio"],
        default="waterfall",
        help=(
            "Input source for carrier extraction.  'waterfall' uses the raw PNG "
            "(upstream of Doppler correction) and is the default.  'audio' uses "
            "the demodulated .ogg (post-AFC) and is useful only as a sanity check."
        ),
    )

    # Audio-specific
    p.add_argument(
        "--strong-threshold-db",
        type=float,
        default=DEFAULT_STRONG_THRESHOLD_DB,
        help="[audio] dB above median noise floor to label a frame 'strong'.",
    )

    # Waterfall-specific
    p.add_argument(
        "--exclude-freq-below-khz",
        type=float,
        default=DEFAULT_EXCLUDE_FREQ_BELOW_KHZ,
        help=(
            "[waterfall] Exclude columns with freq < this value (kHz from centre) "
            "from the per-row argmax.  Use to skip low-frequency birdies/RFI."
        ),
    )
    p.add_argument(
        "--exclude-freq-above-khz",
        type=float,
        default=DEFAULT_EXCLUDE_FREQ_ABOVE_KHZ,
        help=(
            "[waterfall] Exclude columns with freq > this value (kHz from centre) "
            "from the per-row argmax.  Increase to guard against high-freq RFI or "
            "right-edge rendering artefacts."
        ),
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE_PERCENTILE,
        help=(
            "[waterfall] Keep only rows whose confidence (peak − row median power) "
            "is at or above this percentile of all rows.  80 → top 20%% of rows."
        ),
    )

    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Project root containing data/ and filtered_data/ (default: parent of this script).",
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    args = build_parser().parse_args()
    obs_id: int = args.obs_id
    source: str = args.source

    project_root = args.data_root or Path(__file__).resolve().parent.parent

    print(f"\n=== Doppler Validation — obs {obs_id}  [{source}] ===\n")

    # ------------------------------------------------------------------
    # Locate observation folder
    # ------------------------------------------------------------------
    try:
        obs_folder = find_obs_folder(obs_id, project_root)
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")
    print(f"Observation folder : {obs_folder}")

    derived_dir = obs_folder / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load meta
    # ------------------------------------------------------------------
    meta = load_meta(obs_folder, obs_id)
    obs_start = parse_iso(meta["start"])
    obs_end = parse_iso(meta["end"])
    duration_s = (obs_end - obs_start).total_seconds()
    station_name: str = meta.get("station_name") or "unknown"
    f0_hz: float = float(meta["observation_frequency"])
    delta_days = tle_epoch_delta_days(meta)

    print(f"Station            : {station_name}")
    print(f"Obs start (UTC)    : {obs_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration           : {duration_s:.0f} s")
    print(f"Centre frequency   : {f0_hz/1e6:.4f} MHz")
    print(
        f"TLE epoch delta    : {delta_days:.4f} days"
        if delta_days is not None
        else "TLE epoch delta    : N/A"
    )

    raw_dir = obs_folder / "raw"

    # ------------------------------------------------------------------
    # Step 1: Extract carrier track
    # ------------------------------------------------------------------
    if source == "audio":
        ogg_files = list(raw_dir.glob("*.ogg"))
        if not ogg_files:
            sys.exit(f"ERROR: No .ogg file found in {raw_dir}")
        ogg_path = ogg_files[0]
        print(f"Audio file         : {ogg_path.name}")
        print(
            f"\n[1/4] Extracting carrier track from audio "
            f"(threshold: median+{args.strong_threshold_db:.0f} dB) ..."
        )
        track = extract_carrier_track_audio(
            ogg_path, obs_start, strong_threshold_db=args.strong_threshold_db
        )
        n_total = len(track["timestamps"])
        print(f"   Total frames     : {n_total}")
        print(f"   Noise floor      : {track['noise_floor_db']:.1f} dB")

    else:  # waterfall
        wf_files = list(raw_dir.glob("waterfall_*.png"))
        if not wf_files:
            sys.exit(f"ERROR: No waterfall_*.png found in {raw_dir}")
        wf_path = wf_files[0]
        print(f"Waterfall file     : {wf_path.name}")
        print(
            f"\n[1/4] Extracting carrier track from waterfall "
            f"(OCR calibration, "
            f"band [{args.exclude_freq_below_khz:.0f}, {args.exclude_freq_above_khz:.0f}] kHz, "
            f"conf≥p{args.min_confidence:.0f}) ..."
        )
        try:
            track = extract_carrier_track_waterfall(
                wf_path,
                obs_start,
                duration_s,
                f0_hz=f0_hz,
                exclude_freq_below_khz=args.exclude_freq_below_khz,
                exclude_freq_above_khz=args.exclude_freq_above_khz,
                min_confidence_percentile=args.min_confidence,
            )
        except ValueError as exc:
            sys.exit(f"ERROR: {exc}")
        n_total = len(track["timestamps"])
        print(
            f"   Plot rows        : {n_total}  "
            f"(X ticks: {track['n_x_ticks']}, Y ticks: {track['n_y_ticks']})"
        )
        print(
            f"   X scale          : {track['x_slope_khz_per_px']*1000:.2f} Hz/px  "
            f"intercept={track['x_intercept_khz']:.2f} kHz"
        )
        print(
            f"   Y scale          : {track['y_slope_s_per_px']:.4f} s/row  "
            f"intercept={track['y_intercept_s']:.2f} s"
        )

    n_strong = int(track["strong_mask"].sum())
    frac_strong = n_strong / n_total if n_total else 0.0
    print(f"   Strong frames    : {n_strong}  ({frac_strong:.1%})")

    csv_path = save_carrier_track_csv(track, derived_dir, obs_id)
    print(f"   Saved CSV        : {csv_path}")

    if n_strong < 10:
        hint = "--strong-threshold-db" if source == "audio" else "--min-confidence"
        print(
            f"\nWARNING: Only {n_strong} strong frames — fit will be unreliable. "
            f"Try adjusting {hint}."
        )

    # ------------------------------------------------------------------
    # Step 2: Predict Doppler
    # ------------------------------------------------------------------
    print("\n[2/4] Predicting Doppler shift from TLE + station position ...")
    try:
        predicted_shift = predict_doppler(
            meta, track["timestamps"], track["strong_mask"]
        )
    except Exception as exc:
        sys.exit(f"ERROR during Doppler prediction: {exc}")

    strong_idx = np.where(track["strong_mask"])[0]
    strong_freq = track["peak_freq_hz"][strong_idx]
    pred_range = float(predicted_shift.max() - predicted_shift.min())
    print(
        f"   Predicted shift range : {pred_range:.1f} Hz  "
        f"({predicted_shift.min():.1f} → {predicted_shift.max():.1f} Hz)"
    )

    # ------------------------------------------------------------------
    # Step 3: Linear fit
    # ------------------------------------------------------------------
    print("\n[3/4] Fitting  observed_freq = A + B × predicted_shift ...")
    if n_strong < 2:
        sys.exit("ERROR: Need at least 2 strong frames to fit.")

    fit = fit_doppler(strong_freq, predicted_shift)
    # For waterfall: print A as offset from f0 for interpretability
    a_display = fit["A"] - f0_hz if source == "waterfall" else fit["A"]
    a_label = "A (offset from f₀)" if source == "waterfall" else "A (offset)"
    print(f"   {a_label:22s}: {a_display:.2f} Hz")
    print(f"   B (scale factor)       : {fit['B']:.4f}")
    print(f"   R²                     : {fit['r2']:.4f}")
    print(f"   RMS residual           : {fit['rms_hz']:.2f} Hz")
    print(f"   Interpretation         : {fit['interpretation']}")

    # ------------------------------------------------------------------
    # Step 4: Plot
    # ------------------------------------------------------------------
    print("\n[4/4] Saving plot ...")
    try:
        plot_path = make_plot(
            track=track,
            strong_freq=strong_freq,
            predicted_shift=predicted_shift,
            fit=fit,
            obs_id=obs_id,
            station_name=station_name,
            derived_dir=derived_dir,
            f0_hz=f0_hz,
        )
        print(f"   Saved plot       : {plot_path}")
    except ImportError:
        print("   WARNING: matplotlib not available — plot skipped.")
        plot_path = None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  obs_id            : {obs_id}")
    print(f"  source            : {source}")
    print(f"  station           : {station_name}")
    print(
        f"  TLE epoch delta   : {delta_days:.4f} days"
        if delta_days is not None
        else "  TLE epoch delta   : N/A"
    )
    print(f"  n_strong_frames   : {n_strong}")
    print(f"  {a_label:22s}: {a_display:.2f} Hz")
    print(f"  B (scale factor)  : {fit['B']:.4f}")
    print(f"  R²                : {fit['r2']:.4f}")
    print(f"  RMS residual      : {fit['rms_hz']:.2f} Hz")
    print(f"  Result            : {fit['interpretation']}")
    if plot_path:
        print(f"  Plot              : {plot_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
