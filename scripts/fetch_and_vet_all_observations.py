#!/usr/bin/env python3
"""
Fetch the full SatNOGS observation history for MOVE-II (NORAD 43780), apply a
metadata prefilter, download audio/waterfall for survivors, run spectrogram
vetting, and sort into data/ vs filtered_data/ trees.

  data/move2_{id}_{YYYY-MM-DD}/          # all metadata-pass observations
  filtered_data/move2_{id}_{YYYY-MM-DD}/ # high-quality subset for downstream use
  data/vetting_report_full.csv           # one row per fetched observation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import requests
import soundfile as sf
from scipy.signal import spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_move2_observations import (  # noqa: E402
    API_BASE,
    RateLimitedSession,
    download_file,
    find_local_ogg,
    format_start_for_filename,
    lat_lng_precision_score,
    observation_folder_name,
    parse_iso,
    parse_link_next,
    tle_epoch_from_line1,
    write_meta,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_NORAD_ID = 43780
DEFAULT_API_DELAY = 1.0
DEFAULT_MIN_ALTITUDE = 20.0
DEFAULT_MAX_TLE_AGE_DAYS = 3.0
DEFAULT_MIN_COORD_DECIMALS = 2
DEFAULT_MIN_SNR = 15.0
DEFAULT_MIN_FRAC_STRONG = 0.3
DEFAULT_DOPPLER_EXCURSION_HZ = 200.0
DEFAULT_MIN_FRAC_FOR_DOPPLER_GUESS = 0.03
DEFAULT_DATA_DIR = "data"
DEFAULT_FILTERED_DIR = "filtered_data"
USER_AGENT = "move2-fetch-and-vet/1.0 (research; respectful rate limits)"

REPORT_FIELDS = [
    "obs_id",
    "norad_cat_id",
    "station_name",
    "max_altitude",
    "tle_epoch_delta_days",
    "duration_s",
    "rf_gain",
    "if_gain",
    "bb_gain",
    "fixed_gain",
    "snr_db",
    "frac_time_strong",
    "freq_excursion_hz",
    "doppler_corrected_guess",
    "insufficient_signal",
    "passed_tracked_norad_filter",
    "passed_metadata_filter",
    "passed_vetting",
    "usable_for_doppler",
    "usable_for_amplitude",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "fetch_and_vet.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# API fetch + page cache
# ---------------------------------------------------------------------------


def _cache_state_path(cache_dir: Path) -> Path:
    return cache_dir / "fetch_state.json"


def _page_path(cache_dir: Path, page: int) -> Path:
    return cache_dir / f"page_{page:04d}.json"


def _load_fetch_state(cache_dir: Path) -> dict[str, Any]:
    path = _cache_state_path(cache_dir)
    if not path.is_file():
        return {"complete": False, "next_url": None, "page": 0, "count": 0}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_fetch_state(cache_dir: Path, state: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_state_path(cache_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
        fh.write("\n")


def _load_cached_pages(cache_dir: Path, up_to_page: int) -> list[dict]:
    observations: list[dict] = []
    for page in range(1, up_to_page + 1):
        path = _page_path(cache_dir, page)
        if not path.is_file():
            raise RuntimeError(
                f"Cache incomplete: missing {path} (expected pages 1..{up_to_page})"
            )
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise RuntimeError(f"Cached page {path} is not a JSON list")
        observations.extend(data)
    return observations


def stream_observation_pages(
    session: RateLimitedSession,
    *,
    norad_id: int,
    status: str,
    cache_dir: Path,
    start_before: Optional[str] = None,
    start_after: Optional[str] = None,
) -> Generator[list[dict], None, None]:
    """Yield one page (list of obs dicts) at a time, caching each to disk as it arrives.

    Already-cached pages are replayed first so the caller processes them before
    hitting the network again.  On resume, fetching continues from the next_url
    stored in fetch_state.json.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    state = _load_fetch_state(cache_dir)
    cached_pages = int(state.get("page") or 0)

    # --- replay cached pages first ---
    for page_num in range(1, cached_pages + 1):
        path = _page_path(cache_dir, page_num)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        yield data

    if state.get("complete"):
        logging.info(
            "API cache complete (%d pages, %d obs) — no further network fetch needed",
            cached_pages,
            state.get("count", 0),
        )
        return

    # --- determine starting URL ---
    next_url: Optional[str] = state.get("next_url")
    page = cached_pages

    if next_url:
        url: Optional[str] = next_url
        logging.info(
            "Resuming API fetch from page %d (next_url from cache)",
            page + 1,
        )
    elif page == 0:
        # Use norad_cat_id (tracked TLE object), not satellite__norad_cat_id
        # (transmitter association). SatNOGS honors start/end with this filter;
        # start__gte/start__lte are ignored.
        params = [
            ("norad_cat_id", str(norad_id)),
            ("format", "json"),
        ]
        if status:
            params.append(("status", status))
        if start_after:
            params.append(("start", start_after[:10] if len(start_after) >= 10 else start_after))
        if start_before:
            params.append(("end", start_before[:10] if len(start_before) >= 10 else start_before))
        req = requests.Request("GET", f"{API_BASE}/observations/", params=params)
        url = session.session.prepare_request(req).url
        logging.info(
            "Starting API fetch for tracked norad_cat_id=%d "
            "(status=%s start=%s end=%s)",
            norad_id,
            status or "all",
            start_after or "any",
            start_before or "any",
        )
    else:
        # Cached pages present but no next_url and not marked complete — treat as done.
        logging.info(
            "Cache has %d pages with no next_url; marking complete",
            page,
        )
        state["complete"] = True
        _save_fetch_state(cache_dir, state)
        return

    total_count: int = state.get("count", 0)

    while url:
        page += 1
        logging.info("Fetching page %d ...", page)
        resp = session.get(url, timeout=90)
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected API response type: {type(data)}")
        next_url = parse_link_next(resp.headers.get("Link") or resp.headers.get("link"))
        resp.close()

        # Cache immediately so a crash/interrupt doesn't lose this page.
        with open(_page_path(cache_dir, page), "w", encoding="utf-8") as fh:
            json.dump(data, fh)
            fh.write("\n")

        total_count += len(data)
        state = {
            "complete": next_url is None,
            "next_url": next_url,
            "page": page,
            "count": total_count,
        }
        _save_fetch_state(cache_dir, state)
        logging.info(
            "Page %d: %d obs (running total %d)",
            page,
            len(data),
            total_count,
        )
        yield data
        url = next_url


def _parse_date_bound(value: Optional[str], *, end_of_day: bool = False) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if len(value) == 10:
        dt = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        if end_of_day:
            return dt + timedelta(days=1) - timedelta(seconds=1)
        return dt
    return parse_iso(value)


def _paginate_observations_in_range(
    session: RateLimitedSession,
    *,
    norad_id: int,
    status: str,
    range_start: datetime,
    range_end: datetime,
    cache_subdir: Path | None,
) -> list[dict]:
    """Fetch all API pages for observations with start in [range_start, range_end]."""
    if cache_subdir is not None:
        done_path = cache_subdir / "window_complete.json"
        if done_path.is_file():
            with open(done_path, encoding="utf-8") as fh:
                return json.load(fh)

    # SatNOGS honors start/end (YYYY-MM-DD) with norad_cat_id; start__gte/lte do not.
    # end is exclusive-ish upper bound; bump one day past range_end.date().
    end_day = (range_end + timedelta(days=1)).strftime("%Y-%m-%d")
    params: list[tuple[str, str]] = [
        ("norad_cat_id", str(norad_id)),
        ("format", "json"),
        ("start", range_start.strftime("%Y-%m-%d")),
        ("end", end_day),
    ]
    if status:
        params.append(("status", status))

    req = requests.Request("GET", f"{API_BASE}/observations/", params=params)
    url: Optional[str] = session.session.prepare_request(req).url
    page = 0
    all_obs: list[dict] = []

    while url:
        page += 1
        logging.info(
            "Fetching window %s .. %s page %d ...",
            range_start.date(),
            range_end.date(),
            page,
        )
        resp = session.get(url, timeout=90)
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected API response type: {type(data)}")
        next_url = parse_link_next(resp.headers.get("Link") or resp.headers.get("link"))
        resp.close()
        all_obs.extend(data)
        url = next_url

    if cache_subdir is not None:
        cache_subdir.mkdir(parents=True, exist_ok=True)
        with open(cache_subdir / "window_complete.json", "w", encoding="utf-8") as fh:
            json.dump(all_obs, fh)
            fh.write("\n")

    return all_obs


def stream_observations_chronological(
    session: RateLimitedSession,
    *,
    norad_id: int,
    status: str,
    cache_dir: Path,
    start_after: Optional[str] = None,
    start_before: Optional[str] = None,
    window_days: int = 30,
) -> Generator[list[dict], None, None]:
    """Yield observation pages oldest-first via forward date windows.

    Each API page is yielded immediately (not buffered for the whole window)
    so downloads can start as soon as the first page arrives. Within each
    window, pages arrive newest-first from SatNOGS; we reverse each page so
    processing prefers older IDs first.
    """
    t0 = _parse_date_bound(start_after) or datetime(2018, 11, 1, tzinfo=timezone.utc)
    t1 = _parse_date_bound(start_before, end_of_day=True) or datetime(
        2025, 7, 19, tzinfo=timezone.utc
    )
    logging.info(
        "Chronological fetch (tracked norad_cat_id): %s → %s in %d-day windows",
        t0.date(),
        t1.date(),
        window_days,
    )

    cur = t0
    while cur <= t1:
        win_end = min(cur + timedelta(days=window_days) - timedelta(seconds=1), t1)
        label = f"{cur.date()}_{win_end.date()}"
        win_cache = cache_dir / "windows" / label
        done_path = win_cache / "window_complete.json"

        if done_path.is_file():
            window_obs = json.loads(done_path.read_text(encoding="utf-8"))
            window_obs.sort(key=lambda o: o.get("start") or "")
            logging.info("Window %s: %d obs (from cache)", label, len(window_obs))
            if window_obs:
                yield window_obs
            cur = win_end + timedelta(seconds=1)
            continue

        # Stream page-by-page; also accumulate for window_complete.json cache.
        end_day = (win_end + timedelta(days=1)).strftime("%Y-%m-%d")
        params: list[tuple[str, str]] = [
            ("norad_cat_id", str(norad_id)),
            ("format", "json"),
            ("start", cur.strftime("%Y-%m-%d")),
            ("end", end_day),
        ]
        if status:
            params.append(("status", status))
        req = requests.Request("GET", f"{API_BASE}/observations/", params=params)
        url: Optional[str] = session.session.prepare_request(req).url
        page = 0
        all_obs: list[dict] = []

        while url:
            page += 1
            logging.info(
                "Fetching window %s .. %s page %d ...",
                cur.date(),
                win_end.date(),
                page,
            )
            resp = session.get(url, timeout=90)
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected API response type: {type(data)}")
            next_url = parse_link_next(
                resp.headers.get("Link") or resp.headers.get("link")
            )
            resp.close()
            all_obs.extend(data)
            # Reverse page so older starts within the page come first.
            page_sorted = sorted(data, key=lambda o: o.get("start") or "")
            if page_sorted:
                yield page_sorted
            url = next_url

        win_cache.mkdir(parents=True, exist_ok=True)
        with open(done_path, "w", encoding="utf-8") as fh:
            json.dump(all_obs, fh)
            fh.write("\n")
        logging.info("Window %s complete: %d observations cached", label, len(all_obs))
        cur = win_end + timedelta(seconds=1)


def find_existing_obs_folder(obs_id: int, search_roots: list[Path]) -> Path | None:
    """Locate move2_{id}_* under any search root (excluding duplicates)."""
    for root in search_roots:
        if not root.is_dir():
            continue
        matches = sorted(root.glob(f"move2_{obs_id}_*"))
        for candidate in matches:
            if candidate.is_dir():
                return candidate
    return None


def observation_is_complete(folder: Path, obs_id: int) -> bool:
    raw = folder / "raw"
    meta = folder / "meta" / f"observation_{obs_id}.json"
    if not meta.is_file():
        return False
    ogg_ok = any(
        p.is_file() and p.stat().st_size > 0 for p in raw.glob(f"satnogs_{obs_id}_*.ogg")
    )
    wf_ok = any(p.is_file() and p.stat().st_size > 0 for p in raw.glob("waterfall_*.png"))
    return ogg_ok and wf_ok


def copy_observation_tree(src: Path, dest: Path) -> None:
    """Copy raw/, meta/, derived/ from an existing move2 tree."""
    dest.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "meta", "derived"):
        s = src / sub
        if not s.is_dir():
            continue
        d = dest / sub
        d.mkdir(parents=True, exist_ok=True)
        for path in s.iterdir():
            if not path.is_file():
                continue
            out = d / path.name
            if out.is_file() and out.stat().st_size > 0:
                continue
            shutil.copy2(path, out)


def reuse_existing_observation(
    obs: dict,
    data_dir: Path,
    search_roots: list[Path],
    *,
    require_tracked_norad_id: Optional[int],
) -> bool:
    """If a complete local tree exists, copy into data_dir. Return True if reused."""
    obs_id = obs["id"]
    dest = data_dir / observation_folder_name(obs)
    if observation_is_complete(dest, obs_id):
        return True

    src = find_existing_obs_folder(obs_id, search_roots)
    if src is None or src.resolve() == dest.resolve():
        return False
    if not observation_is_complete(src, obs_id):
        return False

    if require_tracked_norad_id is not None:
        meta_path = src / "meta" / f"observation_{obs_id}.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("norad_cat_id") != require_tracked_norad_id:
                return False

    logging.info("Reusing existing tree %s -> %s", src, dest)
    copy_observation_tree(src, dest)
    write_meta(obs, dest / "meta")
    return observation_is_complete(dest, obs_id)


# ---------------------------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------------------------


def tle_epoch_delta_days(obs: dict) -> Optional[float]:
    epoch = tle_epoch_from_line1(obs.get("tle1") or "")
    if epoch is None or not obs.get("start"):
        return None
    start = parse_iso(obs["start"])
    return abs((start - epoch).total_seconds()) / 86400.0


def duration_from_timestamps(obs: dict) -> Optional[float]:
    if not obs.get("start") or not obs.get("end"):
        return None
    return (parse_iso(obs["end"]) - parse_iso(obs["start"])).total_seconds()


def passes_tracked_norad_filter(obs: dict, *, require_tracked_norad_id: Optional[int]) -> bool:
    """Require observation.norad_cat_id to match the target satellite (not just transmitter)."""
    if require_tracked_norad_id is None:
        return True
    try:
        tracked = int(obs.get("norad_cat_id"))
    except (TypeError, ValueError):
        return False
    return tracked == require_tracked_norad_id


def passes_metadata_filter(
    obs: dict,
    *,
    min_altitude: float,
    max_tle_age_days: float,
    min_coord_decimals: int,
    require_tracked_norad_id: Optional[int] = None,
) -> bool:
    if not passes_tracked_norad_filter(
        obs, require_tracked_norad_id=require_tracked_norad_id
    ):
        return False
    if obs.get("vetted_status") != "good":
        return False
    if obs.get("waterfall_status") != "with-signal":
        return False
    if obs.get("transmitter_mode") != "CW":
        return False

    elev = obs.get("max_altitude")
    if elev is None or float(elev) < min_altitude:
        return False

    lat, lng = obs.get("station_lat"), obs.get("station_lng")
    if lat is None or lng is None or lat == "" or lng == "":
        return False
    if lat_lng_precision_score(lat, lng) <= min_coord_decimals:
        return False

    delta = tle_epoch_delta_days(obs)
    if delta is None or delta > max_tle_age_days:
        return False

    if not obs.get("waterfall"):
        return False
    if not (obs.get("payload") or obs.get("archive_url")):
        return False

    return True


# ---------------------------------------------------------------------------
# Gain / client metadata
# ---------------------------------------------------------------------------


def _parse_gain(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_radio_metadata(obs: dict) -> dict[str, Any]:
    radio: dict[str, Any] = {}
    raw = obs.get("client_metadata") or "{}"
    try:
        meta = json.loads(raw) if isinstance(raw, str) else (raw or {})
        radio = meta.get("radio") or {}
    except (json.JSONDecodeError, TypeError, AttributeError):
        radio = {}

    rf = _parse_gain(radio.get("rf_gain"))
    if_g = _parse_gain(radio.get("if_gain"))
    bb = _parse_gain(radio.get("bb_gain"))
    fixed = rf is not None or if_g is not None or bb is not None
    return {
        "rf_gain": rf,
        "if_gain": if_g,
        "bb_gain": bb,
        "radio_name": radio.get("name"),
        "fixed_gain": fixed,
    }


# ---------------------------------------------------------------------------
# Download into observation folder
# ---------------------------------------------------------------------------


def ensure_observation_files(
    session: RateLimitedSession,
    obs: dict,
    data_dir: Path,
    *,
    search_roots: list[Path],
) -> dict[str, bool]:
    """Download meta/audio/waterfall into data_dir tree. Skip existing files."""
    folder = data_dir / observation_folder_name(obs)
    raw_dir = folder / "raw"
    meta_dir = folder / "meta"
    derived_dir = folder / "derived"
    folder.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    meta_dir.mkdir(exist_ok=True)
    derived_dir.mkdir(exist_ok=True)

    obs_id = obs["id"]
    write_meta(obs, meta_dir)

    result = {"meta": True, "ogg": False, "waterfall": False}

    start_tag = format_start_for_filename(obs["start"])
    ogg_dest = raw_dir / f"satnogs_{obs_id}_{start_tag}.ogg"
    if ogg_dest.is_file() and ogg_dest.stat().st_size > 0:
        result["ogg"] = True
    else:
        local = find_local_ogg(obs, search_roots)
        audio_url = obs.get("payload") or obs.get("archive_url")
        if local is not None:
            logging.info("Reusing local audio %s -> %s", local, ogg_dest)
            shutil.copy2(local, ogg_dest)
            result["ogg"] = ogg_dest.is_file() and ogg_dest.stat().st_size > 0
        elif audio_url:
            logging.info("Downloading audio for %s", obs_id)
            download_file(session, audio_url, ogg_dest)
            result["ogg"] = ogg_dest.is_file() and ogg_dest.stat().st_size > 0
        else:
            logging.warning("No audio URL for observation %s", obs_id)

    wf_url = obs.get("waterfall")
    wf_dest = raw_dir / f"waterfall_{obs_id}.png"
    if wf_dest.is_file() and wf_dest.stat().st_size > 0:
        result["waterfall"] = True
    elif wf_url:
        logging.info("Downloading waterfall for %s", obs_id)
        try:
            download_file(session, wf_url, wf_dest)
            result["waterfall"] = wf_dest.is_file() and wf_dest.stat().st_size > 0
        except Exception as exc:  # noqa: BLE001
            logging.warning("Waterfall download failed for %s: %s", obs_id, exc)
    else:
        logging.warning("No waterfall URL for observation %s", obs_id)

    return result


# ---------------------------------------------------------------------------
# Spectrogram vetting
# ---------------------------------------------------------------------------


def analyze_audio(
    ogg_path: Path,
    *,
    doppler_excursion_hz: float,
    min_frac_for_doppler_guess: float,
) -> dict[str, Any]:
    data, sr = sf.read(str(ogg_path))
    if data.ndim > 1:
        data = data[:, 0]

    nperseg = 8192
    noverlap = int(nperseg * 0.75)
    f, _t, Sxx = spectrogram(
        data, fs=sr, nperseg=nperseg, noverlap=noverlap, window="hann"
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    band_mask = f < 3000
    band = Sxx_db[band_mask]
    f_band = f[band_mask]

    peak_freq = f_band[band.argmax(axis=0)]
    peak_power = band.max(axis=0)

    noise_floor = float(np.median(peak_power))
    strong = peak_power > noise_floor + 10
    if strong.any():
        freq_excursion_hz = float(peak_freq[strong].max() - peak_freq[strong].min())
    else:
        freq_excursion_hz = 0.0

    snr_db = float(peak_power.max() - noise_floor)
    frac_time_strong = float(strong.mean())
    duration_s = float(len(data) / sr)

    # If so few frames are above the noise floor that the excursion measurement
    # is meaningless, flag as insufficient signal and leave doppler_corrected_guess
    # as None so downstream code can distinguish three cases:
    #   None  — no real signal detected (insufficient_signal=True)
    #   True  — real signal, frequency excursion is small (likely Doppler-corrected)
    #   False — real signal, frequency excursion is large (likely raw / uncorrected)
    insufficient_signal = frac_time_strong < min_frac_for_doppler_guess
    if insufficient_signal:
        doppler_corrected_guess: Optional[bool] = None
    else:
        doppler_corrected_guess = freq_excursion_hz < doppler_excursion_hz

    return {
        "duration_s": round(duration_s, 3),
        "snr_db": round(snr_db, 2),
        "frac_time_strong": round(frac_time_strong, 3),
        "freq_excursion_hz": round(freq_excursion_hz, 2),
        "doppler_corrected_guess": doppler_corrected_guess,
        "insufficient_signal": insufficient_signal,
    }


def compute_usability(
    *,
    snr_db: float,
    frac_time_strong: float,
    doppler_corrected_guess: Optional[bool],
    fixed_gain: bool,
    insufficient_signal: bool,
    min_snr: float,
    min_frac_strong: float,
) -> dict[str, bool]:
    # Observations with insufficient signal are unusable for any purpose.
    if insufficient_signal:
        return {
            "passed_vetting": False,
            "usable_for_doppler": False,
            "usable_for_amplitude": False,
        }
    passed_quality = snr_db >= min_snr and frac_time_strong >= min_frac_strong
    # doppler_corrected_guess is None only when insufficient_signal is True (handled above),
    # so it is always a bool here.
    usable_for_doppler = passed_quality and not doppler_corrected_guess
    usable_for_amplitude = passed_quality and fixed_gain
    passed_vetting = usable_for_doppler or usable_for_amplitude
    return {
        "passed_vetting": passed_vetting,
        "usable_for_doppler": usable_for_doppler,
        "usable_for_amplitude": usable_for_amplitude,
    }


# ---------------------------------------------------------------------------
# filtered_data copy
# ---------------------------------------------------------------------------


def copy_to_filtered(src_folder: Path, filtered_dir: Path, folder_name: str) -> None:
    """Copy observation tree into filtered_data/, skipping existing files."""
    dest = filtered_dir / folder_name
    if not src_folder.is_dir():
        return
    for src_path in src_folder.rglob("*"):
        if not src_path.is_file():
            continue
        rel = src_path.relative_to(src_folder)
        dest_path = dest / rel
        if dest_path.is_file() and dest_path.stat().st_size > 0:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
    # Ensure derived/ exists even if empty
    (dest / "derived").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def empty_report_row(
    obs: dict,
    *,
    passed_metadata: bool,
    passed_tracked_norad: bool = True,
) -> dict[str, Any]:
    radio = parse_radio_metadata(obs)
    delta = tle_epoch_delta_days(obs)
    return {
        "obs_id": obs["id"],
        "norad_cat_id": obs.get("norad_cat_id", ""),
        "station_name": obs.get("station_name"),
        "max_altitude": obs.get("max_altitude"),
        "tle_epoch_delta_days": round(delta, 4) if delta is not None else "",
        "duration_s": duration_from_timestamps(obs) or "",
        "rf_gain": radio["rf_gain"] if radio["rf_gain"] is not None else "",
        "if_gain": radio["if_gain"] if radio["if_gain"] is not None else "",
        "bb_gain": radio["bb_gain"] if radio["bb_gain"] is not None else "",
        "fixed_gain": radio["fixed_gain"],
        "snr_db": "",
        "frac_time_strong": "",
        "freq_excursion_hz": "",
        "doppler_corrected_guess": "",
        "insufficient_signal": "",
        "passed_tracked_norad_filter": passed_tracked_norad,
        "passed_metadata_filter": passed_metadata,
        "passed_vetting": False,
        "usable_for_doppler": False,
        "usable_for_amplitude": False,
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REPORT_FIELDS})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fetch all SatNOGS observations for a NORAD ID, metadata-filter, "
            "download, spectrogram-vet, and write dual data trees + CSV report."
        )
    )
    p.add_argument(
        "--norad-id",
        type=int,
        default=DEFAULT_NORAD_ID,
        help=(
            "SatNOGS API filter norad_cat_id (tracked TLE object during the pass). "
            "Default: 43780 (MOVE-II)."
        ),
    )
    p.add_argument(
        "--require-tracked-norad-id",
        type=int,
        default=None,
        metavar="NORAD",
        help=(
            "After API fetch, only download/vet observations whose norad_cat_id field "
            "(TLE object tracked during the pass) equals this value. "
            "Use 43780 for genuine MOVE-II passes. Omit to keep legacy behaviour."
        ),
    )
    p.add_argument(
        "--api-delay",
        type=float,
        default=DEFAULT_API_DELAY,
        help="Minimum seconds between API requests (default: 1.0)",
    )
    p.add_argument("--min-altitude", type=float, default=DEFAULT_MIN_ALTITUDE)
    p.add_argument(
        "--max-tle-age-days",
        type=float,
        default=DEFAULT_MAX_TLE_AGE_DAYS,
        help="Max |obs_start - TLE epoch| in days (default: 3)",
    )
    p.add_argument(
        "--min-coord-decimals",
        type=int,
        default=DEFAULT_MIN_COORD_DECIMALS,
        help="Require lat/lng precision strictly greater than this (default: 2)",
    )
    p.add_argument("--min-snr", type=float, default=DEFAULT_MIN_SNR)
    p.add_argument(
        "--min-frac-strong",
        type=float,
        default=DEFAULT_MIN_FRAC_STRONG,
    )
    p.add_argument(
        "--doppler-excursion-hz",
        type=float,
        default=DEFAULT_DOPPLER_EXCURSION_HZ,
        help="freq_excursion_hz below this => doppler_corrected_guess (default: 200)",
    )
    p.add_argument(
        "--min-frac-for-doppler-guess",
        type=float,
        default=DEFAULT_MIN_FRAC_FOR_DOPPLER_GUESS,
        help=(
            "Minimum frac_time_strong to trust the doppler_corrected_guess flag. "
            "Below this, doppler_corrected_guess=None and insufficient_signal=True. "
            "(default: 0.03)"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "After metadata filter, process only the first N survivors "
            "(download + vet). All pages are still fetched and cached. "
            "Useful for smoke-test runs."
        ),
    )
    p.add_argument(
        "--status",
        default="good",
        help=(
            "SatNOGS vetted_status filter passed to the API (default: good). "
            "Set to empty string '' to fetch all statuses (much slower)."
        ),
    )
    p.add_argument(
        "--start-before",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Only fetch observations that started before this date "
            "(passed as SatNOGS end= exclusive upper bound). "
            "E.g. '2021-01-01' covers through 2020-12-31."
        ),
    )
    p.add_argument(
        "--start-after",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Only fetch observations that started on/after this date "
            "(passed as SatNOGS start=). E.g. '2018-11-01' for MOVE-II launch."
        ),
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--filtered-dir", default=DEFAULT_FILTERED_DIR)
    p.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "API page cache directory "
            "(default: <data-dir>/api_cache/norad_<id>_status_<status>)"
        ),
    )
    p.add_argument(
        "--from-local",
        action="store_true",
        help=(
            "Skip the API fetch entirely. Load observation metadata from "
            "already-downloaded data/move2_*/meta/observation_*.json files "
            "and vet those. Requires no network access."
        ),
    )
    p.add_argument(
        "--order",
        choices=("newest", "oldest"),
        default="newest",
        help=(
            "API iteration order. 'oldest' walks forward in date windows from "
            "--start-after (recommended for early-epoch tumble work). "
            "'newest' uses global cursor pagination (default legacy behaviour)."
        ),
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Date window size for --order oldest (days per API query).",
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip download/vetting when obs already complete in --data-dir.",
    )
    p.add_argument(
        "--reuse-dirs",
        default="",
        help=(
            "Comma-separated directories to search for existing move2_* trees "
            "before hitting the network (e.g. 'data' to reuse 2018 downloads)."
        ),
    )
    return p


def load_local_observations(data_dir: Path) -> list[dict]:
    """Load all observation JSON files already present under data/move2_*/meta/."""
    obs_list: list[dict] = []
    for path in sorted(data_dir.glob("move2_*/meta/observation_*.json")):
        with open(path, encoding="utf-8") as fh:
            obs_list.append(json.load(fh))
    obs_list.sort(key=lambda o: o.get("start") or "")
    return obs_list


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).resolve()
    filtered_dir = Path(args.filtered_dir).resolve()
    status_slug = args.status.replace(" ", "_") if args.status else "all"
    date_slug = ""
    if args.start_after:
        date_slug += f"_from_{args.start_after.replace('-', '')}"
    if args.start_before:
        date_slug += f"_to_{args.start_before.replace('-', '')}"
    cache_dir = (
        Path(args.cache_dir).resolve()
        if args.cache_dir
        else data_dir / "api_cache" / f"norad_{args.norad_id}_status_{status_slug}{date_slug}"
    )

    setup_logging(data_dir)

    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    reuse_dirs: list[Path] = []
    for part in (args.reuse_dirs or "").split(","):
        part = part.strip()
        if not part:
            continue
        path = Path(part)
        if not path.is_absolute():
            path = workspace_root / path
        reuse_dirs.append(path.resolve())
    search_roots = [workspace_root, data_dir, filtered_dir, *reuse_dirs]

    logging.info(
        "fetch_and_vet: api_satellite__norad=%d require_tracked_norad=%s status=%s "
        "order=%s window_days=%d skip_existing=%s reuse_dirs=%s "
        "api_delay=%.1fs min_alt=%.1f max_tle_days=%.1f min_snr=%.1f min_frac=%.2f "
        "doppler_hz=%.0f min_frac_doppler=%.3f limit=%s",
        args.norad_id,
        args.require_tracked_norad_id if args.require_tracked_norad_id is not None else "any",
        args.status or "all",
        args.order,
        args.window_days,
        args.skip_existing,
        ",".join(str(d) for d in reuse_dirs) or "(none)",
        args.api_delay,
        args.min_altitude,
        args.max_tle_age_days,
        args.min_snr,
        args.min_frac_strong,
        args.doppler_excursion_hz,
        args.min_frac_for_doppler_guess,
        args.limit if args.limit is not None else "all",
    )

    session = RateLimitedSession(min_interval=args.api_delay)
    session.session.headers.update({"User-Agent": USER_AGENT})

    # --- source of observation pages ---
    if args.from_local:
        logging.info(
            "--from-local: loading from %s/move2_*/meta/ (no network)", data_dir
        )
        local_obs = load_local_observations(data_dir)
        if not local_obs:
            logging.error("No local observations found under %s/move2_*/meta/", data_dir)
            return 1
        logging.info("Loaded %d local observations", len(local_obs))
        if args.order == "oldest":
            local_obs.sort(key=lambda o: o.get("start") or "")
        page_source = iter([local_obs])
    elif args.order == "oldest":
        page_source = stream_observations_chronological(
            session,
            norad_id=args.norad_id,
            status=args.status,
            cache_dir=cache_dir,
            start_after=args.start_after,
            start_before=args.start_before,
            window_days=args.window_days,
        )
    else:
        page_source = stream_observation_pages(
            session,
            norad_id=args.norad_id,
            status=args.status,
            cache_dir=cache_dir,
            start_before=args.start_before,
            start_after=args.start_after,
        )

    # --- streaming process: fetch page → filter → download+vet → filtered_data ---
    n_fetched = 0
    n_wrong_tracked_norad = 0
    n_skipped_existing = 0
    n_reused_local = 0
    n_meta = 0
    n_downloaded = 0
    n_vetted_pass = 0
    n_in_filtered = 0
    all_rows: list[dict[str, Any]] = []
    limit_reached = False

    for page_data in page_source:
        if limit_reached:
            break
        for obs in page_data:
            n_fetched += 1
            obs_id = obs["id"]

            tracked_ok = passes_tracked_norad_filter(
                obs, require_tracked_norad_id=args.require_tracked_norad_id
            )
            if not tracked_ok:
                n_wrong_tracked_norad += 1
                all_rows.append(
                    empty_report_row(
                        obs, passed_metadata=False, passed_tracked_norad=False
                    )
                )
                continue

            # --- metadata filter ---
            if not passes_metadata_filter(
                obs,
                min_altitude=args.min_altitude,
                max_tle_age_days=args.max_tle_age_days,
                min_coord_decimals=args.min_coord_decimals,
                require_tracked_norad_id=args.require_tracked_norad_id,
            ):
                all_rows.append(
                    empty_report_row(obs, passed_metadata=False, passed_tracked_norad=True)
                )
                continue

            n_meta += 1
            folder_name = observation_folder_name(obs)
            folder = data_dir / folder_name

            if args.skip_existing and observation_is_complete(folder, obs_id):
                n_skipped_existing += 1
                logging.info("Skip obs %s — already complete in %s", obs_id, folder)
                continue

            reuse_search = [
                root for root in search_roots if root.resolve() != data_dir.resolve()
            ]
            if reuse_existing_observation(
                obs,
                data_dir,
                reuse_search,
                require_tracked_norad_id=args.require_tracked_norad_id,
            ):
                n_reused_local += 1
                logging.info("Reused local tree for obs %s (no network download)", obs_id)

            row = empty_report_row(obs, passed_metadata=True, passed_tracked_norad=True)
            radio = parse_radio_metadata(obs)
            row["rf_gain"] = radio["rf_gain"] if radio["rf_gain"] is not None else ""
            row["if_gain"] = radio["if_gain"] if radio["if_gain"] is not None else ""
            row["bb_gain"] = radio["bb_gain"] if radio["bb_gain"] is not None else ""
            row["fixed_gain"] = radio["fixed_gain"]

            # --- download into data/ ---
            try:
                files = ensure_observation_files(
                    session, obs, data_dir, search_roots=search_roots
                )
                if files["ogg"]:
                    n_downloaded += 1

                start_tag = format_start_for_filename(obs["start"])
                ogg_path = folder / "raw" / f"satnogs_{obs_id}_{start_tag}.ogg"
                if not (ogg_path.is_file() and ogg_path.stat().st_size > 0):
                    logging.warning("obs %s: missing audio after download", obs_id)
                    all_rows.append(row)
                    continue

                # --- spectrogram vetting ---
                metrics = analyze_audio(
                    ogg_path,
                    doppler_excursion_hz=args.doppler_excursion_hz,
                    min_frac_for_doppler_guess=args.min_frac_for_doppler_guess,
                )
                row.update(
                    {
                        "duration_s": metrics["duration_s"],
                        "snr_db": metrics["snr_db"],
                        "frac_time_strong": metrics["frac_time_strong"],
                        "freq_excursion_hz": metrics["freq_excursion_hz"],
                        "doppler_corrected_guess": metrics["doppler_corrected_guess"],
                        "insufficient_signal": metrics["insufficient_signal"],
                    }
                )
                flags = compute_usability(
                    snr_db=metrics["snr_db"],
                    frac_time_strong=metrics["frac_time_strong"],
                    doppler_corrected_guess=metrics["doppler_corrected_guess"],
                    fixed_gain=bool(radio["fixed_gain"]),
                    insufficient_signal=metrics["insufficient_signal"],
                    min_snr=args.min_snr,
                    min_frac_strong=args.min_frac_strong,
                )
                row.update(flags)

                if metrics["insufficient_signal"]:
                    verdict = "NO-SIGNAL"
                elif flags["passed_vetting"]:
                    verdict = "PASS"
                else:
                    verdict = "FAIL"

                excursion_str = (
                    "N/A"
                    if metrics["insufficient_signal"]
                    else f"{metrics['freq_excursion_hz']:.0f}Hz"
                )
                logging.info(
                    "[meta#%d] obs %s: SNR=%.1fdB, excursion=%s, %s",
                    n_meta,
                    obs_id,
                    metrics["snr_db"],
                    excursion_str,
                    verdict,
                )

                # --- copy to filtered_data/ if it passes ---
                if flags["passed_vetting"]:
                    n_vetted_pass += 1
                    copy_to_filtered(folder, filtered_dir, folder_name)
                    n_in_filtered += 1
                    logging.info(
                        "  → copied to filtered_data/ (doppler=%s amplitude=%s)",
                        flags["usable_for_doppler"],
                        flags["usable_for_amplitude"],
                    )

            except Exception as exc:  # noqa: BLE001
                logging.exception("obs %s: vetting failed: %s", obs_id, exc)

            all_rows.append(row)

            # --- --limit caps metadata-passing downloads, not total fetched ---
            if args.limit is not None and n_meta >= args.limit:
                logging.info(
                    "--limit %d reached after %d fetched; stopping.",
                    args.limit,
                    n_fetched,
                )
                limit_reached = True
                break

    # --- write CSV (stable order by obs id) ---
    ordered = sorted(all_rows, key=lambda r: int(r["obs_id"]))
    report_path = data_dir / "vetting_report_full.csv"
    write_report(report_path, ordered)

    n_insufficient = sum(1 for r in all_rows if r.get("insufficient_signal") is True)
    n_usable_doppler = sum(1 for r in all_rows if r.get("usable_for_doppler") is True)
    n_usable_amplitude = sum(1 for r in all_rows if r.get("usable_for_amplitude") is True)

    logging.info(
        "Summary: fetched=%d → wrong_tracked_norad=%d → skipped_existing=%d → "
        "reused_local=%d → passed_metadata=%d → downloaded=%d → "
        "passed_vetting=%d → in_filtered_data=%d",
        n_fetched,
        n_wrong_tracked_norad,
        n_skipped_existing,
        n_reused_local,
        n_meta,
        n_downloaded,
        n_vetted_pass,
        n_in_filtered,
    )
    logging.info(
        "Breakdown: usable_for_doppler=%d  usable_for_amplitude=%d  "
        "insufficient_signal=%d",
        n_usable_doppler,
        n_usable_amplitude,
        n_insufficient,
    )
    logging.info("Wrote report (%d rows) -> %s", len(ordered), report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
