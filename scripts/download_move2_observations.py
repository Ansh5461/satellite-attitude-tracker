#!/usr/bin/env python3
"""
Download "good"-vetted SatNOGS observations for MOVE-II into a structured layout:

  data/move2_{id}_{YYYY-MM-DD}/
    raw/satnogs_{id}_{start}.ogg
    raw/waterfall_{id}.png
    meta/observation_{id}.json
    meta/tle_{id}.txt
    derived/                    # left empty for the analysis pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SAT_ID = "JWFD-9682-9871-6006-7786"
DEFAULT_STATUS = "good"
DEFAULT_LIMIT = 100
DEFAULT_ORDER = "oldest"
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_WORKERS = 3
API_BASE = "https://network.satnogs.org/api"
LAUNCH_DATE = datetime(2018, 12, 3, tzinfo=timezone.utc)
DECAY_DATE = datetime(2025, 7, 18, tzinfo=timezone.utc)
USER_AGENT = "move2-downloader/1.0 (research; respectful rate limits)"
API_MIN_INTERVAL_S = 3.0
FILE_MIN_INTERVAL_S = 0.5
INDEX_FIELDS = [
    "id",
    "start",
    "end",
    "ground_station",
    "station_name",
    "station_lat",
    "station_lng",
    "station_alt",
    "lat_lng_precision",
    "max_altitude",
    "tle_age_hours",
    "transmitter_mode",
    "transmitter_uuid",
    "observation_frequency",
    "status",
    "ogg_saved",
    "waterfall_saved",
    "meta_saved",
    "error",
    "folder",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_iso(ts: str) -> datetime:
    """Parse SatNOGS ISO timestamps (with or without trailing Z)."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def format_start_for_filename(start: str) -> str:
    """2019-01-30T22:54:32Z -> 2019-01-30T22-54-32"""
    dt = parse_iso(start)
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


def observation_folder_name(obs: dict) -> str:
    start_date = parse_iso(obs["start"]).strftime("%Y-%m-%d")
    return f"move2_{obs['id']}_{start_date}"


def decimal_places(value: Any) -> int:
    if value is None:
        return 0
    s = str(value)
    if "." not in s:
        return 0
    return len(s.split(".", 1)[1].rstrip("0") or "0")


def lat_lng_precision_score(lat: Any, lng: Any) -> int:
    """Min decimal places across lat/lng — higher is better (~4+ = GPS-like)."""
    return min(decimal_places(lat), decimal_places(lng))


def tle_epoch_from_line1(tle1: str) -> Optional[datetime]:
    """
    Parse epoch from TLE line 1 field 4: YYDDD.DDDDDDDD
    e.g. '19030.18320388' -> 2019-01-30 ...
    """
    if not tle1:
        return None
    parts = tle1.split()
    if len(parts) < 4:
        return None
    try:
        epoch_str = parts[3]
        yy = int(epoch_str[:2])
        year = 2000 + yy if yy < 57 else 1900 + yy
        day_of_year = float(epoch_str[2:])
        return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)
    except (ValueError, IndexError):
        return None


def tle_age_hours(obs: dict) -> Optional[float]:
    epoch = tle_epoch_from_line1(obs.get("tle1") or "")
    if epoch is None:
        return None
    start = parse_iso(obs["start"])
    return (start - epoch).total_seconds() / 3600.0


def parse_link_next(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    for part in link_header.split(","):
        m = re.search(r'<([^>]+)>;\s*rel="next"', part.strip())
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Rate-limited HTTP session
# ---------------------------------------------------------------------------


class RateLimitedSession:
    def __init__(self, min_interval: float = API_MIN_INTERVAL_S):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.min_interval = min_interval
        self._last_request = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def get(
        self,
        url: str,
        *,
        stream: bool = False,
        timeout: float = 60,
        max_retries: int = 12,
    ) -> requests.Response:
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            self._wait()
            try:
                self._last_request = time.monotonic()
                resp = self.session.get(url, stream=stream, timeout=timeout)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    # Honor Retry-After from SatNOGS. Cap at 1 hour so a
                    # pathological header cannot hang forever, but do NOT
                    # truncate a 40–50 minute ban down to 5 minutes — that
                    # just burns retries and crashes before the ban lifts.
                    MAX_RETRY_AFTER_S = 3600
                    header_wait = float(retry_after) if retry_after else 0.0
                    floor = min(5 * (2**attempt), 300)
                    wait = min(max(header_wait, floor), MAX_RETRY_AFTER_S)
                    eta = datetime.now(timezone.utc) + timedelta(seconds=wait)
                    logging.warning(
                        "429 Too Many Requests for %s — sleeping %.0fs until ~%s UTC "
                        "(attempt %d/%d, Retry-After=%s)",
                        urlparse(url).path,
                        wait,
                        eta.strftime("%H:%M:%S"),
                        attempt + 1,
                        max_retries,
                        retry_after,
                    )
                    resp.close()
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    wait = min(2**attempt, 120)
                    logging.warning(
                        "HTTP %d for %s — retrying in %.0fs",
                        resp.status_code,
                        url,
                        wait,
                    )
                    resp.close()
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_err = exc
                wait = min(2**attempt, 120)
                logging.warning("Network error %s — retrying in %.0fs", exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"Failed GET {url} after {max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Observation discovery
# ---------------------------------------------------------------------------


def iter_observations(
    session: RateLimitedSession,
    *,
    sat_id: str,
    status: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    stop_after: Optional[int] = None,
) -> Iterator[dict]:
    """Yield observations from the cursor-paginated list endpoint.

    If stop_after is set, stop requesting further pages once that many
    observations have been yielded (still finishes the current page).
    """
    params: list[tuple[str, str]] = [
        ("sat_id", sat_id),
        ("status", status),
        ("format", "json"),
    ]
    if start is not None:
        params.append(("start", start.strftime("%Y-%m-%dT%H:%M:%SZ")))
    if end is not None:
        params.append(("end", end.strftime("%Y-%m-%dT%H:%M:%SZ")))

    url = f"{API_BASE}/observations/"
    # Build initial URL with params
    req = requests.Request("GET", url, params=params)
    prepared = session.session.prepare_request(req)
    url = prepared.url

    page = 0
    yielded = 0
    while url:
        page += 1
        logging.info("Fetching observation page %d ...", page)
        resp = session.get(url, timeout=90)
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected API response type: {type(data)}")
        next_url = parse_link_next(resp.headers.get("Link") or resp.headers.get("link"))
        resp.close()
        for obs in data:
            yield obs
            yielded += 1
        if stop_after is not None and yielded >= stop_after:
            logging.info(
                "Reached stop_after=%d (yielded %d); not fetching further pages",
                stop_after,
                yielded,
            )
            break
        url = next_url


def collect_newest(
    session: RateLimitedSession,
    *,
    sat_id: str,
    status: str,
    limit: int,
    start: Optional[datetime],
    end: Optional[datetime],
) -> list[dict]:
    """Default API order is newest-first; collect up to `limit` (0 = all)."""
    stop_after = limit if limit else None
    out: list[dict] = []
    for obs in iter_observations(
        session,
        sat_id=sat_id,
        status=status,
        start=start,
        end=end,
        stop_after=stop_after,
    ):
        out.append(obs)
        if limit and len(out) >= limit:
            break
    return out


def collect_oldest(
    session: RateLimitedSession,
    *,
    sat_id: str,
    status: str,
    limit: int,
    start: Optional[datetime],
    end: Optional[datetime],
) -> list[dict]:
    """
    Adaptive expanding window from launch (or --start) until we have >= limit
    observations, then keep the oldest `limit`. With --limit 0, fetch the full
    window (start..end or launch..decay).

    Note: the SatNOGS list API returns newest-first even inside a date window,
    so each window must be fully paginated before we can pick the oldest rows.
    Windows therefore start small (7 days) to keep early runs cheap.
    """
    window_start = start or LAUNCH_DATE
    hard_end = end or DECAY_DATE

    if limit == 0:
        logging.info(
            "Collecting ALL good observations from %s to %s (unbounded)",
            window_start.date(),
            hard_end.date(),
        )
        all_obs = list(
            iter_observations(
                session,
                sat_id=sat_id,
                status=status,
                start=window_start,
                end=hard_end,
            )
        )
        all_obs.sort(key=lambda o: o["start"])
        return all_obs

    # Expand end in growing steps until we have enough (or hit hard_end).
    # Fetch only the newly added slice on each expansion to save API quota.
    step_days = 7
    cursor = window_start
    window_end = min(window_start + timedelta(days=step_days), hard_end)
    collected: dict[int, dict] = {}

    while True:
        logging.info(
            "Oldest-mode slice %s .. %s (have %d / need %d)",
            cursor.date(),
            window_end.date(),
            len(collected),
            limit,
        )
        for obs in iter_observations(
            session,
            sat_id=sat_id,
            status=status,
            start=cursor,
            end=window_end,
        ):
            collected[obs["id"]] = obs

        if len(collected) >= limit or window_end >= hard_end:
            break

        cursor = window_end
        step_days = min(step_days * 2, 180)
        window_end = min(window_end + timedelta(days=step_days), hard_end)

    ordered = sorted(collected.values(), key=lambda o: o["start"])
    return ordered[:limit]


def apply_filters(
    observations: list[dict],
    *,
    min_elevation: Optional[float],
    max_tle_age_hours: Optional[float],
) -> list[dict]:
    filtered = []
    for obs in observations:
        if min_elevation is not None:
            elev = obs.get("max_altitude")
            if elev is None or float(elev) < min_elevation:
                logging.debug(
                    "Skip %s: max_altitude=%s < %.1f",
                    obs["id"],
                    elev,
                    min_elevation,
                )
                continue
        if max_tle_age_hours is not None:
            age = tle_age_hours(obs)
            if age is None or age > max_tle_age_hours:
                logging.debug(
                    "Skip %s: tle_age_hours=%s > %.1f",
                    obs["id"],
                    age,
                    max_tle_age_hours,
                )
                continue
        filtered.append(obs)
    return filtered


# ---------------------------------------------------------------------------
# Download / layout
# ---------------------------------------------------------------------------


def find_local_ogg(obs: dict, search_roots: list[Path]) -> Optional[Path]:
    """Reuse a previously downloaded .ogg if present locally (e.g. workspace root)."""
    start_tag = format_start_for_filename(obs["start"])
    candidates = [
        f"satnogs_{obs['id']}_{start_tag}.ogg",
        f"satnogs_{obs['id']}.ogg",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for name in candidates:
            path = root / name
            if path.is_file() and path.stat().st_size > 0:
                return path
        # Also scan one level deep for any matching satnogs_{id}_*.ogg
        for path in root.glob(f"**/satnogs_{obs['id']}_*.ogg"):
            if path.is_file() and path.stat().st_size > 0:
                return path
    return None


def download_file(
    session: RateLimitedSession,
    url: str,
    dest: Path,
    *,
    min_interval: float = FILE_MIN_INTERVAL_S,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Unique partial name avoids collisions if a previous run left leftovers.
    partial = dest.with_name(f".{dest.name}.{os.getpid()}.partial")
    old = session.min_interval
    session.min_interval = max(old, min_interval)
    try:
        resp = session.get(url, stream=True, timeout=300)
        try:
            with open(partial, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            resp.close()
        if not partial.is_file():
            raise RuntimeError(f"Download produced no file for {url}")
        if partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Download was empty for {url}")
        # os.replace is more reliable than Path.replace on some network FS mounts
        os.replace(partial, dest)
    finally:
        session.min_interval = old
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass


def write_meta(obs: dict, meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    obs_id = obs["id"]
    obs_path = meta_dir / f"observation_{obs_id}.json"
    with open(obs_path, "w", encoding="utf-8") as fh:
        json.dump(obs, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    tle_path = meta_dir / f"tle_{obs_id}.txt"
    lines = [
        obs.get("tle0") or "",
        obs.get("tle1") or "",
        obs.get("tle2") or "",
    ]
    with open(tle_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")


def observation_complete(folder: Path, obs: dict, *, dry_run: bool) -> bool:
    """True if meta + expected raw files already exist (resumability)."""
    obs_id = obs["id"]
    start_tag = format_start_for_filename(obs["start"])
    meta_ok = (folder / "meta" / f"observation_{obs_id}.json").is_file() and (
        folder / "meta" / f"tle_{obs_id}.txt"
    ).is_file()
    if dry_run:
        return meta_ok
    ogg_ok = (folder / "raw" / f"satnogs_{obs_id}_{start_tag}.ogg").is_file()
    # waterfall may be absent from the API; only require it if URL exists
    wf_url = obs.get("waterfall")
    if wf_url:
        wf_ok = (folder / "raw" / f"waterfall_{obs_id}.png").is_file()
    else:
        wf_ok = True
    # Need at least one audio source available historically; if neither URL, skip audio
    has_audio_url = bool(obs.get("payload") or obs.get("archive_url"))
    if not has_audio_url:
        ogg_ok = True
    return meta_ok and ogg_ok and wf_ok


def process_observation(
    session: RateLimitedSession,
    obs: dict,
    output_dir: Path,
    *,
    dry_run: bool,
    search_roots: list[Path],
) -> dict[str, Any]:
    """Download one observation into the target folder layout. Returns index row."""
    obs_id = obs["id"]
    folder = output_dir / observation_folder_name(obs)
    raw_dir = folder / "raw"
    meta_dir = folder / "meta"
    derived_dir = folder / "derived"

    row: dict[str, Any] = {
        "id": obs_id,
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
        "tle_age_hours": (
            round(age, 3) if (age := tle_age_hours(obs)) is not None else ""
        ),
        "transmitter_mode": obs.get("transmitter_mode"),
        "transmitter_uuid": obs.get("transmitter_uuid"),
        "observation_frequency": obs.get("observation_frequency"),
        "status": obs.get("status"),
        "ogg_saved": False,
        "waterfall_saved": False,
        "meta_saved": False,
        "error": "",
        "folder": str(folder),
    }

    try:
        folder.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(exist_ok=True)
        meta_dir.mkdir(exist_ok=True)
        derived_dir.mkdir(exist_ok=True)

        if observation_complete(folder, obs, dry_run=dry_run):
            logging.info("Skip %s (already complete)", obs_id)
            row["meta_saved"] = True
            start_tag = format_start_for_filename(obs["start"])
            row["ogg_saved"] = (raw_dir / f"satnogs_{obs_id}_{start_tag}.ogg").is_file()
            row["waterfall_saved"] = (raw_dir / f"waterfall_{obs_id}.png").is_file()
            return row

        write_meta(obs, meta_dir)
        row["meta_saved"] = True

        if dry_run:
            logging.info("Dry-run: wrote meta for %s", obs_id)
            return row

        # --- audio ---
        start_tag = format_start_for_filename(obs["start"])
        ogg_dest = raw_dir / f"satnogs_{obs_id}_{start_tag}.ogg"
        if not ogg_dest.is_file():
            local = find_local_ogg(obs, search_roots)
            audio_url = obs.get("payload") or obs.get("archive_url")
            if local is not None:
                logging.info("Reusing local audio %s -> %s", local, ogg_dest)
                shutil.copy2(local, ogg_dest)
                row["ogg_saved"] = True
            elif audio_url:
                logging.info("Downloading audio for %s", obs_id)
                download_file(session, audio_url, ogg_dest)
                row["ogg_saved"] = ogg_dest.is_file() and ogg_dest.stat().st_size > 0
            else:
                logging.warning("No audio URL for observation %s", obs_id)
        else:
            row["ogg_saved"] = True

        # --- waterfall ---
        wf_url = obs.get("waterfall")
        wf_dest = raw_dir / f"waterfall_{obs_id}.png"
        if wf_url and not wf_dest.is_file():
            logging.info("Downloading waterfall for %s", obs_id)
            try:
                download_file(session, wf_url, wf_dest)
                row["waterfall_saved"] = wf_dest.is_file() and wf_dest.stat().st_size > 0
            except Exception as exc:  # noqa: BLE001
                logging.warning("Waterfall download failed for %s: %s", obs_id, exc)
                row["error"] = f"waterfall: {exc}"
        elif wf_dest.is_file():
            row["waterfall_saved"] = True
        elif not wf_url:
            logging.warning("No waterfall URL for observation %s", obs_id)

    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed observation %s: %s", obs_id, exc)
        row["error"] = str(exc)

    return row


# ---------------------------------------------------------------------------
# Index / logging setup
# ---------------------------------------------------------------------------


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "download.log"
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


def write_index(output_dir: Path, rows: list[dict]) -> None:
    path = output_dir / "index.csv"
    # Merge with existing index so re-runs don't wipe older rows from other batches
    existing: dict[str, dict] = {}
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                existing[str(r["id"])] = r
    for row in rows:
        existing[str(row["id"])] = {k: row.get(k, "") for k in INDEX_FIELDS}

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for obs_id in sorted(existing.keys(), key=lambda x: int(x)):
            writer.writerow(existing[obs_id])
    logging.info("Wrote index with %d rows -> %s", len(existing), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download good-vetted SatNOGS MOVE-II observations."
    )
    p.add_argument("--sat-id", default=DEFAULT_SAT_ID)
    p.add_argument("--status", default=DEFAULT_STATUS)
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max observations to keep (0 = all). Default: 100",
    )
    p.add_argument(
        "--order",
        choices=("oldest", "newest"),
        default=DEFAULT_ORDER,
        help="Which observations to keep toward --limit. Default: oldest",
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="ISO start bound (e.g. 2018-12-08). Default: launch date for oldest mode",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="ISO end bound. Default: decay date for oldest mode",
    )
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--min-elevation",
        type=float,
        default=None,
        help="Optional hard filter: require max_altitude >= this (degrees)",
    )
    p.add_argument(
        "--max-tle-age-hours",
        type=float,
        default=None,
        help="Optional hard filter: require TLE age at obs start <= this",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent file downloads (metadata writes are still serial). Default: 3",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write meta/ JSON + TLE; skip audio and waterfall downloads",
    )
    return p


def parse_bound(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if value.endswith("Z"):
        return parse_iso(value)
    # date-only
    if len(value) == 10:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return parse_iso(value if "+" in value or value.endswith("Z") else value + "Z")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    setup_logging(output_dir)

    start = parse_bound(args.start)
    end = parse_bound(args.end)

    logging.info(
        "MOVE-II downloader: sat_id=%s status=%s limit=%s order=%s dry_run=%s",
        args.sat_id,
        args.status,
        args.limit,
        args.order,
        args.dry_run,
    )
    if args.limit == 0:
        logging.warning(
            "limit=0 requested — this may take a long time and hit API rate limits"
        )

    session = RateLimitedSession(min_interval=API_MIN_INTERVAL_S)

    if args.order == "oldest":
        observations = collect_oldest(
            session,
            sat_id=args.sat_id,
            status=args.status,
            limit=args.limit,
            start=start,
            end=end,
        )
    else:
        observations = collect_newest(
            session,
            sat_id=args.sat_id,
            status=args.status,
            limit=args.limit,
            start=start,
            end=end,
        )

    before = len(observations)
    observations = apply_filters(
        observations,
        min_elevation=args.min_elevation,
        max_tle_age_hours=args.max_tle_age_hours,
    )
    logging.info(
        "Collected %d observations (%d after optional filters)",
        before,
        len(observations),
    )

    # Workspace root + output dir for local ogg reuse
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent
    search_roots = [workspace_root, output_dir]

    rows: list[dict] = []

    # Process sequentially for API politeness on metadata; allow parallel file
    # downloads within a single observation is overkill — instead parallelize
    # across observations carefully with a shared session lock via workers.
    # Using a small worker pool; each worker gets its own RateLimitedSession
    # clone to avoid interleaving issues on one Session object.
    def _worker(obs: dict) -> dict:
        # Per-thread session (requests.Session is not fully thread-safe)
        worker_session = RateLimitedSession(min_interval=API_MIN_INTERVAL_S)
        return process_observation(
            worker_session,
            obs,
            output_dir,
            dry_run=args.dry_run,
            search_roots=search_roots,
        )

    workers = 1 if args.dry_run else max(1, args.workers)
    if workers == 1:
        for i, obs in enumerate(observations, 1):
            logging.info(
                "[%d/%d] Processing observation %s (%s)",
                i,
                len(observations),
                obs["id"],
                obs.get("start"),
            )
            rows.append(
                process_observation(
                    session,
                    obs,
                    output_dir,
                    dry_run=args.dry_run,
                    search_roots=search_roots,
                )
            )
    else:
        logging.info("Downloading with %d workers", workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_worker, obs): obs for obs in observations
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                obs = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001
                    logging.exception("Worker failed for %s: %s", obs["id"], exc)
                    row = {
                        "id": obs["id"],
                        "start": obs.get("start"),
                        "end": obs.get("end"),
                        "ground_station": obs.get("ground_station"),
                        "station_name": obs.get("station_name"),
                        "station_lat": obs.get("station_lat"),
                        "station_lng": obs.get("station_lng"),
                        "station_alt": obs.get("station_alt"),
                        "lat_lng_precision": "",
                        "max_altitude": obs.get("max_altitude"),
                        "tle_age_hours": "",
                        "transmitter_mode": obs.get("transmitter_mode"),
                        "transmitter_uuid": obs.get("transmitter_uuid"),
                        "observation_frequency": obs.get("observation_frequency"),
                        "status": obs.get("status"),
                        "ogg_saved": False,
                        "waterfall_saved": False,
                        "meta_saved": False,
                        "error": str(exc),
                        "folder": "",
                    }
                rows.append(row)
                logging.info(
                    "[%d/%d] Finished observation %s",
                    done,
                    len(observations),
                    obs["id"],
                )

    write_index(output_dir, rows)
    ok = sum(1 for r in rows if not r.get("error"))
    logging.info("Done. %d/%d observations without errors.", ok, len(rows))
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
