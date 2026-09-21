#!/usr/bin/env python3
"""
Download genuine tracked MOVE-II observations for the early tumble era.

Uses SatNOGS filter norad_cat_id=43780 (tracked object) with start/end dates.
Skips complete local trees, reuses data/, downloads audio+waterfall only
(no spectrogram vetting).

Default window: 2018-11-01 → 2021-01-01.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_move2_observations import (  # noqa: E402
    API_BASE,
    RateLimitedSession,
    download_file,
    format_start_for_filename,
    observation_folder_name,
    parse_link_next,
    write_meta,
)
from fetch_and_vet_all_observations import (  # noqa: E402
    copy_observation_tree,
    find_existing_obs_folder,
    observation_is_complete,
)

MOVEII_NORAD = 43780


def setup_logging(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "download_tumble_era.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )


def list_observations(
    session: RateLimitedSession,
    *,
    norad_id: int,
    status: str,
    start: str,
    end: str,
) -> list[dict]:
    params = [
        ("norad_cat_id", str(norad_id)),
        ("format", "json"),
        ("start", start),
        ("end", end),
    ]
    if status:
        params.append(("status", status))
    req = requests.Request("GET", f"{API_BASE}/observations/", params=params)
    url = session.session.prepare_request(req).url
    out: list[dict] = []
    page = 0
    while url:
        page += 1
        logging.info("List page %d ...", page)
        resp = session.get(url, timeout=90)
        data = resp.json()
        next_url = parse_link_next(resp.headers.get("Link") or resp.headers.get("link"))
        resp.close()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected API response type: {type(data)}")
        if not data:
            break
        out.extend(data)
        logging.info(
            "  got %d (running %d) newest=%s oldest_on_page=%s",
            len(data),
            len(out),
            data[0].get("start"),
            data[-1].get("start"),
        )
        url = next_url
    return out


def ensure_files(
    session: RateLimitedSession,
    obs: dict,
    folder: Path,
) -> bool:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "raw").mkdir(exist_ok=True)
    (folder / "meta").mkdir(exist_ok=True)
    (folder / "derived").mkdir(exist_ok=True)
    write_meta(obs, folder / "meta")

    oid = obs["id"]
    start_tag = format_start_for_filename(obs["start"])
    ogg = folder / "raw" / f"satnogs_{oid}_{start_tag}.ogg"
    ok = True

    if not (ogg.is_file() and ogg.stat().st_size > 0):
        audio_url = obs.get("payload") or obs.get("archive_url")
        if audio_url:
            try:
                logging.info("Downloading audio %s", oid)
                download_file(session, audio_url, ogg)
            except Exception as exc:  # noqa: BLE001
                logging.warning("audio fail %s: %s", oid, exc)
                ok = False
        else:
            logging.warning("no audio URL for %s", oid)
            ok = False

    wf_url = obs.get("waterfall")
    if wf_url:
        name = Path(urlparse(wf_url).path).name or f"waterfall_{oid}.png"
        wf = folder / "raw" / name
        if not (wf.is_file() and wf.stat().st_size > 0):
            try:
                logging.info("Downloading waterfall %s", oid)
                download_file(session, wf_url, wf)
            except Exception as exc:  # noqa: BLE001
                logging.warning("waterfall fail %s: %s", oid, exc)
                ok = False

    return ok and observation_is_complete(folder, oid)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2018-11-01")
    p.add_argument("--end", default="2021-01-01", help="Exclusive upper bound (SatNOGS end=)")
    p.add_argument("--norad-id", type=int, default=MOVEII_NORAD)
    p.add_argument("--status", default="good")
    p.add_argument("--data-dir", default="data_move2_tracked")
    p.add_argument("--reuse-dirs", default="data")
    p.add_argument("--api-delay", type=float, default=3.0)
    p.add_argument(
        "--list-cache",
        default=None,
        help="Path to full_list.json (default: <data-dir>/api_cache/.../full_list.json)",
    )
    p.add_argument(
        "--refresh-list",
        action="store_true",
        help="Ignore existing full_list.json and re-query the API",
    )
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir).resolve()
    setup_logging(data_dir)
    reuse_dirs = []
    for part in (args.reuse_dirs or "").split(","):
        part = part.strip()
        if not part:
            continue
        path = Path(part)
        if not path.is_absolute():
            path = Path.cwd() / path
        reuse_dirs.append(path.resolve())

    session = RateLimitedSession(min_interval=args.api_delay)
    session.session.headers["User-Agent"] = "move2-tumble-download/1.0"

    cache_dir = data_dir / "api_cache" / "norad_43780_tracked_tumble_era_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    list_path = (
        Path(args.list_cache).resolve()
        if args.list_cache
        else cache_dir / "full_list.json"
    )

    if list_path.is_file() and not args.refresh_list:
        obs_list = json.loads(list_path.read_text(encoding="utf-8"))
        logging.info("Loaded %d obs from list cache %s", len(obs_list), list_path)
    else:
        obs_list = list_observations(
            session,
            norad_id=args.norad_id,
            status=args.status,
            start=args.start,
            end=args.end,
        )
        list_path.write_text(json.dumps(obs_list), encoding="utf-8")
        logging.info("Wrote list cache %s (%d)", list_path, len(obs_list))

    obs_list = [o for o in obs_list if o.get("norad_cat_id") == args.norad_id]
    obs_list.sort(key=lambda o: o.get("start") or "")
    years = Counter((o.get("start") or "")[:4] for o in obs_list)
    logging.info(
        "Ready to download %d obs years=%s range %s → %s",
        len(obs_list),
        dict(years),
        obs_list[0].get("start") if obs_list else None,
        obs_list[-1].get("start") if obs_list else None,
    )

    n_skip = n_reuse = n_dl = n_fail = 0
    for i, obs in enumerate(obs_list, 1):
        oid = obs["id"]
        folder = data_dir / observation_folder_name(obs)
        if observation_is_complete(folder, oid):
            n_skip += 1
            if i % 100 == 0 or i == len(obs_list):
                logging.info(
                    "[%d/%d] progress skip=%d reuse=%d dl=%d fail=%d",
                    i,
                    len(obs_list),
                    n_skip,
                    n_reuse,
                    n_dl,
                    n_fail,
                )
            continue

        src = find_existing_obs_folder(oid, reuse_dirs)
        if src is not None and observation_is_complete(src, oid):
            copy_observation_tree(src, folder)
            write_meta(obs, folder / "meta")
            if observation_is_complete(folder, oid):
                n_reuse += 1
                logging.info("[%d/%d] reused local %s", i, len(obs_list), oid)
                continue

        logging.info("[%d/%d] fetching %s start=%s", i, len(obs_list), oid, obs.get("start"))
        if ensure_files(session, obs, folder):
            n_dl += 1
        else:
            n_fail += 1
            logging.warning("incomplete %s", oid)

    logging.info(
        "DONE listed=%d skip=%d reuse=%d downloaded=%d fail=%d",
        len(obs_list),
        n_skip,
        n_reuse,
        n_dl,
        n_fail,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
