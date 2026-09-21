#!/usr/bin/env python3
"""
Post-download pipeline for the tracked-NORAD MOVE-II dataset.

Runs, in order:
  1. Waterfall + SGP4 Doppler validation (all move2_* under --data-dir)
  2. Amplitude envelope extraction for raw_uncorrected passes
  3. Gap-aware leaderboard v2
  4. Sample-size report (replaces legacy "423" and "32" counts)

Usage (after fetch_move2_tracked.py finishes):

  python scripts/run_move2_tracked_pipeline.py --data-dir data_move2_tracked

Optional — Lomb-Scargle on top candidates:

  python scripts/lomb_scargle_tumble.py \\
    --pipeline-dir data_move2_tracked/amplitude_pipeline \\
    --output-dir data_move2_tracked/lomb_scargle \\
    --top-n 10
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

MOVEII_NORAD = 43780


def count_move2_folders(data_dir: Path, *, norad_id: int | None = None) -> tuple[int, int]:
    """Return (total move2 folders, count with matching norad_cat_id)."""
    total = 0
    matched = 0
    for folder in data_dir.glob("move2_*"):
        if not folder.is_dir():
            continue
        oid = int(folder.name.split("_")[1])
        meta_path = folder / "meta" / f"observation_{oid}.json"
        if not meta_path.is_file():
            continue
        total += 1
        if norad_id is None:
            matched += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("norad_cat_id") == norad_id:
            matched += 1
    return total, matched


def count_from_results(results_csv: Path, *, classification: str | None = None) -> int:
    if not results_csv.is_file():
        return 0
    n = 0
    with open(results_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if classification is None or row.get("classification") == classification:
                n += 1
    return n


def write_sample_report(
    data_dir: Path,
    pipeline_dir: Path,
    results_csv: Path,
    output_path: Path,
) -> dict:
    total_folders, norad_ok = count_move2_folders(data_dir, norad_id=MOVEII_NORAD)
    validated = count_from_results(results_csv)
    raw_unc = count_from_results(results_csv, classification="raw_uncorrected")

    lb_path = pipeline_dir / "leaderboard.csv"
    n_candidates = 0
    if lb_path.is_file():
        with open(lb_path, encoding="utf-8") as fh:
            n_candidates = sum(1 for _ in csv.DictReader(fh))

    report = {
        "moveii_norad": MOVEII_NORAD,
        "data_dir": str(data_dir),
        "genuine_moveii_folders": norad_ok,
        "total_move2_folders": total_folders,
        "waterfall_validated": validated,
        "raw_uncorrected_candidates": raw_unc,
        "amplitude_pipeline_candidates": n_candidates,
        "replaces_legacy_counts": {
            "old_batch_size_423": "waterfall_validated on contaminated data/",
            "old_candidate_set_32": "raw_uncorrected from contaminated validation",
            "new_primary_N": validated,
            "new_candidate_N": raw_unc,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_step(cmd: list[str], *, cwd: Path) -> None:
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate + amplitude pipeline for tracked MOVE-II data.")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_move2_tracked"),
        help="Root containing move2_* folders (from fetch_move2_tracked.py)",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repo root (default: parent of scripts/)",
    )
    ap.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip waterfall SGP4 batch (use existing results.csv)",
    )
    ap.add_argument(
        "--skip-amplitude",
        action="store_true",
        help="Skip amplitude pipeline + leaderboard v2",
    )
    args = ap.parse_args(argv)

    project_root = (args.project_root or Path(__file__).resolve().parent.parent).resolve()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        logging.error("Data dir not found: %s — run fetch_move2_tracked.py first", data_dir)
        return 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    py = sys.executable
    scripts = project_root / "scripts"

    sys.path.insert(0, str(scripts))
    import carrier_doppler_check as cdc

    try:
        rel = str(data_dir.relative_to(project_root))
    except ValueError:
        rel = str(data_dir)
    if rel not in cdc.SEARCH_ROOTS:
        cdc.SEARCH_ROOTS = [rel, *cdc.SEARCH_ROOTS]

    results_csv = data_dir / "waterfall_sgp4_validation" / "results.csv"
    pipeline_dir = data_dir / "amplitude_pipeline"

    if not args.skip_validation:
        run_step(
            [
                py,
                str(scripts / "run_full_waterfall_sgp4_validation.py"),
                "--data-dir",
                str(data_dir),
                "--require-norad-id",
                str(MOVEII_NORAD),
            ],
            cwd=project_root,
        )

    if not args.skip_amplitude:
        run_step(
            [
                py,
                str(scripts / "run_amplitude_pipeline.py"),
                "--data-dir",
                str(data_dir),
                "--results-csv",
                str(results_csv),
            ],
            cwd=project_root,
        )
        run_step(
            [
                py,
                str(scripts / "build_amplitude_leaderboard_v2.py"),
                "--pipeline-dir",
                str(pipeline_dir),
            ],
            cwd=project_root,
        )

    report = write_sample_report(
        data_dir,
        pipeline_dir,
        results_csv,
        data_dir / "move2_sample_sizes.json",
    )

    print("\n" + "=" * 72)
    print("MOVE-II TRACKED SAMPLE SIZES (replaces 423 / 32)")
    print("=" * 72)
    print(f"  Data directory:              {data_dir}")
    print(f"  move2 folders (norad 43780): {report['genuine_moveii_folders']}")
    print(f"  Waterfall validated:         {report['waterfall_validated']}")
    print(f"  raw_uncorrected candidates:  {report['raw_uncorrected_candidates']}  ← new '32'")
    print(f"  amplitude pipeline rows:     {report['amplitude_pipeline_candidates']}")
    print(f"  Full report:                 {data_dir / 'move2_sample_sizes.json'}")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
