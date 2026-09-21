#!/usr/bin/env python3
"""
Build gap-stats-based amplitude leaderboard (v2) for MOVE-II pipeline candidates.

Joins data/amplitude_pipeline/leaderboard.csv with gap_stats.csv, computes
amplitude_quality = R² × (1 − frac_gap), flags amplitude_risk_v2 from waterfall
carrier-track keying gaps (not audio frac_time_strong), and prints a comparison
report against the legacy audio-vetting flags.

MOVE-II CW beacon timing (WARR MOVE-II docs): callsign every ~40 s, short beep
every ~10 s, BPSK telemetry ~every 60 s.  Expected keyed-off gaps are sub-second
to a few seconds within a burst; inter-beep silence is O(10 s).  Thresholds:
  • frac_gap > 0.30  — carrier absent on >30% of waterfall rows
  • longest_gap_s > 12 — longer than one inter-beep period (10 s) plus margin
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Gap-based risk thresholds (tuned to MOVE-II 10 s beep / 40 s callsign pattern)
DEFAULT_FRAC_GAP_THRESH = 0.30
DEFAULT_LONGEST_GAP_S_THRESH = 12.0
DEFAULT_OLD_FRAC_STRONG_THRESH = 0.30


def load_csv_by_obs_id(path: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[int(row["obs_id"])] = row
    return out


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _risk_v2(frac_gap: float, longest_gap_s: float, *, frac_thresh: float, gap_thresh: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if frac_gap > frac_thresh:
        reasons.append(f"frac_gap>{frac_thresh:g}")
    if longest_gap_s > gap_thresh:
        reasons.append(f"longest_gap>{gap_thresh:g}s")
    return bool(reasons), reasons


def build_rows(
    leaderboard: dict[int, dict[str, str]],
    gap_stats: dict[int, dict[str, str]],
    *,
    frac_gap_thresh: float,
    longest_gap_thresh: float,
    old_frac_thresh: float,
) -> list[dict]:
    merged: list[dict] = []
    missing_gap = []
    for obs_id, lb in leaderboard.items():
        gs = gap_stats.get(obs_id)
        if gs is None:
            missing_gap.append(obs_id)
            continue

        r2 = _f(lb, "R2")
        frac_gap = _f(gs, "frac_gap")
        longest_gap_s = _f(gs, "longest_gap_s")
        frac_time_strong = lb.get("frac_time_strong", "")
        frac_strong = float(frac_time_strong) if frac_time_strong not in ("", None) else None

        amplitude_quality = r2 * (1.0 - frac_gap)
        risk_v2, risk_v2_reasons = _risk_v2(
            frac_gap, longest_gap_s, frac_thresh=frac_gap_thresh, gap_thresh=longest_gap_thresh
        )
        old_risk = str(lb.get("amplitude_risk", "")).lower() in ("true", "1", "yes")

        old_frac_risky = frac_strong is not None and frac_strong < old_frac_thresh
        new_frac_clean = frac_gap <= frac_gap_thresh
        frac_metric_disagree = old_frac_risky and new_frac_clean

        old_frac_clean = frac_strong is not None and frac_strong >= old_frac_thresh
        new_frac_risky = frac_gap > frac_gap_thresh
        frac_metric_disagree_rev = old_frac_clean and new_frac_risky

        risk_disagree = old_risk != risk_v2

        merged.append(
            {
                "obs_id": obs_id,
                "station": lb.get("station", ""),
                "B": _f(lb, "B"),
                "R2": r2,
                "n_samples": int(_f(lb, "n_samples")),
                "snr_db": lb.get("snr_db", ""),
                "frac_time_strong": frac_time_strong,
                "frac_gap": frac_gap,
                "longest_gap_s": longest_gap_s,
                "median_gap_s": _f(gs, "median_gap_s"),
                "n_gaps": int(_f(gs, "n_gaps")),
                "amplitude_quality": amplitude_quality,
                "amplitude_risk": old_risk,
                "risk_reasons": lb.get("risk_reasons", ""),
                "amplitude_risk_v2": risk_v2,
                "risk_v2_reasons": ";".join(risk_v2_reasons),
                "risk_flags_disagree": risk_disagree,
                "frac_metric_disagree": frac_metric_disagree or frac_metric_disagree_rev,
                "frac_disagree_kind": (
                    "old_risky_new_clean"
                    if frac_metric_disagree
                    else ("old_clean_new_risky" if frac_metric_disagree_rev else "")
                ),
            }
        )

    if missing_gap:
        print(f"WARNING: {len(missing_gap)} obs in leaderboard missing from gap_stats: {missing_gap}")

    merged.sort(key=lambda r: r["amplitude_quality"], reverse=True)
    for i, row in enumerate(merged, 1):
        row["rank_v2"] = i
    return merged


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "rank_v2",
        "obs_id",
        "station",
        "B",
        "R2",
        "n_samples",
        "snr_db",
        "frac_time_strong",
        "frac_gap",
        "longest_gap_s",
        "median_gap_s",
        "n_gaps",
        "amplitude_quality",
        "amplitude_risk",
        "risk_reasons",
        "amplitude_risk_v2",
        "risk_v2_reasons",
        "risk_flags_disagree",
        "frac_metric_disagree",
        "frac_disagree_kind",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            out = dict(row)
            out["amplitude_quality"] = round(out["amplitude_quality"], 6)
            out["frac_gap"] = round(out["frac_gap"], 4)
            out["longest_gap_s"] = round(out["longest_gap_s"], 3)
            out["median_gap_s"] = round(out["median_gap_s"], 3)
            w.writerow(out)


def print_report(
    rows: list[dict],
    *,
    frac_gap_thresh: float,
    longest_gap_thresh: float,
    old_frac_thresh: float,
    pipeline_dir: Path,
) -> None:
    print("\n" + "=" * 118)
    print("AMPLITUDE LEADERBOARD v2 — sorted by amplitude_quality = R² × (1 − frac_gap)")
    print("=" * 118)
    print(
        f"  amplitude_risk_v2 thresholds: frac_gap > {frac_gap_thresh:g}  OR  "
        f"longest_gap_s > {longest_gap_thresh:g} s"
    )
    print(
        f"  (MOVE-II beacon: ~10 s beep / ~40 s callsign / ~60 s BPSK — "
        f"longest_gap > {longest_gap_thresh:g} s flags inter-beacon dropouts)"
    )
    print(
        f"  Legacy amplitude_risk: audio SNR < 15 dB OR frac_time_strong < {old_frac_thresh:g}"
    )
    print()
    hdr = (
        f"  {'rk':>3}  {'obs_id':>10}  {'station':<22}  {'R²':>6}  {'qual':>6}  "
        f"{'frac_old':>8}  {'frac_gap':>8}  {'lg_gap':>6}  {'risk_old':>8}  {'risk_v2':>7}"
    )
    print(hdr)
    print("  " + "-" * 114)

    for row in rows[:10]:
        frac_old = row["frac_time_strong"] if row["frac_time_strong"] != "" else "—"
        if isinstance(frac_old, str) and frac_old != "—":
            frac_old_s = f"{float(frac_old):8.3f}"
        else:
            frac_old_s = f"{'—':>8}"
        old_s = "RISKY" if row["amplitude_risk"] else "ok"
        v2_s = "RISKY" if row["amplitude_risk_v2"] else "ok"
        mark = " *" if row["risk_flags_disagree"] else "  "
        print(
            f"  {row['rank_v2']:>3}  {row['obs_id']:>10}  {row['station'][:22]:<22}  "
            f"{row['R2']:>6.4f}  {row['amplitude_quality']:>6.4f}  "
            f"{frac_old_s}  {row['frac_gap']:>8.4f}  {row['longest_gap_s']:>6.1f}  "
            f"{old_s:>8}  {v2_s:>7}{mark}"
        )

    n_disagree = sum(1 for r in rows if r["risk_flags_disagree"])
    n_frac_disagree = [r for r in rows if r["frac_metric_disagree"]]
    n_old_risky_new_clean = [r for r in rows if r["frac_disagree_kind"] == "old_risky_new_clean"]
    n_old_clean_new_risky = [r for r in rows if r["frac_disagree_kind"] == "old_clean_new_risky"]

    print()
    print(f"  Total candidates: {len(rows)}  |  risk flag disagreements (old vs v2): {n_disagree}")
    print(f"  frac_time_strong vs frac_gap metric disagreements: {len(n_frac_disagree)}")
    print("=" * 118)

    if n_old_risky_new_clean:
        print("\n--- VISUAL CHECK: old frac_time_strong risky, new frac_gap clean ---")
        print("    (audio vetting flagged weak signal fraction; waterfall gaps look fine)\n")
        for row in sorted(n_old_risky_new_clean, key=lambda r: r["amplitude_quality"], reverse=True):
            plot = pipeline_dir / f"obs_{row['obs_id']}" / f"amplitude_{row['obs_id']}.png"
            plot_note = str(plot) if plot.is_file() else f"{plot}  [plot not found — re-run pipeline]"
            print(
                f"  obs {row['obs_id']:>10}  qual={row['amplitude_quality']:.4f}  "
                f"frac_strong={float(row['frac_time_strong']):.3f}  frac_gap={row['frac_gap']:.4f}  "
                f"longest_gap={row['longest_gap_s']:.1f}s  old_risk={row['risk_reasons']}"
            )
            print(f"    → {plot_note}")

    if n_old_clean_new_risky:
        print("\n--- INVERSE: old frac_time_strong clean, new frac_gap risky ---\n")
        for row in sorted(n_old_clean_new_risky, key=lambda r: r["amplitude_quality"], reverse=True):
            plot = pipeline_dir / f"obs_{row['obs_id']}" / f"amplitude_{row['obs_id']}.png"
            plot_note = str(plot) if plot.is_file() else f"{plot}  [plot not found]"
            print(
                f"  obs {row['obs_id']:>10}  qual={row['amplitude_quality']:.4f}  "
                f"frac_strong={float(row['frac_time_strong']):.3f}  frac_gap={row['frac_gap']:.4f}  "
                f"longest_gap={row['longest_gap_s']:.1f}s  v2_risk={row['risk_v2_reasons']}"
            )
            print(f"    → {plot_note}")

    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Build gap-based amplitude leaderboard v2.")
    p.add_argument(
        "--pipeline-dir",
        type=Path,
        default=Path("data/amplitude_pipeline"),
        help="Directory containing leaderboard.csv and gap_stats.csv",
    )
    p.add_argument("--frac-gap-thresh", type=float, default=DEFAULT_FRAC_GAP_THRESH)
    p.add_argument("--longest-gap-s", type=float, default=DEFAULT_LONGEST_GAP_S_THRESH)
    p.add_argument("--old-frac-thresh", type=float, default=DEFAULT_OLD_FRAC_STRONG_THRESH)
    args = p.parse_args()

    pipeline_dir = args.pipeline_dir.resolve()
    leaderboard = load_csv_by_obs_id(pipeline_dir / "leaderboard.csv")
    gap_stats = load_csv_by_obs_id(pipeline_dir / "gap_stats.csv")

    rows = build_rows(
        leaderboard,
        gap_stats,
        frac_gap_thresh=args.frac_gap_thresh,
        longest_gap_thresh=args.longest_gap_s,
        old_frac_thresh=args.old_frac_thresh,
    )
    out_path = pipeline_dir / "leaderboard_v2.csv"
    write_csv(out_path, rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")
    print_report(
        rows,
        frac_gap_thresh=args.frac_gap_thresh,
        longest_gap_thresh=args.longest_gap_s,
        old_frac_thresh=args.old_frac_thresh,
        pipeline_dir=pipeline_dir,
    )


if __name__ == "__main__":
    main()
