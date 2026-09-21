#!/usr/bin/env python3
"""
Follow-up Lomb-Scargle analyses requested after initial tumble pass review.

  1. obs 14669931 — overlay pass-specific WARR CW cadence (10/40/60 s) on amplitude
     + raw waterfall; compare LS peak cluster to nominal harmonics at grid resolution.
  2. Maintain excluded observation list (14650494 = CW/noise artifact).
  3. Re-run LS with quadratic/cubic detrend on long-period false negatives.
  4. Query SatNOGS for pre-2020 NORAD 43780 observations.

Outputs under data/lomb_scargle/ by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lomb_scargle_tumble import (  # noqa: E402
    CW_BEEP_PERIOD_S,
    CW_BPSK_PERIOD_S,
    CW_CALLSIGN_PERIOD_S,
    analyse_obs,
    find_npz,
    load_leaderboard_v2,
    result_to_json,
    run_ls,
)

# WARR MOVE-II nominal beacon cadence (exact, not approximate)
WARR_CADENCE = {
    "beep": CW_BEEP_PERIOD_S,
    "callsign": CW_CALLSIGN_PERIOD_S,
    "bpsk": CW_BPSK_PERIOD_S,
}

MOVEII_NORAD = 43780

DEFAULT_EXCLUDED = [
    {
        "obs_id": 14650494,
        "reason": "CW/noise artifact: 154 cycles at 2.1 s, weakest FAP, no coherent envelope",
    },
    {
        "obs_id": 14669931,
        "reason": "NORAD 15595 (GEOSAT) not MOVE-II 43780; 150 MHz Mode-V pass mis-tracked",
    },
]

CW_ALIAS_CHECK_IDS = [14652843, 14675355]

EPOCH_2018_IDS = [349561, 349779, 354023, 359214, 360071]

DETREND_RETRY_IDS = [
    14667894,
    14670075,
    14674812,
    14652843,
    14675160,
    14675355,
]

SATNOGS_API = "https://network.satnogs.org/api/observations/"


def write_excluded_obs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["obs_id", "reason"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def load_excluded_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    with open(path, encoding="utf-8") as fh:
        return {int(r["obs_id"]) for r in csv.DictReader(fh)}


def _score_phase(c: float, phi: float, t: np.ndarray, y: np.ndarray) -> float:
    marks = np.zeros_like(y)
    for g in np.arange(phi, t.max() + c, c):
        marks += np.exp(-0.5 * ((t - g) / 0.5) ** 2)
    if marks.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(y, marks)[0, 1])


def phase_align_cadence(t: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    for name, period in WARR_CADENCE.items():
        phases = np.linspace(0.0, period, int(period / 0.05) + 1)
        scores = [_score_phase(period, p, t, y) for p in phases]
        best_i = int(np.argmax(scores))
        out[name] = {
            "period_s": period,
            "phase_s": float(phases[best_i]),
            "envelope_corr": float(scores[best_i]),
        }
    return out


def build_grid_lines(t_min: float, t_max: float, alignment: dict) -> list[dict]:
    lines: list[dict] = []
    for kind, meta in alignment.items():
        period = meta["period_s"]
        g = meta["phase_s"]
        while g <= t_max + 0.01:
            if g >= t_min - 0.01:
                lines.append({"t_rel_s": float(g), "kind": kind, "period_s": period})
            g += period
    lines.sort(key=lambda x: x["t_rel_s"])
    return lines


def ls_period_resolution(t: np.ndarray, y: np.ndarray, *, detrend_deg: int = 1) -> float:
    ls = run_ls(t, y, detrend_deg=detrend_deg)
    period = ls["period"]
    mask = (period > 35) & (period < 65)
    if mask.sum() < 2:
        return float("nan")
    return float(np.median(np.abs(np.diff(period[mask]))))


def harmonic_table(
    peaks: list[float], resolution_s: float
) -> list[dict]:
    refs = [(base * m, f"{base}s×{m}") for base in (10, 40, 60) for m in range(1, 8)]
    rows = []
    for p in peaks:
        within = [
            {"label": lab, "ref_period_s": ref, "delta_s": round(p - ref, 3)}
            for ref, lab in refs
            if abs(p - ref) <= resolution_s
        ]
        nearest = sorted((abs(p - ref), ref, lab, p - ref) for ref, lab in refs)[:3]
        rows.append(
            {
                "peak_period_s": p,
                "within_resolution": within,
                "nearest": [
                    {"ref_period_s": ref, "label": lab, "delta_s": round(d, 3)}
                    for _, ref, lab, d in nearest
                ],
            }
        )
    return rows


def audit_norad_candidates(
    pipeline_dir: Path,
    project_root: Path,
    *,
    target_norad: int = MOVEII_NORAD,
) -> tuple[list[dict], dict]:
    """Cross-check norad_cat_id for all amplitude-pipeline candidates."""
    lb_path = pipeline_dir / "leaderboard.csv"
    candidate_ids: list[int] = []
    with open(lb_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            candidate_ids.append(int(row["obs_id"]))

    cache_dir = project_root / "data" / "api_cache" / f"norad_{target_norad}_status_good"
    api_index: dict[int, dict] = {}
    if cache_dir.is_dir():
        for page in cache_dir.glob("page_*.json"):
            for obs in json.loads(page.read_text(encoding="utf-8")):
                api_index[obs["id"]] = obs

    def _find_folder(oid: int) -> Path | None:
        for folder in (project_root / "data").glob(f"move2_{oid}_*"):
            if folder.is_dir():
                return folder
        return None

    rows: list[dict] = []
    for oid in candidate_ids:
        folder = _find_folder(oid)
        local: dict = {}
        if folder:
            meta_path = folder / "meta" / f"observation_{oid}.json"
            if meta_path.is_file():
                local = json.loads(meta_path.read_text(encoding="utf-8"))
        tle_norad = None
        if folder:
            tle_path = folder / "meta" / f"tle_{oid}.txt"
            if tle_path.is_file():
                for line in tle_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("1 "):
                        tle_norad = int(line.split()[1].replace("U", ""))
                        break
        api = api_index.get(oid, {})
        local_norad = local.get("norad_cat_id")
        api_norad = api.get("norad_cat_id")
        rows.append(
            {
                "obs_id": oid,
                "local_norad": local_norad,
                "api_norad": api_norad,
                "tle_norad": tle_norad,
                "match_moveii_local": local_norad == target_norad,
                "match_moveii_api": api_norad == target_norad if api_norad is not None else None,
                "local_api_agree": (
                    local_norad == api_norad if api_norad is not None else None
                ),
                "station": local.get("station_name") or api.get("station_name", ""),
                "start": local.get("start") or api.get("start", ""),
                "freq_hz": local.get("observation_frequency"),
                "transmitter": (local.get("transmitter_description") or "")[:50],
                "in_api_cache": oid in api_index,
            }
        )

    audit_csv = pipeline_dir / "norad_audit.csv"
    with open(audit_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    all423 = 0
    moveii423 = 0
    for folder in (project_root / "data").glob("move2_*"):
        oid = int(folder.name.split("_")[1])
        meta_path = folder / "meta" / f"observation_{oid}.json"
        if not meta_path.is_file():
            continue
        all423 += 1
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("norad_cat_id") == target_norad:
            moveii423 += 1

    cache_total = 0
    cache_moveii_field = 0
    if cache_dir.is_dir():
        for page in cache_dir.glob("page_*.json"):
            for obs in json.loads(page.read_text(encoding="utf-8")):
                cache_total += 1
                if obs.get("norad_cat_id") == target_norad:
                    cache_moveii_field += 1

    summary = {
        "target_norad": target_norad,
        "n_candidates": len(rows),
        "n_candidates_moveii_tracked": sum(1 for r in rows if r["match_moveii_local"]),
        "n_candidates_not_moveii": sum(1 for r in rows if not r["match_moveii_local"]),
        "n_local_api_disagree": sum(
            1 for r in rows if r["local_api_agree"] is False
        ),
        "batch_423_total": all423,
        "batch_423_moveii_tracked": moveii423,
        "api_cache_pages": cache_dir.name if cache_dir.is_dir() else None,
        "api_cache_obs_count": cache_total,
        "api_cache_norad_field_eq_target": cache_moveii_field,
        "root_cause": (
            "SatNOGS fetch filters on satellite__norad_cat_id (transmitter association), "
            "but observation.norad_cat_id is the TLE object tracked during the pass — "
            "often a different satellite. Local JSON matches API cache (no join bug). "
            "Downstream gap: run_full_waterfall_sgp4_validation.discover_obs_ids() and "
            "run_amplitude_pipeline load_raw_uncorrected() never filter norad_cat_id==43780."
        ),
        "confirmed_moveii_candidate_ids": [
            r["obs_id"] for r in rows if r["match_moveii_local"]
        ],
        "mismatch_ids": [r["obs_id"] for r in rows if not r["match_moveii_local"]],
    }
    summary_path = pipeline_dir / "norad_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows, summary


def analyse_cw_timing(
    obs_id: int,
    pipeline_dir: Path,
    output_dir: Path,
    project_root: Path,
    *,
    detrend_deg: int = 1,
    peak_periods: list[float] | None = None,
) -> dict:
    npz_path = find_npz(pipeline_dir, obs_id)
    if npz_path is None:
        raise FileNotFoundError(f"Missing npz for obs {obs_id}")

    d = np.load(npz_path)
    t_all = d["t_rel_s"].astype(float)
    y_all = d["amplitude_pathloss_corrected"].astype(float)
    gap = d["is_gap"].astype(bool)
    valid = ~gap
    t = t_all[valid]
    y = y_all[valid]
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    ls, _, _, _ = analyse_obs(
        obs_id, npz_path, gap_mode="masked", detrend_deg=detrend_deg
    )
    peaks = (
        peak_periods
        if peak_periods is not None
        else [pk["period_s"] for pk in ls["top3"]]
    )
    resolution = ls_period_resolution(t, y, detrend_deg=detrend_deg)

    alignment = phase_align_cadence(t, y)
    grid_lines = build_grid_lines(float(t.min()), float(t.max()), alignment)
    harm = harmonic_table(peaks, resolution)

    # envelope minima for measured pass cadence
    win = 51 if len(y) > 51 else max(5, len(y) // 2 * 2 + 1)
    ys = savgol_filter(y, win, 3)
    min_idx, _ = find_peaks(-ys, distance=max(3, int(5.0 / np.median(np.diff(t)))))
    min_t = t[min_idx]
    min_dt = np.diff(min_t) if len(min_t) > 1 else np.array([])

    # key-on edges from gap mask
    rise = np.where(valid[1:] & gap[:-1])[0] + 1
    rise_t = np.sort(t_all[rise]) if len(rise) else np.array([])
    rise_dt = np.diff(rise_t) if len(rise_t) > 1 else np.array([])

    result = {
        "obs_id": obs_id,
        "detrend_deg": detrend_deg,
        "warr_nominal_periods_s": WARR_CADENCE,
        "phase_alignment": alignment,
        "ls_resolution_period_s": round(resolution, 4),
        "ls_peaks_s": peaks,
        "ls_top3_periods_s": [pk["period_s"] for pk in ls["top3"]],
        "harmonic_alignment": harm,
        "envelope_minima_median_interval_s": (
            round(float(np.median(min_dt)), 3) if len(min_dt) else None
        ),
        "key_on_median_interval_s": (
            round(float(np.median(rise_dt)), 3) if len(rise_dt) else None
        ),
        "grid_line_count": len(grid_lines),
    }

    timing_path = output_dir / f"obs_{obs_id}_cw_timing.json"
    with open(timing_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    plot_cw_waterfall_overlay(
        obs_id,
        t_all,
        y_all,
        gap,
        grid_lines,
        peaks,
        resolution,
        project_root,
        output_dir / f"obs_{obs_id}_cw_waterfall_overlay.png",
    )
    return result


def plot_cw_waterfall_overlay(
    obs_id: int,
    t_all: np.ndarray,
    y_all: np.ndarray,
    gap: np.ndarray,
    grid_lines: list[dict],
    ls_peaks: list[float],
    ls_resolution: float,
    project_root: Path,
    output_path: Path,
) -> None:
    from carrier_doppler_check import find_obs_folder

    obs_folder = find_obs_folder(obs_id, project_root)
    wf_path = next((obs_folder / "raw").glob("waterfall_*.png"), None)
    meta_path = obs_folder / "derived" / f"carrier_track_{obs_id}_meta.json"
    y_slope = y_intercept = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        y_slope = float(meta["y_slope_s_per_px"])
        y_intercept = float(meta["y_intercept_s"])

    colors = {"beep": "#2ca02c", "callsign": "#ff7f0e", "bpsk": "#9467bd"}

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.6, 1.0, 0.7])

    # --- Waterfall panel ---
    ax_wf = fig.add_subplot(gs[0])
    if wf_path and wf_path.is_file():
        img = np.array(Image.open(wf_path))
        t_min, t_max = float(np.sort(t_all)[0]), float(np.sort(t_all)[-1])
        if y_slope is not None and y_intercept is not None:
            # waterfall y-axis: pixel row -> t_rel_s via linear OCR fit
            h = img.shape[0]
            row_t = y_intercept + y_slope * np.arange(h)
            row_mask = (row_t >= t_min - 2) & (row_t <= t_max + 2)
            rows = np.where(row_mask)[0]
            if len(rows) > 10:
                img_crop = img[rows.min() : rows.max() + 1]
                t_top = float(row_t[rows.min()])
                t_bot = float(row_t[rows.max()])
                if t_top > t_bot:
                    t_top, t_bot = t_bot, t_top
                ax_wf.imshow(
                    img_crop,
                    aspect="auto",
                    origin="upper",
                    extent=[0, img_crop.shape[1], t_bot, t_top],
                )
            else:
                ax_wf.imshow(img, aspect="auto", origin="upper")
        else:
            ax_wf.imshow(img, aspect="auto", origin="upper")
        ax_wf.set_ylabel("Time in pass [s]")
        ax_wf.set_title(f"obs {obs_id} — raw SatNOGS waterfall (time axis from OCR fit)")
    else:
        ax_wf.text(0.5, 0.5, "Waterfall PNG not found", ha="center", va="center")
    for line in grid_lines:
        c = colors.get(line["kind"], "gray")
        ax_wf.axhline(line["t_rel_s"], color=c, lw=0.6, alpha=0.55)

    # --- Amplitude + cadence ---
    ax_amp = fig.add_subplot(gs[1], sharex=ax_wf if wf_path else None)
    valid = gap == 0
    ax_amp.plot(t_all[valid], y_all[valid], lw=0.7, color="steelblue", label="A_corr")
    if gap.any():
        ax_amp.plot(t_all[~valid], y_all[~valid], "x", color="salmon", ms=2, alpha=0.6)
    for kind, period in WARR_CADENCE.items():
        kind_lines = [ln for ln in grid_lines if ln["kind"] == kind]
        for ln in kind_lines:
            ax_amp.axvline(
                ln["t_rel_s"],
                color=colors[kind],
                lw=0.7,
                alpha=0.45,
                label=f"{period:.0f}s {kind}" if ln is kind_lines[0] else None,
            )
    ax_amp.set_ylabel("A_corr [dB]")
    ax_amp.set_xlabel("Time in pass [s]")
    ax_amp.legend(fontsize=7, ncol=3, loc="upper right")
    ax_amp.grid(True, alpha=0.25)
    ax_amp.set_title(
        "Path-loss-corrected amplitude with phase-aligned WARR cadence "
        "(10 s beep / 40 s callsign / 60 s BPSK)"
    )

    # --- LS peak cluster vs harmonics ---
    ax_h = fig.add_subplot(gs[2])
    refs = [(base * m, f"{base}×{m}") for base in (10, 40, 60) for m in range(1, 7)]
    ref_p = [r[0] for r in refs]
    ref_l = [r[1] for r in refs]
    ax_h.scatter(ref_p, np.zeros(len(ref_p)), marker="|", s=120, color="gray", label="nominal harmonics")
    for i, lab in enumerate(ref_l):
        if refs[i][0] in (10, 40, 60):
            ax_h.text(ref_p[i], 0.02, lab, fontsize=6, ha="center", color="gray")
    ax_h.scatter(ls_peaks, [0.5] * len(ls_peaks), s=80, color="red", zorder=5, label="LS top-3")
    for p in ls_peaks:
        ax_h.annotate(f"{p:.1f}s", (p, 0.5), xytext=(0, 8), textcoords="offset points", fontsize=8, color="red")
    ax_h.axvspan(40 - ls_resolution, 40 + ls_resolution, alpha=0.08, color="orange")
    ax_h.axvspan(60 - ls_resolution, 60 + ls_resolution, alpha=0.08, color="purple")
    ax_h.set_yticks([])
    ax_h.set_xlabel("Period [s]")
    ax_h.set_xlim(5, 75)
    ax_h.set_title(
        f"LS peak cluster vs WARR harmonics  (grid ΔP ≈ {ls_resolution:.2f} s — shaded ±1 bin at 40/60 s)"
    )
    ax_h.legend(fontsize=7, loc="upper right")
    ax_h.grid(True, alpha=0.25)

    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_detrend_sensitivity(
    pipeline_dir: Path,
    output_dir: Path,
    obs_ids: list[int],
) -> list[dict]:
    lb = {r["obs_id"]: r for r in load_leaderboard_v2(pipeline_dir / "leaderboard_v2.csv")}
    rows: list[dict] = []
    for obs_id in obs_ids:
        npz = find_npz(pipeline_dir, obs_id)
        if npz is None:
            continue
        linear_peak = None
        for deg in (1, 2, 3):
            ls, t_all, _, is_gap = analyse_obs(
                obs_id, npz, gap_mode="masked", detrend_deg=deg
            )
            summary = result_to_json(
                obs_id,
                lb.get(obs_id, {}),
                ls,
                "masked",
                int((~is_gap).sum()),
                len(t_all),
            )
            if deg == 1:
                linear_peak = summary["peak_period_s"]
            summary["detrend_deg"] = deg
            summary["delta_peak_vs_linear"] = (
                round(summary["peak_period_s"] - linear_peak, 3)
                if linear_peak is not None
                else None
            )
            rows.append(summary)
    out_csv = output_dir / "detrend_sensitivity.csv"
    fields = [
        "obs_id",
        "detrend_deg",
        "pass_length_s",
        "n_cycles",
        "credible",
        "peak_period_s",
        "delta_peak_vs_linear",
        "fap_peak",
        "peak1_cw_alias",
        "peak2_period_s",
        "peak3_period_s",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            t3 = r.get("top3_peaks", [])
            w.writerow(
                {
                    "obs_id": r["obs_id"],
                    "detrend_deg": r["detrend_deg"],
                    "pass_length_s": r.get("pass_length_s"),
                    "n_cycles": r.get("n_cycles"),
                    "credible": r.get("credible"),
                    "peak_period_s": r["peak_period_s"],
                    "delta_peak_vs_linear": r.get("delta_peak_vs_linear"),
                    "fap_peak": r["fap_peak"],
                    "peak1_cw_alias": t3[0].get("cw_alias_flag", "") if t3 else "",
                    "peak2_period_s": t3[1]["period_s"] if len(t3) > 1 else "",
                    "peak3_period_s": t3[2]["period_s"] if len(t3) > 1 else "",
                }
            )
    return rows


def run_epoch2018_cubic_ls(
    pipeline_dir: Path,
    output_dir: Path,
    obs_ids: list[int],
) -> list[dict]:
    """Cubic-detrend LS on 2018 raw_uncorrected passes (linear vs cubic comparison)."""
    rows: list[dict] = []
    for obs_id in obs_ids:
        npz = find_npz(pipeline_dir, obs_id)
        if npz is None:
            print(f"  obs {obs_id}: missing npz — skip")
            continue
        for deg in (1, 3):
            ls, t_all, _, is_gap = analyse_obs(
                obs_id, npz, gap_mode="masked", detrend_deg=deg
            )
            summary = result_to_json(
                obs_id,
                {},
                ls,
                "masked",
                int((~is_gap).sum()),
                len(t_all),
            )
            summary["detrend_deg"] = deg
            rows.append(summary)
            out_json = output_dir / f"obs_{obs_id}_ls_deg{deg}.json"
            serialisable = {
                k: v
                for k, v in summary.items()
                if k not in ("frequency", "power", "period")
            }
            out_json.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

    out_csv = output_dir / "epoch2018_cubic_ls.csv"
    fields = [
        "obs_id",
        "detrend_deg",
        "pass_length_s",
        "n_cycles",
        "credible",
        "peak_period_s",
        "fap_peak",
        "peak1_cw_alias",
        "peak2_period_s",
        "peak3_period_s",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            t3 = r.get("top3_peaks", [])
            w.writerow(
                {
                    "obs_id": r["obs_id"],
                    "detrend_deg": r["detrend_deg"],
                    "pass_length_s": r.get("pass_length_s"),
                    "n_cycles": r.get("n_cycles"),
                    "credible": r.get("credible"),
                    "peak_period_s": r["peak_period_s"],
                    "fap_peak": r["fap_peak"],
                    "peak1_cw_alias": t3[0].get("cw_alias_flag", "") if t3 else "",
                    "peak2_period_s": t3[1]["period_s"] if len(t3) > 1 else "",
                    "peak3_period_s": t3[2]["period_s"] if len(t3) > 1 else "",
                }
            )
    return rows


def scan_local_pre2020(project_root: Path, end_before: str = "2020-01-01") -> list[dict]:
    """Collect pre-2020 observations from locally downloaded move2_* folders."""
    rows: list[dict] = []
    data_dir = project_root / "data"
    for folder in sorted(data_dir.glob("move2_*")):
        meta_files = list((folder / "meta").glob("observation_*.json"))
        if not meta_files:
            continue
        meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
        start = meta.get("start") or ""
        if not start or start >= end_before:
            continue
        rows.append(
            {
                "obs_id": meta["id"],
                "start": start,
                "end": meta.get("end"),
                "station_name": meta.get("station_name"),
                "norad_cat_id": meta.get("norad_cat_id"),
                "waterfall_status": meta.get("waterfall_status"),
                "has_waterfall": bool(meta.get("waterfall")),
                "local_download": True,
            }
        )
    return rows


def query_satnogs_pre2020(
    norad_id: int = 43780,
    end_before: str = "2020-01-01",
    *,
    cache_dir: Path | None = None,
) -> list[dict]:
    """Load pre-2020 good-status observations from API page cache, else paginate live."""
    rows: list[dict] = []
    if cache_dir and cache_dir.is_dir():
        for page in sorted(cache_dir.glob("page_*.json")):
            batch = json.loads(page.read_text(encoding="utf-8"))
            for obs in batch:
                start = obs.get("start") or ""
                if start and start < end_before:
                    rows.append(
                        {
                            "obs_id": obs["id"],
                            "start": start,
                            "end": obs.get("end"),
                            "station_name": obs.get("station_name"),
                            "norad_cat_id": obs.get("norad_cat_id"),
                            "waterfall_status": obs.get("waterfall_status"),
                            "has_waterfall": bool(obs.get("waterfall")),
                        }
                    )
        if rows:
            return rows

    params = {"norad_cat_id": norad_id, "status": "good", "end__lt": end_before}
    url: str | None = SATNOGS_API
    sess = requests.Session()
    sess.headers["User-Agent"] = "move2-ls-followup/1.0 (research)"
    import time

    while url:
        resp = sess.get(url, params=params if url == SATNOGS_API else None, timeout=60)
        if resp.status_code == 429:
            time.sleep(5.0)
            continue
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for obs in batch:
            rows.append(
                {
                    "obs_id": obs["id"],
                    "start": obs.get("start"),
                    "end": obs.get("end"),
                    "station_name": obs.get("station_name"),
                    "norad_cat_id": obs.get("norad_cat_id"),
                    "waterfall_status": obs.get("waterfall_status"),
                    "has_waterfall": bool(obs.get("waterfall")),
                }
            )
        url = resp.links.get("next", {}).get("url")
        params = None
        time.sleep(1.5)
    return rows


def write_pre2020_report(rows: list[dict], output_dir: Path, local_ids: set[int]) -> Path:
    out = output_dir / "satnogs_pre2020_43780.csv"
    fields = [
        "obs_id",
        "start",
        "end",
        "station_name",
        "norad_cat_id",
        "waterfall_status",
        "has_waterfall",
        "local_download",
    ]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["start"] or ""):
            r = dict(r)
            r["local_download"] = int(r["obs_id"]) in local_ids
            w.writerow(r)
    summary = output_dir / "satnogs_pre2020_43780_summary.json"
    summary.write_text(
        json.dumps(
            {
                "norad_id": 43780,
                "end_before": "2020-01-01",
                "n_api_results": len(rows),
                "n_local": sum(1 for r in rows if r["obs_id"] in local_ids),
                "n_not_local": sum(1 for r in rows if r["obs_id"] not in local_ids),
                "date_range": [
                    min((r["start"] for r in rows if r["start"]), default=None),
                    max((r["start"] for r in rows if r["start"]), default=None),
                ],
            },
            indent=2,
        )
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="LS follow-up analyses for MOVE-II tumble study.")
    ap.add_argument("--pipeline-dir", type=Path, default=Path("data/amplitude_pipeline"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/lomb_scargle"))
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--skip-satnogs", action="store_true")
    args = ap.parse_args()

    pipeline_dir = args.pipeline_dir.resolve()
    output_dir = args.output_dir.resolve()
    project_root = args.project_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_path = pipeline_dir / "excluded_obs.csv"
    write_excluded_obs(excluded_path, DEFAULT_EXCLUDED)
    print(f"Wrote exclusion list → {excluded_path}")

    _, audit = audit_norad_candidates(pipeline_dir, project_root)
    print(
        f"NORAD audit: {audit['n_candidates_moveii_tracked']}/{audit['n_candidates']} "
        f"candidates track MOVE-II; {audit['n_candidates_not_moveii']} mismatches"
    )
    print(f"  → {pipeline_dir / 'norad_audit.csv'}")
    print(f"  → {pipeline_dir / 'norad_audit_summary.json'}")

    for obs_id in CW_ALIAS_CHECK_IDS:
        ls_deg3, _, _, _ = analyse_obs(
            obs_id,
            find_npz(pipeline_dir, obs_id),
            gap_mode="masked",
            detrend_deg=3,
        )
        cubic_peak = ls_deg3["top3"][0]["period_s"] if ls_deg3["top3"] else float("nan")
        timing = analyse_cw_timing(
            obs_id,
            pipeline_dir,
            output_dir,
            project_root,
            detrend_deg=3,
            peak_periods=[cubic_peak] + [pk["period_s"] for pk in ls_deg3["top3"][1:3]],
        )
        print(
            f"obs {obs_id} CW alias check (cubic peak={cubic_peak:.2f}s): "
            f"ΔP≈{timing['ls_resolution_period_s']:.2f}s, "
            f"within_res={timing['harmonic_alignment'][0]['within_resolution'] or 'NONE'}"
        )
        print(f"  overlay → {output_dir / f'obs_{obs_id}_cw_waterfall_overlay.png'}")

    epoch_rows = run_epoch2018_cubic_ls(pipeline_dir, output_dir, EPOCH_2018_IDS)
    print(f"2018 epoch cubic LS → {output_dir / 'epoch2018_cubic_ls.csv'} ({len(epoch_rows)} rows)")

    detrend_rows = run_detrend_sensitivity(pipeline_dir, output_dir, DETREND_RETRY_IDS)
    print(f"Detrend sensitivity → {output_dir / 'detrend_sensitivity.csv'} ({len(detrend_rows)} rows)")

    if not args.skip_satnogs:
        print("Scanning local + SatNOGS cache for pre-2020 NORAD 43780 observations …")
        api_rows = scan_local_pre2020(project_root)
        source = "local move2 folders"
        if not api_rows:
            cache_dir = project_root / "data" / "api_cache" / "norad_43780_status_good"
            api_rows = query_satnogs_pre2020(cache_dir=cache_dir)
            source = "api cache" if api_rows else "live API"
        local_ids = {
            int(p.name.split("_")[1])
            for p in (project_root / "data").glob("move2_*")
            if p.is_dir()
        }
        report = write_pre2020_report(api_rows, output_dir, local_ids)
        summary_path = output_dir / "satnogs_pre2020_43780_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["source"] = source
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Pre-2020 report → {report}  ({len(api_rows)} rows, source={source})")


if __name__ == "__main__":
    main()
