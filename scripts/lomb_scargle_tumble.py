#!/usr/bin/env python3
"""
Lomb-Scargle periodogram analysis of MOVE-II CubeSat tumble modulation.

Reads leaderboard_v2.csv, takes the top-N observations by amplitude_quality,
loads their path-loss-corrected amplitude series (A_corr = amplitude_pathloss_corrected)
from the .npz files, and runs astropy LombScargle on each.

Frequency grid: 1/period_max_s .. 1/period_min_s  (default 1 s .. 300 s),
spanning the plausible tumble range while covering the ~10 s CW keying artefact.

For obs 351032 specifically, runs once with the 17.5 s gap excluded (masked) and
once with it included (interpolated-through) to quantify the impact.

Outputs per obs:  <output_dir>/obs_<id>_ls.png  +  <output_dir>/obs_<id>_ls.json
Aggregate table:  <output_dir>/aggregate_ls.csv
Gap-sensitivity:  <output_dir>/351032_gap_sensitivity.json  +  _comparison.png
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

try:
    from astropy.timeseries import LombScargle
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False
    warnings.warn("astropy not found; falling back to scipy.signal.lombscargle (no FAP)")

import csv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CW keying artefact windows (periods in seconds)
CW_BEEP_PERIOD_S = 10.0        # ~10 s inter-beep silence
CW_CALLSIGN_PERIOD_S = 40.0    # ~40 s callsign cycle
CW_BPSK_PERIOD_S = 60.0        # ~60 s BPSK telemetry cycle
# ±10 % window, harmonics 1–3 only — keeps the flag meaningful
CW_ALIAS_WINDOW = 0.10
CW_HARMONICS = [1, 2, 3]

# A detection is "credible" when the pass spans ≥ this many full cycles
MIN_CYCLES_FOR_CREDIBLE = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flag_cw_alias(period_s: float) -> str:
    """Return a note if the period is suspiciously close to a CW keying harmonic.

    Uses ±10 % window around harmonics 1–3 only of each reference period.
    """
    for base_period in [CW_BEEP_PERIOD_S, CW_CALLSIGN_PERIOD_S, CW_BPSK_PERIOD_S]:
        for mult in CW_HARMONICS:
            cw_p = base_period * mult
            if abs(period_s - cw_p) / cw_p <= CW_ALIAS_WINDOW:
                return f"possible_CW_alias(base={base_period}s×{mult})"
    return ""


def detrend_poly(t: np.ndarray, y: np.ndarray, deg: int = 1) -> np.ndarray:
    """Subtract a polynomial trend from y(t). Returns residuals."""
    coeffs = np.polyfit(t, y, deg)
    return y - np.polyval(coeffs, t)


def run_ls(
    t: np.ndarray,
    y: np.ndarray,
    period_min_s: float = 1.0,
    period_max_s: float = 300.0,
    samples_per_peak: int = 10,
    detrend_deg: int = 0,        # 0 = subtract mean only, 1 = linear, 2 = quadratic
) -> dict:
    """
    Run Lomb-Scargle on (t [s], y [dB]).
    Returns dict with keys: frequency, power, top3, fap_peak, peak_power.
    """
    freq_min = 1.0 / period_max_s
    freq_max = 1.0 / period_min_s
    n_pts = len(t)

    # Detrend if requested (removes slow orbital/elevation systematic)
    if detrend_deg > 0:
        y = detrend_poly(t, y, deg=detrend_deg)

    if ASTROPY_OK:
        ls = LombScargle(t, y)
        # Use astropy autopower with our freq bounds — samples_per_peak controls resolution
        frequency, power = ls.autopower(
            minimum_frequency=freq_min,
            maximum_frequency=freq_max,
            samples_per_peak=samples_per_peak,
        )
        # False-alarm probability for the global peak — use Baluev (analytical, non-zero)
        peak_power = float(power.max())
        try:
            fap_peak = float(ls.false_alarm_probability(peak_power, method="baluev",
                                                         maximum_frequency=freq_max,
                                                         minimum_frequency=freq_min,
                                                         samples_per_peak=samples_per_peak))
        except Exception:
            fap_peak = float("nan")
    else:
        # Fallback: scipy.signal.lombscargle (no built-in FAP)
        frequency = np.linspace(freq_min, freq_max, int(samples_per_peak * n_pts))
        ang_freq = 2 * np.pi * frequency
        y_norm = y - y.mean()
        power = np.zeros(len(ang_freq))
        from scipy.signal import lombscargle as _sp_ls
        power = _sp_ls(t, y_norm, ang_freq, normalize=True)
        peak_power = float(power.max())
        fap_peak = float("nan")

    period = 1.0 / frequency

    # Find top-3 peaks using scipy find_peaks on the power spectrum
    peak_idx, peak_props = find_peaks(power, height=0)
    if len(peak_idx) < 1:
        # Edge-case: no detected peaks (monotone?), just take global max
        peak_idx = np.array([int(np.argmax(power))])

    sorted_peak_idx = peak_idx[np.argsort(power[peak_idx])[::-1]]
    top3_idx = sorted_peak_idx[:3]

    top3 = []
    for idx in top3_idx:
        p_s = float(period[idx])
        pw = float(power[idx])
        cw_flag = _flag_cw_alias(p_s)
        # Individual FAP for each top-3 peak
        if ASTROPY_OK:
            try:
                fap_i = float(ls.false_alarm_probability(pw, method="baluev",
                                                           maximum_frequency=freq_max,
                                                           minimum_frequency=freq_min,
                                                           samples_per_peak=samples_per_peak))
            except Exception:
                fap_i = float("nan")
        else:
            fap_i = float("nan")
        top3.append({
            "period_s": round(p_s, 4),
            "frequency_hz": round(float(frequency[idx]), 6),
            "power": round(pw, 6),
            "fap": fap_i,
            "cw_alias_flag": cw_flag,
        })

    return {
        "frequency": frequency,
        "power": power,
        "period": period,
        "peak_power": peak_power,
        "fap_peak": fap_peak,
        "top3": top3,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_obs(
    obs_id: int,
    station: str,
    t_all: np.ndarray,
    y_all: np.ndarray,
    is_gap: np.ndarray,
    ls_result: dict,
    output_path: Path,
    *,
    label: str = "gap-masked",
    ls_result_interp: dict | None = None,
    amplitude_quality: float = float("nan"),
    frac_gap: float = float("nan"),
    longest_gap_s: float = float("nan"),
) -> None:
    """Two-panel plot: amplitude series (top) + periodogram (bottom)."""
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.6])
    ax_amp = fig.add_subplot(gs[0])
    ax_ls = fig.add_subplot(gs[1])

    # --- Amplitude time series ---
    non_gap_mask = is_gap == 0
    ax_amp.plot(t_all[non_gap_mask], y_all[non_gap_mask], lw=0.8, color="steelblue",
                label="A_corr (valid)")
    if is_gap.any():
        ax_amp.plot(t_all[~non_gap_mask], y_all[~non_gap_mask], "x",
                    color="salmon", ms=3, alpha=0.7, label="gap (interp)")
    ax_amp.set_xlabel("Time [s]")
    ax_amp.set_ylabel("A_corr [dB]")
    ax_amp.set_title(
        f"obs {obs_id}  |  {station}  |  qual={amplitude_quality:.4f}  "
        f"frac_gap={frac_gap:.4f}  longest_gap={longest_gap_s:.1f} s"
    )
    ax_amp.legend(fontsize=8, loc="upper right")
    ax_amp.grid(True, alpha=0.3)

    # --- Periodogram ---
    period = ls_result["period"]
    power = ls_result["power"]
    ax_ls.semilogx(period, power, lw=0.8, color="steelblue", alpha=0.85, label=label)

    if ls_result_interp is not None:
        ax_ls.semilogx(ls_result_interp["period"], ls_result_interp["power"],
                       lw=0.8, color="darkorange", alpha=0.7, linestyle="--",
                       label="gap-interpolated")

    # Mark top-3 peaks for the primary result
    colors_pk = ["red", "orange", "gold"]
    for i, pk in enumerate(ls_result["top3"]):
        p_s = pk["period_s"]
        pw = pk["power"]
        fap_str = f"FAP={pk['fap']:.2e}" if not np.isnan(pk["fap"]) else ""
        cw_str = " ⚠" if pk["cw_alias_flag"] else ""
        ax_ls.axvline(p_s, color=colors_pk[i], lw=1.2, linestyle=":", alpha=0.9)
        ax_ls.annotate(
            f"#{i+1} {p_s:.1f}s\n{fap_str}{cw_str}",
            xy=(p_s, pw),
            xytext=(p_s * 1.15, pw * 0.97),
            fontsize=7,
            color=colors_pk[i],
            arrowprops=dict(arrowstyle="->", color=colors_pk[i], lw=0.8),
        )

    # CW keying reference lines
    for cw_p, cw_lbl in [(CW_BEEP_PERIOD_S, "10s beep"),
                          (CW_CALLSIGN_PERIOD_S, "40s cs"),
                          (CW_BPSK_PERIOD_S, "60s BPSK")]:
        ax_ls.axvline(cw_p, color="gray", lw=0.7, linestyle="--", alpha=0.5)
        ax_ls.text(cw_p * 1.02, ax_ls.get_ylim()[1] * 0.95, cw_lbl,
                   fontsize=6, color="gray", va="top")

    fap_str = (f"  FAP={ls_result['fap_peak']:.2e}" if not np.isnan(ls_result['fap_peak']) else "")
    ax_ls.set_xlabel("Period [s]")
    ax_ls.set_ylabel("LS Power (normalised)")
    ax_ls.set_title(
        f"Lomb-Scargle Periodogram ({label})  |  peak: "
        f"{ls_result['top3'][0]['period_s']:.2f} s{fap_str}"
    )
    ax_ls.legend(fontsize=8)
    ax_ls.grid(True, alpha=0.3)
    ax_ls.set_xlim(period.min() * 0.9, period.max() * 1.1)

    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_gap_comparison(
    obs_id: int,
    ls_masked: dict,
    ls_interp: dict,
    output_path: Path,
) -> None:
    """Side-by-side periodogram comparison for gap-masked vs interpolated for obs 351032."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    for ax, ls_r, label, color in [
        (axes[0], ls_masked, "gap-masked (excluded)", "steelblue"),
        (axes[1], ls_interp, "gap-interpolated (included)", "darkorange"),
    ]:
        ax.semilogx(ls_r["period"], ls_r["power"], lw=0.8, color=color)
        for i, pk in enumerate(ls_r["top3"]):
            ax.axvline(pk["period_s"], color=["red","orange","gold"][i],
                       lw=1.2, linestyle=":", alpha=0.9)
            ax.text(pk["period_s"] * 1.05, pk["power"] * 0.98,
                    f"{pk['period_s']:.1f}s", fontsize=7,
                    color=["red","orange","gold"][i])
        for cw_p, cw_lbl in [(10, "10s"), (40, "40s"), (60, "60s")]:
            ax.axvline(cw_p, color="gray", lw=0.7, linestyle="--", alpha=0.5)
        fap_s = (f"FAP={ls_r['fap_peak']:.2e}" if not np.isnan(ls_r['fap_peak']) else "")
        ax.set_title(f"obs {obs_id} — {label}\npeak={ls_r['top3'][0]['period_s']:.2f}s  {fap_s}")
        ax.set_xlabel("Period [s]")
        ax.set_ylabel("LS Power")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(ls_r["period"].min() * 0.9, ls_r["period"].max() * 1.1)

    fig.suptitle(
        f"obs {obs_id} — Gap sensitivity: does the 17.5 s gap change the peak period?",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def load_excluded_ids(path: Path | None, pipeline_dir: Path) -> set[int]:
    candidates = [path, pipeline_dir / "excluded_obs.csv"]
    for p in candidates:
        if p is None or not p.is_file():
            continue
        with open(p, encoding="utf-8") as fh:
            return {int(r["obs_id"]) for r in csv.DictReader(fh) if r.get("obs_id")}
    return set()


def load_leaderboard_v2(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["amplitude_quality"] = float(row["amplitude_quality"])
            row["obs_id"] = int(row["obs_id"])
            row["frac_gap"] = float(row["frac_gap"])
            row["longest_gap_s"] = float(row["longest_gap_s"])
            rows.append(row)
    rows.sort(key=lambda r: r["amplitude_quality"], reverse=True)
    return rows


def find_npz(pipeline_dir: Path, obs_id: int) -> Path | None:
    p = pipeline_dir / f"obs_{obs_id}" / f"amplitude_{obs_id}.npz"
    return p if p.is_file() else None


def analyse_obs(
    obs_id: int,
    npz_path: Path,
    *,
    gap_mode: str = "masked",   # "masked" | "interp"
    period_min_s: float = 1.0,
    period_max_s: float = 300.0,
    samples_per_peak: int = 10,
    detrend_deg: int = 1,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load .npz, prepare arrays according to gap_mode, run LS.
    Returns (ls_result, t_all, y_all, is_gap).
    """
    d = np.load(npz_path)
    t_all = d["t_rel_s"].astype(float)
    y_all = d["amplitude_pathloss_corrected"].astype(float)
    is_gap = d["is_gap"].astype(bool)

    if gap_mode == "masked":
        valid = ~is_gap
        t_used = t_all[valid]
        y_used = y_all[valid]
    else:
        t_used = t_all
        y_used = y_all

    ls_result = run_ls(t_used, y_used, period_min_s=period_min_s,
                       period_max_s=period_max_s, samples_per_peak=samples_per_peak,
                       detrend_deg=detrend_deg)
    # Attach pass length to result for credibility calculation
    # timestamps are stored in descending order (latest sample first)
    ls_result["pass_length_s"] = float(abs(t_all[-1] - t_all[0]))
    return ls_result, t_all, y_all, is_gap


def result_to_json(
    obs_id: int,
    lb_row: dict,
    ls_result: dict,
    gap_mode: str,
    n_used: int,
    n_total: int,
) -> dict:
    peak_period = ls_result["top3"][0]["period_s"] if ls_result["top3"] else float("nan")
    pass_length = ls_result.get("pass_length_s", float("nan"))
    n_cycles = (pass_length / peak_period
                if (not np.isnan(peak_period) and peak_period > 0) else float("nan"))
    credible = bool(n_cycles >= MIN_CYCLES_FOR_CREDIBLE) if not np.isnan(n_cycles) else False
    return {
        "obs_id": obs_id,
        "station": lb_row.get("station", ""),
        "amplitude_quality": lb_row.get("amplitude_quality", float("nan")),
        "R2": float(lb_row.get("R2", float("nan"))),
        "frac_gap": lb_row.get("frac_gap", float("nan")),
        "longest_gap_s": lb_row.get("longest_gap_s", float("nan")),
        "gap_mode": gap_mode,
        "n_samples_total": n_total,
        "n_samples_used": n_used,
        "pass_length_s": round(pass_length, 2) if not np.isnan(pass_length) else None,
        "n_cycles": round(n_cycles, 1) if not np.isnan(n_cycles) else None,
        "credible": credible,
        "peak_period_s": peak_period,
        "peak_frequency_hz": ls_result["top3"][0]["frequency_hz"] if ls_result["top3"] else float("nan"),
        "peak_power": ls_result["peak_power"],
        "fap_peak": ls_result["fap_peak"],
        "top3_peaks": ls_result["top3"],
    }


# ---------------------------------------------------------------------------
# Aggregate comparison table
# ---------------------------------------------------------------------------

def write_aggregate_csv(rows: list[dict], output_path: Path) -> None:
    fields = [
        "obs_id", "station", "amplitude_quality", "frac_gap", "longest_gap_s",
        "gap_mode", "n_samples_used", "pass_length_s", "n_cycles", "credible",
        "peak_period_s", "peak_frequency_hz", "peak_power", "fap_peak",
        "peak2_period_s", "peak2_fap",
        "peak3_period_s", "peak3_fap",
        "peak1_cw_alias", "peak2_cw_alias", "peak3_cw_alias",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            top3 = r.get("top3_peaks", [])
            def _pk(i, key, default=float("nan")):
                return top3[i].get(key, default) if len(top3) > i else default
            w.writerow({
                "obs_id": r["obs_id"],
                "station": r["station"],
                "amplitude_quality": round(float(r["amplitude_quality"]), 6),
                "frac_gap": round(float(r["frac_gap"]), 4),
                "longest_gap_s": round(float(r["longest_gap_s"]), 3),
                "gap_mode": r["gap_mode"],
                "n_samples_used": r["n_samples_used"],
                "pass_length_s": r.get("pass_length_s", ""),
                "n_cycles": r.get("n_cycles", ""),
                "credible": r.get("credible", ""),
                "peak_period_s": round(float(r["peak_period_s"]), 3),
                "peak_frequency_hz": round(float(r["peak_frequency_hz"]), 6),
                "peak_power": round(float(r["peak_power"]), 6),
                "fap_peak": r["fap_peak"],
                "peak2_period_s": round(float(_pk(1, "period_s")), 3),
                "peak2_fap": _pk(1, "fap"),
                "peak3_period_s": round(float(_pk(2, "period_s")), 3),
                "peak3_fap": _pk(2, "fap"),
                "peak1_cw_alias": _pk(0, "cw_alias_flag", ""),
                "peak2_cw_alias": _pk(1, "cw_alias_flag", ""),
                "peak3_cw_alias": _pk(2, "cw_alias_flag", ""),
            })


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(agg_rows: list[dict], gap_sensitivity: dict | None) -> None:
    print("\n" + "=" * 115)
    print("LOMB-SCARGLE TUMBLE ANALYSIS SUMMARY  (linear detrended, gap-masked)")
    print("=" * 115)
    hdr = (f"{'obs_id':>12}  {'station':<28}  {'qual':>6}  {'peak_P(s)':>9}  "
           f"{'n_cyc':>5}  {'credible':>8}  {'FAP':>10}  {'CW alias':<30}")
    print(hdr)
    print("-" * 115)
    primary_rows = [r for r in agg_rows if r.get("gap_mode") != "interp"]
    for r in primary_rows:
        fap = r["fap_peak"]
        fap_str = f"{fap:.2e}" if (isinstance(fap, (int, float)) and not np.isnan(fap)) else "   n/a"
        note = (r["top3_peaks"][0].get("cw_alias_flag", "") if r["top3_peaks"] else "")
        n_cyc = r.get("n_cycles", "")
        n_cyc_s = f"{n_cyc:.1f}" if isinstance(n_cyc, (int, float)) and not np.isnan(n_cyc) else "—"
        cred = "YES" if r.get("credible") else "no"
        cred_mark = " ✓" if r.get("credible") else "  "
        print(
            f"  {r['obs_id']:>10}  {r['station'][:28]:<28}  "
            f"{float(r['amplitude_quality']):>6.4f}  "
            f"{float(r['peak_period_s']):>9.2f}  {n_cyc_s:>5}  "
            f"{cred:>8}{cred_mark}  {fap_str:>10}  {note:<30}"
        )
    print("=" * 115)

    if gap_sensitivity:
        p_masked = gap_sensitivity["masked"]["peak_period_s"]
        p_interp = gap_sensitivity["interp"]["peak_period_s"]
        delta = abs(p_masked - p_interp)
        fap_m = gap_sensitivity["masked"]["fap_peak"]
        fap_i = gap_sensitivity["interp"]["fap_peak"]
        print(f"\nobs {gap_sensitivity['masked']['obs_id']} gap-sensitivity "
              f"(longest_gap = {gap_sensitivity['masked']['longest_gap_s']} s):")
        print(f"  masked (gap excluded)    → peak period = {p_masked:.2f} s  "
              f"  FAP = {fap_m:.2e}")
        print(f"  interp (gap included)    → peak period = {p_interp:.2f} s  "
              f"  FAP = {fap_i:.2e}")
        print(f"  Δ period = {delta:.3f} s  → {gap_sensitivity.get('verdict', '?')}")

    # Cross-check: credible peaks only
    credible_rows = [r for r in primary_rows if r.get("credible")]
    periods_cred = [r["peak_period_s"] for r in credible_rows
                    if isinstance(r["peak_period_s"], (int, float)) and not np.isnan(r["peak_period_s"])]
    if periods_cred:
        arr = np.array(periods_cred)
        med = np.median(arr)
        n_agree = int(np.sum(np.abs(arr - med) / med <= 0.20))
        print(f"\nCredible detections ({len(arr)} with ≥{MIN_CYCLES_FOR_CREDIBLE} cycles):")
        print(f"  periods = {np.round(arr, 1).tolist()}")
        print(f"  median = {med:.2f} s,  std = {np.std(arr):.2f} s,  range = [{arr.min():.2f}, {arr.max():.2f}] s")
        print(f"  {n_agree}/{len(arr)} within ±20 % of median → "
              f"{'consistent tumble rate' if n_agree >= max(2, int(len(arr)*0.6)) else 'scattered — check harmonics or rate evolution'}")
    elif primary_rows:
        print(f"\nNo credible detections (≥{MIN_CYCLES_FOR_CREDIBLE} cycles) found. "
              "Consider longer observations or narrower period search range.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Lomb-Scargle tumble analysis for MOVE-II observations.")
    ap.add_argument("--pipeline-dir", type=Path, default=Path("data/amplitude_pipeline"),
                    help="Directory containing leaderboard_v2.csv and obs_*/ subdirs")
    ap.add_argument("--output-dir", type=Path, default=Path("data/lomb_scargle"),
                    help="Output directory for plots, JSON, and CSV")
    ap.add_argument("--top-n", type=int, default=10,
                    help="Number of top observations by amplitude_quality to analyse")
    ap.add_argument("--period-min-s", type=float, default=1.0,
                    help="Minimum period to search [s]")
    ap.add_argument("--period-max-s", type=float, default=300.0,
                    help="Maximum period to search [s]")
    ap.add_argument("--samples-per-peak", type=int, default=10,
                    help="Astropy LombScargle samples_per_peak (frequency resolution)")
    ap.add_argument("--detrend-deg", type=int, default=1,
                    help="Polynomial detrend degree before LS (0=mean-subtract, 1=linear, 2=quadratic, 3=cubic)")
    ap.add_argument("--exclude-csv", type=Path, default=None,
                    help="CSV with obs_id column — excluded from analysis (default: pipeline/excluded_obs.csv if present)")
    ap.add_argument("--gap351032-id", type=int, default=351032,
                    help="Observation ID to run gap-sensitivity analysis on")
    args = ap.parse_args()

    pipeline_dir = args.pipeline_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded = load_excluded_ids(args.exclude_csv, pipeline_dir)
    if excluded:
        print(f"  Excluding {len(excluded)} obs from analysis: {sorted(excluded)}")

    lb_path = pipeline_dir / "leaderboard_v2.csv"
    if not lb_path.is_file():
        raise FileNotFoundError(f"leaderboard_v2.csv not found at {lb_path}")

    all_lb = load_leaderboard_v2(lb_path)
    if excluded:
        all_lb = [r for r in all_lb if r["obs_id"] not in excluded]
    top_rows = all_lb[: args.top_n]

    # Ensure obs 351032 is loaded (may not be in top N)
    gap_obs_id = args.gap351032_id
    all_obs_ids_to_run = [r["obs_id"] for r in top_rows]
    lb_by_id = {r["obs_id"]: r for r in all_lb}

    # Build merged list: top-N + always include gap_obs_id
    if gap_obs_id not in all_obs_ids_to_run:
        if gap_obs_id in lb_by_id:
            print(f"  obs {gap_obs_id} not in top-{args.top_n}; adding it for gap-sensitivity analysis.")
        else:
            print(f"  WARNING: obs {gap_obs_id} not found in leaderboard — skipping gap-sensitivity.")
            gap_obs_id = None

    aggregate_json: list[dict] = []

    print(f"\nRunning LS analysis on top-{args.top_n} observations …")
    print(f"  Period range: {args.period_min_s:.1f} s — {args.period_max_s:.1f} s")
    print(f"  Output dir:   {output_dir}\n")

    for rank, lb_row in enumerate(top_rows, 1):
        obs_id = lb_row["obs_id"]
        npz_path = find_npz(pipeline_dir, obs_id)
        if npz_path is None:
            print(f"  [{rank:02d}] obs {obs_id}  SKIPPED — .npz not found")
            continue

        print(f"  [{rank:02d}] obs {obs_id}  station={lb_row['station'][:30]}  "
              f"qual={lb_row['amplitude_quality']:.4f} …", end=" ")

        ls_result, t_all, y_all, is_gap = analyse_obs(
            obs_id, npz_path,
            gap_mode="masked",
            period_min_s=args.period_min_s,
            period_max_s=args.period_max_s,
            samples_per_peak=args.samples_per_peak,
            detrend_deg=args.detrend_deg,
        )

        n_total = len(t_all)
        n_used = int((~is_gap).sum())

        # If this is also the gap-sensitivity obs, run the interp variant
        ls_interp = None
        if obs_id == gap_obs_id:
            ls_interp, _, _, _ = analyse_obs(
                obs_id, npz_path,
                gap_mode="interp",
                period_min_s=args.period_min_s,
                period_max_s=args.period_max_s,
                samples_per_peak=args.samples_per_peak,
                detrend_deg=args.detrend_deg,
            )

        # Plot
        plot_path = output_dir / f"obs_{obs_id}_ls.png"
        plot_obs(
            obs_id, lb_row["station"],
            t_all, y_all, is_gap,
            ls_result, plot_path,
            label="gap-masked",
            ls_result_interp=ls_interp,
            amplitude_quality=lb_row["amplitude_quality"],
            frac_gap=lb_row["frac_gap"],
            longest_gap_s=lb_row["longest_gap_s"],
        )

        # JSON summary
        summary = result_to_json(obs_id, lb_row, ls_result, "masked", n_used, n_total)
        summary_path = output_dir / f"obs_{obs_id}_ls.json"
        # Serialisable copy (drop numpy arrays)
        def _ser(v):
            if isinstance(v, float) and np.isnan(v):
                return None
            return v
        serialisable = {k: _ser(v) for k, v in summary.items() if k not in ("frequency","power","period")}
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else float(x))

        aggregate_json.append(summary)
        top_pk = ls_result["top3"][0] if ls_result["top3"] else {}
        print(f"peak={top_pk.get('period_s', '?'):.2f}s  "
              f"FAP={top_pk.get('fap', float('nan')):.2e}  "
              f"{'⚠ CW?' if top_pk.get('cw_alias_flag') else 'ok'}")

    # -------------------------------------------------------------------
    # Gap-sensitivity analysis for obs 351032 (if not already in top-N)
    # -------------------------------------------------------------------
    gap_sensitivity = None
    if gap_obs_id is not None and gap_obs_id not in [r["obs_id"] for r in top_rows]:
        npz_path = find_npz(pipeline_dir, gap_obs_id)
        if npz_path:
            print(f"\n  Running gap-sensitivity for obs {gap_obs_id} …")
            lb_row_g = lb_by_id[gap_obs_id]

            ls_masked, t_all, y_all, is_gap = analyse_obs(
                gap_obs_id, npz_path, gap_mode="masked",
                period_min_s=args.period_min_s,
                period_max_s=args.period_max_s,
                samples_per_peak=args.samples_per_peak,
                detrend_deg=args.detrend_deg,
            )
            ls_interp, _, _, _ = analyse_obs(
                gap_obs_id, npz_path, gap_mode="interp",
                period_min_s=args.period_min_s,
                period_max_s=args.period_max_s,
                samples_per_peak=args.samples_per_peak,
                detrend_deg=args.detrend_deg,
            )

            n_total = len(t_all)
            n_used_masked = int((~is_gap).sum())

            for ls_r, mode, n_u in [(ls_masked, "masked", n_used_masked),
                                     (ls_interp, "interp", n_total)]:
                s = result_to_json(gap_obs_id, lb_row_g, ls_r, mode, n_u, n_total)
                aggregate_json.append(s)

            # Combined plot saved to obs_<id>_ls.png
            plot_obs(
                gap_obs_id, lb_row_g["station"],
                t_all, y_all, is_gap,
                ls_masked,
                output_dir / f"obs_{gap_obs_id}_ls.png",
                label="gap-masked",
                ls_result_interp=ls_interp,
                amplitude_quality=lb_row_g["amplitude_quality"],
                frac_gap=lb_row_g["frac_gap"],
                longest_gap_s=lb_row_g["longest_gap_s"],
            )
            gap_sensitivity = {
                "masked": result_to_json(gap_obs_id, lb_row_g, ls_masked, "masked", n_used_masked, n_total),
                "interp": result_to_json(gap_obs_id, lb_row_g, ls_interp, "interp", n_total, n_total),
            }
        else:
            print(f"  WARNING: .npz for obs {gap_obs_id} not found — skipping gap-sensitivity.")

    elif gap_obs_id is not None and gap_obs_id in [r["obs_id"] for r in top_rows]:
        # Already processed above; run the interp variant for the sensitivity JSON
        npz_path = find_npz(pipeline_dir, gap_obs_id)
        lb_row_g = lb_by_id[gap_obs_id]
        if npz_path:
            ls_interp_only, t_all, y_all, is_gap = analyse_obs(
                gap_obs_id, npz_path, gap_mode="interp",
                period_min_s=args.period_min_s,
                period_max_s=args.period_max_s,
                samples_per_peak=args.samples_per_peak,
                detrend_deg=args.detrend_deg,
            )
            n_total = len(t_all)
            s_interp = result_to_json(gap_obs_id, lb_row_g, ls_interp_only, "interp", n_total, n_total)
            # Find the masked result we already computed
            s_masked = next((r for r in aggregate_json if r["obs_id"] == gap_obs_id), None)
            if s_masked:
                aggregate_json.append(s_interp)
                gap_sensitivity = {"masked": s_masked, "interp": s_interp}
                # Comparison plot
                ls_masked_reloaded, _, _, _ = analyse_obs(
                    gap_obs_id, npz_path, gap_mode="masked",
                    period_min_s=args.period_min_s,
                    period_max_s=args.period_max_s,
                    samples_per_peak=args.samples_per_peak,
                    detrend_deg=args.detrend_deg,
                )
                plot_gap_comparison(
                    gap_obs_id, ls_masked_reloaded, ls_interp_only,
                    output_dir / f"{gap_obs_id}_gap_comparison.png",
                )

    # -------------------------------------------------------------------
    # Gap-sensitivity JSON
    # -------------------------------------------------------------------
    if gap_sensitivity:
        sens_path = output_dir / f"{gap_obs_id}_gap_sensitivity.json"
        p_masked = gap_sensitivity["masked"]["peak_period_s"]
        p_interp = gap_sensitivity["interp"]["peak_period_s"]
        delta = abs(p_masked - p_interp)
        gap_sensitivity["delta_period_s"] = round(delta, 4)
        gap_sensitivity["verdict"] = (
            "gap_safe_to_include" if delta < 2.0 else "gap_shifts_peak_be_cautious"
        )
        with open(sens_path, "w", encoding="utf-8") as fh:
            def _jdefault(x):
                if isinstance(x, float) and np.isnan(x):
                    return None
                return float(x) if isinstance(x, (np.floating, np.integer)) else x
            json.dump(gap_sensitivity, fh, indent=2, default=_jdefault)
        print(f"\n  Gap-sensitivity JSON → {sens_path}")

    # -------------------------------------------------------------------
    # Aggregate CSV
    # -------------------------------------------------------------------
    # Flatten top3 into the aggregate list
    flat_agg = []
    for r in aggregate_json:
        flat_agg.append(r)
    agg_csv_path = output_dir / "aggregate_ls.csv"
    write_aggregate_csv(flat_agg, agg_csv_path)
    print(f"  Aggregate CSV  → {agg_csv_path}")

    # -------------------------------------------------------------------
    # Multi-panel summary figure (all top-N periodograms on one page)
    # -------------------------------------------------------------------
    primary_results = [r for r in aggregate_json if r.get("gap_mode") == "masked"]
    n_panels = len(primary_results)
    if n_panels > 0:
        ncols = 2
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), constrained_layout=True)
        axes_flat = np.array(axes).flatten() if n_panels > 1 else [axes]

        for i, r in enumerate(primary_results):
            ax = axes_flat[i]
            obs_id = r["obs_id"]
            npz_path = find_npz(pipeline_dir, obs_id)
            if npz_path is None:
                ax.set_visible(False)
                continue
            ls_r, _, _, _ = analyse_obs(
                obs_id, npz_path, gap_mode="masked",
                period_min_s=args.period_min_s,
                period_max_s=args.period_max_s,
                samples_per_peak=args.samples_per_peak,
                detrend_deg=args.detrend_deg,
            )
            ax.semilogx(ls_r["period"], ls_r["power"], lw=0.7, color="steelblue")
            for j, pk in enumerate(ls_r["top3"]):
                ax.axvline(pk["period_s"], color=["red","orange","gold"][j],
                           lw=1.0, linestyle=":", alpha=0.85)
                ax.text(pk["period_s"] * 1.03, pk["power"] * 0.98,
                        f"{pk['period_s']:.1f}s", fontsize=6,
                        color=["red","orange","gold"][j])
            for cw_p in [10, 40, 60]:
                ax.axvline(cw_p, color="gray", lw=0.5, linestyle="--", alpha=0.4)
            station_short = r["station"][:22]
            peak_p = r["top3_peaks"][0]["period_s"] if r["top3_peaks"] else float("nan")
            fap = r["fap_peak"]
            fap_s = f"FAP={fap:.1e}" if not (isinstance(fap, float) and np.isnan(fap)) else ""
            ax.set_title(f"obs {obs_id}\n{station_short}  peak={peak_p:.1f}s  {fap_s}",
                         fontsize=8)
            ax.set_xlabel("Period [s]", fontsize=7)
            ax.set_ylabel("LS Power", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(args.period_min_s * 0.9, args.period_max_s * 1.1)

        for j in range(n_panels, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("MOVE-II Lomb-Scargle Periodograms — Top-N observations (gap-masked)",
                     fontsize=12, fontweight="bold")
        summary_fig_path = output_dir / "all_periodograms.png"
        fig.savefig(summary_fig_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Summary figure → {summary_fig_path}")

    print_summary(aggregate_json, gap_sensitivity)
    print(f"Done. All outputs in {output_dir}\n")


if __name__ == "__main__":
    main()
