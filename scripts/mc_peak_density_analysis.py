#!/usr/bin/env python3
"""Collect LS peak periods from MC aliasing trials; Poisson significance + KDE histogram."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import mc_aliasing_165s as mc  # noqa: E402
from mc_aliasing_165s import (  # noqa: E402
    BATCH_SIZE,
    COINC_WINDOW_S,
    NEAR165_ZONE_S,
    N_TRIALS,
    OBS_A,
    OBS_B,
    SMOOTHNESS_SETTINGS,
    HARMONICS,
    TARGET_PERIOD_S,
    make_cholesky,
    make_freq_grid,
    precompute_ls_matrices,
    run_mc_cell,
    near165_coincidence_mask,
    _find_pipeline_dir,
    load_valid_timestamps,
)

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "archival_tle_validation" / "gp_history"


def poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for Poisson(lam)."""
    if k <= 0:
        return 1.0
    cdf = sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k))
    return 1.0 - cdf


def uniform_null_lambda_per_trial(period_min: float = 1.0, period_max: float = 300.0) -> float:
    """
    P(near-165 hit) under i.i.d. uniform peaks on [period_min, period_max]:
      peak_A in [163,167] AND |peak_B - peak_A| < COINC_WINDOW_S
    """
    zone_lo = TARGET_PERIOD_S - NEAR165_ZONE_S
    zone_hi = TARGET_PERIOD_S + NEAR165_ZONE_S
    span = period_max - period_min
    p_a = (zone_hi - zone_lo) / span
    p_b = COINC_WINDOW_S / span
    return p_a * p_b


def empirical_null_p_hit_per_trial(
    fraction_A_in_zone: float,
    fraction_B_in_zone: float,
    *,
    coinc_window_s: float = COINC_WINDOW_S,
    near165_zone_s: float = NEAR165_ZONE_S,
) -> float:
    """
    P(near-165 coincidence) under independent synthetic passes whose marginal
    near-165 landing rates match the MC KDE fractions:

      P(A ∈ zone) × P(B ∈ zone) × P(|A−B| < w | A,B ∈ zone)

    With uniform A,B on [165−zone, 165+zone], P(|A−B| < w) ≈ 2w / zone_width.
    """
    zone_width = 2.0 * near165_zone_s
    p_agree_given_both = min(1.0, 2.0 * coinc_window_s / zone_width)
    return fraction_A_in_zone * fraction_B_in_zone * p_agree_given_both


def main() -> None:
    n_trials = 120_000
    mc.N_TRIALS = n_trials
    pipeline_dir = _find_pipeline_dir()
    t_A = load_valid_timestamps(OBS_A, pipeline_dir)
    t_B = load_valid_timestamps(OBS_B, pipeline_dir)

    # Confirm timestamps vs any corrected NPZ
    gp_npz = OUTPUT / "amplitude_pipeline" / f"obs_{OBS_A}" / f"amplitude_{OBS_A}.npz"
    ts_note = "unchanged (TLE-independent t_rel_s from waterfall OCR)"
    if gp_npz.is_file():
        z_old = np.load(pipeline_dir / f"obs_{OBS_A}" / f"amplitude_{OBS_A}.npz")
        z_new = np.load(gp_npz)
        diff = np.max(np.abs(z_old["t_rel_s"] - z_new["t_rel_s"]))
        ts_note = f"max |Δt_rel| old vs gp_history NPZ = {diff:.2e} s (identical: {diff==0})"

    freqs_A, per_A = make_freq_grid(t_A)
    freqs_B, per_B = make_freq_grid(t_B)
    cos_A, sin_A = precompute_ls_matrices(t_A, freqs_A)
    cos_B, sin_B = precompute_ls_matrices(t_B, freqs_B)

    T_MAX = max(t_A[-1], t_B[-1]) * 1.01
    t_dense = np.linspace(0.0, T_MAX, mc.N_DENSE)
    t_ctrl = np.linspace(0.0, T_MAX, mc.N_CTRL)
    rng = np.random.default_rng(seed=20241204)

    all_peak_A: list[float] = []
    all_peak_B: list[float] = []
    cell_stats: list[dict] = []
    n_near165_total = 0

    print(f"MC peak collection: {n_trials} trials × {len(SMOOTHNESS_SETTINGS)*len(HARMONICS)} cells")
    print(f"Timestamp check: {ts_note}")
    t0 = time.perf_counter()

    for smooth_label, cfg in SMOOTHNESS_SETTINGS.items():
        L = make_cholesky(cfg["l_s"], cfg["sigma"], t_ctrl)
        for harm_label, harm_N in HARMONICS.items():
            pk_A, pk_B, _saved = run_mc_cell(
                t_A, t_B, cos_A, sin_A, per_A, cos_B, sin_B, per_B,
                L, t_ctrl, t_dense, harm_N, rng, f"{smooth_label}|{harm_label}",
            )
            all_peak_A.append(pk_A)
            all_peak_B.append(pk_B)
            near = near165_coincidence_mask(pk_A, pk_B)
            n_hit = int(near.sum())
            n_near165_total += n_hit
            cell_stats.append({
                "smoothness": smooth_label,
                "harmonic": harm_label,
                "n_trials": n_trials,
                "coinc_window_s": COINC_WINDOW_S,
                "near165_zone_s": NEAR165_ZONE_S,
                "near165_hits": n_hit,
                "hit_rate": n_hit / n_trials,
            })
            print(f"  {smooth_label} | {harm_label}: near165 hits = {n_hit}/{n_trials}")

    elapsed = time.perf_counter() - t0
    peaks_A = np.concatenate(all_peak_A)
    peaks_B = np.concatenate(all_peak_B)

    # Histogram 1s bins, 1-300s
    bins = np.arange(1, 301, 1)
    hist_A, _ = np.histogram(peaks_A, bins=bins)
    hist_B, _ = np.histogram(peaks_B, bins=bins)
    bin_centers = bins[:-1] + 0.5
    total_A = len(peaks_A)
    density_163_167_A = peaks_A[(peaks_A >= 163) & (peaks_A <= 167)].size / total_A
    density_163_167_B = peaks_B[(peaks_B >= 163) & (peaks_B <= 167)].size / len(peaks_B)

    lam_trial_uniform = uniform_null_lambda_per_trial()
    lam_cell_uniform = lam_trial_uniform * n_trials
    lam_total_uniform = lam_cell_uniform * len(cell_stats)
    p_poisson_uniform = poisson_sf(n_near165_total, lam_total_uniform)

    p_hit_empirical = empirical_null_p_hit_per_trial(density_163_167_A, density_163_167_B)
    lam_cell_empirical = p_hit_empirical * n_trials
    lam_total_empirical = lam_cell_empirical * len(cell_stats)
    p_poisson_empirical = poisson_sf(n_near165_total, lam_total_empirical)

    result = {
        "timestamp_check": ts_note,
        "mc_design": {
            "n_trials_per_cell": n_trials,
            "n_cells": len(cell_stats),
            "total_trials": n_trials * len(cell_stats),
            "coinc_window_s": COINC_WINDOW_S,
            "near165_zone_s": NEAR165_ZONE_S,
            "target_period_s": TARGET_PERIOD_S,
            "ls_period_range_s": [1.0, 300.0],
        },
        "observed_near165_hits_total": n_near165_total,
        "uniform_null": {
            "description": "Naive baseline: peaks uniform on [1,300]s, coincidence window applied to full span",
            "p_hit_per_trial": lam_trial_uniform,
            "lambda_expected_per_cell": lam_cell_uniform,
            "lambda_expected_total_12_cells": lam_total_uniform,
            "poisson_p_ge_observed": p_poisson_uniform,
        },
        "empirical_null": {
            "description": (
                "Marginal near-165 landing fractions from MC KDE × independent-pass "
                "agreement probability within coincidence window"
            ),
            "fraction_A_in_163_167": density_163_167_A,
            "fraction_B_in_163_167": density_163_167_B,
            "p_agree_given_both_in_zone": min(1.0, 2.0 * COINC_WINDOW_S / (2.0 * NEAR165_ZONE_S)),
            "p_hit_per_trial": p_hit_empirical,
            "lambda_expected_per_cell": lam_cell_empirical,
            "lambda_expected_total_12_cells": lam_total_empirical,
            "poisson_p_ge_observed": p_poisson_empirical,
        },
        "empirical_peak_density": {
            "n_peaks_A": int(total_A),
            "n_peaks_B": int(len(peaks_B)),
            "fraction_A_in_163_167": density_163_167_A,
            "fraction_B_in_163_167": density_163_167_B,
            "histogram_bin_width_s": 1.0,
            "histogram_range_s": [1, 300],
        },
        "per_cell": cell_stats,
        "elapsed_s": elapsed,
    }

    out_dir = OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mc_poisson_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "mc_peak_histograms.npz",
        bin_centers=bin_centers,
        hist_peak_A=hist_A,
        hist_peak_B=hist_B,
        peaks_A=peaks_A.astype(np.float32),
        peaks_B=peaks_B.astype(np.float32),
    )
    print(f"\nTotal near165 hits: {n_near165_total}")
    print(f"λ_expected (uniform null, 12 cells): {lam_total_uniform:.4f}")
    print(f"P(Poisson ≥ {n_near165_total} | uniform λ): {p_poisson_uniform:.4f}")
    print(f"λ_expected (empirical null, 12 cells): {lam_total_empirical:.6f}")
    print(f"P(Poisson ≥ {n_near165_total} | empirical λ): {p_poisson_empirical:.6e}")
    print(f"Empirical frac peak_A ∈ [163,167]: {density_163_167_A:.6f}")
    print(f"Empirical frac peak_B ∈ [163,167]: {density_163_167_B:.6f}")
    print(f"Elapsed: {elapsed:.0f} s")
    print(f"Wrote {out_dir / 'mc_poisson_analysis.json'}")


if __name__ == "__main__":
    from lomb_scargle_tumble import find_npz  # noqa: F401 — used in ts check
    main()
