#!/usr/bin/env python3
"""
Monte Carlo aliasing test: could the 165.0 s Lomb-Scargle agreement between
obs 349779 (Dec 4, Binar-SN-01) and obs 359214 (Dec 11, KO2F-VHF-1) arise
by chance from aliasing of an unknown, evolving satellite spin rate?

ALGORITHM
----------
1. Load real (non-gap) timestamps from the amplitude-pipeline NPZ files.
   Exact vectors — no interpolation, same as the main pipeline.

2. GP prior on log(f_spin(t)) with squared-exponential kernel.
   Frequency range: 0.3–2 Hz.  Four length scales: choppy → very smooth.

3. For N = 10 000 trials per (smoothness × harmonic) cell:
     a. Draw one GP trajectory f_spin(t).
     b. φ(t_k) = 2π ∫₀^{t_k} f_spin dτ at each pass's real timestamps.
     c. signal = cos(N_sym · φ) + Gaussian noise.
     d. Run Lomb-Scargle on the same frequency grid as the main pipeline.
     e. Record peak period for each pass independently.

4. Report (per cell):
     • P(|peak_A − peak_B| < 0.005 s)   — any period
     • P(same agreement AND that period falls within ±2 s of 165.0 s)
       (peaks must agree with each other first; zone check is wide, not razor-thin)

PERFORMANCE NOTE
-----------------
scipy.signal.lombscargle takes ~865 ms per call on this system for
7 683 frequencies × 814 observations — 57 h if called per-trial.
Solution: precompute cos/sin projection matrices once per pass, then replace
all individual LS calls with a single BLAS matrix-multiply per batch.
This "DFT-approximation LS" (no tau phase correction) finds the same peak
as proper LS in practice (peak position, not power scale, is what matters here).
Total wall time for N=10 000 × 12 cells: ~3–4 minutes.

OUTPUT (data/mc_aliasing/)
--------------------------
  coincidence_table.txt / .csv
  example_periodograms_{N1,N2,N4}.png
  spin_trajectory_examples.png
  top_coincident_trajectories.txt
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from lomb_scargle_tumble import find_npz  # noqa: E402  (existing pipeline helper)

# ── Configuration ──────────────────────────────────────────────────────────────

OBS_A = 349779   # Dec 4  2018, Binar-SN-01   (47% gap, 814 valid pts)
OBS_B = 359214   # Dec 11 2018, KO2F-VHF-1    (22% gap, 1203 valid pts)

# LS parameters — exact match to main pipeline (lomb_scargle_tumble.run_ls defaults)
PERIOD_MIN_S     = 1.0
PERIOD_MAX_S     = 300.0
SAMPLES_PER_PEAK = 10      # df = 1 / (baseline × samples_per_peak)

TARGET_PERIOD_S  = 165.0
COINC_WINDOW_S   = 0.005   # |peak_A − peak_B| — two passes must agree
NEAR165_ZONE_S   = 2.0     # matched agreement must fall within ±this of 165 s

# Spin-frequency prior
SPIN_F_MIN_HZ = 0.30       # period ~3.3 s
SPIN_F_MAX_HZ = 2.00       # period ~0.5 s
MU_LOG_F = 0.5 * (np.log(SPIN_F_MIN_HZ) + np.log(SPIN_F_MAX_HZ))  # geometric mean

NOISE_SCALE = 0.50         # additive Gaussian noise std (signal amplitude = 1)
N_TRIALS    = 10_000
BATCH_SIZE  = 400          # trajectories per batch (memory/speed tradeoff)

# GP discretisation
N_CTRL  = 55   # evenly-spaced GP control points over pass duration
N_DENSE = 8_000  # integration grid points (~0.10 s spacing over 800 s)

# Smoothness: GP squared-exponential kernel length scale (seconds) + log-f sigma
SMOOTHNESS_SETTINGS: dict[str, dict] = {
    "choppy  (l=30 s)":     {"l_s":  30.0, "sigma": 0.55},
    "moderate(l=120 s)":    {"l_s": 120.0, "sigma": 0.55},
    "smooth  (l=350 s)":    {"l_s": 350.0, "sigma": 0.45},
    "v-smooth(l=900 s)":    {"l_s": 900.0, "sigma": 0.30},
}

# Antenna-symmetry harmonic cases
HARMONICS: dict[str, int] = {
    "N=1 (cos φ)":   1,
    "N=2 (cos 2φ)":  2,
    "N=4 (cos 4φ)":  4,
}

N_TRAJ_EXAMPLES   = 5   # draws to show in spin-trajectory figure
N_PERIOD_EXAMPLES = 2   # periodogram pairs to show per smoothness in example figure
N_TOP_PRINT       = 6   # top coincident trajectories to print


# ── Data loading ───────────────────────────────────────────────────────────────

def _find_pipeline_dir() -> Path:
    for cand in [
        _SCRIPT_DIR.parent / "data" / "amplitude_pipeline",
        Path.cwd() / "data" / "amplitude_pipeline",
    ]:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "Cannot find data/amplitude_pipeline/. "
        "Run from the project root or scripts/ directory."
    )


def load_valid_timestamps(obs_id: int, pipeline_dir: Path) -> np.ndarray:
    """Non-gap timestamps from NPZ, sorted and zero-shifted. Reuses find_npz."""
    npz = find_npz(pipeline_dir, obs_id)
    if npz is None:
        raise FileNotFoundError(f"amplitude_{obs_id}.npz not found in {pipeline_dir}")
    d   = np.load(npz)
    t_v = np.sort(d["t_rel_s"].astype(float)[~d["is_gap"].astype(bool)])
    return t_v - t_v[0]


# ── LS frequency grid ──────────────────────────────────────────────────────────

def make_freq_grid(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Frequency and period arrays matching astropy autopower with SAMPLES_PER_PEAK.
    Returns (freqs_hz, periods_s); periods are in DECREASING order.
    """
    baseline = t[-1] - t[0]
    df       = 1.0 / (baseline * SAMPLES_PER_PEAK)
    freqs    = np.arange(1.0 / PERIOD_MAX_S, 1.0 / PERIOD_MIN_S + 0.5 * df, df)
    return freqs, 1.0 / freqs


# ── Vectorised batch LS (DFT approximation) ────────────────────────────────────

def precompute_ls_matrices(
    t_obs: np.ndarray,
    freqs_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute cos/sin projection matrices for vectorised Lomb-Scargle.

    power[b, f] ∝  (cos_mat[f, :] @ y[b, :])²  +  (sin_mat[f, :] @ y[b, :])²

    This is the DFT-approximation LS (no tau phase correction).  The peak
    POSITION matches proper LS to within the grid resolution for all practical
    signals; only the power scale differs.

    Returns cos_mat, sin_mat each of shape (N_freq, N_obs).
    """
    phase   = 2.0 * np.pi * np.outer(freqs_hz, t_obs)  # (N_freq, N_obs)
    return np.cos(phase), np.sin(phase)


def batch_ls_peak_periods(
    Y: np.ndarray,              # (batch, N_obs)  — mean-subtracted
    cos_mat: np.ndarray,        # (N_freq, N_obs)
    sin_mat: np.ndarray,        # (N_freq, N_obs)
    periods: np.ndarray,        # (N_freq,) decreasing
) -> np.ndarray:
    """
    Vectorised batch Lomb-Scargle peak-finding via matrix multiplication.
    Returns peak period for each trial: shape (batch,).
    """
    cos_proj = cos_mat @ Y.T    # (N_freq, batch)
    sin_proj = sin_mat @ Y.T    # (N_freq, batch)
    power    = cos_proj ** 2 + sin_proj ** 2   # (N_freq, batch)
    return periods[power.argmax(axis=0)]       # (batch,)


# ── GP trajectory machinery ────────────────────────────────────────────────────

def make_cholesky(l_s: float, sigma: float, t_ctrl: np.ndarray) -> np.ndarray:
    """Cholesky of squared-exponential GP covariance (with jitter)."""
    d2  = ((t_ctrl[:, None] - t_ctrl[None, :]) / l_s) ** 2
    K   = sigma ** 2 * np.exp(-0.5 * d2) + 1e-6 * np.eye(len(t_ctrl))
    return np.linalg.cholesky(K)


def draw_f_spin_batch(
    L: np.ndarray,
    t_ctrl: np.ndarray,
    t_dense: np.ndarray,
    rng: np.random.Generator,
    batch: int,
) -> np.ndarray:
    """
    Draw `batch` GP trajectories for log(f_spin), exponentiate, and
    linearly interpolate to the dense integration grid.
    Returns f_spin_dense: shape (batch, N_DENSE) in Hz.
    """
    z          = rng.standard_normal((len(t_ctrl), batch))
    log_f_ctrl = np.clip(
        MU_LOG_F + L @ z,
        np.log(SPIN_F_MIN_HZ), np.log(SPIN_F_MAX_HZ),
    ).T                                                # (batch, N_CTRL)

    # Vectorised linear interpolation: t_ctrl → t_dense
    bins  = np.searchsorted(t_ctrl, t_dense, side="right") - 1
    bins  = np.clip(bins, 0, len(t_ctrl) - 2)
    alpha = (t_dense - t_ctrl[bins]) / (t_ctrl[bins + 1] - t_ctrl[bins])

    log_f_dense = log_f_ctrl[:, bins] * (1.0 - alpha) + log_f_ctrl[:, bins + 1] * alpha
    return np.exp(log_f_dense)                         # (batch, N_DENSE)


def compute_phase_at_timestamps(
    f_dense: np.ndarray,   # (batch, N_DENSE)
    t_dense: np.ndarray,
    t_obs: np.ndarray,
) -> np.ndarray:
    """
    φ(t_obs) = 2π ∫₀^{t_obs} f_spin dτ for every trajectory.
    Returns phase: shape (batch, N_OBS).
    """
    dt          = np.diff(t_dense)
    f_mid       = 0.5 * (f_dense[:, :-1] + f_dense[:, 1:])
    cum_phase   = np.empty((len(f_dense), len(t_dense)))
    cum_phase[:, 0] = 0.0
    cum_phase[:, 1:] = 2.0 * np.pi * np.cumsum(f_mid * dt, axis=1)

    # Vectorised linear interpolation: t_dense → t_obs
    bins  = np.searchsorted(t_dense, t_obs, side="right") - 1
    bins  = np.clip(bins, 0, len(t_dense) - 2)
    alpha = (t_obs - t_dense[bins]) / (t_dense[bins + 1] - t_dense[bins])
    return cum_phase[:, bins] * (1.0 - alpha) + cum_phase[:, bins + 1] * alpha


# ── MC engine ─────────────────────────────────────────────────────────────────

def run_mc_cell(
    t_A: np.ndarray,
    t_B: np.ndarray,
    cos_A: np.ndarray, sin_A: np.ndarray, per_A: np.ndarray,
    cos_B: np.ndarray, sin_B: np.ndarray, per_B: np.ndarray,
    L: np.ndarray,
    t_ctrl: np.ndarray,
    t_dense: np.ndarray,
    harmonic_N: int,
    rng: np.random.Generator,
    cell_label: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, np.ndarray]]]:
    """
    Run N_TRIALS trials for one (smoothness, harmonic) cell.
    Returns (peak_A, peak_B, saved_coincident_trajectories).
    """
    peak_A = np.empty(N_TRIALS)
    peak_B = np.empty(N_TRIALS)
    saved: list[tuple[int, np.ndarray]] = []

    n_batches = (N_TRIALS + BATCH_SIZE - 1) // BATCH_SIZE
    t_cell = time.perf_counter()

    for b in range(n_batches):
        start = b * BATCH_SIZE
        end   = min(start + BATCH_SIZE, N_TRIALS)
        bsz   = end - start

        f_dense = draw_f_spin_batch(L, t_ctrl, t_dense, rng, bsz)

        phi_A = compute_phase_at_timestamps(f_dense, t_dense, t_A)
        phi_B = compute_phase_at_timestamps(f_dense, t_dense, t_B)

        sig_A = (np.cos(harmonic_N * phi_A)
                 + rng.normal(0.0, NOISE_SCALE, phi_A.shape))
        sig_B = (np.cos(harmonic_N * phi_B)
                 + rng.normal(0.0, NOISE_SCALE, phi_B.shape))

        # Mean-subtract
        sig_A -= sig_A.mean(axis=1, keepdims=True)
        sig_B -= sig_B.mean(axis=1, keepdims=True)

        # Vectorised batch LS
        pk_A = batch_ls_peak_periods(sig_A, cos_A, sin_A, per_A)
        pk_B = batch_ls_peak_periods(sig_B, cos_B, sin_B, per_B)

        peak_A[start:end] = pk_A
        peak_B[start:end] = pk_B

        # Save trajectories where passes agree AND that agreement is near 165 s
        near_165_mask = near165_coincidence_mask(pk_A, pk_B)
        for i in np.where(near_165_mask)[0]:
            saved.append((start + i, f_dense[i].copy()))

        # Progress every 5 batches
        if (b + 1) % 5 == 0 or (b + 1) == n_batches:
            elapsed = time.perf_counter() - t_cell
            done    = end / N_TRIALS
            eta     = (elapsed / done) * (1 - done) if done > 0 else 0
            print(
                f"    {cell_label}  [{end:>5}/{N_TRIALS}]  "
                f"near-165 hits so far: {len(saved)}  "
                f"ETA {eta:.0f}s",
                flush=True,
            )

    return peak_A, peak_B, saved


# ── Statistics ─────────────────────────────────────────────────────────────────

def near165_coincidence_mask(
    peak_A: np.ndarray,
    peak_B: np.ndarray,
) -> np.ndarray:
    """
    True when the two passes agree on a period AND that agreement is near 165 s.

    Step 1: |peak_A − peak_B| < COINC_WINDOW_S  (they agree with each other)
    Step 2: |peak_A − 165| < NEAR165_ZONE_S     (that agreement is in the 165 s zone)

    When matched, peak_A ≈ peak_B so either peak works for the zone check.
    """
    matched = np.abs(peak_A - peak_B) < COINC_WINDOW_S
    near_165_zone = np.abs(peak_A - TARGET_PERIOD_S) < NEAR165_ZONE_S
    return matched & near_165_zone


def cell_stats(peak_A: np.ndarray, peak_B: np.ndarray) -> dict:
    n         = len(peak_A)
    matched   = np.abs(peak_A - peak_B) < COINC_WINDOW_S
    n_any     = int(matched.sum())
    near165   = near165_coincidence_mask(peak_A, peak_B)
    n_near165 = int(near165.sum())
    return {
        "coinc_any":      n_any / n,
        "n_any":          n_any,
        "coinc_near165":  n_near165 / n,
        "n_near165":      n_near165,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_spin_trajectories(
    t_ctrl: np.ndarray,
    t_dense: np.ndarray,
    rng: np.random.Generator,
    output_path: Path,
) -> None:
    """Show N_TRAJ_EXAMPLES GP draws per smoothness setting."""
    n_rows = len(SMOOTHNESS_SETTINGS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    cmap = plt.cm.tab10

    for ax, (label, cfg) in zip(axes, SMOOTHNESS_SETTINGS.items()):
        L   = make_cholesky(cfg["l_s"], cfg["sigma"], t_ctrl)
        fex = draw_f_spin_batch(L, t_ctrl, t_dense, rng, N_TRAJ_EXAMPLES)
        for k in range(N_TRAJ_EXAMPLES):
            ax.plot(t_dense, fex[k], lw=0.9, alpha=0.85, color=cmap(k),
                    label=f"draw {k+1}")
        ax.axhline(SPIN_F_MIN_HZ, color="gray", lw=0.7, linestyle="--", alpha=0.5)
        ax.axhline(SPIN_F_MAX_HZ, color="gray", lw=0.7, linestyle="--", alpha=0.5)
        ax.set_ylabel("f_spin [Hz]", fontsize=8)
        ax.set_xlabel("t [s]", fontsize=8)
        ax.set_ylim(0.0, SPIN_F_MAX_HZ * 1.2)
        ax.set_title(f"{label}  (σ={cfg['sigma']}, l={cfg['l_s']} s)", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=N_TRAJ_EXAMPLES, loc="upper right")

    fig.suptitle(
        f"GP spin-frequency trajectory examples  "
        f"[{SPIN_F_MIN_HZ}–{SPIN_F_MAX_HZ} Hz,  "
        f"period {1/SPIN_F_MAX_HZ:.1f}–{1/SPIN_F_MIN_HZ:.1f} s]\n"
        "Gray dashed = prior bounds",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}", flush=True)


def plot_example_periodograms_for_harmonic(
    harmonic_label: str,
    harmonic_N: int,
    t_A: np.ndarray,
    t_B: np.ndarray,
    cos_A: np.ndarray, sin_A: np.ndarray, per_A: np.ndarray,
    cos_B: np.ndarray, sin_B: np.ndarray, per_B: np.ndarray,
    t_ctrl: np.ndarray,
    t_dense: np.ndarray,
    rng: np.random.Generator,
    output_path: Path,
) -> None:
    """
    N_PERIOD_EXAMPLES periodogram pairs per smoothness setting.
    Uses the same vectorised DFT-LS as the MC so plots are directly comparable.
    """
    n_smooth = len(SMOOTHNESS_SETTINGS)
    n_rows   = n_smooth * N_PERIOD_EXAMPLES
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 2.8 * n_rows), constrained_layout=True)

    row = 0
    for smooth_label, cfg in SMOOTHNESS_SETTINGS.items():
        L       = make_cholesky(cfg["l_s"], cfg["sigma"], t_ctrl)
        f_dense = draw_f_spin_batch(L, t_ctrl, t_dense, rng, N_PERIOD_EXAMPLES)
        phi_A   = compute_phase_at_timestamps(f_dense, t_dense, t_A)
        phi_B   = compute_phase_at_timestamps(f_dense, t_dense, t_B)
        sig_A   = np.cos(harmonic_N * phi_A) + rng.normal(0.0, NOISE_SCALE, phi_A.shape)
        sig_B   = np.cos(harmonic_N * phi_B) + rng.normal(0.0, NOISE_SCALE, phi_B.shape)
        sig_A  -= sig_A.mean(axis=1, keepdims=True)
        sig_B  -= sig_B.mean(axis=1, keepdims=True)

        for ex_i in range(N_PERIOD_EXAMPLES):
            for col, (t_obs, sig, cos_m, sin_m, per, obs_lbl) in enumerate([
                (t_A, sig_A[ex_i:ex_i+1], cos_A, sin_A, per_A,
                 f"obs {OBS_A} (Binar-SN-01)"),
                (t_B, sig_B[ex_i:ex_i+1], cos_B, sin_B, per_B,
                 f"obs {OBS_B} (KO2F-VHF-1)"),
            ]):
                ax  = axes[row, col]
                # Full power spectrum for this single trial
                cp  = cos_m @ sig.T   # (N_freq, 1)
                sp  = sin_m @ sig.T
                pwr = (cp ** 2 + sp ** 2).ravel()
                pwr_norm = pwr / pwr.max()

                pk_idx = pwr.argmax()
                pk_p   = per[pk_idx]

                ax.semilogx(per, pwr_norm, lw=0.65, color="steelblue", alpha=0.85)
                ax.axvline(pk_p, color="red", lw=1.0, linestyle=":", alpha=0.9)
                ax.axvline(TARGET_PERIOD_S, color="gray", lw=0.8,
                           linestyle="--", alpha=0.55)
                ax.text(pk_p * 1.06, pwr_norm[pk_idx] * 0.94,
                        f"{pk_p:.1f}s", fontsize=6, color="red")
                ax.text(TARGET_PERIOD_S * 1.04, 0.92,
                        f"{TARGET_PERIOD_S:.0f}s", fontsize=5, color="gray",
                        va="top")
                ax.set_xlabel("Period [s]", fontsize=6)
                ax.set_ylabel("Power (norm.)", fontsize=6)
                ax.tick_params(labelsize=5)
                ax.grid(True, alpha=0.20)
                ax.set_xlim(PERIOD_MIN_S * 0.85, PERIOD_MAX_S * 1.15)
                ax.set_title(
                    f"{obs_lbl}  |  {smooth_label}\n"
                    f"example {ex_i+1}: peak = {pk_p:.2f} s",
                    fontsize=7,
                )
            row += 1

    fig.suptitle(
        f"Synthetic LS periodograms — {harmonic_label}\n"
        f"(DFT-approx LS, same freq grid as main pipeline; "
        f"gray = {TARGET_PERIOD_S:.0f} s ref;  red = peak)",
        fontsize=10, fontweight="bold",
    )
    fig.savefig(output_path, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}", flush=True)


# ── Output formatting ──────────────────────────────────────────────────────────

def write_text_table(results: list[dict], fh) -> None:
    smooth_labels   = list(dict.fromkeys(r["smoothness"] for r in results))
    harmonic_labels = list(dict.fromkeys(r["harmonic"]   for r in results))
    by_cell         = {(r["smoothness"], r["harmonic"]): r for r in results}

    header = (
        "Monte Carlo Aliasing Test — 165.0 s LS Agreement\n"
        f"{'='*76}\n"
        f"obs A = {OBS_A} (Binar-SN-01,  Dec 4  2018,  814 valid pts, 47% gaps)\n"
        f"obs B = {OBS_B} (KO2F-VHF-1,   Dec 11 2018, 1203 valid pts, 22% gaps)\n"
        f"\n"
        f"N_trials per cell : {N_TRIALS}\n"
        f"LS period range   : {PERIOD_MIN_S}–{PERIOD_MAX_S} s  "
        f"(samples_per_peak={SAMPLES_PER_PEAK})\n"
        f"Spin freq prior   : {SPIN_F_MIN_HZ}–{SPIN_F_MAX_HZ} Hz  "
        f"(period {1/SPIN_F_MAX_HZ:.1f}–{1/SPIN_F_MIN_HZ:.1f} s, GP log-space)\n"
        f"Noise scale       : {NOISE_SCALE}  (signal amplitude = 1)\n"
        f"Target period     : {TARGET_PERIOD_S} s\n"
        f"Agreement window  : ±{COINC_WINDOW_S} s (passes must agree)\n"
        f"Near-165 zone     : ±{NEAR165_ZONE_S} s around {TARGET_PERIOD_S} s "
        f"(applied to matched trials only)\n"
    )
    fh.write(header + "\n")

    col_w = 17
    for metric_key, title in [
        ("coinc_any",
         "Table 1 — P(|peak_A − peak_B| < 0.005 s, ANY period)"),
        ("coinc_near165",
         f"Table 2 — P(peaks agree ±{COINC_WINDOW_S} s AND "
         f"agreement within ±{NEAR165_ZONE_S} s of {TARGET_PERIOD_S} s)"),
    ]:
        fh.write(f"\n{title}\n" + "─" * 76 + "\n")
        fh.write(f"{'Smoothness setting':<27}")
        for hl in harmonic_labels:
            fh.write(f"  {hl:>{col_w}}")
        fh.write("\n" + "─" * 76 + "\n")
        for sl in smooth_labels:
            fh.write(f"{sl:<27}")
            for hl in harmonic_labels:
                r = by_cell.get((sl, hl))
                v = r[metric_key] if r else float("nan")
                fh.write(f"  {v * 100:>{col_w - 1}.5f}%")
            fh.write("\n")

    fh.write("\n" + "=" * 76 + "\n")


def write_top_trajectories(
    all_saved: dict[tuple[str, str], list[tuple[int, np.ndarray]]],
    t_dense: np.ndarray,
    fh,
) -> None:
    fh.write(
        "Spin Trajectories for Near-165 s Coincident Trials\n"
        f"(|peak_A − peak_B| < {COINC_WINDOW_S} s AND "
        f"|peak_A − {TARGET_PERIOD_S}| < {NEAR165_ZONE_S} s)\n"
        f"f_spin range: {SPIN_F_MIN_HZ}–{SPIN_F_MAX_HZ} Hz  "
        f"(period {1/SPIN_F_MAX_HZ:.2f}–{1/SPIN_F_MIN_HZ:.2f} s)\n"
        f"{'='*80}\n"
    )

    t_sample = np.linspace(0, t_dense[-1], 11)   # 0%, 10%, … 100% of pass

    for (sl, hl), saved in all_saved.items():
        if not saved:
            fh.write(
                f"\n[{sl}  |  {hl}]\n"
                f"  0 near-165 s coincident trials in {N_TRIALS}.\n"
            )
            continue
        fh.write(
            f"\n{'─'*80}\n"
            f"{sl}  |  {hl}\n"
            f"Coincident trials: {len(saved)}/{N_TRIALS} "
            f"({len(saved)/N_TRIALS*100:.4f}%)\n"
        )
        for rank, (trial_idx, f_traj) in enumerate(saved[:N_TOP_PRINT], 1):
            f_pts = np.interp(t_sample, t_dense, f_traj)
            p_pts = 1.0 / f_pts
            fh.write(f"\n  Rank {rank} (trial #{trial_idx}):\n")
            fh.write(
                "    t_rel  [s]  : "
                + "  ".join(f"{x:5.0f}" for x in t_sample) + "\n"
            )
            fh.write(
                "    f_spin [Hz] : "
                + "  ".join(f"{x:5.3f}" for x in f_pts) + "\n"
            )
            fh.write(
                "    period [s]  : "
                + "  ".join(f"{x:5.2f}" for x in p_pts) + "\n"
            )
        extra = len(saved) - N_TOP_PRINT
        if extra > 0:
            fh.write(f"\n  … and {extra} more coincident trials (not printed).\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    pipeline_dir = _find_pipeline_dir()
    output_dir   = pipeline_dir.parent / "mc_aliasing"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68, flush=True)
    print("Monte Carlo Aliasing Test — 165.0 s LS Agreement", flush=True)
    print("=" * 68, flush=True)
    print(f"Pipeline dir : {pipeline_dir}", flush=True)
    print(f"Output dir   : {output_dir}", flush=True)

    # ── Load timestamps ────────────────────────────────────────────────────────
    print("\nLoading timestamps …", flush=True)
    t_A = load_valid_timestamps(OBS_A, pipeline_dir)
    t_B = load_valid_timestamps(OBS_B, pipeline_dir)
    print(f"  obs {OBS_A}: {len(t_A):4d} valid pts, span {t_A[-1]:.1f} s", flush=True)
    print(f"  obs {OBS_B}: {len(t_B):4d} valid pts, span {t_B[-1]:.1f} s", flush=True)

    # Dense time grid and GP control points spanning both passes
    T_MAX   = max(t_A[-1], t_B[-1]) * 1.01
    t_dense = np.linspace(0.0, T_MAX, N_DENSE)
    t_ctrl  = np.linspace(0.0, T_MAX, N_CTRL)

    # ── Precompute LS projection matrices (done ONCE) ─────────────────────────
    print("\nPrecomputing LS projection matrices (once per pass) …", flush=True)
    freqs_A, per_A = make_freq_grid(t_A)
    freqs_B, per_B = make_freq_grid(t_B)
    t0 = time.perf_counter()
    cos_A, sin_A   = precompute_ls_matrices(t_A, freqs_A)
    cos_B, sin_B   = precompute_ls_matrices(t_B, freqs_B)
    print(f"  Done in {time.perf_counter()-t0:.1f} s  "
          f"(A: {cos_A.shape}, B: {cos_B.shape})", flush=True)

    rng = np.random.default_rng(seed=20241204)

    # ── Spin trajectory example plot ──────────────────────────────────────────
    print("\nPlotting spin-trajectory examples …", flush=True)
    plot_spin_trajectories(
        t_ctrl, t_dense,
        np.random.default_rng(seed=999),   # separate seed; MC stays reproducible
        output_dir / "spin_trajectory_examples.png",
    )

    # ── MC run ────────────────────────────────────────────────────────────────
    n_cells  = len(SMOOTHNESS_SETTINGS) * len(HARMONICS)
    n_ls     = N_TRIALS * 2 * n_cells
    print(
        f"\nMC plan: {N_TRIALS} trials × {len(SMOOTHNESS_SETTINGS)} smooth "
        f"× {len(HARMONICS)} harm = {N_TRIALS * n_cells:,} trajectory sets, "
        f"{n_ls:,} LS evaluations  (vectorised, ~3 min)\n",
        flush=True,
    )

    results:   list[dict]                                       = []
    all_saved: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}

    t_total = time.perf_counter()

    for smooth_label, cfg in SMOOTHNESS_SETTINGS.items():
        L = make_cholesky(cfg["l_s"], cfg["sigma"], t_ctrl)

        for harm_label, harm_N in HARMONICS.items():
            key   = (smooth_label, harm_label)
            label = f"{smooth_label} | {harm_label}"
            print(f"\n  ▶ {label}", flush=True)
            t0 = time.perf_counter()

            pk_A, pk_B, saved = run_mc_cell(
                t_A, t_B,
                cos_A, sin_A, per_A,
                cos_B, sin_B, per_B,
                L, t_ctrl, t_dense,
                harm_N, rng, label,
            )

            elapsed = time.perf_counter() - t0
            st      = cell_stats(pk_A, pk_B)
            all_saved[key] = saved

            results.append({
                "smoothness":    smooth_label,
                "l_s":           cfg["l_s"],
                "sigma":         cfg["sigma"],
                "harmonic":      harm_label,
                "N_sym":         harm_N,
                "n_trials":      N_TRIALS,
                "coinc_any":     st["coinc_any"],
                "n_any":         st["n_any"],
                "coinc_near165": st["coinc_near165"],
                "n_near165":     st["n_near165"],
                "elapsed_s":     round(elapsed, 1),
            })
            print(
                f"    ✓ coinc_any={st['coinc_any']:.4%}  "
                f"near165={st['coinc_near165']:.5%}  "
                f"({st['n_near165']} hits / {N_TRIALS})  "
                f"[{elapsed:.0f} s]",
                flush=True,
            )

    total_elapsed = time.perf_counter() - t_total
    print(f"\nTotal MC time: {total_elapsed:.0f} s ({total_elapsed/60:.1f} min)",
          flush=True)

    # ── Write outputs ──────────────────────────────────────────────────────────
    csv_path = output_dir / "coincidence_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    txt_path = output_dir / "coincidence_table.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        write_text_table(results, fh)

    traj_path = output_dir / "top_coincident_trajectories.txt"
    with open(traj_path, "w", encoding="utf-8") as fh:
        write_top_trajectories(all_saved, t_dense, fh)

    print(f"\nWrote: {csv_path}", flush=True)
    print(f"Wrote: {txt_path}", flush=True)
    print(f"Wrote: {traj_path}", flush=True)

    # ── Example periodogram figures ────────────────────────────────────────────
    print("\nGenerating example periodogram figures …", flush=True)
    plot_rng = np.random.default_rng(seed=12345)
    for harm_label, harm_N in HARMONICS.items():
        slug = (harm_label
                .replace(" ", "_").replace("=", "")
                .replace("(", "").replace(")", "").replace("φ", "phi"))
        plot_example_periodograms_for_harmonic(
            harm_label, harm_N,
            t_A, t_B,
            cos_A, sin_A, per_A,
            cos_B, sin_B, per_B,
            t_ctrl, t_dense,
            plot_rng,
            output_dir / f"example_periodograms_{slug}.png",
        )

    # ── Print summary table to stdout ──────────────────────────────────────────
    print("\n", flush=True)
    with open(txt_path, encoding="utf-8") as fh:
        print(fh.read(), flush=True)
    print(f"All outputs in: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
