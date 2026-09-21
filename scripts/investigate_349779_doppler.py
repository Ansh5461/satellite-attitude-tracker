#!/usr/bin/env python3
"""349779 Doppler anomaly closeout: residuals vs time, segment R², carrier-track QC."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from carrier_doppler_check import (  # noqa: E402
    fit_doppler,
    load_meta,
    predict_doppler,
)

PROJECT = _SCRIPT_DIR.parent
OBS_IDS = [349561, 349779, 351032, 354023]
OUT_DIR = PROJECT / "data" / "archival_tle_validation" / "gp_history" / "doppler_349779_investigation"

# gp_history matched TLEs (epoch 18340.69668349 for 349779)
GP_TLE = {
    349561: (
        "1 43780U 18099Y   18340.69668349 -.00000372  00000-0 -28802-4 0  9994",
        "2 43780  97.7683  49.8182 0011803 257.1086 102.8804 14.94846885   424",
    ),
    349779: (
        "1 43780U 18099Y   18340.69668349 -.00000372  00000-0 -28802-4 0  9994",
        "2 43780  97.7683  49.8182 0011803 257.1086 102.8804 14.94846885   424",
    ),
    351032: (
        "1 43780U 18099Y   18340.69668349 -.00000372  00000-0 -28802-4 0  9994",
        "2 43780  97.7683  49.8182 0011803 257.1086 102.8804 14.94846885   424",
    ),
    354023: (
        "1 43780U 18099Y   18341.90156131 +.00000220 +00000-0 +25547-4 0  9993",
        "2 43780 097.7669 051.0119 0012325 252.6507 107.3366 14.94849007000603",
    ),
}


def find_obs_folder(obs_id: int) -> Path:
    for root in (PROJECT / "data_move2_tracked", PROJECT / "data"):
        for cand in root.iterdir() if root.is_dir() else []:
            if cand.is_dir() and cand.name.startswith(f"move2_{obs_id}_"):
                return cand
    raise FileNotFoundError(obs_id)


def load_carrier_csv(obs_id: int) -> dict:
    folder = find_obs_folder(obs_id)
    csv_path = folder / "derived" / f"carrier_track_{obs_id}.csv"
    meta_path = folder / "derived" / f"carrier_track_{obs_id}_meta.json"
    rows = []
    with open(csv_path, encoding="utf-8") as fh:
        import csv
        for row in csv.DictReader(fh):
            rows.append(row)
    t_rel = np.array([float(r["t_rel_s"]) for r in rows])
    peak_freq = np.array([float(r["peak_freq_hz"]) for r in rows])
    confidence = np.array([float(r["peak_power_db_relative"]) for r in rows])
    strong = np.array([bool(int(r["strong"])) for r in rows])
    timestamps = [
        datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00"))
        for r in rows
    ]
    meta_json = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return {
        "t_rel_s": t_rel,
        "peak_freq_hz": peak_freq,
        "confidence": confidence,
        "strong_mask": strong,
        "timestamps": timestamps,
        "ocr_meta": meta_json,
    }


def segment_r2(t: np.ndarray, y: np.ndarray, n_segments: int = 4) -> list[dict]:
    """R² per contiguous time segment (equal sample count)."""
    n = len(t)
    chunk = n // n_segments
    out = []
    for i in range(n_segments):
        lo = i * chunk
        hi = n if i == n_segments - 1 else (i + 1) * chunk
        seg_t = t[lo:hi]
        seg_y = y[lo:hi]
        if len(seg_y) < 3:
            continue
        ss_tot = float(np.sum((seg_y - seg_y.mean()) ** 2))
        ss_res = float(np.sum(seg_y ** 2))  # placeholder, overwritten below
        # fit line through segment vs index (residual structure)
        coeffs = np.polyfit(seg_t, seg_y, 1)
        fitted = coeffs[0] * seg_t + coeffs[1]
        resid = seg_y - fitted
        ss_res = float(np.sum(resid ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out.append({
            "segment": i + 1,
            "t_lo_s": float(seg_t.min()),
            "t_hi_s": float(seg_t.max()),
            "n": int(len(seg_y)),
            "rms_hz": float(np.sqrt(np.mean(resid ** 2))),
            "r2_trend": r2,
        })
    return out


def doppler_fit_residuals(obs_id: int) -> dict:
    folder = find_obs_folder(obs_id)
    meta = load_meta(folder, obs_id)
    meta["tle1"], meta["tle2"] = GP_TLE[obs_id]
    track = load_carrier_csv(obs_id)
    f0 = float(meta["observation_frequency"])

    strong_idx = np.where(track["strong_mask"])[0]
    strong_freq = track["peak_freq_hz"][strong_idx]
    strong_t = track["t_rel_s"][strong_idx]
    pred = predict_doppler(meta, track["timestamps"], track["strong_mask"])
    fit = fit_doppler(strong_freq, pred)
    residuals = fit["residuals"]
    fitted = fit["fitted"]

    # QC metrics from waterfall OCR stage
    all_conf = track["confidence"]
    strong_conf = all_conf[track["strong_mask"]]
    qc = {
        "n_rows_total": int(len(all_conf)),
        "n_strong": int(len(strong_idx)),
        "frac_strong": float(len(strong_idx) / len(all_conf)),
        "conf_median_all": float(np.median(all_conf)),
        "conf_p80_all": float(np.percentile(all_conf, 80)),
        "conf_median_strong": float(np.median(strong_conf)),
        "conf_p10_strong": float(np.percentile(strong_conf, 10)),
        "y_slope_s_per_px": track["ocr_meta"].get("y_slope_s_per_px"),
        "n_y_ticks": track["ocr_meta"].get("n_y_ticks"),
    }

    # Residual segment stats (equal time quarters)
    t_min, t_max = strong_t.min(), strong_t.max()
    seg_edges = np.linspace(t_min, t_max, 5)
    seg_stats = []
    for i in range(4):
        mask = (strong_t >= seg_edges[i]) & (strong_t < seg_edges[i + 1] if i < 3 else strong_t <= seg_edges[i + 1])
        if mask.sum() < 3:
            continue
        seg_res = residuals[mask]
        seg_obs = strong_freq[mask]
        ss_res = float(np.sum(seg_res ** 2))
        ss_tot = float(np.sum((seg_obs - seg_obs.mean()) ** 2))
        seg_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        seg_stats.append({
            "quarter": i + 1,
            "t_lo_s": float(seg_edges[i]),
            "t_hi_s": float(seg_edges[i + 1]),
            "n": int(mask.sum()),
            "rms_residual_hz": float(np.sqrt(np.mean(seg_res ** 2))),
            "mean_residual_hz": float(np.mean(seg_res)),
            "R2_fit": seg_r2,
        })

    return {
        "obs_id": obs_id,
        "B": float(fit["B"]),
        "R2": float(fit["r2"]),
        "rms_hz": float(fit["rms_hz"]),
        "n_strong": int(len(strong_idx)),
        "qc": qc,
        "residual_segments": seg_stats,
        "strong_t": strong_t,
        "residuals": residuals,
        "fitted": fitted,
        "strong_freq": strong_freq,
        "f0_hz": f0,
        "station": meta.get("station_name", ""),
    }


def plot_residuals(results: dict[int, dict], highlight: int = 349779) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.ravel()
    for ax, obs_id in zip(axes, OBS_IDS):
        r = results[obs_id]
        t = r["strong_t"]
        res = r["residuals"]
        color = "crimson" if obs_id == highlight else "steelblue"
        ax.scatter(t, res, s=8, alpha=0.6, color=color, edgecolors="none")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(
            f"obs {obs_id}  B={r['B']:.3f}  R²={r['R2']:.4f}  RMS={r['rms_hz']:.0f} Hz",
            fontsize=10,
        )
        ax.set_xlabel("t_rel (s)")
        ax.set_ylabel("Doppler fit residual (Hz)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Doppler residuals vs time (gp_history TLE, strong rows only)", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "residuals_vs_time_all4.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Highlight 349779 with rolling RMS
    r = results[highlight]
    t = r["strong_t"]
    res = r["residuals"]
    order = np.argsort(t)
    t, res = t[order], res[order]
    win = max(15, len(t) // 20)
    roll_rms = np.array([
        np.sqrt(np.mean(res[max(0, i - win // 2): min(len(res), i + win // 2 + 1)] ** 2))
        for i in range(len(res))
    ])
    fig2, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax0.scatter(t, res, s=10, alpha=0.7, color="crimson", edgecolors="none")
    ax0.axhline(0, color="gray", lw=0.8, ls="--")
    ax0.set_ylabel("Residual (Hz)")
    ax0.set_title(f"obs {highlight} Doppler residuals — B={r['B']:.3f}, R²={r['R2']:.4f}")
    ax0.grid(True, alpha=0.3)
    ax1.plot(t, roll_rms, color="darkorange", lw=1.5)
    ax1.set_xlabel("t_rel (s)")
    ax1.set_ylabel(f"Rolling RMS (Hz, win≈{win})")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = OUT_DIR / f"residuals_vs_time_{highlight}_detail.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    return out


def main() -> None:
    results = {oid: doppler_fit_residuals(oid) for oid in OBS_IDS}
    plot_residuals(results)

    report = {
        "tle_source": "gp_history matched epochs",
        "observations": {},
        "349779_vs_others": {},
    }
    for oid, r in results.items():
        report["observations"][str(oid)] = {
            k: v for k, v in r.items()
            if k not in ("strong_t", "residuals", "fitted", "strong_freq")
        }

    ref = results[349779]
    others = [results[o] for o in OBS_IDS if o != 349779]
    report["349779_vs_others"] = {
        "B_349779": ref["B"],
        "B_others_mean": float(np.mean([o["B"] for o in others])),
        "R2_349779": ref["R2"],
        "R2_others_mean": float(np.mean([o["R2"] for o in others])),
        "rms_349779_hz": ref["rms_hz"],
        "rms_others_mean_hz": float(np.mean([o["rms_hz"] for o in others])),
        "frac_strong_349779": ref["qc"]["frac_strong"],
        "frac_strong_others": {str(o["obs_id"]): o["qc"]["frac_strong"] for o in others},
        "conf_median_strong_349779": ref["qc"]["conf_median_strong"],
        "conf_median_strong_others": {str(o["obs_id"]): o["qc"]["conf_median_strong"] for o in others},
        "worst_quarter_R2_349779": min(s["R2_fit"] for s in ref["residual_segments"]),
        "best_quarter_R2_349779": max(s["R2_fit"] for s in ref["residual_segments"]),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "investigation_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_json}")
    print(f"Plots: {OUT_DIR}/residuals_vs_time_*.png")


if __name__ == "__main__":
    main()
