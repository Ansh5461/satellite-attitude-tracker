#!/usr/bin/env python3
"""Recompute empirical-null Poisson significance from saved MC peak histogram NPZ."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from mc_peak_density_analysis import (  # noqa: E402
    empirical_null_p_hit_per_trial,
    poisson_sf,
    uniform_null_lambda_per_trial,
)
from mc_aliasing_165s import COINC_WINDOW_S, NEAR165_ZONE_S, N_TRIALS  # noqa: E402

GP_DIR = _SCRIPT_DIR.parent / "data" / "archival_tle_validation" / "gp_history"
N_CELLS = 12
N_TRIALS_PER_CELL = 120_000
OBSERVED_HITS = 3


def main() -> None:
    npz_path = GP_DIR / "mc_peak_histograms.npz"
    z = np.load(npz_path)
    peaks_A = z["peaks_A"]
    peaks_B = z["peaks_B"]

    zone_lo = 165.0 - NEAR165_ZONE_S
    zone_hi = 165.0 + NEAR165_ZONE_S
    f_A = float(((peaks_A >= zone_lo) & (peaks_A <= zone_hi)).mean())
    f_B = float(((peaks_B >= zone_lo) & (peaks_B <= zone_hi)).mean())

    p_uniform = uniform_null_lambda_per_trial()
    p_emp = empirical_null_p_hit_per_trial(f_A, f_B)

    lam_uniform = p_uniform * N_TRIALS_PER_CELL * N_CELLS
    lam_emp = p_emp * N_TRIALS_PER_CELL * N_CELLS

    result = {
        "observed_near165_hits_total": OBSERVED_HITS,
        "empirical_peak_density": {
            "fraction_A_in_163_167": f_A,
            "fraction_B_in_163_167": f_B,
        },
        "uniform_null": {
            "p_hit_per_trial": p_uniform,
            "lambda_expected_total_12_cells": lam_uniform,
            "poisson_p_ge_observed": poisson_sf(OBSERVED_HITS, lam_uniform),
        },
        "empirical_null": {
            "p_agree_given_both_in_zone": min(1.0, 2.0 * COINC_WINDOW_S / (2.0 * NEAR165_ZONE_S)),
            "p_hit_per_trial": p_emp,
            "lambda_expected_total_12_cells": lam_emp,
            "poisson_p_ge_observed": poisson_sf(OBSERVED_HITS, lam_emp),
        },
    }

    out_path = GP_DIR / "mc_poisson_analysis.json"
    if out_path.is_file():
        existing = json.loads(out_path.read_text())
        existing["uniform_null"] = result["uniform_null"]
        existing["empirical_null"] = result["empirical_null"]
        existing["observed_near165_hits_total"] = OBSERVED_HITS
        existing["empirical_peak_density"].update(result["empirical_peak_density"])
        out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    else:
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nUpdated {out_path}")


if __name__ == "__main__":
    main()
