"""Validate light-time/Doppler physics and DEM topographic Doppler effects.

This is a physics-level check of the central approximation used by the mapper:
surface fields are evaluated with apparent station positions anchored at the
SRP instead of exact SPICE calls per surface point. It also quantifies the
open question in REPORT.md: how large DEM-induced Doppler shifts are.
Measurements only: no pass/fail criterion.

The epochs are Stockert looks, so the geometry is Stockert's; the output,
validation/results/doppler_dem_physics_stockert.json, is named after it.
"""

from __future__ import annotations

import argparse

import healpy as hp
import numpy as np
from astropy.time import Time

import validation_common as vc

import doppler_equator as de
from spice_setup import furnsh_kernels


EPOCHS = [
    "2025-06-21T08:59:29",
    "2025-09-11T08:05:44",
    "2025-09-16T13:23:26",
]
# The epochs are Stockert looks. (doppler_equator_alignment.TX_STATION names
# the transmitter, but importing that module needs CuPy, which this
# geometry-only check does not.)
TX_NAME = "DWINGELOO"
RX_NAME = "STOCKERT"


def anchored_lt_dlt(rx_time: float, points: np.ndarray, window_s: float, *,
                    tx_name: str, rx_name: str) -> tuple[np.ndarray, np.ndarray]:
    R0_rx, R0_tx, c = de.apparent_station_positions(rx_time, tx_name, rx_name)
    R1_rx, R1_tx, _ = de.apparent_station_positions(rx_time + window_s,
                                                    tx_name, rx_name)
    lt0 = (np.linalg.norm(points - R0_rx, axis=1) +
           np.linalg.norm(points - R0_tx, axis=1)) / c
    lt1 = (np.linalg.norm(points - R1_rx, axis=1) +
           np.linalg.norm(points - R1_tx, axis=1)) / c
    return lt0, (lt1 - lt0) / window_s


def sample_nearside(nside: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    v = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
    # Avoid the extreme limb for exact-vs-anchored statistics; the mapper masks
    # the pathological degeneracy separately.
    v = v[v[:, 0] > 0.05]
    if len(v) > limit:
        v = v[rng.choice(len(v), limit, replace=False)]
    return v


def validate_epoch(epoch: str, nside: int, limit: int, window_s: float,
                   frequency_hz: float, use_dem: bool,
                   rng: np.random.Generator) -> dict:
    rx_time = de.et_from_astropy(Time(epoch, scale="utc"))
    dirs = sample_nearside(nside, limit, rng)
    p_ell = de.moon_surface_points(dirs, use_dem=False)
    p_dem = de.moon_surface_points(dirs, use_dem=True) if use_dem else None

    # Exact per-point SPICE light time on the same (sphere) surface.
    lt_exact = de.moonPointLightTime_BCK(rx_time, dirs, TX_NAME, RX_NAME,
                                         use_dem=False)
    lt_end_exact = de.moonPointLightTime_BCK(rx_time + window_s, dirs, TX_NAME,
                                             RX_NAME, use_dem=False)
    dlt_exact = (lt_end_exact - lt_exact) / window_s
    lt_anchor, dlt_anchor = anchored_lt_dlt(rx_time, p_ell, window_s,
                                            tx_name=TX_NAME, rx_name=RX_NAME)

    out = {
        "epoch": epoch,
        "n_points": int(len(dirs)),
        "window_s": float(window_s),
        "anchored_lt_error_s": vc.finite_stats(lt_anchor - lt_exact),
        "anchored_doppler_error_hz": vc.finite_stats(
            -(dlt_anchor - dlt_exact) * frequency_hz),
    }

    if use_dem and p_dem is not None:
        lt_ell, dlt_ell = anchored_lt_dlt(rx_time, p_ell, window_s,
                                          tx_name=TX_NAME, rx_name=RX_NAME)
        lt_dem, dlt_dem = anchored_lt_dlt(rx_time, p_dem, window_s,
                                          tx_name=TX_NAME, rx_name=RX_NAME)
        topo_delay_us = (lt_dem - lt_ell) * 1e6
        topo_doppler_hz = -(dlt_dem - dlt_ell) * frequency_hz
        out["dem_topo_delay_us"] = vc.finite_stats(topo_delay_us)
        out["dem_topo_doppler_hz"] = vc.finite_stats(topo_doppler_hz)
        out["dem_topo_doppler_by_delay_percentile_hz"] = {
            "near_all": vc.finite_stats(topo_doppler_hz),
            "outer_delay_10pct": vc.finite_stats(
                topo_doppler_hz[np.abs(topo_delay_us) >= np.quantile(np.abs(topo_delay_us), 0.90)]
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--frequency-hz", type=float, default=1299.5e6)
    parser.add_argument("--no-dem", action="store_true")
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    vc.ensure_dirs()
    furnsh_kernels()
    use_dem = not args.no_dem
    if use_dem:
        de.load_lola_dem()

    rng = np.random.default_rng(args.seed)
    epochs = [
        validate_epoch(e, args.nside, args.limit, args.window_s,
                       args.frequency_hz, use_dem, rng)
        for e in EPOCHS
    ]
    payload = {
        "purpose": "exact SPICE vs anchored light-time/Doppler fields; DEM topographic Doppler magnitude",
        "criterion": "none: measurements only",
        "note": (
            "Compare the anchored Doppler error with the rim-calibration scale "
            "(mHz); it is not tested here. DEM topographic Doppler is reported "
            "as a physical term."
        ),
        "tx_name": TX_NAME, "rx_name": RX_NAME,
        "epochs": epochs,
    }
    path = vc.write_json(f"doppler_dem_physics_{RX_NAME.lower()}.json", payload)
    print(f"wrote {vc.report_path(path)}")
    for row in epochs:
        ad = row["anchored_doppler_error_hz"]
        print(f"{row['epoch']}: anchored Doppler max_abs={ad['max_abs']:.4g} Hz, "
              f"p95_abs={ad['p95_abs']:.4g} Hz")
        if "dem_topo_doppler_hz" in row:
            td = row["dem_topo_doppler_hz"]
            print(f"  DEM topo Doppler max_abs={td['max_abs']:.4g} Hz, "
                  f"p95_abs={td['p95_abs']:.4g} Hz")


if __name__ == "__main__":
    main()
