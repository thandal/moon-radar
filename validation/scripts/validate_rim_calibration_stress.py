"""Stress-test the rim Doppler calibration with synthetic DD images.

This is the self-contained validation version of the rim edge-shape bias check
(REPORT 8.2). It carries the synthetic rim-caustic forward model and the exact
production rim estimator, sweeps a known injected Doppler delta through a set of
scattering/smear/contrast stressors on two real SPICE geometries, and writes a
compact JSON summary plus CSV/PNG under validation/results.

Model summary
-------------
The per-look Doppler calibration delta is the mean of the up-rim and down-rim
half-power edge offsets (``measure_rim_offset``). A symmetric edge-shape
displacement cancels into the rim *spread*; a bias in delta can only come from
up/down asymmetry. Each synthetic DD image is built on the real SPICE geometry:

  - per delay column, the annulus Doppler profile is the exact projected-sphere
    rim caustic 1/sqrt((dlt-a)(b-dlt)), bin-integrated analytically, with rim
    positions a, b from the real equator curves;
  - a quasi-specular scattering-law falloff in delay, plus an optional brightness
    gradient across the disk (up rim vs down rim asymmetry);
  - convolution with the window's sinc^2 Doppler resolution kernel (1/T wide),
    plus an optional extra Gaussian smear (fading / intra-look drift), with
    matching row-correlated speckle on echo and noise floor;
  - an injected Doppler offset delta shifting all painted content.

Two real geometries bracket the smear-to-bin regimes seen in the data:
2025-06-21 (66 s looks, ~6 mHz bins) and 2025-09-11 (34 s looks, ~1.8 mHz bins).

Usage (from the repo root):
    .conda/bin/python validation/scripts/validate_rim_calibration_stress.py
"""

from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage
import cspyce as csp
from astropy.time import Time
from astropy import units as au

import validation_common as vc

# validation_common puts the repo root on sys.path; import production code after.
import doppler_equator_alignment as dea


F0 = 1299.5e6

GEOMETRIES = {
    # label: (rx ISO epoch, window length s) -- real look epochs
    "2025-06-21 (66s, 6mHz bins)": ("2025-06-21T08:59:29", 66.0),
    "2025-09-11 (34s, 1.8mHz bins)": ("2025-09-11T08:05:44", 34.0),
}


def build_geometry(iso, T):
    """Axes + rate-corrected equator curves, exactly as compute_dd_image /
    process_file produce them."""
    rx_time = dea.et_from_astropy(Time(iso, scale="utc"))
    fs = 0.25e6
    lead = 20
    t_end = dea.MOON_RADIUS / dea.ak.c * 2
    ddelay = 1.0 / fs
    dvs = np.arange(-lead, int(round(t_end.to_value(au.s) * fs))) * ddelay

    _, _, v_term = csp.edterm("UMBRAL", "DWINGELOO", "MOON", rx_time,
                              "MOON_ME", dea.AB_COR, "STOCKERT", 1000)
    _, dlt_term = dea.moonPointDLT_BCK(rx_time, v_term, "DWINGELOO", "STOCKERT")
    dlt_shifts = np.linspace(dlt_term.min(), dlt_term.max(), 3000)

    rate = dea.srp_dlt_rate_bck(rx_time, T, "DWINGELOO", "STOCKERT")
    lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = \
        dea.compute_doppler_equator_velocity(rx_time, n_points=500,
                                             rx_duration_s=T, dlt_rate_srp=rate)
    return dict(rx_time=rx_time, T=T, dvs=dvs, dlt_shifts=dlt_shifts,
                lt_min=lt_min_eq, delay_up=delay_up, dlt_up=dlt_up,
                delay_down=delay_down, dlt_down=dlt_down)


def correlated_speckle(shape, h, rng):
    """Unit-mean exponential speckle, row-correlated by field kernel h."""
    z = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    z = scipy.ndimage.convolve1d(z.real, h, axis=0, mode="wrap") + \
        1j * scipy.ndimage.convolve1d(z.imag, h, axis=0, mode="wrap")
    return np.abs(z) ** 2 / (2.0 * np.sum(h ** 2))


def synth_log_A(geom, delta_dlt, asym=0.0, smear_extra_hz=0.0,
                contrast=1.5, scatter_n=1.5, rng=None):
    """Synthetic log_A on the geometry's axes with content shifted by
    +delta_dlt (the chain-residual convention of the pipeline)."""
    dvs, dlt_shifts = geom["dvs"], geom["dlt_shifts"]
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    n_dop, n_del = len(dlt_shifts), len(dvs)

    # rims per column from the real equator curves (delays are monotonic
    # along each branch from the SRP to the limb)
    up_d, up_v = geom["delay_up"], geom["dlt_up"]
    dn_d, dn_v = geom["delay_down"], geom["dlt_down"]
    o_u, o_d = np.argsort(up_d), np.argsort(dn_d)
    a = np.interp(dvs, up_d[o_u], up_v[o_u], left=np.nan, right=np.nan)
    b = np.interp(dvs, dn_d[o_d], dn_v[o_d], left=np.nan, right=np.nan)
    a, b = a + delta_dlt, b + delta_dlt
    echo_cols = np.isfinite(a) & np.isfinite(b) & (b - a > 4 * ddlt)

    # quasi-specular + diffuse falloff with delay (cos(theta) of the annulus)
    c_kms = csp.clight()
    two_r_c = 2 * dea.MOON_RADIUS.to_value(au.km) / c_kms
    cosq = np.clip(1.0 - np.maximum(dvs, 0.0) / two_r_c, 0.0, 1.0)
    S = 0.02 + cosq ** scatter_n

    # bin-integrated caustic: int dr / sqrt((r-a)(b-r)) = asin((2r-a-b)/(b-a))
    edges = np.concatenate([dlt_shifts - ddlt / 2, [dlt_shifts[-1] + ddlt / 2]])
    P = np.zeros((n_dop, n_del), dtype=np.float64)
    cols = np.nonzero(echo_cols)[0]
    Fa = np.arcsin(np.clip((2 * edges[:, None] - a[None, cols] - b[None, cols])
                           / (b[None, cols] - a[None, cols]), -1.0, 1.0))
    prof = (Fa[1:] - Fa[:-1]) / np.pi  # column-normalized annulus power
    # up/down brightness asymmetry: linear gradient across the disk
    if asym != 0.0:
        x = np.clip((2 * dlt_shifts[:, None] - a[None, cols] - b[None, cols])
                    / (b[None, cols] - a[None, cols]), -1.0, 1.0)
        prof = prof * (1.0 + asym * x)
    P[:, cols] = prof * S[None, cols]

    # Doppler resolution kernel of the T-second window: field sinc(f T),
    # power sinc^2; optional extra Gaussian smear (fading, drift)
    df_row = ddlt * F0
    half = int(np.ceil(4.0 / (geom["T"] * df_row)))
    m = np.arange(-half, half + 1)
    h = np.sinc(m * df_row * geom["T"])
    if smear_extra_hz > 0:
        g = np.exp(-0.5 * (m * df_row / smear_extra_hz) ** 2)
        h = np.convolve(h, g / g.sum(), mode="same")
    k_pow = h ** 2 / np.sum(h ** 2)
    P = scipy.ndimage.convolve1d(P, k_pow, axis=0, mode="constant")

    # plateau-to-floor contrast in log_A units: log amplitude ratio
    plateau = np.median(P[P > 0]) if np.any(P > 0) else 1.0
    N0 = plateau / np.exp(2.0 * contrast)

    h_field = h / np.sqrt(np.sum(h ** 2))
    I = (P * correlated_speckle(P.shape, h_field, rng)
         + N0 * correlated_speckle(P.shape, h_field, rng))
    return 0.5 * np.log(I)


def recover_delta(log_A, geom):
    """The exact iterative rim calibration from process_file."""
    dlt_shifts, dvs = geom["dlt_shifts"], geom["dvs"]
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    delta, rim = 0.0, None
    for _ in range(3):
        r = dea.measure_rim_offset(log_A, dlt_shifts - delta, dvs,
                                   geom["lt_min"], geom["lt_min"],
                                   geom["delay_up"], geom["dlt_up"],
                                   geom["delay_down"], geom["dlt_down"])
        if r is None:
            break
        rim = r
        delta += r["delta_dlt"]
        if abs(r["delta_dlt"]) < 0.5 * ddlt:
            break
    return (delta, rim) if rim is not None else (None, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-real", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    vc.ensure_dirs()
    rng = np.random.default_rng(args.seed)
    deltas_hz = np.array([-80e-3, -40e-3, -15e-3, 0.0, 15e-3, 40e-3, 80e-3])
    variants = {
        "base": {},
        "asym_plus_0p5": {"asym": 0.5},
        "asym_minus_0p5": {"asym": -0.5},
        "smear_30mHz": {"smear_extra_hz": 30e-3},
        "weak_contrast_0p6": {"contrast": 0.6},
        "flat_law_n0p5": {"scatter_n": 0.5},
    }

    rows = []
    for geometry, (iso, window_s) in GEOMETRIES.items():
        geom = build_geometry(iso, window_s)
        for variant, kwargs in variants.items():
            for true_hz in deltas_hz:
                recovered = []
                failures = 0
                for _ in range(args.n_real):
                    log_A = synth_log_A(
                        geom, -true_hz / F0, rng=rng, **kwargs)
                    delta, _rim = recover_delta(log_A, geom)
                    if delta is None:
                        failures += 1
                    else:
                        recovered.append(-delta * F0)
                rec_mean = float(np.mean(recovered)) if recovered else np.nan
                rec_std = float(np.std(recovered)) if recovered else np.nan
                rows.append({
                    "geometry": geometry,
                    "variant": variant,
                    "delta_true_hz": true_hz,
                    "recovered_mean_hz": rec_mean,
                    "recovered_std_hz": rec_std,
                    "bias_hz": rec_mean - true_hz if np.isfinite(rec_mean) else np.nan,
                    "n_recovered": len(recovered),
                    "n_failed": failures,
                })

    csv_path = vc.RESULTS_DIR / "rim_calibration_stress.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_variant = {}
    for variant in variants:
        biases = [r["bias_hz"] for r in rows
                  if r["variant"] == variant and np.isfinite(r["bias_hz"])]
        by_variant[variant] = vc.finite_stats(np.asarray(biases) * 1e3)
    payload = {
        "purpose": "synthetic rim calibration bias under scattering/smear/contrast stressors",
        "bias_units": "mHz",
        "n_real_per_case": args.n_real,
        "by_variant_bias_mhz": by_variant,
    }
    json_path = vc.write_json("rim_calibration_stress.json", payload)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in variants:
        pts = [r for r in rows if r["variant"] == variant and np.isfinite(r["bias_hz"])]
        x = np.array([r["delta_true_hz"] for r in pts]) * 1e3
        y = np.array([r["bias_hz"] for r in pts]) * 1e3
        ax.scatter(x, y, s=16, alpha=0.75, label=variant)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_xlabel("injected delta (mHz)")
    ax.set_ylabel("bias: recovered - injected (mHz)")
    ax.set_title("Rim calibration synthetic stress")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    png_path = vc.RESULTS_DIR / "rim_calibration_stress.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    print(f"wrote {vc.report_path(csv_path)}")
    print(f"wrote {vc.report_path(json_path)}")
    print(f"wrote {vc.report_path(png_path)}")
    for variant, stats in by_variant.items():
        if stats["n"]:
            print(f"{variant}: max_abs_bias={stats['max_abs']:.2f} mHz, "
                  f"p95_abs={stats['p95_abs']:.2f} mHz")


if __name__ == "__main__":
    main()
