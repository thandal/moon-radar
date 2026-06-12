"""
Rim edge-shape bias validation: synthetic echoes with known delta through
the rim estimator.

The per-look Doppler calibration delta is the mean of the up-rim and
down-rim half-power edge offsets (measure_rim_offset). A symmetric
edge-shape displacement cancels into the rim *spread* by construction; a
bias in delta can only come from up/down asymmetry. This script quantifies
that bias with synthetic DD images built on the real SPICE geometry:

  - per delay column, the annulus Doppler profile is the exact projected-
    sphere rim caustic 1/sqrt((dlt-a)(b-dlt)), bin-integrated analytically,
    with rim positions a, b from the real equator curves;
  - a quasi-specular scattering-law falloff in delay, and an optional
    brightness gradient across the disk (up rim vs down rim asymmetry);
  - convolution with the window's sinc^2 Doppler resolution kernel
    (1/T wide), plus an optional extra Gaussian smear (fading / intra-look
    drift), with matching row-correlated speckle on echo and noise floor;
  - an injected Doppler offset delta shifting all painted content.

Two real geometries bracket the smear-to-bin regimes seen in the data:
2025-06-21 (66 s looks, ~6 mHz bins, edge smear ~2.5 rows) and the
2025-09-11 morning captures (34 s looks, ~1.8 mHz bins, smear ~16 rows --
where the strict contrast gates starve on real data).

The recovered delta uses the exact iterative loop from process_file.

Usage (from the repo root):
    .conda/bin/python rim_bias_validation.py
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import scipy.ndimage
import cspyce as csp
from astropy.time import Time
from astropy import units as au
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

import doppler_equator_alignment as dea

F0 = 1299.5e6
OUT_DIR = os.path.join(os.path.dirname(__file__), "results/RIM_BIAS")

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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(20260612)

    deltas_hz = np.array([-80e-3, -40e-3, -15e-3, 0.0, 15e-3, 40e-3, 80e-3])
    base = dict(asym=0.0, smear_extra_hz=0.0, contrast=1.5)
    variants = {
        "base": dict(base),
        "asym 0.5": dict(base, asym=0.5),
        "asym -0.5": dict(base, asym=-0.5),
        "smear 30mHz": dict(base, smear_extra_hz=30e-3),
        "weak contrast 0.6": dict(base, contrast=0.6),
        "flat law n=0.5": dict(base, scatter_n=0.5),
    }
    N_REAL = {"base": 6, "default": 4}

    results = []  # (geom_label, variant, delta_true_hz, recovered_hz or nan)
    for glabel, (iso, T) in GEOMETRIES.items():
        geom = build_geometry(iso, T)
        ddlt = geom["dlt_shifts"][1] - geom["dlt_shifts"][0]
        print(f"\n=== {glabel}: bin {ddlt*F0*1e3:.2f} mHz, "
              f"1/T smear {1.0/T/(ddlt*F0):.1f} rows ===")
        for vlabel, kw in variants.items():
            n_real = N_REAL.get(vlabel, N_REAL["default"])
            d_sweep = deltas_hz if vlabel == "base" else deltas_hz[1:-1:2]
            for d_hz in d_sweep:
                # pipeline convention: rim_delta_hz = -delta_dlt * f0
                inj = -d_hz / F0
                rec, fails = [], 0
                for _ in range(n_real):
                    log_A = synth_log_A(geom, inj, rng=rng, **kw)
                    delta, rim = recover_delta(log_A, geom)
                    if delta is None:
                        fails += 1
                        continue
                    rec.append(-delta * F0)
                    results.append((glabel, vlabel, d_hz, rec[-1]))
                if rec:
                    rec = np.array(rec)
                    bias = rec.mean() - d_hz
                    print(f"  {vlabel:20s} true {d_hz*1e3:+6.1f} mHz: "
                          f"recovered {rec.mean()*1e3:+7.2f} +/- "
                          f"{rec.std()*1e3:5.2f} mHz  bias {bias*1e3:+6.2f} mHz"
                          + (f"  ({fails}/{n_real} gate-fail)" if fails else ""))
                else:
                    print(f"  {vlabel:20s} true {d_hz*1e3:+6.1f} mHz: "
                          f"all {n_real} gate-fail")
                    results.append((glabel, vlabel, d_hz, np.nan))

    # ---- summary + plot ----
    import csv
    csv_path = os.path.join(OUT_DIR, "rim_bias_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("geometry,variant,delta_true_hz,recovered_hz\n")
        for row in results:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\nWrote {csv_path}")

    fig, axes = pl.subplots(1, len(GEOMETRIES), figsize=(13, 5), sharey=True)
    for ax, glabel in zip(np.atleast_1d(axes), GEOMETRIES):
        for vlabel in variants:
            pts = [(d, r) for g, v, d, r in results
                   if g == glabel and v == vlabel and np.isfinite(r)]
            if not pts:
                continue
            d = sorted(set(p[0] for p in pts))
            mb = [np.mean([r for dd, r in pts if dd == di]) - di for di in d]
            ax.plot(np.array(d) * 1e3, np.array(mb) * 1e3, "o-", ms=4,
                    label=vlabel)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(glabel, fontsize=10)
        ax.set_xlabel("injected delta (mHz)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("bias: recovered - injected (mHz)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    pl.tight_layout()
    png = os.path.join(OUT_DIR, "rim_bias_validation.png")
    pl.savefig(png, dpi=130)
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
