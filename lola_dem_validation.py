"""
LOLA DEM projection validation (REPORT 8.4): quantify the ellipsoid->DEM
mapping change and verify the projected maps move the way the terrain says.

Part 1 (geometry only): the delay/Doppler displacement field between the
ellipsoid and DEM surfaces at each session's reference epoch -- this *is*
the mapping systematic the DEM removes. Reports delay-pixel statistics
(REPORT claims ~7 px ~ 4 km at the extremes) and saves the field as a PNG.

Part 2 (single-look A/B): process one real look twice (use_dem off/on),
band-pass both maps on a common lon/lat grid, and cross-correlate ROIs
around high-relief features. The same DD image content must land displaced
by the prediction of the local first-order mapping Jacobian:
    dx = -J^-1 [dlt, ddlt],  J = d(lt, dlt)/d(east, north).
Agreement in direction and km-scale magnitude shows the DEM projection
moves real map content correctly, not just the model field.

Usage (from the repo root):
    .conda/bin/python lola_dem_validation.py [--nside 400] [--skip-ab]
"""

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl
from astropy.time import Time

import doppler_equator_alignment as dea
import registration_analysis as ra

OUT_DIR = os.path.join(os.path.dirname(__file__), "results/LOLA_DEM")
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data.camras.nl/lunar-radar")
RX_FILE = (DATA_ROOT + "/2025-09-16/stockert_radar_2025_09_16_13_23_26"
           "_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta")

# One reference epoch per session (real look epochs, as in rim_bias_validation).
EPOCHS = {
    "2025-06-21": "2025-06-21T08:59:29",
    "2025-09-11": "2025-09-11T08:05:44",
    "2025-09-16": "2025-09-16T13:23:26",
}
DDELAY_S = 1.0 / 0.25e6           # DD image delay column width (4 us)
T_WINDOW = 30.0                   # representative window for dlt differences

# High-relief / flat ROIs for the A/B feature-displacement check
# (selenographic east lon, lat, half-width deg).
ROIS = {
    "Tycho highlands": (-11.4, -43.3, 7.0),
    "Copernicus": (-20.1, 9.6, 7.0),
    "Mare Imbrium (flat)": (-17.0, 32.0, 7.0),
}


def lt_dlt_fields(rx_time, p, T=T_WINDOW):
    """Anchored-station (lt, window-difference dlt) for surface points p."""
    R0_rx, R0_tx, c = dea.apparent_station_positions(rx_time)
    R1_rx, R1_tx, _ = dea.apparent_station_positions(rx_time + T)
    lt0 = (np.linalg.norm(p - R0_rx, axis=-1) + np.linalg.norm(p - R0_tx, axis=-1)) / c
    lt1 = (np.linalg.norm(p - R1_rx, axis=-1) + np.linalg.norm(p - R1_tx, axis=-1)) / c
    return lt0, (lt1 - lt0) / T


def displacement_field(rx_time, nside):
    """Per-pixel (delay shift s, dlt shift, predicted east/north km) of the
    DEM mapping vs the ellipsoid mapping, near side only."""
    npix = hp.nside2npix(nside)
    v = np.array(hp.pix2vec(nside, np.arange(npix))).T
    near = v[:, 0] > 0.05
    u = v[near]
    p_ell = dea.moon_surface_points(u)
    p_dem = dea.moon_surface_points(u, use_dem=True)

    lt_e, dlt_e = lt_dlt_fields(rx_time, p_ell)
    lt_d, dlt_d = lt_dlt_fields(rx_time, p_dem)
    dlt_shift = dlt_d - dlt_e
    lt_shift = lt_d - lt_e

    # Local mapping Jacobian on the ellipsoid: d(lt, dlt) per km east/north.
    n = u / np.linalg.norm(u, axis=1, keepdims=True)
    e_lon = np.cross([0.0, 0.0, 1.0], n)
    e_lon /= np.linalg.norm(e_lon, axis=1, keepdims=True)
    e_lat = np.cross(n, e_lon)
    eps = 2.0  # km
    J = np.empty(u.shape[:1] + (2, 2))
    for k, e_vec in enumerate((e_lon, e_lat)):
        lt_p, dlt_p = lt_dlt_fields(rx_time, dea.moon_surface_points(p_ell + eps * e_vec))
        lt_m, dlt_m = lt_dlt_fields(rx_time, dea.moon_surface_points(p_ell - eps * e_vec))
        J[:, 0, k] = (lt_p - lt_m) / (2 * eps)
        J[:, 1, k] = (dlt_p - dlt_m) / (2 * eps)
    # Where the projection degenerates (Doppler equator / SRP) the inverse
    # blows up; those pixels are masked by multiplicity in the maps anyway.
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    good = np.abs(det) > np.quantile(np.abs(det), 0.05)
    disp = np.full((u.shape[0], 2), np.nan)
    rhs = np.stack([lt_shift, dlt_shift], axis=-1)
    disp[good] = -np.linalg.solve(J[good], rhs[good][..., np.newaxis])[..., 0]

    full = {}
    for name, vals in (("lt_shift_s", lt_shift), ("dlt_shift", dlt_shift),
                       ("east_km", disp[:, 0]), ("north_km", disp[:, 1])):
        m = np.full(npix, np.nan)
        m[near] = vals
        full[name] = m
    return full


def part1(nside):
    print("=== Part 1: ellipsoid->DEM displacement field (the mapping systematic) ===")
    for sess, iso in EPOCHS.items():
        rx_time = dea.et_from_astropy(Time(iso, scale="utc"))
        f = displacement_field(rx_time, nside)
        dpx = f["lt_shift_s"] / DDELAY_S
        ok = np.isfinite(dpx)
        absd = np.abs(dpx[ok])
        r_km = np.hypot(f["east_km"], f["north_km"])
        print(f"  {sess}: delay shift median {np.median(absd):.2f} px, "
              f"p95 {np.quantile(absd, 0.95):.2f} px, max {absd.max():.2f} px; "
              f">1 px on {100*np.mean(absd > 1):.0f}% of the disk; "
              f"surface displacement p95 {np.nanquantile(r_km, 0.95):.1f} km")
        m = np.where(ok, dpx, hp.UNSEEN)
        pl.close("all")
        fig = pl.figure(figsize=(10, 10))
        hp.orthview(m, title=f"DEM-ellipsoid delay shift (px of 4 us), {sess}",
                    flip="geo", fig=fig, half_sky=True, min=-8, max=8,
                    cmap="RdBu_r", xsize=1600)
        hp.graticule()
        png = os.path.join(OUT_DIR, f"delay_shift_px_{sess}.png")
        pl.savefig(png, dpi=130)
        pl.close(fig)
        print(f"    saved {png}")


def roi_slice(lon_axis, lat_axis, lon0, lat0, half):
    ix = np.where(np.abs(lon_axis - lon0) <= half)[0]
    iy = np.where(np.abs(lat_axis - lat0) <= half)[0]
    return np.ix_(iy, ix)


def part2(nside):
    print("\n=== Part 2: single-look A/B (same DD image, ellipsoid vs DEM projection) ===")
    rows = {}
    for label, use_dem in (("ELLIPSOID", False), ("DEM", True)):
        rows[label] = dea.process_file(RX_FILE, DATA_ROOT,
                                       os.path.join(OUT_DIR, label),
                                       nside=nside, use_dem=use_dem)
    print(f"  SRP elevation {rows['DEM']['srp_elevation_km']:+.3f} km "
          f"(topo delay {rows['DEM']['srp_topo_delay_us']:+.2f} us)")

    step = 0.075
    extent = 55.0
    lon_axis = np.arange(-extent, extent + step / 2, step)
    lat_axis = np.arange(-extent, extent + step / 2, step)
    lo_px, hi_px = 0.3 / step, 2.5 / step
    grids = {}
    for label in ("ELLIPSOID", "DEM"):
        r = rows[label]
        grids[label] = ra.grid_map(np.load(r["map_npy"]), lon_axis, lat_axis,
                                   multiplicity=np.load(r["mult_npy"]))
    band_e = ra.bandpass(grids["ELLIPSOID"], lo_px, hi_px)
    band_d = ra.bandpass(grids["DEM"], lo_px, hi_px)

    # Calibrate the xcorr sign convention with a known synthetic shift.
    probe = np.roll(band_e, 3, axis=1)  # content moved +3 px in lon
    dy0, dx0, _, _ = ra.xcorr_offset(band_e, probe, 10, 4)
    sgn = np.sign(dx0) * 3 / 3  # +1 if (a, shifted b) reports +shift
    assert abs(abs(dx0) - 3) < 0.5, "xcorr sign probe failed"

    # Predicted displacement field on the same lon/lat grid.
    rx_time = dea.et_from_astropy(Time(EPOCHS["2025-09-16"], scale="utc"))
    f = displacement_field(rx_time, nside)
    lon_g, lat_g = np.meshgrid(lon_axis, lat_axis)
    th = np.radians(90.0 - lat_g).ravel()
    ph = np.radians(lon_g).ravel()
    pred_e = hp.get_interp_val(np.nan_to_num(f["east_km"]), th, ph).reshape(lon_g.shape)
    pred_n = hp.get_interp_val(np.nan_to_num(f["north_km"]), th, ph).reshape(lon_g.shape)

    print(f"  {'ROI':22s} {'measured (E,N) km':>22s} {'predicted (E,N) km':>22s} "
          f"{'corr':>6s} {'signif':>7s}")
    search_px = int(1.5 / step)
    for name, (lon0, lat0, half) in ROIS.items():
        sl = roi_slice(lon_axis, lat_axis, lon0, lat0, half)
        a, b = band_e[sl], band_d[sl]
        if np.count_nonzero(a) < 100 or np.count_nonzero(b) < 100:
            print(f"  {name:22s} (outside valid map region)")
            continue
        dy, dx, peak, signif = ra.xcorr_offset(a, b, search_px, int(0.5 / step))
        me = sgn * dx * step * ra.KM_PER_DEG * np.cos(np.radians(lat0))
        mn = sgn * dy * step * ra.KM_PER_DEG
        pe = np.nanmean(np.where(a != 0, pred_e[sl], np.nan))
        pn = np.nanmean(np.where(a != 0, pred_n[sl], np.nan))
        print(f"  {name:22s} ({me:+6.2f}, {mn:+6.2f})        "
              f"({pe:+6.2f}, {pn:+6.2f})        {peak:6.3f} {signif:7.2f}")

    # Side-by-side visual of the two band-passed maps around Tycho.
    sl = roi_slice(lon_axis, lat_axis, *ROIS["Tycho highlands"][:2], 10.0)
    fig, axes = pl.subplots(1, 2, figsize=(14, 7))
    for ax, band, ttl in ((axes[0], band_e, "ellipsoid"), (axes[1], band_d, "LOLA DEM")):
        sub = band[sl]
        s = np.nanstd(sub[sub != 0])
        ax.imshow(sub, origin="lower", vmin=-3 * s, vmax=3 * s, cmap="gray")
        ax.set_title(f"Tycho region, {ttl} projection")
    pl.tight_layout()
    png = os.path.join(OUT_DIR, "tycho_ab.png")
    pl.savefig(png, dpi=130)
    pl.close(fig)
    print(f"  saved {png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, default=400)
    parser.add_argument("--skip-ab", action="store_true",
                        help="geometry-only (no GPU/raw-data processing)")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    dea.load_lola_dem()
    part1(args.nside)
    if not args.skip_ab:
        part2(args.nside)


if __name__ == "__main__":
    main()
