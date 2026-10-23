"""
LOLA DEM projection validation (REPORT 8.4): quantify the ellipsoid->DEM
mapping change and measure whether the projected maps move the way the
terrain says.

Measurements only: this script has no pass/fail criterion. It reports every
region it measures, with the correlation's significance and whether it is a
lock (validation_common.LOCK_SIGNIFICANCE, a labelling convention).

Part 1 (geometry only): the change in delay when the surface carries LOLA
terrain instead of being the smooth sphere, at each session's reference
epoch -- the mapping systematic the DEM removes. Reports delay-pixel
statistics both as the raw change and relative to the terrain at the SRP
(process_file measures delay from there when it maps with the DEM), and
saves the raw field as a PNG. The surface displacement in km needs the
inverse of the mapping Jacobian, which blows up where the mapping is
degenerate; the degeneracy mask comes from the maps Part 2 makes, so the km
statistics are reported there, not here.

Part 2 (single-look A/B): process one real look twice, use_dem=True then
use_dem=False with the first pass's Doppler calibration, so the two maps
are one DD image under two projections. Both passes apply the look's timing
and chain-frequency offsets, measured as the batch measures them; the
script stops if that measurement fails the stacker's timing gates
(freq_offset_hunt.measurement_problems), since production would not use
such a look. Both maps are masked to one set of pixels -- mapped and not
degenerate (multiplicity <= MAX_MULTIPLICITY) in both passes -- then
band-passed on a common lon/lat grid, and regions around chosen features
are cross-correlated. The same DD image content should land displaced by
the prediction of the local first-order mapping Jacobian:
    dx = -J^-1 [dlt_change, ddlt_change],  J = d(lt, dlt)/d(east, north),
averaged over the same pixels (those where the Jacobian is singular have no
prediction and are counted).

Part 2 needs raw SDR data (under data.camras.nl/) and a GPU; Part 1 is
geometry-only. Outputs land under validation/results, named after the
receiving station (the A/B capture's file prefix):
lola_dem_projection_<station>.json, lola_dem_delay_shift_px_<station>_<session>.png,
lola_dem_tycho_ab_<station>.png.

Usage (from the repo root):
    .conda/bin/python validation/scripts/validate_lola_dem_projection.py \
        [--nside 400] [--skip-ab]
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

import validation_common as vc

# validation_common puts the repo root on sys.path; import production code after.
import doppler_equator_alignment as dea
import freq_offset_hunt as foh
import registration_stability as rs
import registration_analysis as ra
import stack_maps as sm
from astropy import units as au

# All artifacts stay under validation/results; Part 2 intermediate map products
# go in a subdirectory so they do not clutter the summary outputs.
AB_DIR = vc.RESULTS_DIR / "lola_dem_ab"
DATA_ROOT = str(vc.REPO_ROOT / "data.camras.nl/lunar-radar")
RX_FILE = (DATA_ROOT + "/2025-09-16/stockert_radar_2025_09_16_13_23_26"
           "_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta")

# Stations: the transmitter, and the receiver named by the A/B capture's file
# prefix (an unknown prefix is an error); the prefix names the outputs. The
# Part 1 epochs are captures by the same receiver.
TX_NAME = dea.TX_STATION
RX_STATION = os.path.basename(RX_FILE).split("_")[0]
RX_NAME = dea.STATION_RX[RX_STATION]

# Timing search half-width, samples: the batch's own default.
MAX_SHIFT = rs.DEFAULT_MAX_SHIFT

# One reference epoch per session (real look epochs).
EPOCHS = {
    "2025-06-21": "2025-06-21T08:59:29",
    "2025-09-11": "2025-09-11T08:05:44",
    "2025-09-16": "2025-09-16T13:23:26",
}
DDELAY_S = 1.0 / 0.25e6           # DD image delay column width (4 us)
T_WINDOW = 30.0                   # representative window for Part 1's dlt

# Regions for the A/B feature-displacement measurement (selenographic east
# lon, lat, half-width deg). Every one is reported.
ROIS = {
    "Tycho highlands": (-11.4, -43.3, 7.0),
    "Copernicus": (-20.1, 9.6, 7.0),
    "Mare Imbrium (flat)": (-17.0, 32.0, 7.0),
}


def lt_dlt_fields(rx_time, p, T):
    """Anchored-station (lt, window-difference dlt) for surface points p."""
    R0_rx, R0_tx, c = dea.apparent_station_positions(rx_time, TX_NAME, RX_NAME)
    R1_rx, R1_tx, _ = dea.apparent_station_positions(rx_time + T, TX_NAME, RX_NAME)
    lt0 = (np.linalg.norm(p - R0_rx, axis=-1) + np.linalg.norm(p - R0_tx, axis=-1)) / c
    lt1 = (np.linalg.norm(p - R1_rx, axis=-1) + np.linalg.norm(p - R1_tx, axis=-1)) / c
    return lt0, (lt1 - lt0) / T


def near_side(nside):
    """(npix, near-side mask, near-side unit vectors) of a healpix grid."""
    npix = hp.nside2npix(nside)
    v = np.array(hp.pix2vec(nside, np.arange(npix))).T
    near = v[:, 0] > 0.05
    return npix, near, v[near]


def terrain_shift(rx_time, u, T):
    """Sphere points in directions u, and the change in their (lt,
    window-difference dlt) when the surface carries LOLA terrain."""
    p_ell = dea.moon_surface_points(u, use_dem=False)
    p_dem = dea.moon_surface_points(u, use_dem=True)
    lt_e, dlt_e = lt_dlt_fields(rx_time, p_ell, T)
    lt_d, dlt_d = lt_dlt_fields(rx_time, p_dem, T)
    return p_ell, lt_d - lt_e, dlt_d - dlt_e


def predicted_displacement(rx_time, p_ell, d_lt, d_dlt, T):
    """Surface displacement (east, north km) of content whose (lt, dlt)
    changes by (d_lt, d_dlt), to first order: -J^-1 [d_lt, d_dlt] with
    J = d(lt, dlt)/d(east, north) on the sphere. Where J is singular there is
    no solution: NaN, and the count is returned."""
    n = p_ell / np.linalg.norm(p_ell, axis=1, keepdims=True)
    e_lon = np.cross([0.0, 0.0, 1.0], n)
    e_lon /= np.linalg.norm(e_lon, axis=1, keepdims=True)
    e_lat = np.cross(n, e_lon)
    eps = 2.0  # km
    J = np.empty(p_ell.shape[:1] + (2, 2))
    for k, e_vec in enumerate((e_lon, e_lat)):
        lt_p, dlt_p = lt_dlt_fields(
            rx_time, dea.moon_surface_points(p_ell + eps * e_vec, use_dem=False), T)
        lt_m, dlt_m = lt_dlt_fields(
            rx_time, dea.moon_surface_points(p_ell - eps * e_vec, use_dem=False), T)
        J[:, 0, k] = (lt_p - lt_m) / (2 * eps)
        J[:, 1, k] = (dlt_p - dlt_m) / (2 * eps)
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    singular = det == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        east = -(J[:, 1, 1] * d_lt - J[:, 0, 1] * d_dlt) / det
        north = -(J[:, 0, 0] * d_dlt - J[:, 1, 0] * d_lt) / det
    east[singular] = np.nan
    north[singular] = np.nan
    return east, north, int(singular.sum())


def delay_stats(dpx):
    absd = np.abs(dpx)
    return {"median": float(np.median(absd)), "p95": float(np.quantile(absd, 0.95)),
            "max": float(absd.max()), "frac_gt_1px": float(np.mean(absd > 1))}


def part1(nside):
    """Ellipsoid->DEM delay change per session. Returns per-session stats for
    the JSON summary and saves a PNG of each raw delay-change map."""
    print("=== Part 1: ellipsoid->DEM delay change (the mapping systematic) ===")
    npix, near, u = near_side(nside)
    sessions = []
    for sess, iso in EPOCHS.items():
        rx_time = dea.et_from_astropy(Time(iso, scale="utc"))
        _, lt_shift, _ = terrain_shift(rx_time, u, T_WINDOW)
        _, srp_topo_delay_s = dea.extract_srp_elevation(rx_time, TX_NAME, RX_NAME)
        dpx = lt_shift / DDELAY_S
        if not np.all(np.isfinite(dpx)):
            raise RuntimeError(f"{sess}: {np.sum(~np.isfinite(dpx))} near-side points "
                               "have no terrain delay")
        # process_file measures delay from the terrain at the SRP, so a DEM
        # map moves content by the change relative to that point.
        dpx_rel = (lt_shift + srp_topo_delay_s) / DDELAY_S
        raw, rel = delay_stats(dpx), delay_stats(dpx_rel)
        stats = {"session": sess, "srp_topo_delay_us": srp_topo_delay_s * 1e6,
                 "delay_shift_px": raw, "delay_shift_rel_srp_px": rel}
        print(f"  {sess}: delay change median {raw['median']:.2f} px, "
              f"p95 {raw['p95']:.2f} px, max {raw['max']:.2f} px, "
              f">1 px on {100 * raw['frac_gt_1px']:.0f}% of the near side; "
              f"relative to the SRP terrain: median {rel['median']:.2f}, "
              f"p95 {rel['p95']:.2f}, max {rel['max']:.2f} px")
        m = np.full(npix, hp.UNSEEN)
        m[near] = dpx
        lim = raw["max"]
        pl.close("all")
        fig = pl.figure(figsize=(10, 10))
        hp.orthview(m, title=f"DEM-ellipsoid delay change (px of 4 us), {sess}; "
                             f"colour range +-{lim:.2f} px (the largest)",
                    flip="geo", fig=fig, half_sky=True, min=-lim, max=lim,
                    cmap="RdBu_r", xsize=1600)
        hp.graticule()
        png = vc.RESULTS_DIR / f"lola_dem_delay_shift_px_{RX_STATION}_{sess}.png"
        pl.savefig(png, dpi=130)
        pl.close(fig)
        stats["png"] = vc.report_path(png)
        print(f"    saved {vc.report_path(png)}")
        sessions.append(stats)
    return sessions


def roi_slice(lon_axis, lat_axis, lon0, lat0, half):
    ix = np.where(np.abs(lon_axis - lon0) <= half)[0]
    iy = np.where(np.abs(lat_axis - lat0) <= half)[0]
    return np.ix_(iy, ix)


def measured_offsets():
    """Timing and chain-frequency offsets of RX_FILE, measured as
    registration_stability.process_one measures them (+-MAX_SHIFT search).
    Stops if the measurement fails the stacker's timing gates
    (freq_offset_hunt.measurement_problems): production would not stack
    such a look, so its A/B would not test the mapping production uses.
    Returns the measurement's record (with the offsets to apply) for the
    JSON."""
    rx, tx, fs_q, f_q, rx_start, _, tx_file = dea.load_observation(RX_FILE, DATA_ROOT)
    fs = fs_q.to_value(au.Hz)
    tx_comp = foh.compensated_tx(rx, tx, fs_q, f_q, rx_start,
                                 dea.nominal_tx_emit_start(rx_start),
                                 tx_name=TX_NAME, rx_name=RX_NAME)
    res = foh.measure_offset(rx, tx_comp, fs, dea.chip_samples_of(tx_file, fs),
                             max_shift=MAX_SHIFT)
    print(f"  measured timing {res['shift_refined']:+.2f} samples "
          f"({res['timing_status']}), chain frequency {res['f_centroid']:+.3f} Hz "
          f"(tone snr {res['snr']:.0f}"
          + (", AT THE EDGE OF THE SEARCH BAND)" if res["tone_at_band_edge"] else ")"))
    reasons = foh.measurement_problems(res, MAX_SHIFT, foh.MIN_TONE_SNR)
    if reasons:
        raise SystemExit(f"{os.path.basename(RX_FILE)}: {'; '.join(reasons)}. The stacker "
                         "would leave this look out, so its A/B would not test the "
                         "mapping production uses.")
    return {"shift_refined_samples": res["shift_refined"],
            "timing_status": res["timing_status"],
            "tone_snr": res["snr"], "min_tone_snr": foh.MIN_TONE_SNR,
            "tone_at_band_edge": res["tone_at_band_edge"],
            "max_shift_samples": MAX_SHIFT,
            "tx_extra_offset_s": res["shift_refined"] / fs,
            "freq_offset_hz": res["f_centroid"]}


def part2(nside):
    """Single-look A/B feature displacement. Returns measured-vs-predicted
    displacement for every region plus the SRP terrain terms for the JSON."""
    print("\n=== Part 2: single-look A/B (same DD image, ellipsoid vs DEM projection) ===")
    AB_DIR.mkdir(parents=True, exist_ok=True)
    # Both passes image the capture with the timing and chain-frequency
    # offsets measured as the batch measures them, so the echo sits where
    # production puts it (the terrain shift being tested is measured from
    # the echo's leading edge at lag 0); the two DD images are identical.
    timing = measured_offsets()
    common = dict(tx_name=TX_NAME, rx_name=RX_NAME,
                  tx_extra_offset_s=timing["tx_extra_offset_s"],
                  freq_offset_hz=timing["freq_offset_hz"])
    dem = dea.process_file(RX_FILE, DATA_ROOT, str(AB_DIR / "DEM"), nside,
                           use_dem=True, **common)
    if dem["rim_status"] != "converged" or dem["edge_status"] != "fitted":
        raise SystemExit(f"DEM pass: rim calibration {dem['rim_status']}, edge fit "
                         f"{dem['edge_status']}; the A/B shares a converged Doppler "
                         "calibration and a fitted edge correction between its passes")
    # The sphere pass takes the DEM pass's Doppler calibration and edge-fit
    # offsets rather than measuring its own, so the maps differ only in the
    # projection (and the delay reference below), which is what the
    # prediction models.
    ell = dea.process_file(RX_FILE, DATA_ROOT, str(AB_DIR / "ELLIPSOID"), nside,
                           use_dem=False, rim_delta_hz=float(dem["rim_delta_hz"]),
                           edge_offsets=(float(dem["edge_delay_us"]) * 1e-6,
                                         float(dem["edge_doppler_hz"])),
                           **common)
    rows = {"ELLIPSOID": ell, "DEM": dem}
    srp_topo_delay_s = float(dem["srp_topo_delay_us"]) * 1e-6
    print(f"  SRP elevation {dem['srp_elevation_km']:+.3f} km (topo delay "
          f"{dem['srp_topo_delay_us']:+.2f} us); Doppler calibration "
          f"{1e3 * float(dem['rim_delta_hz']):+.1f} mHz in both passes")

    step = 0.075
    extent = 55.0
    lon_axis = np.arange(-extent, extent + step / 2, step)
    lat_axis = np.arange(-extent, extent + step / 2, step)
    lo_px, hi_px = 0.3 / step, 2.5 / step
    maps = {k: np.load(r["map_npy"]) for k, r in rows.items()}
    mults = {k: np.load(r["mult_npy"]) for k, r in rows.items()}
    npix, near, u = near_side(nside)
    if any(len(m) != npix for m in maps.values()):
        raise RuntimeError(f"the A/B maps are not nside {nside} maps")
    # One mask for both maps and the prediction: pixels mapped and not
    # degenerate (the production degeneracy mask) in both passes. Band-passed
    # on their own masks, the maps' edges would differ, and the
    # cross-correlation can read that as a displacement.
    keep = (sm.degeneracy_mask(maps["ELLIPSOID"], mults["ELLIPSOID"])
            & sm.degeneracy_mask(maps["DEM"], mults["DEM"]))

    def on_keep(values):
        out = np.full(npix, hp.UNSEEN)
        out[keep] = values[keep]
        return out

    # Single-look log maps have a value at every kept pixel (no noise floor
    # removed), so the plain band-pass applies.
    grids = {k: ra.grid_map(on_keep(maps[k]), lon_axis, lat_axis) for k in rows}
    band_e = ra.bandpass(grids["ELLIPSOID"], lo_px, hi_px)
    band_d = ra.bandpass(grids["DEM"], lo_px, hi_px)

    # xcorr_offset(a, b) returns the correction that moves b onto a
    # (registration_analysis.xcorr_offset; test/test_registration_conventions.py),
    # so b's content sits at minus that relative to a's. Check it on a known
    # shift of this data before relying on it.
    probe = ra.xcorr_offset(band_e, np.roll(band_e, 3, axis=1), 10)
    if probe.at_edge or abs(probe.dx + 3) > 0.5:
        raise RuntimeError(f"xcorr_offset sign check: content moved +3 px east "
                           f"gave dx {probe.dx:+.2f} (expected -3)")

    # Predicted displacement for the capture's own epoch and window (as
    # process_file mapped it). The DEM pass measures delay from the terrain
    # at the SRP (lt_ref = lt_min_image - srp_topo_delay_s), so a point's
    # delay changes by lt_shift + srp_topo_delay_s between the passes; the
    # Doppler calibration is shared, so its Doppler changes by dlt_shift.
    rx_time = dea.et_from_astropy(Time(dem["rx_start_utc"], scale="utc"))
    p_ell, lt_shift, dlt_shift = terrain_shift(rx_time, u, float(dem["capture_s"]))
    east, north, n_singular = predicted_displacement(
        rx_time, p_ell, lt_shift + srp_topo_delay_s, dlt_shift, float(dem["capture_s"]))

    # Predictions exist on near_side's pixels without a singular Jacobian;
    # mapped pixels without one are counted, not filled.
    disp = np.full(npix, np.nan)
    disp[near] = np.hypot(east, north)
    disp_km = vc.finite_stats(disp[keep])
    n_kept = int(keep.sum())
    if disp_km["n"] == 0:
        raise RuntimeError("no mapped, non-degenerate pixel has a predicted displacement")
    print(f"  predicted surface displacement over the {disp_km['n']} of {n_kept} mapped, "
          f"non-degenerate pixels that have one: median {disp_km['median']:.2f} km, "
          f"p95 {disp_km['p95_abs']:.2f} km, max {disp_km['max_abs']:.2f} km "
          f"({n_kept - disp_km['n']} without one: beyond the near-side cut or a "
          f"singular Jacobian; {n_singular} near-side pixels have a singular Jacobian)")

    pred = {}
    for name, vals in (("east", east), ("north", north)):
        m = np.full(npix, np.nan)
        m[near] = vals
        pred[name] = ra.grid_map(on_keep(m), lon_axis, lat_axis)
    mapped = np.isfinite(grids["ELLIPSOID"])      # the same cells as the DEM grid
    has_pred = np.isfinite(pred["east"]) & np.isfinite(pred["north"])
    valid = mapped & has_pred
    lat_g = np.broadcast_to(lat_axis[:, None], valid.shape)

    def km(e, n):
        return "(none)" if e is None else f"({e:+6.2f}, {n:+6.2f})"

    print(f"  {'ROI':22s} {'measured (E,N) km':>20s} {'predicted (E,N) km':>20s} "
          f"{'corr':>6s} {'signif':>7s}  lock")
    search_px = int(1.5 / step)
    rois = []
    for name, (lon0, lat0, half) in ROIS.items():
        sl = roi_slice(lon_axis, lat_axis, lon0, lat0, half)
        a, b = band_e[sl], band_d[sl]
        if not (a.any() and b.any()):
            print(f"  {name:22s} no band-passed map content in the region")
            rois.append({"roi": name, "skipped": "no band-passed map content in the region"})
            continue
        x = ra.xcorr_offset(a, b, search_px)
        v = valid[sl]
        # Grid cells with map content but no prediction are left out of the
        # predicted mean (counted).
        rec = {"roi": name, "n_valid_px": int(v.sum()),
               "n_mapped_px_without_prediction": int((mapped[sl] & ~has_pred[sl]).sum()),
               "xcorr_peak": x.peak, "xcorr_signif": x.significance,
               "at_edge": x.at_edge, "lock": vc.is_lock(x)}
        if x.at_edge:
            # The peak is on the edge of the search window: no measurement.
            rec["measured_east_km"] = rec["measured_north_km"] = None
        else:
            # Longitude degrees to km at the mean cos(latitude) of the
            # pixels the correlation sees (map content in either pass).
            seen = (a != 0) | (b != 0)
            cos_lat = float(np.mean(np.cos(np.radians(lat_g[sl][seen]))))
            rec["measured_east_km"] = float(-x.dx * step * ra.KM_PER_DEG * cos_lat)
            rec["measured_north_km"] = float(-x.dy * step * ra.KM_PER_DEG)
        if v.any():
            rec["predicted_east_km"] = float(np.mean(pred["east"][sl][v]))
            rec["predicted_north_km"] = float(np.mean(pred["north"][sl][v]))
        else:
            rec["predicted_east_km"] = rec["predicted_north_km"] = None
        rois.append(rec)
        print(f"  {name:22s} {km(rec['measured_east_km'], rec['measured_north_km']):>20s} "
              f"{km(rec['predicted_east_km'], rec['predicted_north_km']):>20s} "
              f"{x.peak:6.3f} {x.significance:7.2f}  {'yes' if rec['lock'] else 'no'}"
              + ("  AT THE SEARCH EDGE" if x.at_edge else ""))
    print(f"  lock: significance >= {vc.LOCK_SIGNIFICANCE:g} and not at the search "
          "edge (a labelling convention). Measurements only: no pass/fail criterion.")

    # Side-by-side visual of the two band-passed maps around Tycho, on one
    # grey scale (+-3 standard deviations of both together).
    sl = roi_slice(lon_axis, lat_axis, *ROIS["Tycho highlands"][:2], 10.0)
    subs = [band_e[sl], band_d[sl]]
    s = np.std(np.concatenate([x[x != 0] for x in subs]))
    fig, axes = pl.subplots(1, 2, figsize=(14, 7))
    for ax, sub, ttl in zip(axes, subs, ("ellipsoid", "LOLA DEM")):
        ax.imshow(sub, origin="lower", vmin=-3 * s, vmax=3 * s, cmap="gray")
        ax.set_title(f"Tycho region, {ttl} projection (grey scale +-3 std)")
    pl.tight_layout()
    png = vc.RESULTS_DIR / f"lola_dem_tycho_ab_{RX_STATION}.png"
    pl.savefig(png, dpi=130)
    pl.close(fig)
    print(f"  saved {vc.report_path(png)}")
    return {
        "rx_file": os.path.basename(RX_FILE),
        "tx_name": TX_NAME, "rx_name": RX_NAME,
        "timing_measurement": timing,
        "n_mapped_nondegenerate_px": n_kept,
        "rim_status_dem_pass": dem["rim_status"],
        "rim_delta_hz_both_passes": float(dem["rim_delta_hz"]),
        "srp_elevation_km": float(dem["srp_elevation_km"]),
        "srp_topo_delay_us": float(dem["srp_topo_delay_us"]),
        "predicted_surface_disp_km_masked": disp_km,
        "n_singular_jacobian_px": n_singular,
        "lock_significance": vc.LOCK_SIGNIFICANCE,
        "rois": rois,
        "tycho_png": vc.report_path(png),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nside", type=int, default=400)
    parser.add_argument("--skip-ab", action="store_true",
                        help="geometry-only (no GPU/raw-data processing)")
    args = parser.parse_args()
    vc.ensure_dirs()
    dea.load_lola_dem()

    payload = {
        "purpose": ("ellipsoid->DEM delay change (the systematic the DEM removes) "
                    "and a single-look A/B measurement of how real map content "
                    "moves against the local mapping Jacobian's prediction"),
        "criterion": "none: measurements only",
        # Part 1's epochs are this receiver's looks; Part 2 images its capture.
        "tx_name": TX_NAME, "rx_name": RX_NAME,
        "nside": args.nside,
        "part1_delay_change": part1(args.nside),
    }
    payload["part2_single_look_ab"] = ("not run (--skip-ab)" if args.skip_ab
                                       else part2(args.nside))

    path = vc.write_json(f"lola_dem_projection_{RX_STATION}.json", payload)
    print(f"\nwrote {vc.report_path(path)}")


if __name__ == "__main__":
    main()
