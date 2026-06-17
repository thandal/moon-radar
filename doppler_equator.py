"""
Doppler Equator and bistatic Moon-bounce geometry (SPICE).

Light times, Doppler (d lt/dt), bistatic specular point, window-averaged
Doppler with differential-rate correction, anchored apparent station
positions, and the Doppler-equator boundary-curve methods.

Conventions:
  * dlt is delta-light-time; doppler = -dlt * frequency. "up" branches are
    the approaching (dlt-min) side in every method.
  * Doppler derivatives are central finite differences of the total light
    time. SPICE light times from this kernel chain are quantized at ~2e-11 s,
    so short-baseline derivatives are noisy (~0.25 Hz at L-band for
    dt=0.05 s): pass a wide dt (e.g. half the integration window) when values
    label Doppler bins, or use the window-average identity (rate_corrected_dlt).

This module does not furnish SPICE kernels; callers must furnsh a kernel set
including observatories and observatory radii (see observatory_radii.tpc)
before calling anything here.
"""

import os
import re

import numpy as np
import healpy as hp
import cspyce as csp

AB_COR = "LT"
EARTH_FRAME = "ITRF93"
DLT_DT = 0.05


# ---------------------------------------------------------------------------
# SPICE helpers
# ---------------------------------------------------------------------------
def et_from_astropy(t):
    try:
        return float(t)
    except (TypeError, ValueError):
        # astropy's default ISO precision is 3 (milliseconds) -- a plain
        # str2et(t.utc.value) would silently quantize sub-ms epochs (e.g.
        # measured us-level timing corrections). Render at ns precision.
        tu = t.utc.replicate()
        tu.precision = 9
        return csp.str2et(tu.isot)


_MOON_RADII = None

def moon_radii():
    """Cached MOON RADII lookup (bodvrd is surprisingly hot in inner loops)."""
    global _MOON_RADII
    if _MOON_RADII is None:
        _MOON_RADII = csp.bodvrd("MOON", "RADII")
    return _MOON_RADII


# ---------------------------------------------------------------------------
# LOLA DEM (LRO LOLA GDR cylindrical PDS IMG; see fetch_lola_dem.sh)
# ---------------------------------------------------------------------------
_LOLA_INTERPOLATOR = None
_LOLA_DEM_PATH = None
LOLA_DEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "lola_dem")


def _parse_pds_label(lbl_path):
    """Minimal PDS3 'KEY = value' line parser (enough for the LOLA GDR labels)."""
    keys = {}
    with open(lbl_path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*\^?([A-Z][A-Z0-9_]*)\s*=\s*(.+?)\s*$", line)
            if m:
                keys.setdefault(m.group(1), m.group(2).strip().strip('"'))
    return keys


def _pds_number(value):
    """Numeric value of a PDS field, dropping a trailing unit like '<deg>'."""
    return float(value.split("<")[0])


def find_lola_dem(dem_dir=None):
    """Path of the highest-resolution ldem_<ppd>.img in lola_dem/, or None."""
    dem_dir = LOLA_DEM_DIR if dem_dir is None else dem_dir
    if not os.path.isdir(dem_dir):
        return None
    found = []
    for name in os.listdir(dem_dir):
        m = re.match(r"ldem_(\d+)\.img$", name, re.IGNORECASE)
        if m:
            found.append((int(m.group(1)), os.path.join(dem_dir, name)))
    return max(found)[1] if found else None


def load_lola_dem(dem_path=None):
    """Load a LOLA GDR cylindrical DEM (.img + detached .lbl) and build the
    global elevation interpolator behind moon_surface_points(use_dem=True).

    With no path, picks the highest-resolution DEM in lola_dem/ (see
    fetch_lola_dem.sh). Idempotent per path; returns the loaded path.

    The GDR grid is pixel-registered simple cylindrical in the DE421
    mean-Earth/polar frame (= MOON_ME here); heights are meters relative to
    the 1737.4 km sphere -- the same sphere as the SPICE MOON RADII, so
    elevations add radially onto the ellipsoid surface.
    """
    global _LOLA_INTERPOLATOR, _LOLA_DEM_PATH
    from scipy.interpolate import RegularGridInterpolator

    if dem_path is None:
        dem_path = find_lola_dem()
        if dem_path is None:
            raise FileNotFoundError(
                f"no ldem_*.img in {LOLA_DEM_DIR}; run ./fetch_lola_dem.sh")
    dem_path = os.path.abspath(dem_path)
    if dem_path == _LOLA_DEM_PATH:
        return dem_path

    lbl = _parse_pds_label(os.path.splitext(dem_path)[0] + ".lbl")
    if lbl["SAMPLE_TYPE"] != "LSB_INTEGER" or lbl["SAMPLE_BITS"] != "16":
        raise ValueError(f"unsupported sample format for {dem_path}")
    lines = int(lbl["LINES"])
    samples = int(lbl["LINE_SAMPLES"])
    scale_km = _pds_number(lbl["SCALING_FACTOR"]) / 1000.0
    res = _pds_number(lbl["MAP_RESOLUTION"])           # pix/deg
    max_lat = _pds_number(lbl["MAXIMUM_LATITUDE"])
    west_lon = _pds_number(lbl["WESTERNMOST_LONGITUDE"])
    ref_radius_km = _pds_number(lbl["OFFSET"]) / 1000.0
    if abs(ref_radius_km - moon_radii()[0]) > 1e-6:
        raise ValueError(f"DEM reference radius {ref_radius_km} km != SPICE "
                         f"MOON radius {moon_radii()[0]} km")

    dn = np.fromfile(dem_path, dtype="<i2")
    if dn.size != lines * samples:
        raise ValueError(f"{dem_path}: {dn.size} samples != {lines}x{samples}")
    elev_km = dn.reshape(lines, samples).astype(np.float32)
    elev_km *= np.float32(scale_km)

    # Pixel-center axes; line 1 is the northernmost row -> flip lat ascending.
    lats = np.radians(max_lat - (np.arange(lines) + 0.5) / res)[::-1]
    elev_km = elev_km[::-1]
    lons = np.radians(west_lon + (np.arange(samples) + 0.5) / res)
    # Wrap one column around each lon edge so queries anywhere in [0, 2pi)
    # interpolate across the seam instead of extrapolating.
    lons = np.concatenate(([lons[-1] - 2 * np.pi], lons, [lons[0] + 2 * np.pi]))
    elev_km = np.concatenate((elev_km[:, -1:], elev_km, elev_km[:, :1]), axis=1)

    # fill_value=None extrapolates the sub-pixel sliver beyond the polar
    # pixel-center rows (|lat| > max_lat - 0.5/res).
    _LOLA_INTERPOLATOR = RegularGridInterpolator(
        (lats, lons), elev_km, bounds_error=False, fill_value=None)
    _LOLA_DEM_PATH = dem_path
    return dem_path


def get_lola_elevation(p_moon):
    """LOLA elevation (km above the 1737.4 km sphere) along each MOON_ME
    direction in p_moon (..., 3). Vector magnitudes are ignored."""
    if _LOLA_INTERPOLATOR is None:
        raise RuntimeError("LOLA DEM not loaded; call load_lola_dem() first")
    p = np.asarray(p_moon, dtype=float)
    lat = np.arctan2(p[..., 2], np.hypot(p[..., 0], p[..., 1]))
    lon = np.mod(np.arctan2(p[..., 1], p[..., 0]), 2 * np.pi)
    elev = _LOLA_INTERPOLATOR(np.stack([lat, lon], axis=-1))
    return elev.reshape(p.shape[:-1])


def moon_surface_points(p_moon, use_dem=False):
    """Radial projection of vectors onto the lunar surface.

    use_dem=False: the SPICE ellipsoid (the 1737.4 km sphere). use_dem=True:
    sphere + LOLA topography when a DEM is loaded (ellipsoid fallback
    otherwise). The specular zoom and the equator/rim curves keep the
    default: the minimum-light-time search needs the smooth convex surface
    (terrain would trap the zoom in local minima), and the rim calibration
    is differential in Doppler where topography is second-order.
    """
    r = moon_radii()
    if not use_dem or _LOLA_INTERPOLATOR is None:
        return csp.edpnt_vector(p_moon, r[0], r[1], r[2])
    p = np.asarray(p_moon, dtype=float)
    u = p / np.linalg.norm(p, axis=-1, keepdims=True)
    return u * (r[0] + get_lola_elevation(u))[..., np.newaxis]


def extract_srp_elevation(rx_time, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """LOLA elevation at the ellipsoid SRP and its two-way topographic delay.

    The SRP stays the smooth-ellipsoid anchor; this samples the terrain
    under it. The echo's true minimum delay leads the ellipsoid prediction
    by ~2h/c (incidence at the SRP is near-normal and the bistatic angle is
    small, so the cos factors are within <1 % of 1) -- subtracting this from
    a measured per-look timing offset isolates the SDR hardware jitter.

    Returns (elevation_km, delay_shift_s), delay_shift_s negative below the
    reference sphere.
    """
    srp = specular_point_bck(rx_time, tx_name, rx_name)
    elev_km = float(get_lola_elevation(srp))
    return elev_km, 2.0 * elev_km / csp.clight()


def moonPointLightTime_BCK(rx_time, p_moon, tx_name="DWINGELOO", rx_name="STOCKERT",
                           use_dem=False):
    """Two-leg TX-surface-RX light time as a function of receive ET."""
    p_surf = moon_surface_points(p_moon, use_dem=use_dem)
    _, lt_rx = csp.spkcpt_vector(p_surf, "MOON", "MOON_ME", rx_time,
                                  EARTH_FRAME, "OBSERVER", AB_COR, rx_name)
    _, lt_tx = csp.spkcpo_vector(tx_name, rx_time - lt_rx, EARTH_FRAME,
                                  "TARGET", AB_COR, p_surf, "MOON", "MOON_ME")
    return lt_rx + lt_tx


def moonPointLightTime_FWD(tx_time, p_moon, tx_name="DWINGELOO", rx_name="STOCKERT",
                           use_dem=False):
    """Two-leg TX-surface-RX light time as a function of transmit ET."""
    p_surf = moon_surface_points(p_moon, use_dem=use_dem)
    _, lt_tx = csp.spkcpt_vector(p_surf, "MOON", "MOON_ME", tx_time,
                                  EARTH_FRAME, "TARGET", "X" + AB_COR, tx_name)
    _, lt_rx = csp.spkcpo_vector(rx_name, tx_time + lt_tx, EARTH_FRAME,
                                  "OBSERVER", "X" + AB_COR, p_surf, "MOON", "MOON_ME")
    return lt_tx + lt_rx


def moonPointDLT_BCK(rx_time, p_moon, tx_name="DWINGELOO", rx_name="STOCKERT",
                     dt=DLT_DT, use_dem=False):
    """Compute light time and fractional Doppler from d(total light time)/d(rx ET)."""
    lt = moonPointLightTime_BCK(rx_time, p_moon, tx_name, rx_name, use_dem)
    lt_plus = moonPointLightTime_BCK(rx_time + dt, p_moon, tx_name, rx_name, use_dem)
    lt_minus = moonPointLightTime_BCK(rx_time - dt, p_moon, tx_name, rx_name, use_dem)
    dlt = (lt_plus - lt_minus) / (2 * dt)
    return lt, dlt


def moonPointDLT_FWD(tx_time, p_moon, tx_name="DWINGELOO", rx_name="STOCKERT",
                     dt=DLT_DT, use_dem=False):
    """Compute light time and fractional Doppler from d(total light time)/d(tx ET)."""
    lt = moonPointLightTime_FWD(tx_time, p_moon, tx_name, rx_name, use_dem)
    lt_plus = moonPointLightTime_FWD(tx_time + dt, p_moon, tx_name, rx_name, use_dem)
    lt_minus = moonPointLightTime_FWD(tx_time - dt, p_moon, tx_name, rx_name, use_dem)
    dlt = (lt_plus - lt_minus) / (2 * dt)
    return lt, dlt


def subpoint_average_guess(rx_time, tx_name="DWINGELOO", rx_name="STOCKERT"):
    srp_rx, trg, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                                 "MOON_ME", AB_COR, rx_name)
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", trg,
                               "MOON_ME", AB_COR, tx_name)
    return moon_surface_points((srp_rx + srp_tx) / 2.0)


def _specular_zoom(lt_fn, x0, start_step_km=100.0, stop_step_km=0.05):
    """Minimum-light-time surface point by iterative tangent-plane grid zoom.

    Each iteration evaluates a 5x5 local grid in ONE vectorized SPICE call
    (the surface projection inside lt_fn removes the radial degree of
    freedom), then recenters and shrinks. ~9 vectorized calls replace the
    hundreds of scalar calls Nelder-Mead needed. The final grid spacing
    (~50 m) bounds the light-time error by ~1e-12 s near the quadratic
    minimum.
    """
    g = np.linspace(-2, 2, 5)
    gx, gy = np.meshgrid(g, g)
    offs = np.column_stack([gx.ravel(), gy.ravel()])
    x = np.asarray(x0, dtype=float)
    step = start_step_km
    while step > stop_step_km:
        n = x / np.linalg.norm(x)
        e1 = np.cross(n, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(n, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        pts = x + step * (offs[:, :1] * e1 + offs[:, 1:] * e2)
        lts = lt_fn(pts)
        x = moon_surface_points(pts[np.argmin(lts)])
        step /= 2.5
    return x


def specular_point_bck(rx_time, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Minimum two-leg light-time point on the lunar ellipsoid for receive ET."""
    x0 = subpoint_average_guess(rx_time, tx_name, rx_name)
    return _specular_zoom(
        lambda pts: moonPointLightTime_BCK(rx_time, pts, tx_name, rx_name), x0)


def specular_point_fwd(tx_time, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Minimum two-leg light-time point on the lunar ellipsoid for transmit ET."""
    srp_tx, trg, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", tx_time,
                                 "MOON_ME", "X" + AB_COR, tx_name)
    srp_rx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", trg,
                               "MOON_ME", AB_COR, rx_name)
    x0 = moon_surface_points((srp_tx + srp_rx) / 2.0)
    return _specular_zoom(
        lambda pts: moonPointLightTime_FWD(tx_time, pts, tx_name, rx_name), x0)


def moonSRP_DLT_BCK(rx_time, tx_name="DWINGELOO", rx_name="STOCKERT", dt=DLT_DT):
    """Compute (lt, dlt) for the bistatic minimum-delay surface point."""
    return moonPointDLT_BCK(rx_time, specular_point_bck(rx_time, tx_name, rx_name),
                            tx_name, rx_name, dt=dt)


def moonSRP_DLT_FWD(tx_time, tx_name="DWINGELOO", rx_name="STOCKERT", dt=DLT_DT):
    """Compute (lt, dlt) for the bistatic minimum-delay surface point."""
    return moonPointDLT_FWD(tx_time, specular_point_fwd(tx_time, tx_name, rx_name),
                            tx_name, rx_name, dt=dt)


# NOTE on numerical noise: SPICE light times from this kernel chain are
# quantized at ~2e-11 s (~6 mm of path). Finite-difference Doppler over a
# short baseline amplifies that: with dt=0.05 s the dlt noise is ~2e-10
# (~0.25 Hz at L-band, tens of Doppler bins). Derivative baselines should be
# as wide as the physics allows; the helpers below use the full RX window.

def srp_dlt_rate_bck(rx_time, rx_duration_s, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Average d(dlt)/dt of the tracked SRP over the RX window.

    Wide (T/2) stencils suppress the light-time granularity noise. Prefer
    passing the dlt_rate_srp returned by compute_dd_image to the equator
    functions so the correction uses exactly the rate that was compensated
    in the image; this is the fallback when no image is available."""
    _, dlt0 = moonSRP_DLT_BCK(rx_time, tx_name, rx_name, dt=rx_duration_s / 2)
    _, dlt1 = moonSRP_DLT_BCK(rx_time + rx_duration_s, tx_name, rx_name,
                              dt=rx_duration_s / 2)
    return (dlt1 - dlt0) / rx_duration_s


def rate_corrected_dlt(rx_time, p_moon, rx_duration_s, dlt_rate_srp,
                       tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Window-averaged dlt of surface points, in the image's start-anchored axis.

    The DD image compensates the SRP Doppler *rate* over the integration
    window, so a surface point whose dlt rate differs from the SRP's keeps a
    residual chirp and its correlation energy centers at its window-averaged
    Doppler. In the start-anchored dlt row axis that is
        dlt_eff = mean_t[dlt_p(t)] - dlt_rate_srp * T/2
    and the window mean is computed exactly (and with ~1 mHz noise, see note
    above) from the light-time identity mean(d lt/dt) = (lt(t0+T)-lt(t0))/T.
    Physical magnitude ~ +/-0.015 Hz at the limbs for T=66 s (see
    test/test_midpoint_fix.py); dlt_eff equals dlt(t0) at the SRP itself."""
    lt0 = moonPointLightTime_BCK(rx_time, p_moon, tx_name, rx_name)
    lt1 = moonPointLightTime_BCK(rx_time + rx_duration_s, p_moon, tx_name, rx_name)
    return (lt1 - lt0) / rx_duration_s - dlt_rate_srp * rx_duration_s / 2


# ---------------------------------------------------------------------------
# Step 1: Doppler equator sampling
# ---------------------------------------------------------------------------
def compute_doppler_equator(rx_time, n_delay_bins=500, nside=200,
                            tx_name="DWINGELOO", rx_name="STOCKERT"):
    """
    Compute the Doppler equator: for each delay bin, find the max and min DLT
    among visible moon surface points.

    NOTE on naming: dlt is delta-light-time, so doppler = -dlt * frequency.
    dlt_max is the fastest-RECEDING edge (down_doppler, most negative
    frequency shift); dlt_min is the approaching edge (up_doppler).

    Returns:
        lt_min: minimum light time (at SRP)
        delay_centers: (n_bins,) array of delay values (seconds from SRP)
        dlt_max: (n_bins,) max DLT per delay bin (down_doppler equator)
        dlt_min: (n_bins,) min DLT per delay bin (up_doppler equator)
    """
    NPIX = hp.nside2npix(nside)
    v = np.array(hp.pix2vec(nside, np.arange(NPIX))).T
    # Keep only the near side (MOON_ME +X points toward Earth). Far-side
    # points all have delay > 2R/c so they never enter the visible delay
    # range, but without this filter they double the binning span and halve
    # the bin resolution.
    v = v[v[:, 0] > 0]

    lt, dlt = moonPointDLT_BCK(rx_time, v, tx_name, rx_name)

    # Bin by delay (relative to minimum light time = SRP)
    lt_min = lt.min()
    lt_max = lt.max()
    delay_edges = np.linspace(0, lt_max - lt_min, n_delay_bins + 1)
    delay_centers = (delay_edges[:-1] + delay_edges[1:]) / 2

    delay_rel = lt - lt_min
    bin_idx = np.clip(np.digitize(delay_rel, delay_edges) - 1, 0, n_delay_bins - 1)

    dlt_max = np.full(n_delay_bins, np.nan)
    dlt_min = np.full(n_delay_bins, np.nan)
    for b in range(n_delay_bins):
        mask = bin_idx == b
        if mask.any():
            dlt_max[b] = dlt[mask].max()
            dlt_min[b] = dlt[mask].min()

    return lt_min, delay_centers, dlt_max, dlt_min


def compute_doppler_equator_velocity(rx_time, n_points=500,
                                     tx_name="DWINGELOO", rx_name="STOCKERT",
                                     rx_duration_s=None, dlt_rate_srp=None):
    """
    Compute the Doppler equator using the SRP velocity vector to determine
    the Doppler axis of rotation, then trace extreme-Doppler paths from SRP
    to the limb.

    The SRP velocity in the Moon body-fixed frame reveals the apparent
    rotation direction (the surface region the SRP drifts toward is rotating
    toward the radar, i.e. it is the up-Doppler / dlt-min side). The Doppler
    axis is perpendicular to both the SRP normal and the tangential SRP
    velocity. The extreme-DLT boundary curves lie along the directions
    perpendicular to this axis.

    If rx_duration_s is given, the dlt values are shifted to the
    window-averaged position where the correlation energy actually lands in
    the DD image (see rate_corrected_dlt).

    Returns:
        lt_min: minimum light time (at SRP)
        delay_up: (n_points,) delay values for up_doppler (approaching) branch
        dlt_up: (n_points,) DLT values for up_doppler branch (dlt minima)
        delay_down: (n_points,) delay values for down_doppler (receding) branch
        dlt_down: (n_points,) DLT values for down_doppler branch (dlt maxima)
    """
    radii = moon_radii()

    # Bistatic minimum-delay point and its apparent motion on the Moon.
    srp = specular_point_bck(rx_time, tx_name, rx_name)
    srp_hat = srp / np.linalg.norm(srp)

    dt = 1.0
    srp2 = specular_point_bck(rx_time + dt, tx_name, rx_name)
    v_srp = (srp2 - srp) / dt

    # Project velocity onto tangent plane at SRP
    v_tangent = v_srp - np.dot(v_srp, srp_hat) * srp_hat
    v_tangent_hat = v_tangent / np.linalg.norm(v_tangent)

    # Doppler axis: perpendicular to both SRP normal and tangential velocity
    doppler_axis = np.cross(srp_hat, v_tangent_hat)
    doppler_axis /= np.linalg.norm(doppler_axis)

    # Max Doppler direction: perpendicular to doppler_axis in tangent plane
    # (same as v_tangent_hat direction)
    max_doppler_dir = v_tangent_hat

    # Trace paths from SRP to limb along max and min Doppler directions
    angles = np.linspace(0, np.pi / 2 * 0.99, n_points)

    # Up-Doppler: along max_doppler_dir
    p_up = (np.outer(np.cos(angles), srp_hat) +
            np.outer(np.sin(angles), max_doppler_dir))
    # Down-Doppler: opposite direction
    p_down = (np.outer(np.cos(angles), srp_hat) +
              np.outer(np.sin(angles), -max_doppler_dir))

    # Snap to ellipsoid surface
    p_up = csp.edpnt_vector(p_up, radii[0], radii[1], radii[2])
    p_down = csp.edpnt_vector(p_down, radii[0], radii[1], radii[2])

    # Compute (lt, dlt) for both branches
    lt_up, dlt_up = moonPointDLT_BCK(rx_time, p_up, tx_name, rx_name)
    lt_down, dlt_down = moonPointDLT_BCK(rx_time, p_down, tx_name, rx_name)

    if rx_duration_s is not None:
        if dlt_rate_srp is None:
            dlt_rate_srp = srp_dlt_rate_bck(rx_time, rx_duration_s, tx_name, rx_name)
        dlt_up = rate_corrected_dlt(rx_time, p_up, rx_duration_s,
                                    dlt_rate_srp, tx_name, rx_name)
        dlt_down = rate_corrected_dlt(rx_time, p_down, rx_duration_s,
                                      dlt_rate_srp, tx_name, rx_name)

    lt_min = min(lt_up.min(), lt_down.min())
    delay_up = lt_up - lt_min
    delay_down = lt_down - lt_min

    return lt_min, delay_up, dlt_up, delay_down, dlt_down


def compute_doppler_equator_terminator(rx_time, n_terminator=1000,
                                       n_points=500,
                                       tx_name="DWINGELOO",
                                       rx_name="STOCKERT",
                                       rx_duration_s=None, dlt_rate_srp=None):
    """
    Compute the Doppler equator using the terminator to find the min/max
    Doppler surface points, then sample along the great circle arc from
    each extreme through the SRP.

    Branch naming matches compute_doppler_equator_velocity: "up" is the
    up-Doppler (approaching, dlt-min) branch, "down" is the down-Doppler
    (receding, dlt-max) branch. Remember doppler = -dlt * frequency.

    Steps:
        1. Sample the terminator to find the points with max and min DLT.
        2. Find the bistatic minimum-delay point (SRP).
        3. Trace from SRP toward the min-DLT terminator point (up_doppler),
           and toward the max-DLT terminator point (down_doppler).
        4. Compute (delay, DLT) along each arc.

    Returns:
        lt_min: minimum light time (at SRP)
        delay_up: (n_points,) delay values for up_doppler (approaching) branch
        dlt_up: (n_points,) DLT values for up_doppler branch (dlt minima)
        delay_down: (n_points,) delay values for down_doppler (receding) branch
        dlt_down: (n_points,) DLT values for down_doppler branch (dlt maxima)
    """
    radii = moon_radii()

    # Sample terminator points and find min/max DLT directions
    _, _, v_term = csp.edterm("UMBRAL", tx_name, "MOON", rx_time,
                              "MOON_ME", AB_COR, rx_name, n_terminator)
    _, dlt_term = moonPointDLT_BCK(rx_time, v_term, tx_name, rx_name)

    p_max_dlt = v_term[np.argmax(dlt_term)]  # receding extreme (down_doppler)
    p_min_dlt = v_term[np.argmin(dlt_term)]  # approaching extreme (up_doppler)

    srp = specular_point_bck(rx_time, tx_name, rx_name)
    srp_hat = srp / np.linalg.norm(srp)

    p_max_hat = p_max_dlt / np.linalg.norm(p_max_dlt)
    p_min_hat = p_min_dlt / np.linalg.norm(p_min_dlt)

    # Project onto tangent plane at SRP to get arc directions
    up_dir = p_min_hat - np.dot(p_min_hat, srp_hat) * srp_hat
    up_dir /= np.linalg.norm(up_dir)
    down_dir = p_max_hat - np.dot(p_max_hat, srp_hat) * srp_hat
    down_dir /= np.linalg.norm(down_dir)

    # Angular distance from SRP to each terminator extreme
    angle_up = np.arccos(np.clip(np.dot(srp_hat, p_min_hat), -1, 1))
    angle_down = np.arccos(np.clip(np.dot(srp_hat, p_max_hat), -1, 1))

    # Sample arcs from SRP to each extreme
    angles_up = np.linspace(0, angle_up, n_points)
    angles_down = np.linspace(0, angle_down, n_points)

    p_up = (np.outer(np.cos(angles_up), srp_hat) +
            np.outer(np.sin(angles_up), up_dir))
    p_down = (np.outer(np.cos(angles_down), srp_hat) +
              np.outer(np.sin(angles_down), down_dir))

    # Snap to ellipsoid surface
    p_up = csp.edpnt_vector(p_up, radii[0], radii[1], radii[2])
    p_down = csp.edpnt_vector(p_down, radii[0], radii[1], radii[2])

    # Compute (lt, dlt) for both branches
    lt_up, dlt_up = moonPointDLT_BCK(rx_time, p_up, tx_name, rx_name)
    lt_down, dlt_down = moonPointDLT_BCK(rx_time, p_down, tx_name, rx_name)

    if rx_duration_s is not None:
        if dlt_rate_srp is None:
            dlt_rate_srp = srp_dlt_rate_bck(rx_time, rx_duration_s, tx_name, rx_name)
        dlt_up = rate_corrected_dlt(rx_time, p_up, rx_duration_s,
                                    dlt_rate_srp, tx_name, rx_name)
        dlt_down = rate_corrected_dlt(rx_time, p_down, rx_duration_s,
                                      dlt_rate_srp, tx_name, rx_name)

    lt_min = min(lt_up.min(), lt_down.min())
    delay_up = lt_up - lt_min
    delay_down = lt_down - lt_min

    return lt_min, delay_up, dlt_up, delay_down, dlt_down


def apparent_station_positions(rx_time, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Light-time-consistent station positions in MOON_ME, anchored at the SRP.

    Returns (R_rx, R_tx, c) such that the two-leg light time of a near-side
    surface point p is (|p - R_rx| + |p - R_tx|) / c. Exact at the SRP by
    construction; elsewhere the error comes from the bounce-epoch variation
    across the disk (<= 2R/c ~ 12 ms), which moves the stations by metres
    and the light time by <~2e-8 s -- negligible against the sample period
    and, being nearly constant over a window, against window-averaged dlt.

    Measured against the exact per-point field (spkcpt_vector/spkcpo_vector
    over the same points) at nside 400: lt agrees to ~1 ns median / ~4 ns
    max near-side -- ~5000x below one 20 us delay bin (50 kHz) and far under
    the +-47 mHz rim residual. So this lets the full-disk (lt, dlt) fields
    be evaluated in numpy instead of per-point SPICE: ~0.22 s vs ~61 s per
    full-disk evaluation (~275x; SPICE costs ~32 us/point), and the
    projection evaluates the field at >=2 epochs per look. The vector SPICE
    forms loop in C but still run the full light-time iteration per point,
    so they don't close the gap -- this replaces the per-point geometry with
    two fixed apparent stations and a closed-form two-leg distance."""
    srp = specular_point_bck(rx_time, tx_name, rx_name)
    # Apparent position of the SRP relative to the RX station, in MOON_ME at
    # the bounce epoch (refloc="TARGET").
    s_rx, lt_rx = csp.spkcpt(srp, "MOON", "MOON_ME", rx_time, "MOON_ME",
                             "TARGET", AB_COR, rx_name)
    R_rx = srp - s_rx[:3]
    # Apparent position of the TX station as seen from the SRP at the bounce
    # epoch, in MOON_ME at that epoch (refloc="OBSERVER").
    s_tx, _ = csp.spkcpo(tx_name, rx_time - lt_rx, "MOON_ME", "OBSERVER",
                         AB_COR, srp, "MOON", "MOON_ME")
    R_tx = srp + s_tx[:3]
    return R_rx, R_tx, csp.clight()


# ---------------------------------------------------------------------------
# Upstream-compatible aliases
# ---------------------------------------------------------------------------
compute_doppler_equator_healpix = compute_doppler_equator
