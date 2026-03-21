"""
Doppler Equator Computation

Functions for computing the Doppler equator of the Moon using SPICE ephemeris data.
Provides multiple methods for calculating the boundary curves that separate regions
of different Doppler shifts in delay-Doppler space.
"""

import numpy as np
import cspyce as csp
from astropy import units as au


# ---------------------------------------------------------------------------
# SPICE configuration constants
# ---------------------------------------------------------------------------
AB_COR = "LT"
EARTH_FRAME = "ITRF93"


# ---------------------------------------------------------------------------
# SPICE helpers
# ---------------------------------------------------------------------------
def moonPointDLT_BCK(rx_time, p_moon, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Compute (lt, dlt) for moon surface points using backward ray tracing."""
    moon_radii = csp.bodvrd("MOON", "RADII")
    p_surf = csp.edpnt_vector(p_moon, moon_radii[0], moon_radii[1], moon_radii[2])
    s_rx, lt_rx = csp.spkcpt_vector(p_surf, "MOON", "MOON_ME", rx_time,
                                     EARTH_FRAME, "OBSERVER", AB_COR, rx_name)
    s_tx, lt_tx = csp.spkcpo_vector(tx_name, rx_time - lt_rx, EARTH_FRAME,
                                     "TARGET", AB_COR, p_surf, "MOON", "MOON_ME")
    v1 = csp.dvnorm_vector(s_rx)
    v2 = csp.dvnorm_vector(s_tx)
    c = csp.clight()
    dlt = 1 - np.sqrt((1 - v1/c)/(1 + v1/c)) * np.sqrt((1 - v2/c)/(1 + v2/c))
    return lt_rx + lt_tx, dlt


def moonSRP_DLT_BCK(rx_time, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Compute (lt, dlt) for the sub-radar point."""
    srp_rx, trg, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                                 "MOON_ME", AB_COR, rx_name)
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", trg,
                               "MOON_ME", AB_COR, tx_name)
    srp = (srp_rx + srp_tx) / 2.0
    return moonPointDLT_BCK(rx_time, srp, tx_name, rx_name)


def moonSRP_DLT_FWD(tx_time, tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Compute (lt, dlt) for the sub-radar point, forward direction."""
    moon_radii = csp.bodvrd("MOON", "RADII")
    srp_tx, trg, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", tx_time,
                                 "MOON_ME", "X" + AB_COR, tx_name)
    srp_rx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", trg,
                               "MOON_ME", AB_COR, rx_name)
    srp = (srp_tx + srp_rx) / 2.0
    p_surf = csp.edpnt_vector(srp, moon_radii[0], moon_radii[1], moon_radii[2])
    s_tx, lt_tx = csp.spkcpt_vector(p_surf, "MOON", "MOON_ME", tx_time,
                                     EARTH_FRAME, "TARGET", "X" + AB_COR, tx_name)
    s_rx, lt_rx = csp.spkcpo_vector(rx_name, tx_time + lt_tx, EARTH_FRAME,
                                     "OBSERVER", "X" + AB_COR, p_surf, "MOON", "MOON_ME")
    v1 = csp.dvnorm_vector(s_tx)
    v2 = csp.dvnorm_vector(s_rx)
    c = csp.clight()
    dlt = 1 - np.sqrt((1 - v1/c)/(1 + v1/c)) * np.sqrt((1 - v2/c)/(1 + v2/c))
    return lt_tx + lt_rx, dlt


# ---------------------------------------------------------------------------
# Doppler equator computation methods
# ---------------------------------------------------------------------------

# Best method:
def compute_doppler_equator_velocity(rx_time, n_points=500,
                                     tx_name="DWINGELOO", rx_name="STOCKERT"):
    """
    Compute the Doppler equator using the SRP velocity vector to determine
    the Doppler axis of rotation, then trace extreme-Doppler paths from SRP
    to the limb.

    The SRP velocity in the Moon body-fixed frame reveals the apparent
    rotation direction. The Doppler axis is perpendicular to both the SRP
    normal and the tangential SRP velocity. The extreme-DLT boundary curves
    lie along the directions perpendicular to this axis.

    Returns:
        lt_min: minimum light time (at SRP)
        delay_up: (n_points,) delay values for up_doppler branch
        dlt_up: (n_points,) DLT values for up_doppler branch
        delay_down: (n_points,) delay values for down_doppler branch
        dlt_down: (n_points,) DLT values for down_doppler branch
    """
    moon_radii = csp.bodvrd("MOON", "RADII")

    # Combined SRP (average of TX and RX sub-points)
    srp_rx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                               "MOON_ME", AB_COR, rx_name)
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                               "MOON_ME", AB_COR, tx_name)
    srp = (srp_rx + srp_tx) / 2.0
    srp_hat = srp / np.linalg.norm(srp)

    # SRP velocity via finite differences
    dt = 1.0
    srp_rx2, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time + dt,
                                "MOON_ME", AB_COR, rx_name)
    srp_tx2, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time + dt,
                                "MOON_ME", AB_COR, tx_name)
    srp2 = (srp_rx2 + srp_tx2) / 2.0
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
    p_up = csp.edpnt_vector(p_up, moon_radii[0], moon_radii[1], moon_radii[2])
    p_down = csp.edpnt_vector(p_down, moon_radii[0], moon_radii[1], moon_radii[2])

    # Compute (lt, dlt) for both branches
    lt_up, dlt_up = moonPointDLT_BCK(rx_time, p_up, tx_name, rx_name)
    lt_down, dlt_down = moonPointDLT_BCK(rx_time, p_down, tx_name, rx_name)

    lt_min = min(lt_up.min(), lt_down.min())
    delay_up = lt_up - lt_min
    delay_down = lt_down - lt_min

    return lt_min, delay_up, dlt_up, delay_down, dlt_down


def compute_doppler_equator_healpix(rx_time, n_delay_bins=500, nside=200,
                                    tx_name="DWINGELOO", rx_name="STOCKERT"):
    """
    Compute the Doppler equator: for each delay bin, find the max and min DLT
    among visible moon surface points using HEALPix sampling.

    Returns:
        lt_min: minimum light time (at SRP)
        delay_centers: (n_bins,) array of delay values (seconds from SRP)
        dlt_max: (n_bins,) max DLT per delay bin (up_doppler equator)
        dlt_min: (n_bins,) min DLT per delay bin (down_doppler equator)
    """
    import healpy as hp

    NPIX = hp.nside2npix(nside)
    v = np.array(hp.pix2vec(nside, np.arange(NPIX))).T
    v_near = v[:, 0] > 0
    v = v[v_near]

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


def compute_doppler_equator_terminator(rx_time, n_terminator=1000,
                                       n_points=500,
                                       tx_name="DWINGELOO",
                                       rx_name="STOCKERT"):
    """
    Compute the Doppler equator using the terminator to find the min/max
    Doppler surface points, then sample along the great circle arc from
    each extreme through the SRP.

    Steps:
        1. Sample the terminator to find the points with max and min DLT.
        2. Compute the combined SRP (average of TX and RX sub-points).
        3. Trace from the max-DLT terminator point through SRP (up_doppler),
           and from the min-DLT terminator point through SRP (down_doppler).
        4. Compute (delay, DLT) along each arc.

    Returns:
        lt_min: minimum light time (at SRP)
        delay_up: (n_points,) delay values for up_doppler branch
        dlt_up: (n_points,) DLT values for up_doppler branch
        delay_down: (n_points,) delay values for down_doppler branch
        dlt_down: (n_points,) DLT values for down_doppler branch
    """
    moon_radii = csp.bodvrd("MOON", "RADII")

    # Sample terminator points and find min/max DLT directions
    _, _, v_term = csp.edterm("UMBRAL", tx_name, "MOON", rx_time,
                              "MOON_ME", AB_COR, rx_name, n_terminator)
    _, dlt_term = moonPointDLT_BCK(rx_time, v_term, tx_name, rx_name)

    p_max_dlt = v_term[np.argmax(dlt_term)]  # max Doppler terminator point
    p_min_dlt = v_term[np.argmin(dlt_term)]  # min Doppler terminator point

    # Combined SRP
    srp_rx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                               "MOON_ME", AB_COR, rx_name)
    srp_tx, _, _ = csp.subpnt('INTERCEPT/ELLIPSOID', "MOON", rx_time,
                               "MOON_ME", AB_COR, tx_name)
    srp = (srp_rx + srp_tx) / 2.0
    srp_hat = srp / np.linalg.norm(srp)

    # Direction from SRP toward max-DLT terminator point
    p_max_hat = p_max_dlt / np.linalg.norm(p_max_dlt)
    p_min_hat = p_min_dlt / np.linalg.norm(p_min_dlt)

    # Project onto tangent plane at SRP to get arc directions
    up_dir = p_max_hat - np.dot(p_max_hat, srp_hat) * srp_hat
    up_dir /= np.linalg.norm(up_dir)
    down_dir = p_min_hat - np.dot(p_min_hat, srp_hat) * srp_hat
    down_dir /= np.linalg.norm(down_dir)

    # Angular distance from SRP to each terminator extreme
    angle_up = np.arccos(np.clip(np.dot(srp_hat, p_max_hat), -1, 1))
    angle_down = np.arccos(np.clip(np.dot(srp_hat, p_min_hat), -1, 1))

    # Sample arcs from SRP to each extreme
    angles_up = np.linspace(0, angle_up, n_points)
    angles_down = np.linspace(0, angle_down, n_points)

    p_up = (np.outer(np.cos(angles_up), srp_hat) +
            np.outer(np.sin(angles_up), up_dir))
    p_down = (np.outer(np.cos(angles_down), srp_hat) +
              np.outer(np.sin(angles_down), down_dir))

    # Snap to ellipsoid surface
    p_up = csp.edpnt_vector(p_up, moon_radii[0], moon_radii[1], moon_radii[2])
    p_down = csp.edpnt_vector(p_down, moon_radii[0], moon_radii[1], moon_radii[2])

    # Compute (lt, dlt) for both branches
    lt_up, dlt_up = moonPointDLT_BCK(rx_time, p_up, tx_name, rx_name)
    lt_down, dlt_down = moonPointDLT_BCK(rx_time, p_down, tx_name, rx_name)

    lt_min = min(lt_up.min(), lt_down.min())
    delay_up = lt_up - lt_min
    delay_down = lt_down - lt_min

    return lt_min, delay_up, dlt_up, delay_down, dlt_down
