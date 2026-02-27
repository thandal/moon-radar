"""
Doppler Equator Alignment & Time Offset Search

Computes the DD image once with nominal offsets, then grid-searches over
TX_START_OFFSET / RX_START_OFFSET perturbations by recomputing only
the SPICE-predicted Doppler equator and scoring its alignment with
the DD image edge features.
"""

import os
import numpy as np
import scipy.signal
import scipy.interpolate
import scipy.ndimage
import healpy as hp
from astropy import units as au
from astropy import constants as ak
from astropy import time as at
import cspyce as csp
import cupy
import sigmf
from tqdm import tqdm
from matplotlib import pyplot as pl

# ---------------------------------------------------------------------------
# SPICE setup
# ---------------------------------------------------------------------------
SPICE_KERNEL_DIR = "spice_kernels"
csp.kclear()
for k in ["naif0012.tls", "de440s.bsp", "pck00011.tpc",
           "earth_latest_high_prec.bpc", "moon_pa_de440_200625.bpc",
           "moon_de440_250416.tf", "observatories.bsp", "observatories.tf"]:
    csp.furnsh(f"{SPICE_KERNEL_DIR}/{k}")

AB_COR = "LT"
EARTH_FRAME = "ITRF93"
MOON_RADIUS = 1_737_400.0 * au.m

# ---------------------------------------------------------------------------
# SPICE helpers (copied from notebook)
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
# Step 1: Doppler equator sampling
# ---------------------------------------------------------------------------
def compute_doppler_equator(rx_time, n_delay_bins=500, nside=200,
                            tx_name="DWINGELOO", rx_name="STOCKERT"):
    """
    Compute the Doppler equator: for each delay bin, find the max and min DLT
    among visible moon surface points.

    Returns:
        delay_centers: (n_bins,) array of delay values (seconds from SRP)
        dlt_max: (n_bins,) max DLT per delay bin (up_doppler equator)
        dlt_min: (n_bins,) min DLT per delay bin (down_doppler equator)
    """
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


# ---------------------------------------------------------------------------
# Step 2: Alignment scoring
# ---------------------------------------------------------------------------
def compute_edge_image(log_A):
    """Compute edge features using Sobel filter along the Doppler axis."""
    edge = scipy.ndimage.sobel(log_A, axis=0)  # vertical (Doppler) edges
    return np.abs(edge)


def alignment_score(edge_img, dlt_shifts, delay_values_s, lt_min_image,
                    lt_min_equator, delay_centers, dlt_curve):
    """
    Score how well a Doppler equator curve aligns with the DD image edges.

    Args:
        edge_img: (n_doppler, n_delay) edge-filtered DD image
        dlt_shifts: (n_doppler,) DLT values for each row of the DD image
        delay_values_s: (n_delay,) delay values in seconds for each column
        lt_min_image: minimum light time used to define image delay axis
        lt_min_equator: minimum light time from SPICE equator computation
        delay_centers: (n_bins,) delay offsets from equator's lt_min
        dlt_curve: (n_bins,) DLT values along the equator curve
    """
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    ddelay = delay_values_s[1] - delay_values_s[0]

    # Map equator points to pixel coordinates
    # Delay: equator delay is relative to lt_min_equator. Image delay is relative to lt_min_image.
    delay_in_image = delay_centers + (lt_min_equator - lt_min_image)
    delay_idx = (delay_in_image - delay_values_s[0]) / ddelay
    doppler_idx = (dlt_curve - dlt_shifts[0]) / ddlt

    score = 0.0
    count = 0
    for i in range(len(delay_centers)):
        if np.isnan(dlt_curve[i]):
            continue
        di = int(round(doppler_idx[i]))
        dj = int(round(delay_idx[i]))
        if 0 <= di < edge_img.shape[0] and 0 <= dj < edge_img.shape[1]:
            score += edge_img[di, dj]
            count += 1

    return score / max(count, 1)


# ---------------------------------------------------------------------------
# DD image computation (from notebook)
# ---------------------------------------------------------------------------
def compute_dd_image(rx_samples, tx_samples, sample_rate, frequency,
                     rx_start_astrotime, tx_start_offset, rx_start_offset,
                     tx_name="DWINGELOO", rx_name="STOCKERT"):
    """
    Compute the Delay-Doppler image and associated axes.
    Returns: log_A, dlt_shifts, delay_values_s, lt_min
    """
    rx_duration = len(rx_samples) / sample_rate
    tx_duration = len(tx_samples) / sample_rate

    rx_start_time_s = csp.str2et(rx_start_astrotime.utc.value) + rx_start_offset
    rx_end_time_s = rx_start_time_s + rx_duration.to(au.s).value
    tx_start_time_s = rx_start_time_s + tx_start_offset
    tx_end_time_s = tx_start_time_s + tx_duration.to(au.s).value

    # FWD light times for TX resampling
    lt_tx_start, _ = moonSRP_DLT_FWD(tx_start_time_s, tx_name, rx_name)
    lt_tx_end, _ = moonSRP_DLT_FWD(tx_end_time_s, tx_name, rx_name)

    # Resample TX
    rx_sample_times0 = np.arange(len(rx_samples)) / sample_rate.to(au.Hz).value
    adjusted_tx_times0 = np.linspace(
        tx_start_offset + lt_tx_start,
        tx_start_offset + tx_duration.to(au.s).value + lt_tx_end,
        len(tx_samples), endpoint=False)
    tx_phase_interp = scipy.interpolate.interp1d(
        adjusted_tx_times0, np.unwrap(np.angle(tx_samples)),
        kind='linear', fill_value=np.nan, bounds_error=False)
    tx_resampled = np.exp(1j * tx_phase_interp(rx_sample_times0))
    np.nan_to_num(tx_resampled, copy=False)

    # BCK Doppler compensation
    lt_rx_start, dlt_rx_start = moonSRP_DLT_BCK(rx_start_time_s, tx_name, rx_name)
    lt_rx_end, dlt_rx_end = moonSRP_DLT_BCK(rx_end_time_s, tx_name, rx_name)
    doppler_start = -dlt_rx_start * frequency
    doppler_end = -dlt_rx_end * frequency
    doppler_rate = (doppler_end - doppler_start) / rx_duration

    t_s = np.arange(len(rx_samples)) / sample_rate
    phi_Hz = -(doppler_start + doppler_rate * t_s / 2)
    tx_compensated = tx_resampled * np.exp(-1j * 2 * np.pi * (phi_Hz * t_s).value).T

    # Delay window
    cor_lags = scipy.signal.correlation_lags(len(rx_samples), len(tx_compensated), mode="same") / sample_rate
    t_end = MOON_RADIUS / ak.c * 2
    cor_i_start = np.argwhere(cor_lags >= 0 * au.s)[0][0]
    cor_i_end = np.argwhere(cor_lags >= t_end)[0][0]

    # Terminator for Doppler range
    NTERMINATOR = 1000
    _, _, v_term = csp.edterm("UMBRAL", tx_name, "MOON", rx_start_time_s,
                              "MOON_ME", AB_COR, rx_name, NTERMINATOR)
    _, dlt_term = moonPointDLT_BCK(rx_start_time_s, v_term, tx_name, rx_name)
    dlt_shifts = np.linspace(dlt_term.min(), dlt_term.max(), 3000)
    f_shifts = -dlt_shifts * frequency.to(au.Hz).value - doppler_start.to(au.Hz).value

    # CUDA correlation
    tx_gpu = cupy.asarray(tx_compensated, dtype=cupy.complex64)
    rx_gpu = cupy.asarray(rx_samples, dtype=cupy.complex64)
    ci_start = cor_i_start - len(rx_samples) // 2
    ci_end = cor_i_end - len(rx_samples) // 2
    A = cupy.zeros((len(f_shifts), ci_end - ci_start))
    fft_rx = cupy.fft.fft(rx_gpu)
    tx_shifted = cupy.zeros_like(rx_gpu)
    tx_range = 1j * 2 * cupy.pi / sample_rate.value * cupy.arange(len(tx_compensated))
    for i in tqdm(range(len(f_shifts)), desc="Correlating"):
        tx_shifted[:len(tx_compensated)] = tx_gpu * cupy.exp(f_shifts[i] * tx_range)
        cor = cupy.fft.ifft(fft_rx * cupy.conj(cupy.fft.fft(tx_shifted)))
        A[i] = cupy.abs(cor[ci_start:ci_end])
    A = cupy.asnumpy(A)

    log_A = np.log(A)
    delay_values_s = (cor_lags[cor_i_start:cor_i_end]).to(au.s).value
    lt_min = lt_rx_start  # SRP light time at rx_start

    return log_A, dlt_shifts, delay_values_s, lt_min


# ---------------------------------------------------------------------------
# Step 3: Grid search
# ---------------------------------------------------------------------------
def grid_search(rx_samples, tx_samples, sample_rate, frequency,
                rx_start_astrotime, rx_name="STOCKERT", tx_name="DWINGELOO",
                tx_offsets=None, rx_offsets=None):
    """
    1. Compute DD image once with nominal offsets.
    2. Grid-search over offsets by recomputing only SPICE Doppler equator.
    """
    if tx_offsets is None:
        tx_offsets = np.linspace(0.999, 1.001, 21)
    if rx_offsets is None:
        rx_offsets = np.linspace(-0.001, 0.001, 21)

    # --- Compute DD image once with nominal offsets ---
    #nominal_tx = 1.0
    #nominal_rx = 0.0
    #print("Computing DD image with nominal offsets...")
    #log_A, dlt_shifts, delay_values_s, lt_min_image = compute_dd_image(
    #    rx_samples, tx_samples, sample_rate, frequency,
    #    rx_start_astrotime, nominal_tx, nominal_rx, tx_name, rx_name)

    ## --- Compute edge image ---
    #edge_img = compute_edge_image(log_A)
    #print(f"DD image shape: {log_A.shape}, edge range: [{edge_img.min():.2f}, {edge_img.max():.2f}]")

    # --- Grid search ---
    scores = np.zeros((len(tx_offsets), len(rx_offsets)))

    for i, tx_off in enumerate(tqdm(tx_offsets, desc="TX offset")):
        for j, rx_off in enumerate(rx_offsets):

            # Recompute DD image with these offsets
            log_A, dlt_shifts, delay_values_s, lt_min_image = compute_dd_image(
                rx_samples, tx_samples, sample_rate, frequency,
                rx_start_astrotime, tx_off, rx_off, tx_name, rx_name)
            edge_img = compute_edge_image(log_A)

            # Recompute SPICE ephemeris with these offsets
            rx_time = csp.str2et(rx_start_astrotime.utc.value) + rx_off

            # Compute Doppler equator at this rx_time
            lt_min_eq, delay_centers, dlt_max, dlt_min = compute_doppler_equator(
                rx_time, n_delay_bins=500, nside=100, tx_name=tx_name, rx_name=rx_name)

            # Score both equators (up_doppler and down_doppler)
            s_up_doppler = alignment_score(edge_img, dlt_shifts, delay_values_s,
                                       lt_min_image, lt_min_eq, delay_centers, dlt_max)
            s_down_doppler = alignment_score(edge_img, dlt_shifts, delay_values_s,
                                       lt_min_image, lt_min_eq, delay_centers, dlt_min)
            scores[i, j] = s_up_doppler + s_down_doppler

    return scores, tx_offsets, rx_offsets, log_A, edge_img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_ROOT = "data.camras.nl/lunar-radar/"

    # Pick one observation file
    rx_filename = f"{DATA_ROOT}2025-09-16/stockert_radar_2025_09_16_13_22_02_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta"

    print(f"Loading {rx_filename}...")
    rx_sigmf = sigmf.sigmffile.fromfile(rx_filename, skip_checksum=True)
    rx_samples = rx_sigmf.read_samples().astype("complex64")
    rx_info = rx_sigmf.get_global_info()
    sample_rate = rx_info['core:sample_rate'] / au.s
    rx_captures = rx_sigmf.get_captures()
    frequency = rx_captures[0]['core:frequency'] * au.Hz
    rx_start_astrotime = at.Time(rx_captures[0]['core:datetime'])

    tx_filename = rx_info['core:description'].split(';')[0]
    tx_sigmf = sigmf.sigmffile.fromfile(f"{DATA_ROOT}tx_signals/{tx_filename}", skip_checksum=True)
    tx_samples = tx_sigmf.read_samples().astype("complex64")

    print(f"Samples: rx={len(rx_samples)}, tx={len(tx_samples)}, rate={sample_rate}, freq={frequency}")

    if 0: # Test compute_dd_image and compute_edge_image
        log_A, dlt_shifts, delay_values_s, lt_min_image = compute_dd_image(
            rx_samples, tx_samples, sample_rate, frequency,
            rx_start_astrotime, 1.0, 0.0, "DWINGELOO", "STOCKERT")

        dd_extent = [dlt_shifts[0], dlt_shifts[-1], delay_values_s[-1], delay_values_s[0]]
        pl.figure()
        pl.imshow(log_A.T, aspect='auto', vmax=log_A.max() * 0.8, vmin=log_A.max() * 0.4,
                  extent=dd_extent)
        pl.title("DD Image")
        pl.xlabel("Fractional Doppler Shift")
        pl.ylabel("Delay (s)")
        pl.savefig("results/ALIGNMENT/test_dd_image.png", dpi=150)
        print("Saved results/ALIGNMENT/test_dd_image.png")

        edge_img = compute_edge_image(log_A)
        pl.figure()
        pl.imshow(edge_img.T, aspect='auto', vmax=np.percentile(edge_img, 99),
                  extent=dd_extent)
        pl.title("Edge Image (Sobel)")
        pl.xlabel("Fractional Doppler Shift")
        pl.ylabel("Delay (s)")
        pl.savefig("results/ALIGNMENT/test_edge_image.png", dpi=150)
        print("Saved results/ALIGNMENT/test_edge_image.png")

    if 0: # Test compute_doppler_equator
        rx_time = csp.str2et(rx_start_astrotime.utc.value)
        lt_min_eq, delay_centers, dlt_max, dlt_min = compute_doppler_equator(
            rx_time, n_delay_bins=500, nside=100, tx_name="DWINGELOO", rx_name="STOCKERT")
        print(f"Doppler equator: lt_min={lt_min_eq}, delay_centers={delay_centers}, dlt_max={dlt_max}, dlt_min={dlt_min}")

        # Create a plot of doppler equator
        pl.figure()
        pl.plot(dlt_max, delay_centers, label="dlt_max")
        pl.plot(dlt_min, delay_centers, label="dlt_min")
        pl.xlabel("Fractional Doppler Shift")
        pl.ylabel("Delay (s)")
        pl.gca().invert_yaxis()
        pl.title("Doppler Equator")
        pl.legend()
        pl.savefig("results/ALIGNMENT/test_doppler_equator.png", dpi=150)
        print("Saved results/ALIGNMENT/test_doppler_equator.png")

    if 0: # Test compute_doppler_equator_velocity
        rx_time = csp.str2et(rx_start_astrotime.utc.value)
        lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = compute_doppler_equator_velocity(rx_time)
        print(f"Doppler equator: lt_min={lt_min_eq}, delay_up={delay_up}, dlt_up={dlt_up}, delay_down={delay_down}, dlt_down={dlt_down}")

        # Create a plot of doppler equator
        pl.figure()
        pl.plot(dlt_up, delay_up, label="dlt_up")
        pl.plot(dlt_down, delay_down, label="dlt_down")
        pl.xlabel("Fractional Doppler Shift")
        pl.ylabel("Delay (s)")
        pl.gca().invert_yaxis()
        pl.title("Doppler Equator")
        pl.legend()
        pl.savefig("results/ALIGNMENT/test_doppler_equator_velocity.png", dpi=150)
        print("Saved results/ALIGNMENT/test_doppler_equator_velocity.png")

    if 0: # Test compute_doppler_equator_terminator
        rx_time = csp.str2et(rx_start_astrotime.utc.value)
        lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = compute_doppler_equator_terminator(rx_time)
        print(f"Doppler equator: lt_min={lt_min_eq}, delay_up={delay_up}, dlt_up={dlt_up}, delay_down={delay_down}, dlt_down={dlt_down}")

        # Create a plot of doppler equator
        pl.figure()
        pl.plot(dlt_up, delay_up, label="dlt_up")
        pl.plot(dlt_down, delay_down, label="dlt_down")
        pl.xlabel("Fractional Doppler Shift")
        pl.ylabel("Delay (s)")
        pl.gca().invert_yaxis()
        pl.title("Doppler Equator (Terminator)")
        pl.legend()
        pl.savefig("results/ALIGNMENT/test_doppler_equator_terminator.png", dpi=150)
        print("Saved results/ALIGNMENT/test_doppler_equator_terminator.png")

    if 1: # Compare the three compute_dopper_equator functions
        pl.figure(figsize=(10, 10))

        rx_time = csp.str2et(rx_start_astrotime.utc.value)

        # compute_dopper_equator
        lt_min_eq, delay_centers, dlt_max, dlt_min = compute_doppler_equator(
            rx_time, n_delay_bins=500, nside=100, tx_name="DWINGELOO", rx_name="STOCKERT")
        print(f"Doppler equator: lt_min={lt_min_eq}, delay_centers={delay_centers}, dlt_max={dlt_max}, dlt_min={dlt_min}")
        pl.plot(dlt_max, delay_centers, label="dlt_max")
        pl.plot(dlt_min, delay_centers, label="dlt_min")

        # compute_dopper_equator_velocity
        lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = compute_doppler_equator_velocity(rx_time)
        print(f"Doppler equator: lt_min={lt_min_eq}, delay_up={delay_up}, dlt_up={dlt_up}, delay_down={delay_down}, dlt_down={dlt_down}")
        pl.plot(dlt_up, delay_up, label="dlt_up_velocity")
        pl.plot(dlt_down, delay_down, label="dlt_down_velocity")

        pl.plot(dlt_down.max() - (dlt_up - dlt_up.min()), delay_up, label="flipped dlt_up_velocity")
        pl.plot(dlt_up.min() + (dlt_down.max() - dlt_down), delay_down, label="flipped dlt_down_velocity")

        # compute_dopper_equator_terminator
        lt_min_eq, delay_up_t, dlt_up_t, delay_down_t, dlt_down_t = compute_doppler_equator_terminator(rx_time)
        print(f"Doppler equator (term): lt_min={lt_min_eq}")
        pl.plot(dlt_up_t, delay_up_t, label="dlt_up_terminator")
        pl.plot(dlt_down_t, delay_down_t, label="dlt_down_terminator")
        
        pl.xlabel("Fractional Doppler Shift")
        pl.ylabel("Delay (s)")
        pl.gca().invert_yaxis()
        pl.title("Doppler Equator")
        pl.legend()
        pl.savefig("results/ALIGNMENT/test_doppler_equator_comparison.png", dpi=150)
        print("Saved results/ALIGNMENT/test_doppler_equator_comparison.png")


    if 0: # Draw the doppler equator on the DD image
        # Convert equator delay to DD image delay reference frame
        equator_delay_in_image = delay_centers + (lt_min_eq - lt_min_image)

        dd_extent = [dlt_shifts[0], dlt_shifts[-1], delay_values_s[-1], delay_values_s[0]]
        pl.figure()
        pl.imshow(log_A.T, aspect='auto', vmax=log_A.max() * 0.8, vmin=log_A.max() * 0.4,
                  extent=dd_extent)
        pl.title("DD Image with Doppler Equator")
        pl.xlabel("Fractional Doppler Shift")
        pl.ylabel("Delay (s)")
        pl.plot(dlt_max, equator_delay_in_image, ",", label="dlt_max", color='red')
        pl.plot(dlt_min, equator_delay_in_image, ",", label="dlt_min", color='cyan')
        pl.legend()
        pl.savefig("results/ALIGNMENT/test_dd_image_doppler_equator.png", dpi=150)
        print("Saved results/ALIGNMENT/test_dd_image_doppler_equator.png")

    if 0: # Draw the doppler equator on the edge image
        # Convert equator delay to edge image delay reference frame
        equator_delay_in_image = delay_centers + (lt_min_eq - lt_min_image)

        edge_extent = [dlt_shifts[0], dlt_shifts[-1], delay_values_s[-1], delay_values_s[0]]
        pl.figure()
        pl.imshow(edge_img.T, aspect='auto', vmax=np.percentile(edge_img, 99),
                  extent=edge_extent)
        pl.title("Edge Image with Doppler Equator")
        pl.xlabel("Fractional Doppler Shift")
        pl.ylabel("Delay (s)")
        pl.plot(dlt_max, equator_delay_in_image, ",", label="dlt_max", color='red')
        pl.plot(dlt_min, equator_delay_in_image, ",", label="dlt_min", color='cyan')
        pl.legend()
        pl.savefig("results/ALIGNMENT/test_edge_image_doppler_equator.png", dpi=150)
        print("Saved results/ALIGNMENT/test_edge_image_doppler_equator.png")

    if 0: # Test alignment_score by shifting the edge image slightly.
        tx_shifts = range(-2, 3)
        rx_shifts = range(-2, 3)
        scores_up_doppler = np.zeros((len(tx_shifts), len(rx_shifts)))
        scores_down_doppler = np.zeros((len(tx_shifts), len(rx_shifts)))
        for i in tx_shifts:
            for j in rx_shifts:
                edge_img_shifted = np.roll(edge_img, (i, j), axis=(0, 1))
                score_up_doppler = alignment_score(edge_img_shifted, dlt_shifts, delay_values_s,
                                             lt_min_image, lt_min_eq, delay_centers, dlt_max)
                score_down_doppler = alignment_score(edge_img_shifted, dlt_shifts, delay_values_s,
                                             lt_min_image, lt_min_eq, delay_centers, dlt_min)
                scores_up_doppler[i, j] = score_up_doppler
                scores_down_doppler[i, j] = score_down_doppler

        # Show heatmaps
        fig, axes = pl.subplots(1, 2, figsize=(20, 6))
        shift_extent = [tx_shifts[0], tx_shifts[-1], rx_shifts[0], rx_shifts[-1]]
        im = axes[0].imshow(scores_up_doppler.T, origin='lower', aspect='auto', extent=shift_extent)
        im = axes[1].imshow(scores_down_doppler.T, origin='lower', aspect='auto', extent=shift_extent)
        pl.colorbar(im, ax=axes[0])
        pl.colorbar(im, ax=axes[1])
        axes[0].set_title("Alignment Score (up_doppler)")
        axes[1].set_title("Alignment Score (down_doppler)")
        axes[0].set_xlabel("TX Shift")
        axes[0].set_ylabel("RX Shift")
        axes[1].set_xlabel("TX Shift")
        axes[1].set_ylabel("RX Shift")
        pl.tight_layout()
        pl.savefig("results/ALIGNMENT/test_alignment_score.png", dpi=150)
        print("Saved results/ALIGNMENT/test_alignment_score.png")

    if 0: # Test compute_doppler_equator_velocity
        lt_min_vel, delay_up_vel, dlt_up_vel, delay_down_vel, dlt_down_vel = \
            compute_doppler_equator_velocity(rx_time, n_points=500,
                                             tx_name="DWINGELOO", rx_name="STOCKERT")
        print(f"Velocity method: lt_min={lt_min_vel}")

        # Plot on DD image
        delay_up_in_image = delay_up_vel + (lt_min_vel - lt_min_image)
        delay_down_in_image = delay_down_vel + (lt_min_vel - lt_min_image)

        dd_extent = [dlt_shifts[0], dlt_shifts[-1], delay_values_s[-1], delay_values_s[0]]
        pl.figure()
        pl.imshow(log_A.T, aspect='auto', vmax=log_A.max() * 0.8, vmin=log_A.max() * 0.4,
                  extent=dd_extent)
        pl.title("DD Image with Velocity Doppler Equator")
        pl.xlabel("Fractional Doppler Shift")
        pl.ylabel("Delay (s)")
        pl.plot(dlt_up_vel, delay_up_in_image, ",", label="up_doppler (vel)", color='red')
        pl.plot(dlt_down_vel, delay_down_in_image, ",", label="down_doppler (vel)", color='cyan')
        pl.legend()
        pl.savefig("results/ALIGNMENT/test_dd_velocity_equator.png", dpi=150)
        print("Saved results/ALIGNMENT/test_dd_velocity_equator.png")

    if 0: # Test compute_doppler_equator_terminator
        lt_min_term, delay_centers_term, dlt_max_term, dlt_min_term = \
            compute_doppler_equator_terminator(rx_time, n_terminator=1000,
                                               n_delay_bins=500,
                                               tx_name="DWINGELOO", rx_name="STOCKERT")
        print(f"Terminator method: lt_min={lt_min_term}")

        # Plot on DD image
        equator_delay_term = delay_centers_term + (lt_min_term - lt_min_image)

        dd_extent = [dlt_shifts[0], dlt_shifts[-1], delay_values_s[-1], delay_values_s[0]]
        pl.figure()
        pl.imshow(log_A.T, aspect='auto', vmax=log_A.max() * 0.8, vmin=log_A.max() * 0.4,
                  extent=dd_extent)
        pl.title("DD Image with Terminator Doppler Equator")
        pl.xlabel("Fractional Doppler Shift")
        pl.ylabel("Delay (s)")
        pl.plot(dlt_max_term, equator_delay_term, ",", label="up_doppler (term)", color='red')
        pl.plot(dlt_min_term, equator_delay_term, ",", label="down_doppler (term)", color='cyan')
        pl.legend()
        pl.savefig("results/ALIGNMENT/test_dd_terminator_equator.png", dpi=150)
        print("Saved results/ALIGNMENT/test_dd_terminator_equator.png")

    if 0: # Compare all three methods on DD image
        dd_extent = [dlt_shifts[0], dlt_shifts[-1], delay_values_s[-1], delay_values_s[0]]
        fig, axes = pl.subplots(1, 3, figsize=(24, 6))

        for ax, title in zip(axes, ["HEALPix", "Velocity", "Terminator"]):
            ax.imshow(log_A.T, aspect='auto', vmax=log_A.max() * 0.8, vmin=log_A.max() * 0.4,
                      extent=dd_extent)
            ax.set_title(title)
            ax.set_xlabel("Fractional Doppler Shift")
            ax.set_ylabel("Delay (s)")

        # HEALPix
        eq_delay_hp = delay_centers + (lt_min_eq - lt_min_image)
        axes[0].plot(dlt_max, eq_delay_hp, ",", color='red')
        axes[0].plot(dlt_min, eq_delay_hp, ",", color='cyan')

        # Velocity
        axes[1].plot(dlt_up_vel, delay_up_in_image, ",", color='red')
        axes[1].plot(dlt_down_vel, delay_down_in_image, ",", color='cyan')

        # Terminator
        axes[2].plot(dlt_max_term, equator_delay_term, ",", color='red')
        axes[2].plot(dlt_min_term, equator_delay_term, ",", color='cyan')

        pl.tight_layout()
        pl.savefig("results/ALIGNMENT/test_compare_methods.png", dpi=150)
        print("Saved results/ALIGNMENT/test_compare_methods.png")
                             
    if 0: # Grid search
        tx_offsets = np.linspace(0.999, 1.001, 21)   # ±1 ms around 1.0s
        rx_offsets = np.linspace(-0.001, 0.001, 21)   # ±1 ms around 0s

        scores, tx_offs, rx_offs, log_A, edge_img = grid_search(
            rx_samples, tx_samples, sample_rate, frequency,
            rx_start_astrotime,
            tx_offsets=tx_offsets, rx_offsets=rx_offsets)

        # Find optimum
        best = np.unravel_index(np.argmax(scores), scores.shape)
        print(f"\nBest TX_START_OFFSET: {tx_offs[best[0]]:.6f} s")
        print(f"Best RX_START_OFFSET: {rx_offs[best[1]]:.6f} s")
        print(f"Best score: {scores[best[0], best[1]]:.4f}")

        # Save results
        os.makedirs("results/ALIGNMENT", exist_ok=True)

        # Heatmap
        fig, axes = pl.subplots(1, 3, figsize=(20, 6))

        im = axes[0].imshow(scores.T, origin='lower', aspect='auto',
                             extent=[tx_offs[0]*1000, tx_offs[-1]*1000,
                                     rx_offs[0]*1000, rx_offs[-1]*1000])
        axes[0].set_xlabel("TX_START_OFFSET (ms)")
        axes[0].set_ylabel("RX_START_OFFSET (ms)")
        axes[0].set_title("Alignment Score")
        axes[0].plot(tx_offs[best[0]]*1000, rx_offs[best[1]]*1000, 'r*', markersize=15)
        pl.colorbar(im, ax=axes[0])

        axes[1].imshow(log_A.T, aspect='auto',
                        vmax=log_A.max() * 0.8, vmin=log_A.max() * 0.4)
        axes[1].set_title("DD Image (log)")

        axes[2].imshow(edge_img.T, aspect='auto',
                        vmax=np.percentile(edge_img, 99))
        axes[2].set_title("Edge Image (Sobel)")

        pl.tight_layout()
        pl.savefig("results/ALIGNMENT/alignment_search.png", dpi=150)
        print("Saved results/ALIGNMENT/alignment_search.png")
