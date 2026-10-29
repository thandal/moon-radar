"""
Delay-Doppler imaging and per-look calibration.

Per-look core of the pipeline: compute_dd_image (TX phase-interpolation
resampling, SRP Doppler-rate compensation, GPU block-sum correlation), rim calibration
(measure_rim_offset: per-look Doppler residual delta and rim-spread
diagnostic), the edge fit (fit_edge_offsets: per-look delay and Doppler
offsets against a model image from SPICE and LOLA), surface projection with
degeneracy masking, and the batch entry point process_file. See WALKTHROUGH.md for the end-to-end flow and REPORT.md
for conventions and the error budget.
"""

import json
import math
import os
import re
import numpy as np
import scipy.signal
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
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
from spice_setup import furnsh_kernels, kernel_fingerprint
furnsh_kernels()

from doppler_equator import (
    AB_COR, EARTH_FRAME, DLT_DT,
    et_from_astropy, moon_radii, moon_surface_points,
    load_lola_dem, get_lola_elevation, extract_srp_elevation, terrain_model, TERRAIN_MODELS,
    moonPointLightTime_BCK, moonPointLightTime_FWD,
    moonPointDLT_BCK, moonPointDLT_FWD,
    subpoint_average_guess, specular_point_bck, specular_point_fwd,
    moonSRP_DLT_BCK, moonSRP_DLT_FWD,
    srp_dlt_rate_bck, rate_corrected_dlt, apparent_station_positions,
    compute_doppler_equator, compute_doppler_equator_velocity,
    srp_velocity_analytic,
)

MOON_RADIUS = 1_737_400.0 * au.m

# Points averaged per map pixel along each axis in lunar_projection
# (SUPERSAMPLE**2 per pixel). Test on 2026-10-20, the eight 125 s 2026-10-11
# looks per channel at nside 400: the correlation between two independent
# half-stacks (detail that repeats; speckle does not) at 12-20 km features
# was 0.067 / 0.145 / 0.194 for 1 / 4 / 16 points per pixel (opposite sense), i.e.
# 2.4x and 3.4x the signal-to-noise of one point; same sense 2.1x and 3.1x.
# A pixel holds only ~4 independent cells (125 s Doppler rows ~1.2 km apart;
# delay resolution ~6 km), so more points should add little.
SUPERSAMPLE = 4

# Delay samples the DD window leads the SRP delay by (compute_dd_image), and
# how many of them, at the start of the window, give the noise floor
# (noise_floor_log_amp). The echo's correlation response starts one chip
# before its leading edge, so the noise columns are echo-free only for
# codes with fewer than max_chip_samples_for_noise() samples per chip;
# process_file refuses slower codes rather than fold echo into the floor.
LEAD_SAMPLES = 20
NOISE_COLUMNS = 6

# Block-sum correlation (correlate_blocks): the power series of the
# within-block Doppler phase stops when the remainder is below
# DD_SERIES_TOL of the summed magnitude; one part of the delay window holds
# at most DD_TABLE_BYTES of block sums on the GPU; blocks and lags are
# processed DD_BLOCKS_PER_BATCH and DD_LAGS_PER_BATCH at a time.
DD_SERIES_TOL = 1e-5
DD_TABLE_BYTES = 256 * 2 ** 20
DD_BLOCKS_PER_BATCH = 1024
DD_LAGS_PER_BATCH = 128

# RX file prefix (the station's capture files) -> SPICE station name.
# STOCKERT is the 25 m bistatic receiver; ATA recorded the 2025-09-16 session
# (prefix "ata_radar_*"); WSRT (Westerbork) the 2026-10-11 session (prefix
# "wsrt_*"); DWINGELOO is the transmitter's own (monostatic) receiver.
# Every script maps stations through this table; an unknown station is an
# error.
STATION_RX = {"stockert": "STOCKERT", "ata": "ATA", "wsrt": "WSRT",
              "dwingeloo": "DWINGELOO"}
TX_STATION = "DWINGELOO"

# Rim calibration's scan/reference-strip anchor (Hz; calibrate_rim): the
# legacy 10-row window at its native 5.5 mHz row pitch (the 09-16/06-21
# looks), so the strips keep the placement that behaves on real profiles and
# only scale up at finer pitches (investigations/
# rim_window_recalibration_2026-07-03.md).
RIM_CAPTURE_HZ = 0.055

# Edge fit (fit_edge_offsets): the incidence range, along the look's Doppler
# equator, whose echo edges are fitted. Below 10 deg the two edges close in
# on the specular glint, whose brightness the generic scattering law does
# not model; beyond 80 deg the edge is the limb, set by terrain shadowing,
# which the model leaves out. Model rows per image row (the model is shifted
# by fractions of a row while fitting).
EDGE_FIT_INCIDENCE_DEG = (10.0, 80.0)
# The edge fit searches delay over +- the larger of one chip and this: the
# delay of the local terrain at the SRP that the terrain model leaves out.
# On the Moon's smooth shape (the maps' model since 2026-10-27) that terrain
# has an rms of 1.1 km within 60 deg of the disk centre (fit_moon_figure.py),
# about 7 us of two-way delay; 20 us covers +-3 km. One chip (20-40 us at
# 25-50 kchip/s) already did; at 125 kchip/s (8 us) it did not, and on
# 2026-10-24 WSRT 19:27:32's fit stopped at the limit at +9.7 us.
EDGE_TERRAIN_DELAY_S = 20e-6
EDGE_MODEL_OVERSAMPLE = 4

# Captures shorter than this are not processed (registration_stability.py
# records them as skipped).
MIN_CAPTURE_S = 20.0

# The degeneracy limit on lunar_projection's multiplicity (one definition,
# with the gridding helpers that apply it).
from registration_analysis import MAX_MULTIPLICITY  # noqa: E402

# The TX file's core:datetime is the waveform *generation* time, not the
# emission epoch (it is off by ~25 h on the 2025-06-21 dataset and ~88 days
# on 2025-09-16; see investigations/test_tx_start.py). By convention the
# transmission starts this long after the RX recording starts; the measured
# per-look timing offset (freq_offset_hunt.measure_offset) absorbs the rest.
# Every stage that builds the TX model must use nominal_tx_emit_start.
TX_EMIT_DELAY_S = 1.0


def nominal_tx_emit_start(rx_start):
    """Nominal TX emission epoch for a capture starting at rx_start (astropy
    Time): TX_EMIT_DELAY_S after it."""
    return rx_start + TX_EMIT_DELAY_S * au.s


def max_chip_samples_for_noise():
    """Longest chip (samples) for which the noise columns stay clear of the
    echo: the last noise column, LEAD_SAMPLES - NOISE_COLUMNS + 1 samples
    before lag 0, must precede the chip-wide correlation response by at
    least one sample (the per-look timing precision, REPORT 3.1)."""
    return LEAD_SAMPLES - NOISE_COLUMNS


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
        lt_min_image: light time the image's lag 0 refers to (process_file's lt_ref)
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

    # No curve point inside the image: there is no score, not a score of 0.
    return score / count if count else float("nan")


def measure_rim_offset(log_A, dlt_shifts, delay_values_s, lt_min_image,
                       lt_min_eq, delay_up, dlt_up, delay_down, dlt_down,
                       delay_min_s=0.0015, n_cols_avg=3,
                       inner_off=(10, 50), outer_off=(30, 90),
                       min_contrast=0.4, min_samples=30, col_parity=None,
                       delta_capture_dlt=None):
    """Per-look Doppler self-calibration from the horseshoe rim positions.

    The specular-tone centroid calibrates Doppler at the SRP only, and is
    fading-limited to ~tens of mHz. The degenerate stripe amplifies exactly
    that residual (surface displacement ~ sqrt(residual/curvature)), showing
    up as the dark-wedge / seam-fan asymmetry. The rim of the horseshoe is a
    direct probe: a signed chain residual shifts BOTH rims by the same dlt,
    so delta = mean(up-rim offset, down-rim offset); their half-difference
    diagnoses rate/curvature error (rim spread).

    For each sampled delay along the predicted equator curves, the image's
    Doppler profile is scanned outward from inside the rim and the half-power
    edge crossing is located; offsets are medianed over delays.

    Returns dict with delta_dlt (add to predicted dlt, i.e. sample the image
    at dlt_shifts - delta_dlt), spread_dlt, per-branch medians and sample
    counts -- or None if the rim is too weak to measure (e.g. a same-sense look).
    """
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    # Keep the scan/reference geometry fixed in Doppler-frequency units, not
    # rows: with the default 10-row scan start, fine-bin epochs (1.8 mHz
    # bins, the 09-11 morning looks) covered only ~18 mHz of inward delta --
    # a true edge further inside the prediction was silently censored
    # (one-sided medians; REPORT 3.2). delta_capture_dlt anchors the row
    # geometry in dlt units (process_file passes 0.055 Hz / f0 = the legacy
    # 10-row window at its native 5.5 mHz pitch); all four offsets scale
    # together so the reference strips keep their physical Doppler placement
    # relative to the rim. Deeper |delta| capture comes from the caller's
    # convergence loop, NOT from scaling this window further: strips deeper
    # than ~55..275 mHz measurably bias the half-power threshold on real
    # profiles (investigations/rim_window_recalibration_2026-07-03.md).
    if delta_capture_dlt is not None and inner_off[0] * ddlt < delta_capture_dlt:
        s = delta_capture_dlt / (inner_off[0] * ddlt)
        inner_off = (int(np.ceil(inner_off[0] * s)), int(np.ceil(inner_off[1] * s)))
        outer_off = (int(np.ceil(outer_off[0] * s)), int(np.ceil(outer_off[1] * s)))
    ddelay = delay_values_s[1] - delay_values_s[0]
    n_dop, n_del = log_A.shape

    def branch_offsets(delay_curve, dlt_curve, outward):
        offs = []
        delay_img = delay_curve + (lt_min_eq - lt_min_image)
        for k in range(len(delay_curve)):
            if col_parity is not None and (k % 2) != col_parity:
                continue
            if delay_img[k] < delay_min_s:
                continue
            j = int(round((delay_img[k] - delay_values_s[0]) / ddelay))
            if not (n_cols_avg <= j < n_del - n_cols_avg):
                continue
            r_pred = (dlt_curve[k] - dlt_shifts[0]) / ddlt
            prof = log_A[:, j - n_cols_avg:j + n_cols_avg + 1].mean(axis=1)
            ri = int(round(r_pred))
            i0, i1 = sorted([ri - outward * inner_off[1], ri - outward * inner_off[0]])
            o0, o1 = sorted([ri + outward * outer_off[0], ri + outward * outer_off[1]])
            if min(i0, o0) < 0 or max(i1, o1) >= n_dop:
                continue
            inner = np.median(prof[i0:i1])
            outer = np.median(prof[o0:o1])
            if inner - outer < min_contrast:
                continue
            mid = (inner + outer) / 2
            # scan outward from inside the rim for the half-power crossing
            scan = np.arange(ri - outward * inner_off[0], ri + outward * outer_off[1],
                             outward)
            vals = prof[scan]
            below = np.nonzero(vals < mid)[0]
            if len(below) == 0 or below[0] == 0:
                continue
            b = below[0]
            # linear interpolation of the crossing
            frac = (vals[b - 1] - mid) / max(vals[b - 1] - vals[b], 1e-9)
            r_edge = scan[b - 1] + outward * frac
            offs.append((r_edge - r_pred) * ddlt)
        return np.array(offs)

    # up branch = approaching = low dlt: outward is -rows; down branch: +rows
    e_up = branch_offsets(delay_up, dlt_up, -1)
    e_down = branch_offsets(delay_down, dlt_down, +1)
    if len(e_up) < min_samples or len(e_down) < min_samples:
        return None
    med_up, med_down = np.median(e_up), np.median(e_down)
    return {
        "delta_dlt": (med_up + med_down) / 2,
        "spread_dlt": (med_down - med_up) / 2,
        "up_dlt": med_up, "down_dlt": med_down,
        "n_up": len(e_up), "n_down": len(e_down),
    }


def rim_seed_search(log_A, dlt_shifts, delay_values_s, lt_min_image,
                    lt_min_eq, delay_up, dlt_up, delay_down, dlt_down,
                    delta_capture_dlt=None, max_offset_dlt=None, **rim_kwargs):
    """Coarse re-acquisition when the rim gates fail at zero offset.

    A tone-centroid deep fade can displace the compensated echo by hundreds
    of mHz (observed +270 mHz on the 2025-09-16 14:04:14 look) -- far beyond
    the single-pass capture, so the reference strips land off the true rim
    and every column fails the contrast gates; the convergence crawl never
    gets a first step and the look goes uncertified with an uncorrected
    Doppler warp. Sweep trial offsets on a grid as coarse as the inward
    capture and return a seed (the winning trial refined by its own
    measurement) for the normal convergence loop, or None if no trial passes
    the gates (e.g. no-signal captures -- the gates reject pure noise).
    The winner is the trial with the most gate-passing columns, an
    edge-quality measure that spurious partial locks score poorly on.

    Returns None or a dict: seed_dlt, n (the winner's gate-passing columns)
    and runner_up_n (that of the best trial landing at least one sweep step
    from the winner -- a different rim position; 0 if none), so the caller
    can record how clear the win was. Equal counts go to the trial tried
    first (smallest |offset|, + before -). The seed is only a starting
    point: process_file applies nothing unless the rim then passes its
    gates at the seeded offset.
    """
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    step = delta_capture_dlt if delta_capture_dlt is not None else 10 * ddlt
    if max_offset_dlt is None:
        max_offset_dlt = 12 * step  # +-660 mHz at L-band; > worst fade seen
    trials = []
    for k in range(1, int(np.ceil(max_offset_dlt / step)) + 1):
        for sgn in (+1, -1):
            trial = sgn * k * step
            r = measure_rim_offset(log_A, dlt_shifts - trial, delay_values_s,
                                   lt_min_image, lt_min_eq, delay_up, dlt_up,
                                   delay_down, dlt_down,
                                   delta_capture_dlt=delta_capture_dlt,
                                   **rim_kwargs)
            if r is not None:
                trials.append((r["n_up"] + r["n_down"], trial + r["delta_dlt"]))
    if not trials:
        return None
    trials.sort(key=lambda t: -t[0])
    n_best, seed = trials[0]
    # The runner-up is the best trial that converged on a different rim
    # position: neighbouring trials usually land on the winner's own rim.
    other = [n for n, d in trials[1:] if abs(d - seed) >= step]
    return {"seed_dlt": seed, "n": n_best, "runner_up_n": other[0] if other else 0}


# ---------------------------------------------------------------------------
# DD image computation (from notebook)
# ---------------------------------------------------------------------------
def calibrate_rim(log_A, dlt_shifts, delay_values_s, lt_ref, lt_min_eq,
                  delay_up, dlt_up, delay_down, dlt_down, delta_capture_dlt):
    """process_file's rim self-calibration: the residual chain Doppler,
    measured from the horseshoe rims (measure_rim_offset).

    If the rims fail their gates at zero offset (no signal, or a tone deep
    fade displaced the echo beyond the gates' reach), one coarse
    re-acquisition sweep (rim_seed_search) runs; its seed is used only if
    the rims then pass the gates there. The calibration then converges: the
    half-power edge finder is not perfectly linear in the offset (typically
    2 passes); the cap of 12 lets censored looks (true |delta| beyond the
    single-pass capture) crawl in at the measured ~15-30 mHz/pass -- a
    225 mHz offset recovers on real data. Finally the rims are re-measured
    at the applied offset.

    delta_capture_dlt anchors the scan/reference-strip geometry (process_file
    passes RIM_CAPTURE_HZ, 55 mHz = the legacy 10-row window at its native
    5.5 mHz row pitch,
    so the strips keep the placement that behaves on real profiles and only
    scale up at finer pitches; the earlier 80 mHz value moved delta by tens
    of mHz on normal-pitch looks -- A/B over the 09-16 + 09-11 blocks,
    2026-07-03: look-to-look RMS of the total chain offset 4.5/4.6 mHz vs
    32/8 mHz; investigations/rim_window_recalibration_2026-07-03.md).

    Returns a dict: delta_dlt (0.0 when failed: nothing is applied), status
    ("converged": the rims pass at the applied offset and leave under half a
    row; "not_converged": they do not; "failed": no pass got through the
    gates), rim_first and rim_final (measure_rim_offset results or None),
    seed (rim_seed_search's result or None) and passes.
    """
    ddlt_bin = dlt_shifts[1] - dlt_shifts[0]

    def measure_rim(delta):
        return measure_rim_offset(log_A, dlt_shifts - delta, delay_values_s,
                                  lt_ref, lt_min_eq, delay_up, dlt_up,
                                  delay_down, dlt_down,
                                  delta_capture_dlt=delta_capture_dlt)

    rim_first = rim_final = seed = None
    passes = 0
    delta_dlt = 0.0
    rim_i = measure_rim(0.0)
    if rim_i is None:
        seed = rim_seed_search(log_A, dlt_shifts, delay_values_s, lt_ref,
                               lt_min_eq, delay_up, dlt_up, delay_down,
                               dlt_down, delta_capture_dlt=delta_capture_dlt)
        if seed is not None:
            rim_i = measure_rim(seed["seed_dlt"])
            if rim_i is not None:
                delta_dlt = seed["seed_dlt"]
    while rim_i is not None and passes < 12:
        rim_first = rim_i if rim_first is None else rim_first
        delta_dlt += rim_i["delta_dlt"]
        passes += 1
        if abs(rim_i["delta_dlt"]) < 0.5 * ddlt_bin:
            break
        rim_i = measure_rim(delta_dlt)
    if rim_first is None:
        status, delta_dlt = "failed", 0.0
    else:
        rim_final = measure_rim(delta_dlt)
        converged = (rim_final is not None and
                     abs(rim_final["delta_dlt"]) < 0.5 * ddlt_bin)
        status = "converged" if converged else "not_converged"
    return {"delta_dlt": delta_dlt, "status": status, "rim_first": rim_first,
            "rim_final": rim_final, "seed": seed, "passes": passes}


def edge_model_image(n_rows, dlt_shifts, delay_values_s, lt_ref, rx_time_s, *,
                     rx_duration_s, dlt_rate_srp, chip_samples, tx_overlap_s,
                     f_hz, delay_range_s, doppler_range_hz, tx_name, rx_name,
                     use_dem="figure"):
    """The DD image a look should give near its echo edges, from external
    inputs only: SPICE geometry, the LOLA terrain, a generic scattering law
    (cos(incidence)^1.5) and uniform reflectivity -- no radar map. For
    fit_edge_offsets.

    Surface points are placed in the image as lunar_projection places them
    (the same station-position light-time field, lag 0 at lt_ref, rows
    in the image's start-anchored, rate-compensated Doppler), except that
    each point's Doppler drift over the look is kept: the look is split into
    sub-intervals, enough that a point drifts by less than one Doppler
    resolution cell (1 / tx_overlap_s) within each. The sum is convolved with
    the code's delay response (triangle_power, one chip either side) and the
    coherent integration's Doppler response (sinc^2 with its sidelobes).

    Only a band of surface either side of the look's Doppler equator is
    modelled: wide enough to cover every row a fit window can reach, with
    the windows' widest search (edge slope x (one chip + delay_range_s),
    plus doppler_range_hz and three resolution cells) and a 20 % margin.

    Returns (M, edge_lo, edge_hi, incidence_deg): M is power (arbitrary
    scale) on EDGE_MODEL_OVERSAMPLE rows per image row, float32;
    edge_lo/edge_hi the outermost rows reached by surface points in each
    delay column (image-row units, window-mean Doppler; NaN where none);
    incidence_deg the incidence on the Doppler equator at each column's
    delay (NaN where none).
    """
    over = EDGE_MODEL_OVERSAMPLE
    n_cols = len(delay_values_s)
    ddelay = delay_values_s[1] - delay_values_s[0]
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    row_hz = abs(ddlt) * f_hz
    res_hz = 1.0 / tx_overlap_s
    chip_s = chip_samples * ddelay
    R = moon_radii()[0]
    c_km = csp.clight()
    T = rx_duration_s

    # The Doppler equator: the great circle through the SRP along its
    # apparent motion (as compute_doppler_equator_velocity).
    tm = rx_time_s + T / 2
    s_hat = specular_point_bck(tm, tx_name, rx_name)
    s_hat = s_hat / np.linalg.norm(s_hat)
    v_tan, _, _ = srp_velocity_analytic(tm, tx_name, rx_name)
    v_tan = v_tan - (v_tan @ s_hat) * s_hat
    w_hat = v_tan / np.linalg.norm(v_tan)
    u_hat = np.cross(s_hat, w_hat)

    def lt_field(p, sta):
        R_rx, R_tx, c = sta
        return (np.linalg.norm(p - R_rx, axis=1) + np.linalg.norm(p - R_tx, axis=1)) / c

    # Limb Doppler and drift, from points along the equator.
    a_probe = np.radians(np.arange(-85.0, 85.1, 5.0))
    p_probe = moon_surface_points(np.cos(a_probe)[:, None] * s_hat
                                  + np.sin(a_probe)[:, None] * w_hat)
    sta3 = [apparent_station_positions(rx_time_s + k * T / 2, tx_name, rx_name)
            for k in range(3)]
    lt3 = [lt_field(p_probe, s) for s in sta3]
    dlt_halves = [(lt3[k + 1] - lt3[k]) / (T / 2) for k in range(2)]
    drift_hz = 2 * np.max(np.abs(dlt_halves[1] - dlt_halves[0] - dlt_rate_srp * T / 2)) * f_hz
    n_sub = max(1, int(np.ceil(drift_hz / res_hz)))
    limb_hz = (abs((lt3[2][-1] - lt3[0][-1]) - (lt3[2][0] - lt3[0][0])) / T * f_hz
               / 2 / np.sin(a_probe[-1]))

    # Band half-width vs angle along the equator: a window reaching W Hz
    # inside the edge takes points out to R sqrt(2 W sin(a) / limb_hz) (sphere,
    # edge Doppler limb_hz sin(a)); W = slope (one chip + delay_range_s) + ...
    # with slope = limb_hz c cos(a) / (2 R sin(a)) Hz/s.
    def band_km(a):
        sa, ca = np.abs(np.sin(a)), np.cos(a)
        two_w_sin = (2 * limb_hz * c_km * ca / (2 * R) * (chip_s + delay_range_s)
                     + 2 * (doppler_range_hz + 3 * res_hz) * sa)
        return 1.2 * R * np.sqrt(np.clip(two_w_sin / limb_hz, 0.0, 2.0))

    # Point spacing: one delay sample of surface at grazing (c / (2 fs)) or
    # one Doppler row at the SRP (row_hz R / limb_hz), whichever is smaller:
    # neighbouring points then land in the same or adjacent cells (they are
    # spread bilinearly between cells).
    step_km = min(c_km * ddelay / 2, row_hz * R / limb_hz)
    alpha = np.arange(-np.radians(95.0), np.radians(95.0), step_km / R)
    epochs = rx_time_s + np.arange(n_sub + 1) * T / n_sub
    sta = [apparent_station_positions(t, tx_name, rx_name) for t in epochs]

    M = np.zeros((n_rows * over, n_cols))
    edge_hi = np.full(n_cols, -np.inf)
    edge_lo = np.full(n_cols, np.inf)
    inc_sum = np.zeros(n_cols)
    inc_n = np.zeros(n_cols)
    for a0 in range(0, len(alpha), 256):
        al = alpha[a0:a0 + 256]
        d_max = band_km(al).max()
        dist = np.arange(-d_max, d_max + step_km / 2, step_km) / R
        A_, D_ = np.meshgrid(al, dist, indexing="ij")
        A_, D_ = A_.ravel(), D_.ravel()
        cd = np.cos(D_)
        v = ((cd * np.cos(A_))[:, None] * s_hat + (cd * np.sin(A_))[:, None] * w_hat
             + np.sin(D_)[:, None] * u_hat)
        cosi = cd * np.cos(A_)                   # sphere normal vs the line of sight
        vis = cosi > 0.0
        v, cosi, A_, D_, cd = v[vis], cosi[vis], A_[vis], D_[vis], cd[vis]
        p = moon_surface_points(v, use_dem=use_dem)
        del v
        lts = [lt_field(p, s) for s in sta]
        col = (lts[0] - lt_ref - delay_values_s[0]) / ddelay
        weight = cosi ** 1.5 * cd / n_sub        # law x area element
        c0 = np.floor(col).astype(np.int64)
        fc = col - c0
        on_eq = np.abs(D_) < step_km / R
        j = np.rint(col[on_eq]).astype(np.int64)
        ok = (j >= 0) & (j < n_cols)
        np.add.at(inc_sum, j[ok], np.abs(A_[on_eq][ok]))
        np.add.at(inc_n, j[ok], 1)
        row_mean = np.zeros(len(col))
        for k in range(n_sub):
            dlt = ((lts[k + 1] - lts[k]) / (T / n_sub)
                   - dlt_rate_srp * (k + 0.5) * T / n_sub)
            row = (dlt - dlt_shifts[0]) / ddlt
            row_mean += row / n_sub
            y = row * over
            r0 = np.floor(y).astype(np.int64)
            fr = y - r0
            for dr, wr in ((0, 1 - fr), (1, fr)):
                for dc, wc in ((0, 1 - fc), (1, fc)):
                    rr, cc = r0 + dr, c0 + dc
                    ok = (rr >= 0) & (rr < n_rows * over) & (cc >= 0) & (cc < n_cols)
                    np.add.at(M.ravel(), rr[ok] * n_cols + cc[ok], (weight * wr * wc)[ok])
        j = np.rint(col).astype(np.int64)
        ok = (j >= 0) & (j < n_cols)
        np.maximum.at(edge_hi, j[ok], row_mean[ok])
        np.minimum.at(edge_lo, j[ok], row_mean[ok])
    M = M.astype(np.float32)
    M = scipy.ndimage.convolve1d(M, triangle_power(chip_samples), axis=1, mode="constant")
    # Sidelobes out to twice the widest window, in model rows.
    null_rows = res_hz / row_hz * over
    n_null = int(np.ceil(2 * (2 * limb_hz * c_km / (2 * R) * chip_s / np.tan(
        np.radians(EDGE_FIT_INCIDENCE_DEG[0])) + doppler_range_hz + 3 * res_hz) / res_hz))
    k = np.arange(-int(n_null * null_rows), int(n_null * null_rows) + 1)
    sinc2 = np.sinc(k / null_rows) ** 2
    M = scipy.signal.fftconvolve(M, (sinc2 / sinc2.sum()).astype(np.float32)[:, None],
                                 mode="same", axes=0).astype(np.float32)
    edge_hi = np.where(np.isfinite(edge_hi), edge_hi, np.nan)
    edge_lo = np.where(np.isfinite(edge_lo), edge_lo, np.nan)
    incidence = np.where(inc_n > 0, np.degrees(inc_sum / np.maximum(inc_n, 1)), np.nan)
    return M, edge_lo, edge_hi, incidence


def fit_edge_offsets(log_A, dlt_shifts, delay_values_s, lt_ref, rx_time_s, *,
                     rx_duration_s, dlt_rate_srp, chip_samples, tx_overlap_s, f_hz,
                     noise_log_amp, tx_name, rx_name, use_dem="figure", n_jackknife=10):
    """Delay and Doppler offsets of a look's echo, from its edges.

    At each delay the echo's Doppler extent ends at the two limbs of the
    Doppler equator. Where those edges sit depends on both the delay and
    the Doppler reference: a delay error moves both edges inward or outward
    by (the edge's slope in Doppler per unit delay) x (the error), most near
    the SRP, where the slope is steepest; a Doppler error moves both edges
    the same way in Doppler, inward at one limb and outward at the other.
    The rim calibration (calibrate_rim) measures only the second, so a
    timing error leaves edges inside or outside where the projection puts
    them, and pixels along the equator read beyond the echo (dark) or
    inside it (bright).

    This fits the image near both edges, over EDGE_FIT_INCIDENCE_DEG, as
    a x model + b x noise floor, the model (edge_model_image) shifted later
    by delay_s and higher in Doppler by doppler_hz. a is free per limb and
    per chip of delay (reflectivity along the equator), b per limb. The fit
    maximizes the single-look speckle likelihood (exponential intensity),
    first on a grid over +- the larger of one chip and EDGE_TERRAIN_DELAY_S
    (the unmodelled terrain at the SRP) and +-3 Doppler resolution cells around
    the given axes, then from the best grid point in windows recentred on
    the shifted edges. The windows hold the model's edge transition (edge
    slope x one chip, plus the Doppler response) and three resolution cells
    beyond it each side. Errors are delete-one jackknife estimates over
    n_jackknife groups of delay columns.

    Inputs as lunar_projection's (dlt_shifts already rim-calibrated; use_dem
    as the projection's), plus noise_log_amp (noise_floor_log_amp of the
    image).

    Returns a dict: status ("fitted"; "at_limit": the best fit is at the
    edge of the search; "failed": fewer than two chips of delay with both
    edges in range), delay_s (+ = the echo is later than lag 0 says: map
    with lt_ref - delay_s), doppler_hz (+ = the echo is at higher Doppler
    than the rows say: map with dlt_shifts + doppler_hz / f_hz), their
    jackknife errors delay_err_s and doppler_err_hz, and n_columns.
    """
    over = EDGE_MODEL_OVERSAMPLE
    n_rows, n_cols = log_A.shape
    ddelay = delay_values_s[1] - delay_values_s[0]
    fs = 1.0 / ddelay
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    hz_per_row = -ddlt * f_hz                    # signed: rows run in dlt order
    row_hz = abs(hz_per_row)
    res_hz = 1.0 / tx_overlap_s
    chip_s = chip_samples * ddelay
    delay_range_s, doppler_range_hz = max(chip_s, EDGE_TERRAIN_DELAY_S), 3 * res_hz
    failed = {"status": "failed", "delay_s": np.nan, "doppler_hz": np.nan,
              "delay_err_s": np.nan, "doppler_err_hz": np.nan, "n_columns": 0}

    M, edge_lo, edge_hi, incidence = edge_model_image(
        n_rows, dlt_shifts, delay_values_s, lt_ref, rx_time_s,
        rx_duration_s=rx_duration_s, dlt_rate_srp=dlt_rate_srp,
        chip_samples=chip_samples, tx_overlap_s=tx_overlap_s, f_hz=f_hz,
        delay_range_s=delay_range_s, doppler_range_hz=doppler_range_hz,
        tx_name=tx_name, rx_name=rx_name, use_dem=use_dem)
    lo_deg, hi_deg = EDGE_FIT_INCIDENCE_DEG
    cols = np.nonzero((incidence >= lo_deg) & (incidence <= hi_deg)
                      & np.isfinite(edge_lo) & np.isfinite(edge_hi))[0]
    if len(cols) < 2 * chip_samples:
        return failed
    # Edge slopes (Hz per s of delay), smoothed over one chip: the edge is
    # only defined to within the delay response.
    n_sm = max(1, int(round(chip_samples)))
    kern = np.ones(n_sm) / n_sm
    slope = {}
    for name, e in (("lo", edge_lo), ("hi", edge_hi)):
        ok = np.isfinite(e)
        g = np.gradient(np.interp(np.arange(n_cols), np.nonzero(ok)[0], e[ok]))
        slope[name] = np.abs(np.convolve(g, kern, "same")) * row_hz * fs
    P = np.exp(2.0 * np.asarray(log_A, dtype=np.float64))
    N0 = np.exp(2.0 * noise_log_amp)
    cols_all = np.arange(n_cols)
    filled = {}
    for name, e in (("lo", edge_lo), ("hi", edge_hi)):
        ok = np.isfinite(e)
        filled[name] = np.interp(cols_all, cols_all[ok], e[ok])

    def cells(delay_s, doppler_hz, search):
        """Window cells (rows, cols, block, limb) around the edges shifted by
        (delay_s, doppler_hz); search widens them by the search ranges."""
        r_l, c_l, b_l, l_l = [], [], [], []
        for limb, name in enumerate(("lo", "hi")):
            centre = (np.interp(cols - delay_s * fs, cols_all, filled[name])
                      + doppler_hz / hz_per_row)
            half = slope[name][cols] * chip_s + 3 * res_hz
            if search:
                half = half + slope[name][cols] * delay_range_s + doppler_range_hz
            half_rows = half / row_hz
            mid = (edge_lo[cols] + edge_hi[cols]) / 2
            lo_r = np.ceil(centre - half_rows).astype(int)
            hi_r = np.floor(centre + half_rows).astype(int)
            if name == "lo":
                hi_r = np.minimum(hi_r, np.floor(mid).astype(int))
            else:
                lo_r = np.maximum(lo_r, np.ceil(mid).astype(int))
            lo_r, hi_r = np.clip(lo_r, 0, n_rows - 1), np.clip(hi_r, 0, n_rows - 1)
            for c, a, b in zip(cols, lo_r, hi_r):
                if b < a:
                    continue
                r = np.arange(a, b + 1)
                r_l.append(r)
                c_l.append(np.full(len(r), c))
                b_l.append(np.full(len(r), limb * n_cols + (c - cols[0]) // n_sm))
                l_l.append(np.full(len(r), limb))
        r, c = np.concatenate(r_l), np.concatenate(c_l)
        blk = np.unique(np.concatenate(b_l), return_inverse=True)[1]
        return r, c, blk, np.concatenate(l_l)

    def make_nll(r, c, blk, limb, keep=None):
        if keep is not None:
            r, c, blk, limb = r[keep], c[keep], blk[keep], limb[keep]
            blk = np.unique(blk, return_inverse=True)[1]
        obs = P[r, c]
        nb = blk.max() + 1
        Mf = M.ravel()
        floor = 1e-2 * N0

        def nll(x):
            delay_s, doppler_hz = x[0] * ddelay, x[1] * res_hz
            y = r * over - doppler_hz / hz_per_row * over
            xx = c - delay_s * fs
            y0, x0 = np.floor(y).astype(np.int64), np.floor(xx).astype(np.int64)
            fy, fx = y - y0, xx - x0
            y0 = np.clip(y0, 0, n_rows * over - 2)
            x0 = np.clip(x0, 0, n_cols - 2)
            m = ((1 - fy) * (1 - fx) * Mf[y0 * n_cols + x0]
                 + (1 - fy) * fx * Mf[y0 * n_cols + x0 + 1]
                 + fy * (1 - fx) * Mf[(y0 + 1) * n_cols + x0]
                 + fy * fx * Mf[(y0 + 1) * n_cols + x0 + 1])
            a = (np.bincount(blk, obs * m, nb)
                 / np.maximum(np.bincount(blk, m * m, nb), 1e-300))
            b = np.ones(2)
            for _ in range(4):
                mu = np.maximum(a[blk] * m + b[limb] * N0, floor)
                w = 1.0 / mu ** 2
                a = np.maximum(np.bincount(blk, w * m * (obs - b[limb] * N0), nb)
                               / np.maximum(np.bincount(blk, w * m * m, nb), 1e-300), 0.0)
                mu = np.maximum(a[blk] * m + b[limb] * N0, floor)
                w = 1.0 / mu ** 2
                b = (np.bincount(limb, w * N0 * (obs - a[blk] * m), 2)
                     / np.bincount(limb, w * N0 * N0, 2))
            mu = np.maximum(a[blk] * m + b[limb] * N0, floor)
            return float(np.sum(obs / mu + np.log(mu)))
        return nll

    # Search in units of delay samples and Doppler resolution cells.
    dmax, omax = delay_range_s * fs, doppler_range_hz / res_hz
    nll = make_nll(*cells(0.0, 0.0, search=True))
    grid = [(d, o) for d in np.linspace(-dmax, dmax, 21) for o in np.linspace(-omax, omax, 13)]
    x0 = np.array(min(grid, key=nll))
    window = cells(x0[0] * ddelay, x0[1] * res_hz, search=False)
    nll = make_nll(*window)
    opt = dict(method="Nelder-Mead", options={"xatol": 0.01, "fatol": 1e-3,
                                              "initial_simplex": [x0, x0 + [1, 0], x0 + [0, 0.25]]})
    best = scipy.optimize.minimize(nll, x0, **opt).x
    # Delete-one jackknife over groups of columns.
    groups = np.array_split(cols, n_jackknife)
    est = []
    for gcols in groups:
        keep = ~np.isin(window[1], gcols)
        est.append(scipy.optimize.minimize(make_nll(*window, keep=keep), best, **{
            **opt, "options": {**opt["options"], "initial_simplex":
                               [best, best + [0.5, 0], best + [0, 0.1]]}}).x)
    est = np.array(est)
    n = len(est)
    err = np.sqrt((n - 1) / n * np.sum((est - est.mean(axis=0)) ** 2, axis=0))
    inside = abs(best[0]) < dmax and abs(best[1]) < omax
    return {"status": "fitted" if inside else "at_limit",
            "delay_s": best[0] * ddelay, "doppler_hz": best[1] * res_hz,
            "delay_err_s": err[0] * ddelay, "doppler_err_hz": err[1] * res_hz,
            "n_columns": len(cols)}


def edge_row(edge):
    """The runs-CSV columns of an edge-fit result (fit_edge_offsets, or one
    inherited from the opposite-sense twin); blank where a value is undefined."""
    def num(x, scale):
        return "" if not np.isfinite(x) else float(x) * scale
    return {"edge_status": edge["status"],
            "edge_delay_us": num(edge["delay_s"], 1e6),
            "edge_delay_err_us": num(edge["delay_err_s"], 1e6),
            "edge_doppler_hz": num(edge["doppler_hz"], 1.0),
            "edge_doppler_err_hz": num(edge["doppler_err_hz"], 1.0),
            "edge_columns": edge["n_columns"]}


def compensated_tx(n_rx, tx_samples, sample_rate, frequency,
                   rx_start_time_s, tx_start_time_s, *, tx_name, rx_name,
                   freq_offset_hz=0.0, chunk=1 << 22):
    """The TX waveform as the receiver should see the specular echo: phase
    resampled onto the RX sample times (FWD SRP light time) and Doppler
    compensated (BCK SRP Doppler, linear rate over the capture, plus
    freq_offset_hz). The one model behind both the timing/frequency search
    (freq_offset_hunt) and the DD image, so a measured offset means the same
    thing in both.

    rx_start_time_s / tx_start_time_s are ET seconds (tx including any
    measured timing offset). Samples where the TX was not on air are zero.
    Returns (tx_compensated complex64, info): info has lt_rx_start,
    dlt_rate_srp, doppler_start_hz and tx_overlap_s (seconds of the capture
    the TX model covers).
    """
    fs = sample_rate.to_value(au.Hz)
    rx_duration_s = n_rx / fs
    tx_duration_s = len(tx_samples) / fs
    rx_end_time_s = rx_start_time_s + rx_duration_s
    tx_end_time_s = tx_start_time_s + tx_duration_s

    # FWD light times for TX resampling
    lt_tx_start, _ = moonSRP_DLT_FWD(tx_start_time_s, tx_name, rx_name)
    lt_tx_end, _ = moonSRP_DLT_FWD(tx_end_time_s, tx_name, rx_name)

    # BCK Doppler compensation. Wide (T/2) derivative stencils keep the
    # light-time granularity noise (~2e-11 s) out of the compensated rate
    # (see the NOTE above srp_dlt_rate_bck).
    lt_rx_start, dlt_rx_start = moonSRP_DLT_BCK(rx_start_time_s, tx_name, rx_name,
                                                dt=rx_duration_s / 2)
    lt_rx_end, dlt_rx_end = moonSRP_DLT_BCK(rx_end_time_s, tx_name, rx_name,
                                            dt=rx_duration_s / 2)
    f_hz = frequency.to_value(au.Hz)
    doppler_start = -dlt_rx_start * f_hz
    doppler_end = -dlt_rx_end * f_hz
    doppler_rate = (doppler_end - doppler_start) / rx_duration_s
    dlt_rate_srp = (dlt_rx_end - dlt_rx_start) / rx_duration_s

    # Resample TX: its phase, interpolated onto the RX sample times.
    # adjusted_tx_times0 maps TX samples to the RX-relative timeline (RX
    # sample times start at 0): TX arrives at absolute time
    # (tx_start_time_s + lt), RX-relative = absolute - rx_start_time_s.
    # (The phase is unwrapped first: its temporaries are the largest here.)
    tx_phase = np.unwrap(np.angle(tx_samples))
    tx_rx_offset = tx_start_time_s - rx_start_time_s
    adjusted_tx_times0 = np.linspace(
        tx_rx_offset + lt_tx_start,
        tx_rx_offset + tx_duration_s + lt_tx_end,
        len(tx_samples), endpoint=False)
    # The times are sorted (linspace) and not modified, so interp1d needs
    # neither its sort nor its copies.
    tx_phase_interp = scipy.interpolate.interp1d(
        adjusted_tx_times0, tx_phase,
        kind='linear', fill_value=np.nan, bounds_error=False,
        copy=False, assume_sorted=True)

    # The compensated waveform, a slice of RX samples at a time:
    # exp(1j * phase) (zero outside the transmission, where there is no
    # signal) times exp(-2j*pi*phi*t). Every step is elementwise, so slicing
    # leaves the result unchanged; whole-capture float64/complex128
    # temporaries would take ~2 GB on the 125 s 2026-10-11 captures.
    tx_compensated = np.empty(n_rx, dtype=np.complex64)
    n_on_air = 0
    for i0 in range(0, n_rx, chunk):
        t_s = np.arange(i0, min(i0 + chunk, n_rx)) / fs
        tx_resampled = 1j * tx_phase_interp(t_s)
        # Outside the transmission there is no signal: zero, and count what
        # is covered (recorded per look; a wrong start time shows up here).
        on_air = np.isfinite(tx_resampled)
        n_on_air += int(on_air.sum())
        np.exp(tx_resampled, out=tx_resampled)
        tx_resampled[~on_air] = 0.0
        phi_hz = -(doppler_start + doppler_rate * t_s / 2 + freq_offset_hz)
        ramp = -1j * 2 * np.pi * (phi_hz * t_s)
        np.exp(ramp, out=ramp)
        tx_resampled *= ramp
        tx_compensated[i0:i0 + len(t_s)] = tx_resampled
    tx_overlap_s = n_on_air / fs
    info = {"lt_rx_start": lt_rx_start, "dlt_rate_srp": dlt_rate_srp,
            "doppler_start_hz": doppler_start, "tx_overlap_s": tx_overlap_s}
    return tx_compensated, info


def delay_window(n_rx, sample_rate):
    """The DD image's delay window: (correlation lags in s as an astropy
    Quantity, first index, end index). It starts LEAD_SAMPLES before lag 0,
    where the echo's leading edge sits once the timing offset is applied,
    and spans the Moon's radius in round-trip delay."""
    cor_lags = scipy.signal.correlation_lags(n_rx, n_rx, mode="same") / sample_rate
    t_end = MOON_RADIUS / ak.c * 2
    cor_i_start = np.argwhere(cor_lags >= -LEAD_SAMPLES / sample_rate)[0][0]
    cor_i_end = np.argwhere(cor_lags >= t_end)[0][0]
    return cor_lags, cor_i_start, cor_i_end


def doppler_span(rx_start_time_s, tx_name, rx_name):
    """(min, max) dlt over the visible disk (its limb, from SPICE's
    terminator points): the DD image's Doppler range."""
    _, _, v_term = csp.edterm("UMBRAL", tx_name, "MOON", rx_start_time_s,
                              "MOON_ME", AB_COR, rx_name, 1000)
    _, dlt_term = moonPointDLT_BCK(rx_start_time_s, v_term, tx_name, rx_name)
    return float(dlt_term.min()), float(dlt_term.max())


def correlate_rows(rx_samples, tx_compensated, fs, f_shifts, lag_start, lag_end):
    """|sum_s rx[(s + lag) % n] conj(tx[s]) exp(-2j pi f s / fs)| for each f in
    f_shifts (rows, Hz) and lag in lag_start..lag_end-1 (columns), one
    full-length FFT correlation per row: compute_dd_image's method until
    2026-10-24, kept to check correlate_blocks. 165-174 s on a 125 s
    capture (RTX 3050 laptop GPU, mains power, 2026-10-24). Returns float32
    (rows, lags)."""
    # CUDA correlation, batched over Doppler rows: chunked 2-D FFTs amortize
    # kernel-launch and plan overhead, and float32 phase ramps replace the
    # complex128 exp of the old per-row loop (phase < 1e4 rad, so float32
    # keeps phase error < 1e-3 rad).
    tx_gpu = cupy.asarray(tx_compensated, dtype=cupy.complex64)
    rx_gpu = cupy.asarray(rx_samples, dtype=cupy.complex64)
    # Circular correlation puts lag -k at index n-k, so gather the window
    # with wrapped indices.
    lag_idx = cupy.asarray(np.arange(lag_start, lag_end) % len(rx_samples))
    A = cupy.zeros((len(f_shifts), lag_end - lag_start), dtype=cupy.float32)
    fft_rx = cupy.fft.fft(rx_gpu)
    t_norm = (cupy.arange(len(tx_compensated), dtype=cupy.float32)
              * cupy.float32(2 * np.pi / fs))
    # 4 rows x ~4 buffers x ~70 MB ~ 1.2 GB peak for a 35 s capture: leaves
    # room for several worker processes to share the GPU. Longer captures
    # (the 125 s 2026-10-11 looks, ~250 MB per row) drop to fewer rows so the
    # peak stays near that size and fits a 4 GB card.
    CHUNK = max(1, min(4, int(40_000_000 // len(rx_samples))))
    for i0 in tqdm(range(0, len(f_shifts), CHUNK), desc="Correlating"):
        fs_chunk = cupy.asarray(f_shifts[i0:i0 + CHUNK], dtype=cupy.float32)
        tx_block = tx_gpu[None, :] * cupy.exp(1j * fs_chunk[:, None] * t_norm[None, :])
        cor = cupy.fft.ifft(fft_rx[None, :] * cupy.conj(cupy.fft.fft(tx_block, axis=1)),
                            axis=1)
        A[i0:i0 + len(fs_chunk)] = cupy.abs(cor[:, lag_idx])
    A = cupy.asnumpy(A)
    # Return pool blocks to the driver so concurrent worker processes can
    # allocate (the per-process pool would otherwise hoard freed memory).
    del tx_gpu, rx_gpu, fft_rx, t_norm
    cupy.get_default_memory_pool().free_all_blocks()
    return A


def series_terms(x_max, tol=DD_SERIES_TOL):
    """Terms of the power series of exp(-i x), |x| <= x_max, that leave a
    remainder below tol."""
    p = 1
    while x_max ** p / math.factorial(p) >= tol:
        p += 1
    return p


def block_layout(n_lags):
    """(block length, FFT size) of correlate_blocks for an n_lags-delay
    window: the smallest power-of-two FFT that holds a block of at least
    n_lags / 4 samples followed by the window (block + n_lags - 1 = FFT
    size). 1179 and 4096 for the 2918-delay window at 250 ksps (4.7 ms
    blocks)."""
    m = 1 << int(np.ceil(np.log2(n_lags + -(-n_lags // 4) - 1)))
    return m - n_lags + 1, m


def chirp_z_plan(f_shifts, n_blocks, t_block, t_offset):
    """A chirp-z transform over blocks, evaluated at the evenly spaced row
    frequencies f_shifts: S[k] = sum_b y[b] exp(-2j pi f_k (b t_block +
    t_offset)), from FFTs (Bluestein: kb = (k^2 + b^2 - (k - b)^2) / 2 makes
    it a convolution). Returns (pre (blocks, numpy complex128), V (FFT
    length, GPU), post (rows, numpy complex128), FFT length):
    S = post * ifft(fft(y * pre, L) * V)[:rows]."""
    k_rows = len(f_shifts)
    f0 = f_shifts[0]
    df = (f_shifts[-1] - f_shifts[0]) / (k_rows - 1)
    if np.abs(f_shifts - (f0 + df * np.arange(k_rows))).max() > 1e-6 * abs(df):
        raise ValueError("correlate_blocks needs evenly spaced row frequencies")
    alpha = df * t_block                  # cycles per (row step x block step)
    lf = 1 << int(np.ceil(np.log2(n_blocks + k_rows - 1)))

    def chirp(nn):                        # exp(+j pi alpha n^2), phase mod 2 in float64
        nn = np.asarray(nn, dtype=np.float64)
        return np.exp(1j * np.pi * np.mod(alpha * nn * nn, 2.0))

    b = np.arange(n_blocks)
    pre = np.exp(-2j * np.pi * np.mod(f0 * b * t_block, 1.0)) * np.conj(chirp(b))
    v = np.zeros(lf, dtype=np.complex128)
    v[:k_rows] = chirp(np.arange(k_rows))
    v[lf - (n_blocks - 1):] = chirp(np.arange(n_blocks - 1, 0, -1))
    post = np.conj(chirp(np.arange(k_rows))) * np.exp(
        -2j * np.pi * np.mod(f_shifts * t_offset, 1.0))
    return pre, cupy.fft.fft(cupy.asarray(v, dtype=cupy.complex64)), post, lf


def _wrapped(x, start, length):
    """x[(start + i) % len(x)] for i < length (a view when nothing wraps)."""
    n = len(x)
    start %= n
    if start + length <= n:
        return x[start:start + length]
    return np.take(x, np.arange(start, start + length) % n)


def correlate_blocks(rx_samples, tx_compensated, fs, f_shifts, lag_start, lag_end, *,
                     n_terms=None, drift_hz_per_s=0.0, t_ref_s=0.0):
    """What correlate_rows computes -- |sum_s rx[(s + lag) % n] conj(tx[s])
    exp(-2j pi f s / fs)| for each f in f_shifts (rows, Hz, evenly spaced)
    and lag in lag_start..lag_end-1 (columns) -- from block sums, ~100x
    faster.

    The capture is cut into blocks of L samples (block_layout). For each
    block, one short FFT correlation gives the sum over the block of
    rx(t + lag) * conj(tx(t)) at every lag at once; the rows are then the
    blocks' sum with each row's Doppler phase at the block centres, a
    chirp-z transform over blocks for each lag. The phase change within a
    block is kept as a power series: block sums weighted by
    ((t - block centre) / L)^p for p < n_terms, combined per row with
    (-2j pi f L / fs)^p / p!. n_terms=None takes enough terms for a
    remainder below DD_SERIES_TOL (5 for |f| <= 13 Hz at L = 1179).
    n_terms=1 (the plain block sum) is wrong for Zadoff-Chu codes: at a
    wrong lag the product is a tone (45 Hz per chip of delay error for the
    25 kchip/s q=1351 code), and block sums fold tones near multiples of the
    block rate into the image (on the 2026-10-11 35 s look, 15 % of cells
    changed by more than 0.1 in log amplitude).

    drift_hz_per_s, t_ref_s: an extra phase exp(-1j pi a (t - t_ref)^2)
    applied at each block centre (a focused image, for
    investigations/focused_dd_prototype.py); its change within a block is
    left out, which is a phase error of at most pi |a| T L / (2 fs) over a
    capture of T s (4e-4 rad for 5e-4 Hz/s on a 125 s capture).

    GPU memory stays below ~0.6 GB: the capture goes to the GPU one batch of
    blocks at a time, and the window is split into parts whose block sums
    take at most DD_TABLE_BYTES. Measured 2026-10-24 on the RTX 3050 laptop
    GPU: 0.67 s against correlate_rows' 45 s on a 35 s capture (mains
    power); 3.9-4.7 s on battery power against 165-174 s on mains on a
    125 s capture (a version that kept the whole capture on the GPU, 2.0 GB,
    took 1.4-1.5 s on mains). The images agreed to float32 rounding.

    Returns (float32 magnitude (rows, lags), info dict: block length, FFT
    size, blocks, terms, the largest within-block phase in rad, the GPU
    pool's peak bytes).
    """
    rx_samples = np.asarray(rx_samples, dtype=np.complex64)
    tx_compensated = np.asarray(tx_compensated, dtype=np.complex64)
    n = len(rx_samples)
    w = lag_end - lag_start
    L, _ = block_layout(w)
    nb = -(-n // L)
    f_shifts = np.asarray(f_shifts, dtype=np.float64)
    k_rows = len(f_shifts)
    # Within a block the phase runs over +-x_max about the block centre.
    x_max = np.pi * np.abs(f_shifts).max() * L / fs
    if n_terms is None:
        n_terms = series_terms(x_max)
    pool = cupy.get_default_memory_pool()
    pool.free_all_blocks()

    t_block, t_centre = L / fs, (L - 1) / 2 / fs
    pre, V, post, lf = chirp_z_plan(f_shifts, nb, t_block, t_centre)
    if drift_hz_per_s:
        t_rel = np.arange(nb) * t_block + t_centre - t_ref_s
        pre = pre * np.exp(-1j * np.pi * drift_hz_per_s * t_rel ** 2)
    pre = cupy.asarray(pre, dtype=cupy.complex64)
    # Time within the block about its centre, in block lengths (|v| <= 1/2).
    v = ((cupy.arange(L) - (L - 1) / 2) / L).astype(cupy.float32)
    theta = -2j * np.pi * f_shifts * L / fs
    scales = [cupy.asarray(theta ** p / math.factorial(p) * post, dtype=cupy.complex64)
              for p in range(n_terms)]

    img = cupy.zeros((k_rows, w), dtype=cupy.complex64)
    part = max(DD_LAGS_PER_BATCH, DD_TABLE_BYTES // (8 * nb))
    for k0 in range(0, w, part):
        k1 = min(w, k0 + part)
        wp = k1 - k0
        m = 1 << int(np.ceil(np.log2(L + wp - 1)))
        P = cupy.empty((nb, wp), dtype=cupy.complex64)   # one term's block sums
        for p in range(n_terms):
            vp = v ** p if p else None
            for b0 in range(0, nb, DD_BLOCKS_PER_BATCH):
                b1 = min(nb, b0 + DD_BLOCKS_PER_BATCH)
                # Block b's rx segment starts at b L + this part's first lag.
                rx_seg = cupy.asarray(_wrapped(rx_samples, b0 * L + lag_start + k0,
                                               (b1 - b0 - 1) * L + m))
                seg = cupy.lib.stride_tricks.as_strided(
                    rx_seg, shape=(b1 - b0, m), strides=(L * 8, 8))
                X = cupy.fft.fft(cupy.ascontiguousarray(seg), axis=1)
                txs = tx_compensated[b0 * L:b1 * L]
                txb = cupy.zeros((b1 - b0) * L, dtype=cupy.complex64)
                txb[:len(txs)] = cupy.asarray(txs)
                txb = txb.reshape(b1 - b0, L)
                if vp is not None:
                    txb *= vp
                X *= cupy.conj(cupy.fft.fft(txb, n=m, axis=1))
                P[b0:b1] = cupy.fft.ifft(X, axis=1)[:, :wp]
                del rx_seg, seg, X, txb
            for l0 in range(0, wp, DD_LAGS_PER_BATCH):
                l1 = min(wp, l0 + DD_LAGS_PER_BATCH)
                Y = cupy.fft.fft(cupy.ascontiguousarray(P[:, l0:l1].T) * pre[None, :],
                                 n=lf, axis=1)
                Y *= V[None, :]
                S = cupy.fft.ifft(Y, axis=1)[:, :k_rows]
                img[:, k0 + l0:k0 + l1] += scales[p][:, None] * S.T
                del Y, S
        del P
    A = cupy.asnumpy(cupy.abs(img))
    info = {"block": L, "fft_size": block_layout(w)[1], "n_blocks": nb,
            "n_terms": n_terms, "x_max_rad": x_max, "gpu_pool_peak_bytes": pool.total_bytes()}
    # Return pool blocks to the driver so concurrent worker processes can
    # allocate (the per-process pool would otherwise hoard freed memory).
    del img, pre, V, v, scales
    pool.free_all_blocks()
    return A, info


def compute_dd_image(rx_samples, tx_samples, sample_rate, frequency,
                     rx_start_astrotime, tx_start_astrotime,
                     tx_start_offset, rx_start_offset, tx_name, rx_name,
                     freq_offset_hz=0.0, method="blocks"):
    """
    Compute the Delay-Doppler image and associated axes.
    Returns: log_A, dlt_shifts, delay_values_s, lt_min, dlt_rate_srp,
    tx_overlap_s

    method: "blocks" (correlate_blocks, the default) or "rows" (the per-row
    full-length correlation used until 2026-10-24, kept for checks); the
    two agree to float32 rounding.

    dlt_rate_srp is the SRP dlt rate whose linear Doppler chirp was
    compensated in the image; pass it to the equator/projection functions so
    their rate correction matches the compensation exactly. tx_overlap_s:
    seconds of the capture the TX model covers (the coherent integration
    time, which sets the Doppler resolution).

    freq_offset_hz: measured TX/RX chain frequency offset (e.g. the specular
    line centroid from freq_offset_hunt). It is added to the compensation so
    the image rows stay labeled by geometric dlt with the chain offset
    removed.
    """
    rx_start_time_s = et_from_astropy(rx_start_astrotime) + rx_start_offset
    tx_start_time_s = et_from_astropy(tx_start_astrotime) + tx_start_offset
    tx_compensated, info = compensated_tx(
        len(rx_samples), tx_samples, sample_rate, frequency,
        rx_start_time_s, tx_start_time_s, tx_name=tx_name, rx_name=rx_name,
        freq_offset_hz=freq_offset_hz)
    lt_rx_start = info["lt_rx_start"]
    dlt_rate_srp = info["dlt_rate_srp"]
    doppler_start_hz = info["doppler_start_hz"]

    cor_lags, cor_i_start, cor_i_end = delay_window(len(rx_samples), sample_rate)
    dlt_lo, dlt_hi = doppler_span(rx_start_time_s, tx_name, rx_name)
    dlt_shifts = np.linspace(dlt_lo, dlt_hi, 3000)
    f_shifts = -dlt_shifts * frequency.to(au.Hz).value - doppler_start_hz

    # The window's lags: ci_start may be negative (leading lags).
    ci_start = cor_i_start - len(rx_samples) // 2
    ci_end = cor_i_end - len(rx_samples) // 2
    fs = sample_rate.to_value(au.Hz)
    if method == "blocks":
        A, _ = correlate_blocks(rx_samples, tx_compensated, fs, f_shifts, ci_start, ci_end)
    elif method == "rows":
        A = correlate_rows(rx_samples, tx_compensated, fs, f_shifts, ci_start, ci_end)
    else:
        raise ValueError(f"unknown correlation method {method!r}")

    log_A = np.log(A)
    delay_values_s = (cor_lags[cor_i_start:cor_i_end]).to(au.s).value
    lt_min = lt_rx_start  # SRP light time at rx_start

    return (log_A, dlt_shifts, delay_values_s, lt_min, dlt_rate_srp,
            info["tx_overlap_s"])


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def load_observation(rx_filename, data_root):
    rx_sigmf = sigmf.sigmffile.fromfile(rx_filename, skip_checksum=True)
    rx_samples = rx_sigmf.read_samples().astype("complex64", copy=False)
    rx_info = rx_sigmf.get_global_info()
    rx_captures = rx_sigmf.get_captures()
    sample_rate = rx_info['core:sample_rate'] / au.s
    frequency = rx_captures[0]['core:frequency'] * au.Hz
    rx_start_astrotime = at.Time(rx_captures[0]['core:datetime'])
    tx_filename = rx_info['core:description'].split(';')[0]
    tx_sigmf = sigmf.sigmffile.fromfile(os.path.join(data_root, "tx_signals", tx_filename),
                                        skip_checksum=True)
    tx_info = tx_sigmf.get_global_info()
    if rx_info['core:sample_rate'] != tx_info['core:sample_rate']:
        raise ValueError(f"RX/TX sample-rate mismatch: {rx_filename}")
    tx_samples = tx_sigmf.read_samples().astype("complex64", copy=False)
    tx_start_astrotime = at.Time(tx_sigmf.get_captures()[0]['core:datetime'])
    return (rx_samples, tx_samples, sample_rate, frequency,
            rx_start_astrotime, tx_start_astrotime, tx_filename)


def lunar_projection(log_A, dlt_shifts, delay_values_s, lt_ref,
                     rx_time_s, nside, *, tx_name, rx_name, chip_samples,
                     doppler_res_dlt, rx_duration_s=None, dlt_rate_srp=None,
                     use_dem="figure", supersample=SUPERSAMPLE, chunk=2_000_000):
    """Project a DD image (log amplitude) onto a healpix map of the Moon.

    lt_ref: the light time the image's lag 0 refers to. The timing offset is
    measured on the specular echo, which comes from the terrain at the SRP,
    so process_file passes the smooth-sphere SRP light time minus the
    terrain's two-way delay there (srp_topo_delay_s); each point's own light
    time is computed on the terrain (use_dem), so both ends of the delay
    difference see the same surface.

    Each map pixel averages the image intensity over supersample**2 points
    spread across the pixel (the centres of its sub-pixels on the
    nside*supersample grid), and stores it back as log amplitude.
    supersample=1 reads only the image cell under the pixel centre -- the
    behaviour before 2026-10-20, which ignored the other cells inside the
    pixel: on the 125 s 2026-10-11 looks a 4.4 km (nside 400) pixel spans
    ~4 independent Doppler rows, and averaging them cuts the speckle like
    extra looks do (see SUPERSAMPLE).

    Returns (val_surface, valid_fraction, multiplicity, cell_area). A
    pixel is valid where at least one of its points falls inside the image;
    the average is over those points. multiplicity: see the note at the end.
    cell_area: ground area (km^2, both sides of the Doppler equator) whose
    echo lands in one image cell at the pixel -- what the cell's power is
    spread over; brightness per unit area is the pixel's intensity divided
    by it (stack_maps.py). It is counted, not derived: the map's points per
    image cell, spread over the image's power response -- along delay the
    code's correlation (a triangle one chip, chip_samples, either side, so
    its square in power), along Doppler the coherent integration (the main
    lobe of sinc^2, first null doppler_res_dlt = 1 / (integration time x
    frequency) either side) -- times the area of a point. Like the pixel's intensity it is the plain mean over the pixel's
    points: the pixel holds the mean of its cells' intensities, each about
    the brightness times its cell's area, so dividing by the mean area
    recovers the brightness. Away from the Doppler equator this is the
    geometric cell area, which grows as 1/distance from it as delay and
    Doppler contours turn parallel; within ~sqrt(2 R x range resolution)
    (~140 km at 25 kchip/s) the delay resolution caps it, which a purely
    geometric area misses (it over-darkens a band along the equator).
    """
    s = int(supersample)
    npix = hp.nside2npix(nside)
    fine_nside = nside * s
    n_fine = hp.nside2npix(fine_nside)
    n_rows, n_cols = log_A.shape
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    ddelay = delay_values_s[1] - delay_values_s[0]
    # Intensity relative to the image peak, float64 (no under/overflow).
    ref = float(np.max(log_A))
    intensity = np.exp(2.0 * (np.asarray(log_A, dtype=np.float64) - ref)).ravel()

    # Station positions are per epoch, not per point: solve them once.
    sta_start = apparent_station_positions(rx_time_s, tx_name, rx_name)
    if rx_duration_s is not None:
        if dlt_rate_srp is None:
            dlt_rate_srp = srp_dlt_rate_bck(rx_time_s, rx_duration_s, tx_name, rx_name)
        sta_end = apparent_station_positions(rx_time_s + rx_duration_s, tx_name, rx_name)

    def lt_field(p, sta):
        R_rx, R_tx, c = sta
        return (np.linalg.norm(p - R_rx, axis=1) +
                np.linalg.norm(p - R_tx, axis=1)) / c

    # Pass 1, in chunks of sub-pixels (the full nside 1600 grid would need
    # several GB): image cell of each sub-pixel centre, summed into its
    # parent pixel. The centre of a sub-pixel lies inside its parent, so
    # vec2pix at the coarse nside finds the parent (checked below: every
    # parent gets exactly s*s sub-pixels).
    cell = np.empty(n_fine, dtype=np.int32)        # -1: outside the image
    sum_i = np.zeros(npix)
    n_in = np.zeros(npix, dtype=np.int64)
    n_sub = np.zeros(npix, dtype=np.int64)
    for i0 in range(0, n_fine, chunk):
        v = np.array(hp.pix2vec(fine_nside, np.arange(i0, min(i0 + chunk, n_fine)))).T
        parent = hp.vec2pix(nside, v[:, 0], v[:, 1], v[:, 2])
        n_sub += np.bincount(parent, minlength=npix)
        # With a LOLA DEM loaded the points carry real topography, so the lt
        # field places each point at its true delay (and, through the
        # window-averaged differences, its true Doppler): the ~7 delay px
        # (~4 km) ellipsoid mapping systematic drops out. The SRP solve
        # behind apparent_station_positions stays on the smooth ellipsoid
        # anchor.
        p = moon_surface_points(v, use_dem=use_dem)
        lt = lt_field(p, sta_start)
        if rx_duration_s is not None:
            # Place each point at its window-averaged Doppler, where the DD
            # image correlation energy actually lands (see rate_corrected_dlt).
            dlt = ((lt_field(p, sta_end) - lt) / rx_duration_s
                   - dlt_rate_srp * rx_duration_s / 2)
        else:
            _, dlt = moonPointDLT_BCK(rx_time_s, v, tx_name, rx_name,
                                      use_dem=use_dem)
        del v, p
        doppler_index = np.rint((dlt - dlt_shifts[0]) / ddlt).astype(np.int64)
        # Column 0 of the image is delay_values_s[0] (slightly negative since
        # the window leads the SRP delay), not the SRP delay itself.
        delay_index = np.rint((lt - lt_ref - delay_values_s[0]) / ddelay).astype(np.int64)
        valid = ((doppler_index >= 0) & (doppler_index < n_rows) &
                 (delay_index >= 0) & (delay_index < n_cols))
        c = np.where(valid, doppler_index * n_cols + delay_index, -1)
        cell[i0:i0 + len(c)] = c
        sum_i += np.bincount(parent[valid], weights=intensity[c[valid]], minlength=npix)
        n_in += np.bincount(parent[valid], minlength=npix)
    if not np.all(n_sub == s * s):
        raise RuntimeError(f"sub-pixel parents: expected {s * s} per pixel, got "
                           f"{n_sub.min()}..{n_sub.max()}")
    ok = n_in > 0
    val_surface = np.full(npix, hp.UNSEEN, dtype=np.float32)
    val_surface[ok] = 0.5 * np.log(sum_i[ok] / n_in[ok]) + ref

    # Mapping degeneracy per pixel: how many pixels' worth of surface share
    # the pixel's DD cells (mean over its points). Near the Doppler equator
    # (and the SRP) the projection collapses and one bright cell smears
    # along a long surface arc -- the bright stripe artifact. High
    # multiplicity marks exactly those pixels, independent of the data, so
    # downstream analysis masks them by construction (MAX_MULTIPLICITY). It
    # counts in pixels of this map's nside (a cell's footprint over s*s
    # sub-pixels per pixel), so with supersample=1 it is the number of pixel
    # centres in the cell, as before; with s > 1, cells smaller than a pixel
    # give values below 1.
    cell_count = np.bincount(cell[cell >= 0], minlength=n_rows * n_cols)
    # Points per cell spread over the image's resolution (see cell_area).
    spread = cell_count.reshape(n_rows, n_cols).astype(np.float64)
    spread = convolve_response(spread, triangle_power(chip_samples), axis=1)
    spread = convolve_response(spread, sinc_power(doppler_res_dlt / ddlt), axis=0).ravel()
    sum_mult = np.zeros(npix)
    sum_spread = np.zeros(npix)
    for i0 in range(0, n_fine, chunk):
        c = cell[i0:i0 + chunk]
        v = np.array(hp.pix2vec(fine_nside, np.arange(i0, i0 + len(c)))).T
        parent = hp.vec2pix(nside, v[:, 0], v[:, 1], v[:, 2])
        inside = c >= 0
        sum_mult += np.bincount(parent[inside], weights=cell_count[c[inside]],
                                minlength=npix)
        sum_spread += np.bincount(parent[inside], weights=spread[c[inside]],
                                  minlength=npix)
    multiplicity = np.zeros(npix, dtype=np.float32)
    multiplicity[ok] = sum_mult[ok] / n_in[ok] / (s * s)
    point_area = 4 * np.pi * moon_radii()[0] ** 2 / n_fine     # km^2
    cell_area = np.full(npix, np.nan, dtype=np.float32)
    cell_area[ok] = point_area * sum_spread[ok] / n_in[ok]
    return val_surface, ok.mean(), multiplicity, cell_area


def _triangle_peak1(half_width):
    k = np.arange(-int(np.ceil(half_width)) + 1, int(np.ceil(half_width)))
    return np.clip(1.0 - np.abs(k) / half_width, 0.0, None) ** 2


def _sinc_lobe_peak1(first_null):
    k = np.arange(-int(np.ceil(first_null)) + 1, int(np.ceil(first_null)))
    return np.sinc(k / first_null) ** 2


def triangle_power(half_width):
    """Power response of a code correlation along delay: the amplitude is a
    triangle reaching zero half_width (one chip) samples either side, so the
    power is its square. Unit sum."""
    w = _triangle_peak1(half_width)
    return w / w.sum()


def sinc_power(first_null):
    """Power response of a coherent integration along Doppler: sinc^2 with
    its first null first_null rows either side (1/T in frequency), the main
    lobe only (it holds 90 % of the power; the rest spreads over many rows).
    Unit sum."""
    w = _sinc_lobe_peak1(first_null)
    return w / w.sum()


def resolution_cells(chip_samples, doppler_res_rows):
    """Image cells in one resolution cell: the delay and Doppler power
    responses of triangle_power and sinc_power summed with their peaks at 1
    instead of normalized, about 2/3 of a chip (chip_samples columns) times
    0.9 of 1/T (doppler_res_rows rows). lunar_projection's cell_area times
    this is the ground area inside one resolution cell at the pixel, which
    does not depend on the image's row and column spacing (cell_area alone
    does)."""
    return float(_triangle_peak1(chip_samples).sum() * _sinc_lobe_peak1(doppler_res_rows).sum())


def convolve_response(a, response, axis):
    """a convolved along axis with a unit-sum response (zero beyond the array
    edges, where no surface lies). A response of one sample leaves a
    unchanged."""
    if len(response) <= 1:
        return a
    return scipy.ndimage.convolve1d(a, response, axis=axis, mode="constant")


def chip_samples_of(tx_file, sample_rate_hz):
    """Samples per chip of the transmitted code, from the CAMRAS TX file name
    (...-fzc<chips per s>-...). Every production waveform carries it; a name
    without it is an error rather than a guess."""
    m = re.search(r"fzc(\d+)", tx_file)
    if not m:
        raise ValueError(f"no chip rate (fzc<chips/s>) in TX file name {tx_file!r}")
    return sample_rate_hz / float(m.group(1))


def noise_floor_log_amp(log_A):
    """Noise floor of a DD image in its own units (log amplitude): the mean
    intensity over the first NOISE_COLUMNS delay columns, which lead the
    echo's leading edge by 60-80 us at 250 ksps -- more than a chip at 25 or
    50 kchip/s (process_file refuses codes with longer chips,
    max_chip_samples_for_noise). It holds only noise if the measured timing
    offset put the echo's leading edge at lag 0. On the 2026-10-11 looks
    this matches the level in Doppler rows beyond the limb, where no surface
    is, to ~10 %."""
    lead = np.asarray(log_A[:, :NOISE_COLUMNS], dtype=np.float64)
    return 0.5 * float(np.log(np.mean(np.exp(2.0 * lead))))


def save_lunar_image(val_surface, out_png, title, vmin=None, vmax=None):
    pl.close('all')
    fig = pl.figure(figsize=(12, 12))
    # xsize sets orthview's projection grid; the default (800) would cap the
    # rendered resolution well below high-nside healpix maps.
    hp.orthview(val_surface, title=title, flip='geo', fig=fig, half_sky=True,
                min=vmin, max=vmax, xsize=2400)
    hp.graticule()
    pl.savefig(out_png, dpi=150)
    pl.close(fig)


def process_file(rx_filename, data_root, out_dir, nside, *, tx_name, rx_name,
                 tx_extra_offset_s, freq_offset_hz, rim_delta_hz=None,
                 edge_offsets=None, save_pngs=False, use_dem="figure",
                 supersample=SUPERSAMPLE):
    """Image, calibrate and map one capture; returns its runs-CSV row.

    tx_extra_offset_s / freq_offset_hz: the measured timing and chain
    frequency offsets (freq_offset_hunt.measure_offset). rim_delta_hz and
    edge_offsets: None to calibrate from this look's own echo edges (opposite-sense
    looks: calibrate_rim, then fit_edge_offsets from there); values to apply as
    given (same-sense looks, from the opposite-sense twin; edge_offsets is the twin's
    applied (delay_s, doppler_hz)). use_dem names the terrain model
    (doppler_equator.terrain_model): "figure", the default, maps on the
    Moon's smooth overall shape (MOON_FIGURE_KM); "sphere" (False) on the
    1737.4 km sphere; "lola" (True) on LOLA's terrain, for comparisons only
    (it imprints LOLA's craters through the cell area), and then a missing
    terrain grid is an error, never a silent fallback.
    """
    print(f"Processing {rx_filename}")
    terrain = terrain_model(use_dem)
    dem_file = os.path.basename(load_lola_dem()) if terrain == "lola" else ""
    (rx_samples, tx_samples, sample_rate, frequency,
     rx_start, tx_start, tx_filename) = load_observation(rx_filename, data_root)
    fs = sample_rate.to_value(au.Hz)
    f_hz = frequency.to_value(au.Hz)
    capture_s, tx_s = len(rx_samples) / fs, len(tx_samples) / fs
    if capture_s < MIN_CAPTURE_S:
        raise ValueError(f"capture {capture_s:.1f} s is shorter than {MIN_CAPTURE_S:g} s")
    chip_samples = chip_samples_of(tx_filename, fs)
    if chip_samples > max_chip_samples_for_noise():
        raise ValueError(
            f"{tx_filename}: {chip_samples:g} samples per chip; the noise columns "
            f"(lags -{LEAD_SAMPLES}..-{LEAD_SAMPLES - NOISE_COLUMNS + 1}) would "
            "hold echo. Widen LEAD_SAMPLES to process this code.")

    tx_emit_start = nominal_tx_emit_start(rx_start)
    (log_A, dlt_shifts, delay_values_s, lt_min_image, dlt_rate_srp,
     tx_overlap_s) = compute_dd_image(
        rx_samples, tx_samples, sample_rate, frequency,
        rx_start, tx_emit_start, tx_extra_offset_s, 0.0, tx_name, rx_name,
        freq_offset_hz=freq_offset_hz)
    del rx_samples, tx_samples

    rx_time_s = et_from_astropy(rx_start)
    # The timing offset was measured on the specular echo, which comes from
    # the terrain at the SRP: lag 0 of the image is that terrain's light
    # time, srp_topo_delay_s (2h/c) ahead of the smooth-sphere SRP's
    # lt_min_image. Everything placed in the image -- the rim curves and
    # the map's points -- is measured from there. (Without terrain the model
    # has no such delay; the A/B then shows the full terrain effect.)
    srp_elevation_km = srp_topo_delay_s = np.nan
    lt_ref = lt_min_image
    if terrain != "sphere":
        srp_elevation_km, srp_topo_delay_s = extract_srp_elevation(
            rx_time_s, tx_name, rx_name, use_dem=terrain)
        lt_ref = lt_min_image - srp_topo_delay_s

    rx_duration_s = capture_s
    lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = compute_doppler_equator_velocity(
        rx_time_s, n_points=500, rx_duration_s=rx_duration_s,
        dlt_rate_srp=dlt_rate_srp, tx_name=tx_name, rx_name=rx_name)

    # Rim self-calibration (calibrate_rim). For same-sense looks the rim is
    # too diffuse to measure -- rim_delta_hz comes from the opposite-sense twin.
    rim_first = rim_final = seed = None
    rim_passes = 0
    if rim_delta_hz is None:
        cal = calibrate_rim(log_A, dlt_shifts, delay_values_s, lt_ref, lt_min_eq,
                            delay_up, dlt_up, delay_down, dlt_down,
                            delta_capture_dlt=RIM_CAPTURE_HZ / f_hz)
        delta_dlt, rim_status = cal["delta_dlt"], cal["status"]
        rim_first, rim_final, seed, rim_passes = (cal["rim_first"], cal["rim_final"],
                                                  cal["seed"], cal["passes"])
    else:
        rim_status = "inherited"
        delta_dlt = -rim_delta_hz / f_hz
    dlt_shifts_cal = dlt_shifts - delta_dlt
    noise_log_amp = noise_floor_log_amp(log_A)

    # Delay and Doppler offsets from the echo edges (fit_edge_offsets): the
    # timing search places lag 0 only to within a few microseconds, which
    # moves both edges, and the rim calibration measures Doppler alone. Not
    # fitted (or failed): nothing is applied, and stack_maps.py rejects the
    # look.
    if edge_offsets is None:
        edge = {"status": "not_run", "delay_s": np.nan, "doppler_hz": np.nan,
                "delay_err_s": np.nan, "doppler_err_hz": np.nan, "n_columns": 0}
        if rim_status in ("converged", "not_converged"):
            edge = fit_edge_offsets(
                log_A, dlt_shifts_cal, delay_values_s, lt_ref, rx_time_s,
                rx_duration_s=rx_duration_s, dlt_rate_srp=dlt_rate_srp,
                chip_samples=chip_samples, tx_overlap_s=tx_overlap_s, f_hz=f_hz,
                noise_log_amp=noise_log_amp, tx_name=tx_name, rx_name=rx_name,
                use_dem=use_dem)
        applied = ((edge["delay_s"], edge["doppler_hz"]) if edge["status"] == "fitted"
                   else (0.0, 0.0))
    else:
        edge = {"status": "inherited", "delay_s": edge_offsets[0],
                "doppler_hz": edge_offsets[1], "delay_err_s": np.nan,
                "doppler_err_hz": np.nan, "n_columns": 0}
        applied = edge_offsets
    lt_proj = lt_ref - applied[0]
    dlt_proj = dlt_shifts_cal + applied[1] / f_hz

    edge_img = compute_edge_image(log_A)
    score = alignment_score(edge_img, dlt_proj, delay_values_s,
                            lt_proj, lt_min_eq, delay_up, dlt_up)
    score += alignment_score(edge_img, dlt_proj, delay_values_s,
                             lt_proj, lt_min_eq, delay_down, dlt_down)

    val_surface, valid_fraction, multiplicity, cell_area = lunar_projection(
        log_A, dlt_proj, delay_values_s, lt_proj,
        rx_time_s, nside, tx_name=tx_name, rx_name=rx_name,
        chip_samples=chip_samples, doppler_res_dlt=1.0 / (tx_overlap_s * f_hz),
        rx_duration_s=rx_duration_s, dlt_rate_srp=dlt_rate_srp,
        use_dem=use_dem, supersample=supersample)
    if not (val_surface != hp.UNSEEN).any():
        raise ValueError("the projection put no map pixel inside the image")

    base = os.path.basename(rx_filename)
    os.makedirs(out_dir, exist_ok=True)
    log_png = os.path.join(out_dir, f"{base}_log_A.png")
    dd_png = os.path.join(out_dir, f"{base}_lunar.png")
    map_npy = os.path.join(out_dir, f"{base}_map.npy")
    mult_npy = os.path.join(out_dir, f"{base}_mapcount.npy")
    area_npy = os.path.join(out_dir, f"{base}_cellarea.npy")
    dd_npz = os.path.join(out_dir, f"{base}_dd.npz")
    if save_pngs:
        # Colour range: from the image's noise floor to its peak.
        pl.imsave(log_png, log_A.T, vmin=noise_log_amp, vmax=float(log_A.max()))
        valid = val_surface != hp.UNSEEN
        save_lunar_image(val_surface, dd_png, base, vmin=noise_log_amp,
                         vmax=float(val_surface[valid].max()))
        # The DD image itself, for replotting (plot_dd.py) and re-projection
        # (reproject_maps.py): rows are rim-calibrated Doppler relative to
        # the SRP, columns round-trip delay since the nominal emission start
        # (including the measured timing offset) -- both before the edge
        # fit's offsets, which are in the row (edge_*) and which
        # reproject_maps.py fits again from here. float16 keeps log
        # amplitude to ~0.004.
        _, dlt_srp = moonSRP_DLT_BCK(rx_time_s, tx_name, rx_name, dt=rx_duration_s / 2)
        np.savez(dd_npz, log_A=log_A.astype(np.float16),
                 doppler_hz=-(dlt_shifts_cal - dlt_srp) * f_hz,
                 delay_s=lt_min_image + tx_extra_offset_s + delay_values_s)
    else:
        log_png = dd_png = dd_npz = ""
    np.save(map_npy, val_surface)  # healpix map (hp.UNSEEN where invalid)
    np.save(mult_npy, multiplicity)  # DD-cell multiplicity (degeneracy mask)
    np.save(area_npy, cell_area)  # ground area per DD cell, km^2 (both sides)

    # Diagnostic only: the TX file timestamp is a generation time, not the
    # emission epoch used above.
    tx_file_minus_rx = (tx_start - rx_start).to_value('s')
    # Captures synthesized by make_circular.py carry its per-capture checks.
    with open(rx_filename, encoding="utf-8") as f:
        rx_global = json.load(f)["global"]

    def hz(dlt):
        return -dlt * f_hz

    return {
        "rx_file": base,
        "tx_file": tx_filename,
        "rx_start_utc": rx_start.utc.value,
        "tx_emit_start_utc": tx_emit_start.utc.value,
        "tx_file_datetime_utc": tx_start.utc.value,
        "tx_file_minus_rx_s": tx_file_minus_rx,
        "sample_rate_hz": fs,
        "frequency_hz": f_hz,
        "tx_extra_offset_s": tx_extra_offset_s,
        "freq_offset_hz": freq_offset_hz,
        # Rim calibration. rim_status: converged / not_converged (the
        # applied offset did not re-measure within half a row) / failed (no
        # pass got through the gates; nothing applied) / inherited (given,
        # from the opposite-sense twin). rim_delta_hz is blank when failed.
        "rim_status": rim_status,
        "rim_delta_hz": "" if rim_status == "failed" else hz(delta_dlt),
        "rim_spread_hz": hz(rim_final["spread_dlt"]) if rim_final else "",
        "rim_residual_hz": hz(rim_final["delta_dlt"]) if rim_final else "",
        "rim_n": (rim_first["n_up"] + rim_first["n_down"]) if rim_first else 0,
        "rim_n_final": (rim_final["n_up"] + rim_final["n_down"]) if rim_final else 0,
        "rim_spread_first_hz": hz(rim_first["spread_dlt"]) if rim_first else "",
        "rim_passes": rim_passes,
        # The coarse sweep's seed (blank if not run or nothing passed) and
        # how many more gate-passing columns it had than the runner-up.
        "rim_seed_hz": hz(seed["seed_dlt"]) if seed else "",
        "rim_seed_margin": (seed["n"] - seed["runner_up_n"]) if seed else "",
        # Edge fit (fit_edge_offsets): fitted / at_limit / failed / not_run
        # (rim failed) / inherited (from the opposite-sense twin). The offsets are
        # applied only when fitted or inherited; errors are jackknife.
        **edge_row(edge),
        "alignment_score": score,
        "valid_lunar_fraction": valid_fraction,
        "use_dem": terrain,
        "dem_file": dem_file,
        "spice_kernels": kernel_fingerprint(),
        "srp_elevation_km": srp_elevation_km,
        "srp_topo_delay_us": srp_topo_delay_s * 1e6,
        "log_png": log_png,
        "lunar_png": dd_png,
        "map_npy": map_npy,
        "mult_npy": mult_npy,
        "dd_npz": dd_npz,
        # Points averaged per map pixel along each axis (lunar_projection).
        "supersample": supersample,
        # Radiometry for stack_maps.py: the image's noise floor (map units,
        # log amplitude) and the per-pixel ground area of a cell.
        "noise_log_amp": noise_log_amp,
        "area_npy": area_npy,
        # The capture as recorded (path under the data root), its length,
        # the TX waveform's length and how much of the capture it covers.
        "rx_meta_path": os.path.relpath(rx_filename, data_root),
        "capture_s": capture_s,
        "tx_s": tx_s,
        "tx_overlap_s": tx_overlap_s,
        # make_circular.py's feed combination (blank for native captures):
        # tone phase coherence between the feeds, and the tone's and the
        # noise's feed amplitude ratios (equal for a circular echo and
        # equal feed gains).
        "circ_tone_coherence": rx_global.get("circ:tone_coherence", ""),
        "circ_amp_ratio": rx_global.get("circ:amp_ratio", ""),
        "circ_noise_amp_ratio": rx_global.get("circ:noise_amp_ratio", ""),
    }
