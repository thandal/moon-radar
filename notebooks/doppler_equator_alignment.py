"""
Doppler Equator Alignment & Time Offset Search

Computes the DD image once with nominal offsets, then grid-searches over
TX_START_OFFSET / RX_START_OFFSET perturbations by recomputing only
the SPICE-predicted Doppler equator and scoring its alignment with
the DD image edge features.
"""

import argparse
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
from spice_setup import furnsh_kernels
furnsh_kernels()

from doppler_equator import (
    AB_COR, EARTH_FRAME, DLT_DT,
    et_from_astropy, moon_radii, moon_surface_points,
    moonPointLightTime_BCK, moonPointLightTime_FWD,
    moonPointDLT_BCK, moonPointDLT_FWD,
    subpoint_average_guess, specular_point_bck, specular_point_fwd,
    moonSRP_DLT_BCK, moonSRP_DLT_FWD,
    srp_dlt_rate_bck, rate_corrected_dlt, apparent_station_positions,
    compute_doppler_equator, compute_doppler_equator_velocity,
)

MOON_RADIUS = 1_737_400.0 * au.m


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


def measure_rim_offset(log_A, dlt_shifts, delay_values_s, lt_min_image,
                       lt_min_eq, delay_up, dlt_up, delay_down, dlt_down,
                       delay_min_s=0.0015, n_cols_avg=3,
                       inner_off=(10, 50), outer_off=(30, 90),
                       min_contrast=0.4, min_samples=30, col_parity=None):
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
    counts -- or None if the rim is too weak to measure (e.g. cross-pol).
    """
    ddlt = dlt_shifts[1] - dlt_shifts[0]
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


# ---------------------------------------------------------------------------
# DD image computation (from notebook)
# ---------------------------------------------------------------------------
def compute_dd_image(rx_samples, tx_samples, sample_rate, frequency,
                     rx_start_astrotime, tx_start_astrotime,
                     tx_start_offset=0.0, rx_start_offset=0.0,
                     tx_name="DWINGELOO", rx_name="STOCKERT",
                     freq_offset_hz=0.0):
    """
    Compute the Delay-Doppler image and associated axes.
    Returns: log_A, dlt_shifts, delay_values_s, lt_min, dlt_rate_srp

    dlt_rate_srp is the SRP dlt rate whose linear Doppler chirp was
    compensated in the image; pass it to the equator/projection functions so
    their rate correction matches the compensation exactly.

    freq_offset_hz: measured TX/RX chain frequency offset (e.g. the specular
    line centroid from freq_offset_hunt). It is added to the compensation so
    the image rows stay labeled by geometric dlt with the chain offset
    removed.
    """
    rx_duration = len(rx_samples) / sample_rate
    tx_duration = len(tx_samples) / sample_rate

    rx_start_time_s = et_from_astropy(rx_start_astrotime) + rx_start_offset
    rx_end_time_s = rx_start_time_s + rx_duration.to(au.s).value
    tx_start_time_s = et_from_astropy(tx_start_astrotime) + tx_start_offset
    tx_end_time_s = tx_start_time_s + tx_duration.to(au.s).value

    # FWD light times for TX resampling
    lt_tx_start, _ = moonSRP_DLT_FWD(tx_start_time_s, tx_name, rx_name)
    lt_tx_end, _ = moonSRP_DLT_FWD(tx_end_time_s, tx_name, rx_name)

    # Resample TX
    # adjusted_tx_times0 maps TX samples to the RX-relative timeline (rx_sample_times0 starts at 0)
    # TX arrives at absolute time (tx_start_time_s + lt), RX-relative = absolute - rx_start_time_s
    tx_rx_offset = tx_start_time_s - rx_start_time_s
    rx_sample_times0 = np.arange(len(rx_samples)) / sample_rate.to(au.Hz).value
    adjusted_tx_times0 = np.linspace(
        tx_rx_offset + lt_tx_start,
        tx_rx_offset + tx_duration.to(au.s).value + lt_tx_end,
        len(tx_samples), endpoint=False)
    tx_phase_interp = scipy.interpolate.interp1d(
        adjusted_tx_times0, np.unwrap(np.angle(tx_samples)),
        kind='linear', fill_value=np.nan, bounds_error=False)
    tx_resampled = np.exp(1j * tx_phase_interp(rx_sample_times0))
    np.nan_to_num(tx_resampled, copy=False)

    # BCK Doppler compensation. Wide (T/2) derivative stencils keep the
    # light-time granularity noise (~2e-11 s) out of the compensated rate
    # (see the NOTE above srp_dlt_rate_bck).
    rx_duration_s = rx_duration.to(au.s).value
    lt_rx_start, dlt_rx_start = moonSRP_DLT_BCK(rx_start_time_s, tx_name, rx_name,
                                                dt=rx_duration_s / 2)
    lt_rx_end, dlt_rx_end = moonSRP_DLT_BCK(rx_end_time_s, tx_name, rx_name,
                                            dt=rx_duration_s / 2)
    doppler_start = -dlt_rx_start * frequency
    doppler_end = -dlt_rx_end * frequency
    doppler_rate = (doppler_end - doppler_start) / rx_duration
    dlt_rate_srp = (dlt_rx_end - dlt_rx_start) / rx_duration_s

    t_s = np.arange(len(rx_samples)) / sample_rate
    phi_Hz = -(doppler_start + doppler_rate * t_s / 2 + freq_offset_hz * au.Hz)
    tx_compensated = tx_resampled * np.exp(-1j * 2 * np.pi * (phi_Hz * t_s).value).T

    # Delay window. Start a few samples negative so the echo's leading edge
    # (which sits exactly at lag 0 once timing offsets are corrected) is not
    # clipped at the window boundary.
    LEAD_SAMPLES = 20
    cor_lags = scipy.signal.correlation_lags(len(rx_samples), len(tx_compensated), mode="same") / sample_rate
    t_end = MOON_RADIUS / ak.c * 2
    cor_i_start = np.argwhere(cor_lags >= -LEAD_SAMPLES / sample_rate)[0][0]
    cor_i_end = np.argwhere(cor_lags >= t_end)[0][0]

    # Terminator for Doppler range
    NTERMINATOR = 1000
    _, _, v_term = csp.edterm("UMBRAL", tx_name, "MOON", rx_start_time_s,
                              "MOON_ME", AB_COR, rx_name, NTERMINATOR)
    _, dlt_term = moonPointDLT_BCK(rx_start_time_s, v_term, tx_name, rx_name)
    dlt_shifts = np.linspace(dlt_term.min(), dlt_term.max(), 3000)
    f_shifts = -dlt_shifts * frequency.to(au.Hz).value - doppler_start.to(au.Hz).value

    # CUDA correlation, batched over Doppler rows: chunked 2-D FFTs amortize
    # kernel-launch and plan overhead, and float32 phase ramps replace the
    # complex128 exp of the old per-row loop (phase < 1e4 rad, so float32
    # keeps phase error < 1e-3 rad).
    tx_gpu = cupy.asarray(tx_compensated, dtype=cupy.complex64)
    rx_gpu = cupy.asarray(rx_samples, dtype=cupy.complex64)
    # ci_start may be negative (leading lags); circular correlation puts lag
    # -k at index n-k, so gather the window with wrapped indices.
    ci_start = cor_i_start - len(rx_samples) // 2
    ci_end = cor_i_end - len(rx_samples) // 2
    lag_idx = cupy.asarray(np.arange(ci_start, ci_end) % len(rx_samples))
    A = cupy.zeros((len(f_shifts), ci_end - ci_start), dtype=cupy.float32)
    fft_rx = cupy.fft.fft(rx_gpu)
    t_norm = (cupy.arange(len(tx_compensated), dtype=cupy.float32)
              * cupy.float32(2 * np.pi / sample_rate.value))
    CHUNK = 4  # 4 rows x ~4 buffers x ~70 MB ~ 1.2 GB peak: leaves room for
               # several worker processes to share the GPU.
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

    log_A = np.log(A)
    delay_values_s = (cor_lags[cor_i_start:cor_i_end]).to(au.s).value
    lt_min = lt_rx_start  # SRP light time at rx_start

    return log_A, dlt_shifts, delay_values_s, lt_min, dlt_rate_srp


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def load_observation(rx_filename, data_root):
    rx_sigmf = sigmf.sigmffile.fromfile(rx_filename, skip_checksum=True)
    rx_samples = rx_sigmf.read_samples().astype("complex64")
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
    tx_samples = tx_sigmf.read_samples().astype("complex64")
    tx_start_astrotime = at.Time(tx_sigmf.get_captures()[0]['core:datetime'])
    return (rx_samples, tx_samples, sample_rate, frequency,
            rx_start_astrotime, tx_start_astrotime, tx_filename)


def candidate_rx_files(data_root, date, limit=None):
    data_dir = os.path.join(data_root, date)
    files = []
    for name in sorted(os.listdir(data_dir)):
        if not name.startswith("stockert") or not name.endswith(".sigmf-meta"):
            continue
        if "_1970_" in name:
            continue
        path = os.path.join(data_dir, name)
        try:
            info = sigmf.sigmffile.fromfile(path, skip_checksum=True).get_global_info()
        except Exception as exc:
            print(f"Skipping unreadable metadata {path}: {exc}")
            continue
        desc = info.get("core:description", "")
        if "zadoff-chu" not in desc or "cw-" in desc or "pulsed" in desc:
            continue
        if date == "2025-09-16" and "30sec" not in desc:
            continue
        files.append(path)
    if limit:
        if len(files) <= limit:
            return files
        idx = np.linspace(0, len(files) - 1, limit).round().astype(int)
        return [files[i] for i in idx]
    return files



def lunar_projection(log_A, dlt_shifts, delay_values_s, lt_min_image,
                     rx_time_s, nside, rx_duration_s=None, dlt_rate_srp=None,
                     tx_name="DWINGELOO", rx_name="STOCKERT"):
    npix = hp.nside2npix(nside)
    v = np.array(hp.pix2vec(nside, np.arange(npix))).T
    p = moon_surface_points(v)

    def lt_field(t):
        R_rx, R_tx, c = apparent_station_positions(t, tx_name, rx_name)
        return (np.linalg.norm(p - R_rx, axis=1) +
                np.linalg.norm(p - R_tx, axis=1)) / c

    lt = lt_field(rx_time_s)
    if rx_duration_s is not None:
        # Place each point at its window-averaged Doppler, where the DD image
        # correlation energy actually lands (see rate_corrected_dlt).
        if dlt_rate_srp is None:
            dlt_rate_srp = srp_dlt_rate_bck(rx_time_s, rx_duration_s, tx_name, rx_name)
        lt_end = lt_field(rx_time_s + rx_duration_s)
        dlt = (lt_end - lt) / rx_duration_s - dlt_rate_srp * rx_duration_s / 2
    else:
        _, dlt = moonPointDLT_BCK(rx_time_s, v, tx_name, rx_name)
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    ddelay = delay_values_s[1] - delay_values_s[0]
    doppler_index = np.rint((dlt - dlt_shifts[0]) / ddlt).astype(int)
    # Column 0 of the image is delay_values_s[0] (slightly negative since the
    # window leads the SRP delay), not the SRP delay itself.
    delay_index = np.rint((lt - lt_min_image - delay_values_s[0]) / ddelay).astype(int)
    valid = ((doppler_index >= 0) & (doppler_index < log_A.shape[0]) &
             (delay_index >= 0) & (delay_index < log_A.shape[1]))
    val_surface = np.full(npix, hp.UNSEEN, dtype=np.float32)
    val_surface[valid] = log_A[doppler_index[valid], delay_index[valid]]
    # Mapping degeneracy per pixel: the number of surface pixels that share
    # the pixel's DD cell. Near the Doppler equator (and the SRP) the
    # projection collapses and one bright cell smears along a long surface
    # arc -- the bright stripe artifact. High multiplicity marks exactly
    # those pixels, independent of the data, so downstream analysis can mask
    # them by construction.
    multiplicity = np.zeros(npix, dtype=np.float32)
    cell = doppler_index[valid] * log_A.shape[1] + delay_index[valid]
    counts = np.bincount(cell)
    multiplicity[valid] = counts[cell]
    return val_surface, valid.mean(), multiplicity


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


def process_file(rx_filename, data_root, out_dir, nside=100,
                 tx_name="DWINGELOO", rx_name="STOCKERT",
                 tx_extra_offset_s=0.0, freq_offset_hz=0.0, save_pngs=True,
                 rim_delta_hz=None):
    print(f"Processing {rx_filename}")
    (rx_samples, tx_samples, sample_rate, frequency,
     rx_start, tx_start, tx_filename) = load_observation(rx_filename, data_root)
    rx_duration = len(rx_samples) / sample_rate
    if rx_duration < 20 * au.s:
        raise ValueError(f"RX duration too short: {rx_duration}")

    # The TX file's core:datetime is the waveform *generation* time, not the
    # emission epoch (it is off by ~25 h on the 2025-06-21 dataset and ~88
    # days on 2025-09-16; see test/test_tx_start.py). By convention the
    # transmission starts 1.0 s after the RX recording starts.
    tx_emit_start = rx_start + 1.0 * au.s

    rx_duration_s = rx_duration.to(au.s).value

    log_A, dlt_shifts, delay_values_s, lt_min_image, dlt_rate_srp = compute_dd_image(
        rx_samples, tx_samples, sample_rate, frequency,
        rx_start, tx_emit_start, tx_extra_offset_s, 0.0, tx_name, rx_name,
        freq_offset_hz=freq_offset_hz)

    edge_img = compute_edge_image(log_A)
    rx_time_s = et_from_astropy(rx_start)
    lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = compute_doppler_equator_velocity(
        rx_time_s, n_points=500, rx_duration_s=rx_duration_s,
        dlt_rate_srp=dlt_rate_srp, tx_name=tx_name, rx_name=rx_name)

    # Rim self-calibration: residual chain Doppler measured from the
    # horseshoe rims (see measure_rim_offset). For cross-pol looks the rim is
    # too diffuse to measure -- pass rim_delta_hz from the co-pol twin.
    f_hz = frequency.to_value(au.Hz)
    rim = None
    rim_residual_dlt = None
    if rim_delta_hz is None:
        # Iterate: the half-power edge finder is not perfectly linear in the
        # offset, so converge the calibration (typically 2 iterations).
        delta_dlt = 0.0
        ddlt_bin = dlt_shifts[1] - dlt_shifts[0]
        for _ in range(3):
            rim_i = measure_rim_offset(log_A, dlt_shifts - delta_dlt, delay_values_s,
                                       lt_min_image, lt_min_eq, delay_up, dlt_up,
                                       delay_down, dlt_down)
            if rim_i is None:
                break
            rim = rim_i if rim is None else rim
            delta_dlt += rim_i["delta_dlt"]
            rim_residual_dlt = rim_i["delta_dlt"]
            if abs(rim_i["delta_dlt"]) < 0.5 * ddlt_bin:
                break
        if rim is not None:
            rim_f = measure_rim_offset(log_A, dlt_shifts - delta_dlt, delay_values_s,
                                       lt_min_image, lt_min_eq, delay_up, dlt_up,
                                       delay_down, dlt_down)
            rim_residual_dlt = rim_f["delta_dlt"] if rim_f else rim_residual_dlt
    else:
        delta_dlt = -rim_delta_hz / f_hz
    dlt_shifts_cal = dlt_shifts - delta_dlt

    score = alignment_score(edge_img, dlt_shifts_cal, delay_values_s,
                            lt_min_image, lt_min_eq, delay_up, dlt_up)
    score += alignment_score(edge_img, dlt_shifts_cal, delay_values_s,
                             lt_min_image, lt_min_eq, delay_down, dlt_down)

    val_surface, valid_fraction, multiplicity = lunar_projection(
        log_A, dlt_shifts_cal, delay_values_s, lt_min_image,
        rx_time_s, nside, rx_duration_s=rx_duration_s, dlt_rate_srp=dlt_rate_srp,
        tx_name=tx_name, rx_name=rx_name)

    base = os.path.basename(rx_filename)
    os.makedirs(out_dir, exist_ok=True)
    log_png = os.path.join(out_dir, f"{base}_log_A.png")
    dd_png = os.path.join(out_dir, f"{base}_lunar.png")
    map_npy = os.path.join(out_dir, f"{base}_map.npy")
    mult_npy = os.path.join(out_dir, f"{base}_mapcount.npy")
    if save_pngs:
        pl.imsave(log_png, log_A.T, vmax=log_A.max() * 0.8, vmin=log_A.max() * 0.4)
        save_lunar_image(val_surface, dd_png, base,
                         vmin=log_A.max() * 0.4, vmax=log_A.max() * 0.8)
    else:
        log_png = dd_png = ""
    np.save(map_npy, val_surface)  # healpix map (hp.UNSEEN where invalid)
    np.save(mult_npy, multiplicity)  # DD-cell multiplicity (degeneracy mask)

    # Diagnostic only: the TX file timestamp is a generation time, not the
    # emission epoch used above.
    tx_file_minus_rx = (tx_start - rx_start).to_value('s')
    return {
        "rx_file": base,
        "tx_file": tx_filename,
        "rx_start_utc": rx_start.utc.value,
        "tx_emit_start_utc": tx_emit_start.utc.value,
        "tx_file_datetime_utc": tx_start.utc.value,
        "tx_file_minus_rx_s": tx_file_minus_rx,
        "sample_rate_hz": sample_rate.to_value(au.Hz),
        "frequency_hz": frequency.to_value(au.Hz),
        "tx_extra_offset_s": tx_extra_offset_s,
        "freq_offset_hz": freq_offset_hz,
        "rim_delta_hz": -delta_dlt * f_hz,
        "rim_spread_hz": (-rim["spread_dlt"] * f_hz) if rim else "",
        "rim_residual_hz": (-rim_residual_dlt * f_hz) if rim_residual_dlt is not None else "",
        "rim_n": (rim["n_up"] + rim["n_down"]) if rim else 0,
        "alignment_score": score,
        "valid_lunar_fraction": valid_fraction,
        "log_png": log_png,
        "lunar_png": dd_png,
        "map_npy": map_npy,
        "mult_npy": mult_npy,
    }


def write_metrics(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in keys) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__), "data.camras.nl/lunar-radar"))
    parser.add_argument("--date", default="2025-09-16")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--nside", type=int, default=100)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results/FIXED_BATCH"))
    parser.add_argument("--single")
    args = parser.parse_args()

    if args.single:
        files = [args.single]
    else:
        files = candidate_rx_files(args.data_root, args.date, args.limit)
    rows = []
    for rx_filename in files:
        try:
            rows.append(process_file(rx_filename, args.data_root, args.out_dir, args.nside))
        except Exception as exc:
            print(f"ERROR processing {rx_filename}: {exc}")
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    write_metrics(rows, metrics_path)
    print(f"Wrote {len(rows)} rows to {metrics_path}")


if __name__ == "__main__":
    main()
