"""Focused delay-Doppler imaging, prototype on one look.

Today's pipeline correlates the whole look at once for each of 3000 Doppler
rows, after removing the specular point's Doppler; points elsewhere drift
in Doppler relative to it (28 mHz median, up to 57 mHz over a 125 s look)
and smear across rows. Here:

  1. same transmitted-code alignment and specular Doppler removal as
     compute_dd_image (copied arithmetic);
  2. correlate in 10 ms blocks -> delay profiles through the look;
  3. for each trial drift rate a, remove the chirp exp(i pi a t'^2) (t'
     from mid-look) and Fourier-transform through time -> one focused
     delay-Doppler image per a (rows ~3 mHz apart);
  4. project: every point of the map reads the image whose a matches its
     own drift (from the geometry), at its delay and mid-look Doppler; 4x4
     points per pixel averaged as in lunar_projection.

Also writes the a = 0 version (no drift correction, block method) to check
it against today's image and map.

Prototype, 2026-10-21, on the 2026-10-11 circular set (paths below are
that run's). Against today's pipeline on the 16 long looks: without the
drift correction it reproduces today's maps (detail correlation 0.95-0.97,
same delay axis, no Doppler offset) in ~50 s per 125 s look; with it,
repeatable detail between independent half-stacks at nside 400 rises 4-22 %
at 9-40 km, mostly along the Doppler direction (REPORT section 6).

Known flaw: a 10 ms block holds only part of a Zadoff-Chu period, where the
code's correlation is not clean, so the strong opposite-sense echo leaks into the
delay columns ahead of the echo and the noise_log_amp written here reads the
opposite-sense floor ~6x too high in power (the same-sense floor is right; the maps themselves
agree with production within 1 %). Use production's floor for stacking.

usage (one look per process; memory creeps up across looks in one process):
    .conda/bin/python investigations/focused_dd_prototype.py OUT_ROOT NSIDE S RX_FILE
"""
import csv
import os
import sys
import time

import numpy as np
import healpy as hp
import scipy.interpolate
import scipy.signal
import cupy
from astropy import units as au

sys.path.insert(0, "/home/than/code/moon-radar")
import doppler_equator_alignment as dea
from doppler_equator_alignment import (et_from_astropy, moonSRP_DLT_FWD, moonSRP_DLT_BCK,
                                       moonPointDLT_BCK, apparent_station_positions,
                                       moon_surface_points, interp_in_slices, csp, AB_COR,
                                       MOON_RADIUS, LEAD_SAMPLES, moon_radii)
import scipy.ndimage
from astropy import constants as ak

BLOCK = 2500               # samples per block (10 ms at 250 ksps)
NFFT_BLOCK = 8192
NFFT_SLOW = 1 << 15        # slow-time FFT length (rows ~3 mHz apart for 125 s)
DALPHA_MAX = 1.2e-4        # Hz/s between trial drift rates (pi da T^2/8 < pi/4 at 125 s)


def tx_compensated(rx_samples, tx_samples, sample_rate, frequency, rx_start, tx_emit_start,
                   tx_start_offset, freq_offset_hz, tx_name, rx_name):
    """compute_dd_image's TX alignment and Doppler compensation, unchanged."""
    rx_duration = len(rx_samples) / sample_rate
    tx_duration = len(tx_samples) / sample_rate
    rx_start_time_s = et_from_astropy(rx_start)
    rx_end_time_s = rx_start_time_s + rx_duration.to(au.s).value
    tx_start_time_s = et_from_astropy(tx_emit_start) + tx_start_offset
    tx_end_time_s = tx_start_time_s + tx_duration.to(au.s).value
    lt_tx_start, _ = moonSRP_DLT_FWD(tx_start_time_s, tx_name, rx_name)
    lt_tx_end, _ = moonSRP_DLT_FWD(tx_end_time_s, tx_name, rx_name)
    tx_rx_offset = tx_start_time_s - rx_start_time_s
    adjusted_tx_times0 = np.linspace(tx_rx_offset + lt_tx_start,
                                     tx_rx_offset + tx_duration.to(au.s).value + lt_tx_end,
                                     len(tx_samples), endpoint=False)
    tx_phase_interp = scipy.interpolate.interp1d(
        adjusted_tx_times0, np.unwrap(np.angle(tx_samples)),
        kind='linear', fill_value=np.nan, bounds_error=False)
    del adjusted_tx_times0
    rx_duration_s = rx_duration.to(au.s).value
    lt_rx_start, dlt_rx_start = moonSRP_DLT_BCK(rx_start_time_s, tx_name, rx_name, dt=rx_duration_s / 2)
    lt_rx_end, dlt_rx_end = moonSRP_DLT_BCK(rx_end_time_s, tx_name, rx_name, dt=rx_duration_s / 2)
    f_hz = frequency.to_value(au.Hz)
    doppler_start = -dlt_rx_start * f_hz
    doppler_end = -dlt_rx_end * f_hz
    doppler_rate = (doppler_end - doppler_start) / rx_duration_s
    dlt_rate_srp = (dlt_rx_end - dlt_rx_start) / rx_duration_s
    fs = sample_rate.to_value(au.Hz)
    # In slices (memory): same per-sample arithmetic as compute_dd_image.
    tx_comp = np.empty(len(rx_samples), dtype=np.complex64)
    for i0 in range(0, len(rx_samples), 1 << 22):
        t_s = np.arange(i0, min(i0 + (1 << 22), len(rx_samples))) / fs
        ph = tx_phase_interp(t_s)
        z = np.nan_to_num(np.exp(1j * ph))
        phi_Hz = -(doppler_start + doppler_rate * t_s / 2 + freq_offset_hz)
        z *= np.exp(-1j * 2 * np.pi * (phi_Hz * t_s))
        tx_comp[i0:i0 + len(t_s)] = z
    del tx_phase_interp
    # delay window and Doppler span, as compute_dd_image
    cor_lags = scipy.signal.correlation_lags(len(rx_samples), len(tx_comp), mode="same") / sample_rate
    t_end = MOON_RADIUS / ak.c * 2
    cor_i_start = np.argwhere(cor_lags >= -LEAD_SAMPLES / sample_rate)[0][0]
    cor_i_end = np.argwhere(cor_lags >= t_end)[0][0]
    delay_values_s = cor_lags[cor_i_start:cor_i_end].to(au.s).value
    _, _, v_term = csp.edterm("UMBRAL", tx_name, "MOON", rx_start_time_s, "MOON_ME", AB_COR, rx_name, 1000)
    _, dlt_term = moonPointDLT_BCK(rx_start_time_s, v_term, tx_name, rx_name)
    return dict(tx_comp=tx_comp, doppler_start=doppler_start,
                dlt_rate_srp=dlt_rate_srp, lt_min=lt_rx_start, delay_values_s=delay_values_s,
                dlt_term=(dlt_term.min(), dlt_term.max()), rx_time_s=rx_start_time_s,
                rx_duration_s=rx_duration_s)


def block_profiles(rx, tx, k0, n_lag, batch=500):
    """C[b, m] = sum_{n in block b} rx[n + k0 + m] conj(tx[n]) on the GPU."""
    n_blocks = len(tx) // BLOCK
    pad_lo, pad_hi = max(0, -k0), k0 + n_lag + BLOCK
    rx_g = cupy.zeros(len(rx) + pad_lo + pad_hi, dtype=cupy.complex64)
    rx_g[pad_lo:pad_lo + len(rx)] = cupy.asarray(rx)
    tx_g = cupy.asarray(tx[:n_blocks * BLOCK]).reshape(n_blocks, BLOCK)
    C = cupy.empty((n_blocks, n_lag), dtype=cupy.complex64)
    span = cupy.arange(BLOCK + n_lag)
    for b0 in range(0, n_blocks, batch):
        b = cupy.arange(b0, min(b0 + batch, n_blocks))
        idx = (b[:, None] * BLOCK + pad_lo + k0) + span[None, :]
        X = cupy.fft.fft(rx_g[idx], NFFT_BLOCK, axis=1)
        Y = cupy.fft.fft(tx_g[b], NFFT_BLOCK, axis=1)
        C[b] = cupy.fft.ifft(X * cupy.conj(Y), axis=1)[:, :n_lag]
        del idx, X, Y
    del rx_g, tx_g
    cupy.get_default_memory_pool().free_all_blocks()
    return C


def focused_image(C, alpha, t_mid, keep, sinc_corr, lag_batch=512):
    """|FT over blocks of C * exp(-i pi alpha t'^2)|^2 for the kept rows,
    -> float32 (rows, lags)."""
    n_blocks, n_lag = C.shape
    t = (cupy.arange(n_blocks) + 0.5) * BLOCK / FS - t_mid
    chirp = cupy.exp(-1j * np.pi * alpha * t ** 2).astype(cupy.complex64)
    out = np.empty((len(keep), n_lag), dtype=np.float32)
    keep_g = cupy.asarray(keep)
    corr_g = cupy.asarray(sinc_corr, dtype=cupy.float32)
    for l0 in range(0, n_lag, lag_batch):
        Z = C[:, l0:l0 + lag_batch] * chirp[:, None]
        F = cupy.fft.fft(Z, NFFT_SLOW, axis=0)[keep_g]
        out[:, l0:l0 + lag_batch] = cupy.asnumpy((cupy.abs(F) ** 2) * corr_g[:, None])
        del Z, F
    return out


def project(image_fn, n_images, alpha_edges, dlt_shifts_cal, delay_values_s, lt_min_image,
            geo, nside, chip_samples, s=4, chunk=2_000_000):
    """lunar_projection with a per-point choice of image by drift-rate bin.
    image_fn(j) -> float32 intensity (rows, lags) for bin j, computed when
    needed (one image in memory at a time)."""
    rx_time_s, T, rate_srp, f_c, tx_name, rx_name = geo
    npix = hp.nside2npix(nside)
    fine_nside = nside * s
    n_fine = hp.nside2npix(fine_nside)
    n_cols = len(delay_values_s)
    n_rows = len(dlt_shifts_cal)
    ddlt = dlt_shifts_cal[1] - dlt_shifts_cal[0]
    ddelay = delay_values_s[1] - delay_values_s[0]
    sta0 = apparent_station_positions(rx_time_s, tx_name, rx_name)
    sta1 = apparent_station_positions(rx_time_s + T / 2, tx_name, rx_name)
    sta2 = apparent_station_positions(rx_time_s + T, tx_name, rx_name)
    ltf = lambda p, sta: (np.linalg.norm(p - sta[0], axis=1) + np.linalg.norm(p - sta[1], axis=1)) / sta[2]
    cell_all = np.empty(n_fine, dtype=np.int32)
    bin_all = np.empty(n_fine, dtype=np.int8)
    parent_all = np.empty(n_fine, dtype=np.int32)
    alphas_seen = []
    for i0 in range(0, n_fine, chunk):
        v = np.array(hp.pix2vec(fine_nside, np.arange(i0, min(i0 + chunk, n_fine)))).T
        parent = hp.vec2pix(nside, v[:, 0], v[:, 1], v[:, 2])
        p = moon_surface_points(v, use_dem=True)
        l0, l1, l2 = ltf(p, sta0), ltf(p, sta1), ltf(p, sta2)
        del p, v
        dlt = (l2 - l0) / T - rate_srp * T / 2
        # Residual drift (Hz/s) after the specular-point compensation, whose
        # frequency ramps at -f_c * rate_srp: -f_c (lt'' - rate_srp).
        alpha = +f_c * (((l2 - l1) - (l1 - l0)) / (T / 2) ** 2 - rate_srp)   # SIGN FLIPPED (test)
        di = np.rint((dlt - dlt_shifts_cal[0]) / ddlt).astype(np.int64)
        ki = np.rint((l0 - lt_min_image - delay_values_s[0]) / ddelay).astype(np.int64)
        valid = (di >= 0) & (di < n_rows) & (ki >= 0) & (ki < n_cols)
        sl = slice(i0, i0 + len(valid))
        cell_all[sl] = np.where(valid, di * n_cols + ki, -1)
        bin_all[sl] = np.clip(np.searchsorted(alpha_edges, alpha) - 1, 0, n_images - 1)
        parent_all[sl] = parent
        alphas_seen.append(alpha[valid][::50])
    inside = cell_all >= 0
    n_in = np.bincount(parent_all[inside], minlength=npix)
    if not np.all(np.bincount(parent_all, minlength=npix) == s * s):
        raise RuntimeError("sub-pixel parents")
    sum_i = np.zeros(npix)
    for j in range(n_images):
        im = image_fn(j).ravel()
        sel = inside & (bin_all == j)
        sum_i += np.bincount(parent_all[sel], weights=im[cell_all[sel]], minlength=npix)
        del im
    ok = n_in > 0
    val = np.full(npix, hp.UNSEEN, dtype=np.float32)
    val[ok] = 0.5 * np.log(sum_i[ok] / n_in[ok])
    cell_count = np.bincount(cell_all[inside], minlength=n_rows * n_cols)
    spread = cell_count.reshape(n_rows, n_cols).astype(np.float64)
    h = max(1, int(round(chip_samples)))
    tri = 1.0 - np.abs(np.arange(-h + 1, h)) / h
    spread = scipy.ndimage.convolve1d(spread, tri / tri.sum(), axis=1, mode="constant")
    hr = max(1, int(round(dea.AREA_ROWS * 8.3e-3 / abs(ddlt * f_c))))
    tri = 1.0 - np.abs(np.arange(-hr, hr + 1)) / (hr + 1)
    spread = scipy.ndimage.convolve1d(spread, tri / tri.sum(), axis=0, mode="constant").ravel()
    c = cell_all[inside]
    sum_mult = np.bincount(parent_all[inside], weights=cell_count[c], minlength=npix)
    sum_inv = np.bincount(parent_all[inside], weights=1.0 / spread[c], minlength=npix)
    mult = np.zeros(npix, dtype=np.float32)
    mult[ok] = np.maximum(1.0, sum_mult[ok] / n_in[ok] / (s * s))
    area = np.full(npix, np.nan, dtype=np.float32)
    area[ok] = 4 * np.pi * moon_radii()[0] ** 2 / n_fine / (sum_inv[ok] / n_in[ok])
    return val, mult, area, np.concatenate(alphas_seen)


def process_look(r, out_dir, nside, s):
    """a = 0 (block method, no drift correction) and focused maps for one
    runs-CSV row; returns two row dicts for stacking."""
    global FS
    os.makedirs(out_dir, exist_ok=True)
    rx_name = r["rx_file"].split("_")[0].upper()
    data_root = "/home/than/code/moon-radar/data/eve2026_circ"
    path = os.path.join(data_root, "2026-10-11", r["rx_file"])
    t0 = time.time()
    (rx, tx, sample_rate, frequency, rx_start, _, _) = dea.load_observation(path, data_root)
    FS = sample_rate.to_value(au.Hz)
    f_c = frequency.to_value(au.Hz)
    shift_s = float(r["applied_shift_samples"]) / FS
    g = tx_compensated(rx, tx, sample_rate, frequency, rx_start, rx_start + 1.0 * au.s, shift_s,
                       float(r["applied_df_hz"]), "DWINGELOO", rx_name)
    del tx
    dv = g["delay_values_s"]
    C = block_profiles(rx, g.pop("tx_comp"), int(round(dv[0] * FS)), len(dv))
    del rx
    f_bins = np.fft.fftfreq(NFFT_SLOW, BLOCK / FS)
    dlt_bins = -(f_bins + g["doppler_start"]) / f_c
    lo, hi = g["dlt_term"]
    margin = 0.02 * (hi - lo)
    keep = np.where((dlt_bins >= lo - margin) & (dlt_bins <= hi + margin))[0]
    keep = keep[np.argsort(dlt_bins[keep])]
    sinc_corr = 1.0 / np.sinc(f_bins[keep] * BLOCK / FS) ** 2
    dlt_cal = dlt_bins[keep] + float(r["rim_delta_hz"]) / f_c       # dlt_shifts - delta_dlt
    T = g["rx_duration_s"]
    geo = (g["rx_time_s"], T, g["dlt_rate_srp"], f_c, "DWINGELOO", rx_name)
    chip = dea.chip_samples_of(r["tx_file"], FS)
    im0 = focused_image(C, 0.0, T / 2, keep, sinc_corr)
    noise = 0.5 * float(np.log(im0[:, :dea.NOISE_COLUMNS].mean()))
    rows = {}
    v0, m0, a0, alphas = project(lambda j: im0, 1, np.array([-np.inf, np.inf]), dlt_cal, dv,
                                 g["lt_min"], geo, nside, chip, s)
    del im0
    a_lo, a_hi = np.percentile(alphas, [0.01, 99.99])
    bank = np.linspace(a_lo, a_hi, int(np.ceil((a_hi - a_lo) / DALPHA_MAX)) + 1)
    edges = np.concatenate(([-np.inf], (bank[1:] + bank[:-1]) / 2, [np.inf]))
    v1, m1, a1, _ = project(lambda j: focused_image(C, bank[j], T / 2, keep, sinc_corr),
                            len(bank), edges, dlt_cal, dv, g["lt_min"], geo, nside, chip, s)
    del C
    cupy.get_default_memory_pool().free_all_blocks()
    for name, (v, m, a) in (("alpha0", (v0, m0, a0)), ("focused", (v1, m1, a1))):
        files = {k: os.path.join(out_dir, f"{name}_{k}.npy") for k in ("map", "mapcount", "cellarea")}
        for (k, f), arr in zip(files.items(), (v, m, a)):
            np.save(f, arr)
        rows[name] = dict(r, map_npy=files["map"], mult_npy=files["mapcount"],
                          area_npy=files["cellarea"], noise_log_amp=noise)
    print(f"  {r['rx_file'][:50]}: {len(bank)} drift rates, {time.time() - t0:.0f} s", flush=True)
    return rows


if __name__ == "__main__":
    # One look per process (memory creeps up across looks in one process):
    # focus_proto.py OUT_ROOT NSIDE S RX_FILE -> OUT_ROOT/<stem>/rows.json
    import json
    out_root, nside, s, rx_file = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    st = rx_file.split("_")[0]
    chan = "chan1" if ".chan1." in rx_file else "chan0"
    run = "/home/than/code/moon-radar/results/EVE2026_CIRC"
    r = [r for r in csv.DictReader(open(f"{run}/registration_runs_{st}_{chan}.csv")) if r["rx_file"] == rx_file][0]
    out = os.path.join(out_root, rx_file.replace(".sigmf-meta", ""))
    dea.load_lola_dem()
    rows = process_look(r, out, nside, s)
    json.dump(rows, open(os.path.join(out, "rows.json"), "w"))
