"""Focused delay-Doppler imaging, prototype on one look.

Production's delay-Doppler image (compute_dd_image) removes only the
specular point's Doppler; points elsewhere drift in Doppler relative to it
(28 mHz median, up to 57 mHz over a 125 s look) and smear across rows.
Here:

  1. same transmitted-code alignment and specular Doppler removal as
     compute_dd_image (its functions compensated_tx, delay_window and
     doppler_span);
  2. for each trial drift rate a, production's block-sum correlation
     (correlate_blocks) with the chirp exp(-i pi a t'^2) applied at each
     block centre (t' from mid-look) -> one focused delay-Doppler image per
     a (rows ROW_HZ = 3.05 mHz apart);
  3. project: every point of the map reads the image whose a matches its
     own drift (from the geometry), at its delay and mid-look Doppler; 4x4
     points per pixel averaged as in lunar_projection.

Also writes the a = 0 version (no drift correction: production's image on
these rows) to check it against production's map.

Prototype, 2026-10-21, tried on the 2026-10-11 circular set. Measured
that day against the pipeline on the 16 long looks: without the drift
correction it reproduced the maps (detail correlation 0.95-0.97, same delay
axis, no Doppler offset) in ~50 s per 125 s look; with it, repeatable
detail between independent half-stacks at nside 400 rose 4-22 % at 9-40 km,
mostly along the Doppler direction (REPORT section 6). Those numbers, the
drift figures above and the noise-floor figures below come from that
day's scratch scripts, kept
in investigations/2026-10-21_scratch/ (its README lists which script gave
which figure; they no longer run as they are). They ran on products made
before that day's pipeline fixes, so the figures must be re-measured
before they are relied on.

Until 2026-10-24 the images came from plain sums over 10 ms blocks, which
fold the Zadoff-Chu code's wrong-delay tones into the image
(correlate_blocks' docstring): the strong opposite-sense echo leaked into the delay
columns ahead of the echo and the opposite-sense noise floor read ~6x too high in
power (the same-sense floor was right; the maps agreed with production within 1 %).
The rows were therefore written with a blank noise_log_amp, and the
2026-10-21 comparison carried production's floor over
(2026-10-21_scratch/make_focused_run.py); all the figures above come from
those blocks. The images now come from correlate_blocks, which keeps the
phase change within a block, and the rows carry the a = 0 image's own
noise floor.

Input: a runs CSV row of a run made since 2026-10-21 (it reads rim_status,
rx_meta_path and the applied timing, frequency and rim corrections), a
opposite-sense (chan1) or same-sense (chan0) capture, and the edge fit's delay and
Doppler offsets, applied as process_file applies them (since 2026-10-24
here; looks without a fitted or inherited edge fit are refused, as
stack_maps.py refuses them).

usage (one look per process; memory creeps up across looks in one process):
    .conda/bin/python investigations/focused_dd_prototype.py OUT_ROOT RUN_DIR \
        DATA_ROOT NSIDE S RX_FILE
"""
import csv
import os
import resource
import sys
import time

import numpy as np
import healpy as hp
from astropy import units as au

# The pipeline modules live in the repository root, one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import doppler_equator_alignment as dea
from doppler_equator_alignment import (et_from_astropy, apparent_station_positions,
                                       moon_surface_points, moon_radii)

# Row spacing (residual frequency): that of the 2026-10-21 runs (the bins of
# a 2^15-point Fourier transform over 10 ms blocks at 250 ksps).
ROW_HZ = 250e3 / (2500 * 2 ** 15)
DALPHA_MAX = 1.2e-4        # Hz/s between trial drift rates (pi da T^2/8 < pi/4 at 125 s)


def tx_compensated(rx_samples, tx_samples, sample_rate, frequency, rx_start, tx_emit_start,
                   tx_start_offset, freq_offset_hz, tx_name, rx_name):
    """compute_dd_image's TX model, delay window and Doppler span (the
    production functions themselves)."""
    rx_start_time_s = et_from_astropy(rx_start)
    tx_comp, info = dea.compensated_tx(
        len(rx_samples), tx_samples, sample_rate, frequency, rx_start_time_s,
        et_from_astropy(tx_emit_start) + tx_start_offset, tx_name=tx_name, rx_name=rx_name,
        freq_offset_hz=freq_offset_hz)
    cor_lags, i0, i1 = dea.delay_window(len(rx_samples), sample_rate)
    return dict(tx_comp=tx_comp, doppler_start=info["doppler_start_hz"],
                dlt_rate_srp=info["dlt_rate_srp"], lt_min=info["lt_rx_start"],
                delay_values_s=cor_lags[i0:i1].to(au.s).value,
                dlt_term=dea.doppler_span(rx_start_time_s, tx_name, rx_name),
                rx_time_s=rx_start_time_s, rx_duration_s=len(rx_samples) / sample_rate.to_value(au.Hz),
                tx_overlap_s=info["tx_overlap_s"])


def focused_image(rx, tx_comp, f_rows, lag0, n_lag, alpha, t_mid):
    """Intensity of the image focused at drift rate alpha (Hz/s), float32
    (rows, lags): production's correlate_blocks with the chirp
    exp(-i pi alpha (t - t_mid)^2) at each block centre."""
    A, _ = dea.correlate_blocks(rx, tx_comp, FS, f_rows, lag0, lag0 + n_lag,
                                drift_hz_per_s=alpha, t_ref_s=t_mid)
    return A * A


def place_points(dlt_proj, delay_values_s, lt_proj, geo, nside, s, chunk=1_000_000):
    """Every fine point's DD cell (-1 outside the image), parent pixel and
    residual drift rate (Hz/s, float32), placed as lunar_projection places
    them (same delay reference; s x s points per pixel). A point inside the
    image with no finite drift rate is an error."""
    rx_time_s, T, rate_srp, f_c, tx_name, rx_name = geo
    fine_nside = nside * s
    n_fine = hp.nside2npix(fine_nside)
    n_cols = len(delay_values_s)
    n_rows = len(dlt_proj)
    ddlt = dlt_proj[1] - dlt_proj[0]
    ddelay = delay_values_s[1] - delay_values_s[0]
    sta0 = apparent_station_positions(rx_time_s, tx_name, rx_name)
    sta1 = apparent_station_positions(rx_time_s + T / 2, tx_name, rx_name)
    sta2 = apparent_station_positions(rx_time_s + T, tx_name, rx_name)
    ltf = lambda p, sta: (np.linalg.norm(p - sta[0], axis=1) + np.linalg.norm(p - sta[1], axis=1)) / sta[2]
    cell_all = np.empty(n_fine, dtype=np.int32)
    parent_all = np.empty(n_fine, dtype=np.int32)
    alpha_all = np.empty(n_fine, dtype=np.float32)
    for i0 in range(0, n_fine, chunk):
        v = np.array(hp.pix2vec(fine_nside, np.arange(i0, min(i0 + chunk, n_fine)))).T
        parent = hp.vec2pix(nside, v[:, 0], v[:, 1], v[:, 2])
        p = moon_surface_points(v, use_dem=True)
        l0, l1, l2 = ltf(p, sta0), ltf(p, sta1), ltf(p, sta2)
        del p, v
        dlt = (l2 - l0) / T - rate_srp * T / 2
        # Residual drift (Hz/s) after the specular-point compensation, whose
        # frequency ramps at -f_c * rate_srp: -f_c (lt'' - rate_srp).
        alpha = -f_c * (((l2 - l1) - (l1 - l0)) / (T / 2) ** 2 - rate_srp)
        di = np.rint((dlt - dlt_proj[0]) / ddlt).astype(np.int64)
        ki = np.rint((l0 - lt_proj - delay_values_s[0]) / ddelay).astype(np.int64)
        valid = (di >= 0) & (di < n_rows) & (ki >= 0) & (ki < n_cols)
        no_rate = valid & ~np.isfinite(alpha)
        if no_rate.any():
            raise ValueError(f"{int(no_rate.sum())} points inside the image have "
                             "no finite drift rate")
        sl = slice(i0, i0 + len(valid))
        cell_all[sl] = np.where(valid, di * n_cols + ki, -1)
        parent_all[sl] = parent
        alpha_all[sl] = alpha
    if not np.all(np.bincount(parent_all, minlength=hp.nside2npix(nside)) == s * s):
        raise RuntimeError("sub-pixel parents")
    return cell_all, parent_all, alpha_all


def cell_counts(cell_all, parent_all, inside, n_in, n_rows, n_cols, nside, s,
                chip_samples, doppler_res_dlt, ddlt):
    """Overlap count and ground area per map pixel, as lunar_projection
    (they depend on where the points fall, not on the image)."""
    npix = hp.nside2npix(nside)
    ok = n_in > 0
    cell_count = np.bincount(cell_all[inside], minlength=n_rows * n_cols)
    spread = cell_count.reshape(n_rows, n_cols).astype(np.float64)
    spread = dea.convolve_response(spread, dea.triangle_power(chip_samples), axis=1)
    spread = dea.convolve_response(spread, dea.sinc_power(doppler_res_dlt / abs(ddlt)),
                                   axis=0).ravel()
    c = cell_all[inside]
    sum_mult = np.bincount(parent_all[inside], weights=cell_count[c], minlength=npix)
    sum_spread = np.bincount(parent_all[inside], weights=spread[c], minlength=npix)
    mult = np.zeros(npix, dtype=np.float32)
    mult[ok] = sum_mult[ok] / n_in[ok] / (s * s)
    area = np.full(npix, np.nan, dtype=np.float32)
    area[ok] = (4 * np.pi * moon_radii()[0] ** 2 / hp.nside2npix(nside * s)
                * sum_spread[ok] / n_in[ok])
    return mult, area


def log_map(sum_i, n_in):
    """0.5 log of the mean intensity over each pixel's points inside the
    image (hp.UNSEEN where none is)."""
    ok = n_in > 0
    val = np.full(len(sum_i), hp.UNSEEN, dtype=np.float32)
    val[ok] = 0.5 * np.log(sum_i[ok] / n_in[ok])
    return val


def process_look(r, data_root, out_dir, nside, s):
    """a = 0 (no drift correction) and focused maps for one production
    runs-CSV row (its timing, frequency, rim and edge-fit corrections, applied
    as process_file applies them); returns two row dicts, both with the a = 0
    image's noise floor (focusing leaves the noise level unchanged; the bank
    images' floors are printed). The points are placed once: the two maps
    share their positions, overlap counts and areas, and differ only in which
    image each point reads."""
    global FS
    status = r["pair_rim_status"] if r["rim_status"] == "inherited" else r["rim_status"]
    if status != "converged":
        raise ValueError(f"{r['rx_file']}: production rim calibration {status}"
                         + (" (its opposite-sense twin's)" if r["rim_status"] == "inherited" else ""))
    if r["edge_status"] not in ("fitted", "inherited"):
        raise ValueError(f"{r['rx_file']}: edge fit {r['edge_status']}: no delay/Doppler "
                         "offsets (stack_maps.py rejects such looks)")
    os.makedirs(out_dir, exist_ok=True)
    rx_name = dea.STATION_RX[r["rx_file"].split("_")[0]]
    path = os.path.join(data_root, r["rx_meta_path"])
    t0 = time.time()
    (rx, tx, sample_rate, frequency, rx_start, _, _) = dea.load_observation(path, data_root)
    FS = sample_rate.to_value(au.Hz)
    f_c = frequency.to_value(au.Hz)
    g = tx_compensated(rx, tx, sample_rate, frequency, rx_start,
                       dea.nominal_tx_emit_start(rx_start), float(r["tx_extra_offset_s"]),
                       float(r["applied_df_hz"]), dea.TX_STATION, rx_name)
    del tx
    dv = g["delay_values_s"]
    tx_comp = g.pop("tx_comp")
    lag0 = int(round(dv[0] * FS))
    # Rows: the multiples of ROW_HZ (residual frequency) over the visible
    # disk's Doppler span plus 2 % either side, in increasing dlt.
    lo, hi = g["dlt_term"]
    margin = 0.02 * (hi - lo)
    f_lo = -(hi + margin) * f_c - g["doppler_start"]
    f_hi = -(lo - margin) * f_c - g["doppler_start"]
    f_rows = np.arange(np.floor(f_hi / ROW_HZ), np.ceil(f_lo / ROW_HZ) - 1, -1) * ROW_HZ
    dlt_rows = -(f_rows + g["doppler_start"]) / f_c
    T = g["rx_duration_s"]
    geo = (g["rx_time_s"], T, g["dlt_rate_srp"], f_c, dea.TX_STATION, rx_name)
    chip = dea.chip_samples_of(r["tx_file"], FS)
    # As process_file: delays from the terrain's light time at the SRP, then
    # the rim correction and the edge fit's delay and Doppler offsets.
    _, topo_s = dea.extract_srp_elevation(g["rx_time_s"], dea.TX_STATION, rx_name)
    edge_delay_s = float(r["edge_delay_us"]) * 1e-6
    edge_doppler_hz = float(r["edge_doppler_hz"])
    lt_proj = g["lt_min"] - topo_s - edge_delay_s
    dlt_proj = dlt_rows + (float(r["rim_delta_hz"]) + edge_doppler_hz) / f_c
    res_dlt = 1.0 / (g["tx_overlap_s"] * f_c)

    cell_all, parent_all, alpha_all = place_points(dlt_proj, dv, lt_proj, geo, nside, s)
    inside = cell_all >= 0
    npix = hp.nside2npix(nside)
    n_in = np.bincount(parent_all[inside], minlength=npix)
    a_lo, a_hi = np.percentile(alpha_all[inside], [0.01, 99.99])
    bank = np.linspace(a_lo, a_hi, int(np.ceil((a_hi - a_lo) / DALPHA_MAX)) + 1)
    edges = (bank[1:] + bank[:-1]) / 2
    # Each inside point's image in the bank (the nearest drift rate).
    bin_all = np.where(inside, np.searchsorted(edges, alpha_all), -1).astype(np.int16)
    del alpha_all

    def add(sum_i, im, sel):
        sum_i += np.bincount(parent_all[sel], weights=im.ravel()[cell_all[sel]],
                             minlength=npix)

    im = focused_image(rx, tx_comp, f_rows, lag0, len(dv), 0.0, T / 2)
    floor0 = dea.noise_floor_log_amp(0.5 * np.log(im))
    sum0 = np.zeros(npix)
    add(sum0, im, inside)
    sum1 = np.zeros(npix)
    floors = []
    for j, a in enumerate(bank):
        im = focused_image(rx, tx_comp, f_rows, lag0, len(dv), a, T / 2)
        floors.append(dea.noise_floor_log_amp(0.5 * np.log(im)))
        add(sum1, im, bin_all == j)
    del im, rx, tx_comp, bin_all
    mult, area = cell_counts(cell_all, parent_all, inside, n_in, len(dlt_proj), len(dv),
                             nside, s, chip, res_dlt, dlt_proj[1] - dlt_proj[0])
    rows = {}
    for name, sum_i in (("alpha0", sum0), ("focused", sum1)):
        files = {k: os.path.join(out_dir, f"{name}_{k}.npy") for k in ("map", "mapcount", "cellarea")}
        for (k, f), arr in zip(files.items(), (log_map(sum_i, n_in), mult, area)):
            np.save(f, arr)
        rows[name] = dict(r, map_npy=files["map"], mult_npy=files["mapcount"],
                          area_npy=files["cellarea"], noise_log_amp=floor0)
    print(f"  {r['rx_file'][:50]}: {len(bank)} drift rates, {time.time() - t0:.0f} s, "
          f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB; "
          f"noise floor (log amplitude) {floor0:.4f} at a = 0, "
          f"{min(floors):.4f}..{max(floors):.4f} over the bank", flush=True)
    return rows


if __name__ == "__main__":
    # One look per process (memory creeps up across looks in one process):
    # focused_dd_prototype.py OUT_ROOT RUN_DIR DATA_ROOT NSIDE S RX_FILE
    #   -> OUT_ROOT/<stem>/rows.json
    import json
    out_root, run, data_root = sys.argv[1], sys.argv[2], sys.argv[3]
    nside, s, rx_file = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    st = rx_file.split("_")[0]
    chans = [c for c in ("chan1", "chan0") if f".{c}." in rx_file]
    if len(chans) != 1:
        raise ValueError(f"{rx_file}: not a chan1 (opposite sense) or chan0 (same sense) capture")
    runs_csv = f"{run}/registration_runs_{st}_{chans[0]}.csv"
    found = [r for r in csv.DictReader(open(runs_csv)) if r["rx_file"] == rx_file]
    if len(found) != 1:
        raise ValueError(f"{rx_file}: {len(found)} rows in {runs_csv}, expected 1")
    r = found[0]
    out = os.path.join(out_root, rx_file.replace(".sigmf-meta", ""))
    dea.load_lola_dem()
    rows = process_look(r, data_root, out, nside, s)
    json.dump(rows, open(os.path.join(out, "rows.json"), "w"))
