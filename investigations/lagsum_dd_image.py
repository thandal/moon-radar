"""
Block-sum ("lag-sum") delay-Doppler image: an A/B against the per-row
correlation on real looks.

Until 2026-10-24 compute_dd_image correlated the whole capture once per
Doppler row (correlate_rows: 3000 rows, two FFTs of the full capture each,
31 M samples on a 125 s look) and kept 2918 of the N delays. Its default is
now correlate_blocks: rx(t + lag) * conj(tx(t)) for every delay in the
window, summed over 4.7 ms blocks (all delays of a block at once by one
4096-point FFT correlation), then summed over blocks with each row's
Doppler phase (a chirp-z transform), at the same 3000 row frequencies. The
phase change within a block is kept as a short power series; with one term
(the plain block sum) the Zadoff-Chu code's wrong-delay tones fold into
the image (correlate_blocks' docstring). This script was written to decide
the switch and now checks it.

Two parts:
  --self-test  synthetic Zadoff-Chu echo, small, on the GPU: correlate_blocks
               (1 term and the automatic count) against correlate_rows.
  --look L     one look of LOOKS: the images from compute_dd_image with
               method="rows" and the default method="blocks", and the plain
               block sum (1 term), from the same inputs; image differences
               and noise floors; then process_file with the rows image and
               with the blocks image, comparing rim calibration, edge fit,
               noise floor and maps; and a third time, the blocks image with
               the rows run's rim and edge offsets (the map difference due to
               the image alone). Timings and peak memory are recorded.

process_file is run with its compute_dd_image swapped for one returning
the precomputed image.

Usage (repo root, one look per process keeps memory flat):
    .conda/bin/python investigations/lagsum_dd_image.py --self-test
    .conda/bin/python investigations/lagsum_dd_image.py --look 09_28_21 \
        [--out-dir results/LAGSUM_TEST]
"""
import argparse
import csv
import json
import os
import resource
import sys
import time

import numpy as np
import healpy as hp
from astropy import units as au

# The pipeline modules live in the repository root, one level up.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import doppler_equator_alignment as dea  # noqa: E402

# The A/B looks: 2026-10-11 Stockert opposite sense (circular set), both rim
# converged and edge fitted in the production run; one 35 s and one 125 s
# capture.
DATA_ROOT = os.path.join(REPO, "data", "eve2026_circ")
RUNS_CSV = os.path.join(REPO, "results", "EVE2026_CIRC", "registration_runs_stockert_chan1.csv")
LOOKS = {
    "09_28_21": "stockert_25m_moon_2026_10_11_09_28_21_1299.650MHz_0.25Msps_ci16_le.chan1.sigmf-meta",
    "09_41_25": "stockert_25m_moon_2026_10_11_09_41_25_1299.650MHz_0.25Msps_ci16_le.chan1.sigmf-meta",
}


def self_test():
    """Synthetic Zadoff-Chu echo (25 kchip/s, q=1351, N=750000 chips, 10
    samples per chip, as the 2026-10-11 code), 2^21 samples: a bright
    specular point, an extended echo behind it, noise. Compares
    correlate_blocks (1 term, automatic) with correlate_rows on 21 rows
    across +-13 Hz."""
    rng = np.random.default_rng(3)
    fs, n, chip, q, n_zc = 250e3, 1 << 21, 10, 1351, 750_000
    lead, w = dea.LEAD_SAMPLES, 2918
    mm = np.arange(n // chip + 1, dtype=np.float64)
    zc = np.exp(-1j * np.pi * q * np.mod(mm * mm, 2 * n_zc) / n_zc)   # even length
    tx = np.repeat(zc, chip)[:n].astype(np.complex64)
    t = np.arange(n) / fs
    rx = 1.0 * tx * np.exp(2j * np.pi * 0.4 * t)                  # specular, delay 0
    for d in rng.integers(5, 2800, 300):                            # extended echo
        f = rng.uniform(-12, 12) * np.sqrt(d / 2800)
        a = (rng.normal() + 1j * rng.normal()) * 0.3 / np.sqrt(1 + d / 50)
        rx += a * np.roll(tx, d) * np.exp(2j * np.pi * f * t)
    rx += 1.0 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    rx = rx.astype(np.complex64)
    f_rows = np.linspace(-13.0, 13.0, 21)
    ref = dea.correlate_rows(rx, tx, fs, f_rows, -lead, w - lead)
    noise = np.sqrt(np.mean(ref[:, :dea.NOISE_COLUMNS] ** 2))
    print(f"self-test: {n} samples, rows {f_rows[0]:+.0f}..{f_rows[-1]:+.0f} Hz, "
          f"noise amplitude {noise:.0f}, peak {ref.max() / noise:.0f}x noise")
    for n_terms in (1, None):
        A, info = dea.correlate_blocks(rx, tx, fs, f_rows, -lead, w - lead, n_terms=n_terms)
        err = np.abs(A - ref)
        lead_pow = np.mean(A[:, :dea.NOISE_COLUMNS] ** 2) / noise ** 2
        print(f"  {info['n_terms']} term(s), {info['block']}-sample blocks: "
              f"max |error| {err.max() / noise:.2e} of the noise amplitude "
              f"({err.max() / ref.max():.1e} of the peak); noise-column power "
              f"{lead_pow:.3f}x the per-row correlation's")


def look_row(look):
    rx_file = LOOKS[look]
    found = [r for r in csv.DictReader(open(RUNS_CSV)) if r["rx_file"] == rx_file]
    if len(found) != 1:
        raise ValueError(f"{rx_file}: {len(found)} rows in {RUNS_CSV}")
    return found[0]


# Filled by the wrapped functions below.
TIMES = {}
IMAGES = {}


def timed(fn, key):
    """fn, adding its run time to TIMES[key] (and keeping its info dict, if
    it returns one, in TIMES[key + '_info'])."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        TIMES[key] = TIMES.get(key, 0.0) + time.perf_counter() - t0
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            TIMES[key + "_info"] = out[1]
        return out
    return wrapper


def all_images(rx_samples, tx_samples, sample_rate, frequency,
               rx_start_astrotime, tx_start_astrotime,
               tx_start_offset, rx_start_offset, tx_name, rx_name,
               freq_offset_hz=0.0):
    """compute_dd_image's outputs with method="rows" and "blocks", and the
    plain block sum (1 term) from the same inputs; returns the rows one.
    The 1-term setup repeats compute_dd_image's (TX model, delay window,
    Doppler rows); its axes are checked against compute_dd_image's."""
    args = (rx_samples, tx_samples, sample_rate, frequency, rx_start_astrotime,
            tx_start_astrotime, tx_start_offset, rx_start_offset, tx_name, rx_name)
    for method in ("rows", "blocks"):
        IMAGES[method] = ORIG_COMPUTE_DD_IMAGE(*args, freq_offset_hz=freq_offset_hz,
                                               method=method)
    rows = IMAGES["rows"]

    rx_start_time_s = dea.et_from_astropy(rx_start_astrotime) + rx_start_offset
    tx_start_time_s = dea.et_from_astropy(tx_start_astrotime) + tx_start_offset
    tx_compensated, info = dea.compensated_tx(
        len(rx_samples), tx_samples, sample_rate, frequency,
        rx_start_time_s, tx_start_time_s, tx_name=tx_name, rx_name=rx_name,
        freq_offset_hz=freq_offset_hz)
    cor_lags, cor_i_start, cor_i_end = dea.delay_window(len(rx_samples), sample_rate)
    dlt_lo, dlt_hi = dea.doppler_span(rx_start_time_s, tx_name, rx_name)
    dlt_shifts = np.linspace(dlt_lo, dlt_hi, 3000)
    f_shifts = -dlt_shifts * frequency.to(au.Hz).value - info["doppler_start_hz"]
    delay_values_s = (cor_lags[cor_i_start:cor_i_end]).to(au.s).value
    if not (np.array_equal(dlt_shifts, rows[1]) and np.array_equal(delay_values_s, rows[2])
            and info["lt_rx_start"] == rows[3] and info["dlt_rate_srp"] == rows[4]):
        raise RuntimeError("1-term setup does not reproduce compute_dd_image's axes")
    n = len(rx_samples)
    A, _ = ORIG_CORRELATE_BLOCKS(rx_samples, tx_compensated, sample_rate.to_value(au.Hz),
                                 f_shifts, cor_i_start - n // 2, cor_i_end - n // 2,
                                 n_terms=1)
    IMAGES["blocks_1term"] = (np.log(A),) + tuple(rows[1:])
    TIMES["f_shifts_abs_max_hz"] = float(np.abs(f_shifts).max())
    return rows


def compare_images(ref, other):
    """Differences of other's log amplitude from ref's."""
    floor = dea.noise_floor_log_amp(ref)
    d = other - ref
    echo = ref > floor + 1.5          # > 4.5x the floor in amplitude
    lead = slice(0, dea.NOISE_COLUMNS)
    return {
        "noise_log_amp": dea.noise_floor_log_amp(other),
        "noise_power_ratio": float(np.exp(2 * (dea.noise_floor_log_amp(other) - floor))),
        "echo_cells": int(echo.sum()),
        "echo_mean_diff": float(d[echo].mean()),
        "echo_max_abs_diff": float(np.abs(d[echo]).max()),
        "noise_cells_median_abs_diff": float(np.median(np.abs(d[:, lead]))),
        "all_cells_p999_abs_diff": float(np.percentile(np.abs(d), 99.9)),
        "all_cells_max_abs_diff": float(np.abs(d).max()),
    }


def compare_maps(row_a, row_b):
    a, b = np.load(row_a["map_npy"]), np.load(row_b["map_npy"])
    both = (a != hp.UNSEEN) & (b != hp.UNSEEN)
    d = np.abs(b[both] - a[both])
    return {"pixels_a": int((a != hp.UNSEEN).sum()), "pixels_b": int((b != hp.UNSEEN).sum()),
            "pixels_both": int(both.sum()),
            "median_abs_diff": float(np.median(d)),
            "p99_abs_diff": float(np.percentile(d, 99)),
            "max_abs_diff": float(d.max()),
            "pixels_over_0.01": int((d > 0.01).sum()),
            "corr": float(np.corrcoef(a[both], b[both])[0, 1])}


ROW_KEYS = ("rim_status", "rim_delta_hz", "rim_spread_hz", "edge_status", "edge_delay_us",
            "edge_delay_err_us", "edge_doppler_hz", "edge_doppler_err_hz", "edge_columns",
            "alignment_score", "noise_log_amp", "valid_lunar_fraction", "tx_overlap_s")


def run_look(look, out_root, nside):
    r = look_row(look)
    path = os.path.join(DATA_ROOT, r["rx_meta_path"])
    rx_name = dea.STATION_RX[r["rx_file"].split("_")[0]]
    out_dir = os.path.join(out_root, look)
    kwargs = dict(tx_name=dea.TX_STATION, rx_name=rx_name,
                  tx_extra_offset_s=float(r["tx_extra_offset_s"]),
                  freq_offset_hz=float(r["freq_offset_hz"]), save_pngs=True)
    print(f"{look}: {r['rx_file']} ({r['capture_s']} s capture)", flush=True)

    # Run 1: the rows image (all three images computed here).
    dea.compute_dd_image = all_images
    t0 = time.perf_counter()
    row_rows = dea.process_file(path, DATA_ROOT, os.path.join(out_dir, "rows"),
                                nside, **kwargs)
    TIMES["process_file_with_all_images_s"] = time.perf_counter() - t0
    # Run 2: the blocks image, same everything else (the image precomputed).
    dea.compute_dd_image = lambda *a, **k: IMAGES["blocks"]
    t0 = time.perf_counter()
    row_blocks = dea.process_file(path, DATA_ROOT, os.path.join(out_dir, "blocks"),
                                  nside, **kwargs)
    TIMES["process_file_image_precomputed_s"] = time.perf_counter() - t0
    # Run 3: the blocks image mapped with the rows run's rim and edge-fit
    # offsets, so its map differs from the rows map by the image alone (the
    # fits move a little with any change to the image, and a shifted point
    # can land in the next cell).
    row_same = dea.process_file(
        path, DATA_ROOT, os.path.join(out_dir, "blocks_rows_offsets"), nside,
        rim_delta_hz=float(row_rows["rim_delta_hz"]),
        edge_offsets=(float(row_rows["edge_delay_us"]) * 1e-6,
                      float(row_rows["edge_doppler_hz"])), **kwargs)
    dea.compute_dd_image = ORIG_COMPUTE_DD_IMAGE

    ref = IMAGES["rows"][0]
    np.savez(os.path.join(out_dir, "images.npz"),
             **{k: v[0].astype(np.float32) for k, v in IMAGES.items()},
             dlt_shifts=IMAGES["rows"][1], delay_values_s=IMAGES["rows"][2])
    summary = {
        "look": look, "rx_file": r["rx_file"], "capture_s": float(r["capture_s"]),
        "nside": nside,
        "times": TIMES,
        "images_vs_rows": {k: compare_images(ref, IMAGES[k][0])
                           for k in ("blocks", "blocks_1term")},
        "rows_noise_log_amp": dea.noise_floor_log_amp(ref),
        "csv_rows": {k: {"recorded_run": r.get(k, ""), "rows": row_rows.get(k),
                         "blocks": row_blocks.get(k)} for k in ROW_KEYS},
        "maps_blocks_vs_rows": compare_maps(row_rows, row_blocks),
        "maps_blocks_rows_offsets_vs_rows": compare_maps(row_rows, row_same),
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float), flush=True)


ORIG_COMPUTE_DD_IMAGE = dea.compute_dd_image
ORIG_CORRELATE_BLOCKS = dea.correlate_blocks
# Time the pieces compute_dd_image calls (module globals, looked up at call
# time).
dea.compensated_tx = timed(dea.compensated_tx, "compensated_tx_s")
dea.correlate_rows = timed(dea.correlate_rows, "correlate_rows_s")
dea.correlate_blocks = timed(dea.correlate_blocks, "correlate_blocks_s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--look", choices=sorted(LOOKS))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "LAGSUM_TEST"))
    ap.add_argument("--nside", type=int, default=400)
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.look:
        dea.load_lola_dem()
        run_look(args.look, args.out_dir, args.nside)
    else:
        ap.error("give --self-test or --look")
