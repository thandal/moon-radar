"""Does fit_edge_offsets recover known delay and Doppler offsets?

For one or more saved looks (their own axes, geometry and noise floor),
builds the model image (edge_model_image), shifts it by a known delay and
Doppler offset, and makes a synthetic look from it the way the real image
is made up:
  - brightness varying along each limb (independently per limb, smooth over
    a few chips of delay, 30 % rms) and across the Doppler equator (smooth
    over 20 Doppler rows, 20 % rms) -- what the fit's per-chip amplitudes
    must absorb, and what could pull the edges if they did not;
  - the image's noise floor;
  - speckle that is correlated as the real image's is: a complex Gaussian
    field filtered by the code's delay response and the integration's
    Doppler response before squaring.
Then fits it as production does and reports the recovered minus injected
offsets, in microseconds and mHz and in units of the fit's own jackknife
errors.

What this does not test: errors of the model itself (the scattering law,
terrain, reflectivity patterns unlike these), since the synthetic look is
made from the same model the fit uses. On real 2026-10-11 looks, swapping
the model's scattering law changed the edge positions by at most a few mHz.

Usage (from the repo root):
  .conda/bin/python validation/scripts/validate_edge_offsets.py \\
      --run-dir results/EVE2026_CIRC --data-root data/eve2026_circ \\
      --station stockert --looks 09:41:25 09:28:21
Output: validation/results/edge_offsets_<run folder>_<station>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import scipy.ndimage
import scipy.signal
from astropy import time as at

import validation_common as vc

import doppler_equator_alignment as dea

# Injected (delay in us, Doppler in Doppler resolution cells) -- inside the
# fit's search (+-1 chip, +-3 cells), of the size seen on the 2026 looks and
# beyond.
CASES = [(0.0, 0.0), (8.0, -1.0), (-15.0, 2.0), (25.0, 0.5)]


def look_inputs(r, run_dir, data_root, rx_name):
    """The fit's inputs for a saved look (axes as reproject_maps.py undoes them)."""
    with open(os.path.join(data_root, r["rx_meta_path"]), encoding="utf-8") as f:
        meta = json.load(f)
    rx_time_s = dea.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
    fs, f_hz, T = float(r["sample_rate_hz"]), float(r["frequency_hz"]), float(r["capture_s"])
    dd = np.load(os.path.join(run_dir, os.path.basename(r["dd_npz"])))
    log_A = dd["log_A"].astype(np.float32)
    _, dlt_srp = dea.moonSRP_DLT_BCK(rx_time_s, dea.TX_STATION, rx_name, dt=T / 2)
    return log_A, dict(
        dlt_shifts=dlt_srp - dd["doppler_hz"] / f_hz,
        delay_values_s=((dd["delay_s"][0] - float(r["tx_extra_offset_s"]))
                        + np.arange(log_A.shape[1]) / fs),
        lt_ref=-float(r["srp_topo_delay_us"]) * 1e-6,
        rx_time_s=rx_time_s,
        kw=dict(rx_duration_s=T,
                dlt_rate_srp=dea.srp_dlt_rate_bck(rx_time_s, T, dea.TX_STATION, rx_name),
                chip_samples=dea.chip_samples_of(r["tx_file"], fs),
                tx_overlap_s=float(r["tx_overlap_s"]), f_hz=f_hz,
                tx_name=dea.TX_STATION, rx_name=rx_name))


def smooth_field(rng, n, length, rms):
    """exp of a Gaussian random field along one axis, correlation ~length."""
    x = scipy.ndimage.gaussian_filter1d(rng.standard_normal(n), length, mode="wrap")
    return np.exp(x / x.std() * rms)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--station", required=True, choices=sorted(dea.STATION_RX))
    p.add_argument("--looks", nargs="+", required=True, help="rx start times, HH:MM:SS")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    rx_name = dea.STATION_RX[args.station]
    dea.load_lola_dem()
    rows = list(csv.DictReader(open(os.path.join(
        args.run_dir, f"registration_runs_{args.station}_chan1.csv"), encoding="utf-8")))
    rng = np.random.default_rng(args.seed)
    out = []
    for hms in args.looks:
        r = next(x for x in rows if x["rx_start_utc"][11:19] == hms)
        log_A, a = look_inputs(r, args.run_dir, args.data_root, rx_name)
        kw = a["kw"]
        n_rows, n_cols = log_A.shape
        fs = 1.0 / (a["delay_values_s"][1] - a["delay_values_s"][0])
        res_hz = 1.0 / kw["tx_overlap_s"]
        hz_per_row = -(a["dlt_shifts"][1] - a["dlt_shifts"][0]) * kw["f_hz"]
        chip_s = kw["chip_samples"] / fs
        M, edge_lo, edge_hi, _ = dea.edge_model_image(
            n_rows, a["dlt_shifts"], a["delay_values_s"], a["lt_ref"], a["rx_time_s"],
            delay_range_s=chip_s, doppler_range_hz=3 * res_hz, **kw)
        over = dea.EDGE_MODEL_OVERSAMPLE
        N0 = np.exp(2 * dea.noise_floor_log_amp(log_A))
        # Echo scale as the real image's: its total power above the floor.
        P_real = np.exp(2.0 * log_A.astype(np.float64))
        scale = (P_real.sum() - N0 * P_real.size) / (M[::over].sum())
        mid = np.where(np.isfinite(edge_lo) & np.isfinite(edge_hi), (edge_lo + edge_hi) / 2, 0)
        # Correlated speckle: amplitude responses along each axis.
        k = np.arange(-int(4 * res_hz / abs(hz_per_row)) - 1, int(4 * res_hz / abs(hz_per_row)) + 2)
        sinc_amp = np.sinc(k * abs(hz_per_row) / res_hz)
        tri = np.clip(1 - np.abs(np.arange(-int(kw["chip_samples"]), int(kw["chip_samples"]) + 1))
                      / kw["chip_samples"], 0, None)
        for delay_us, dop_cells in CASES:
            dop_hz = dop_cells * res_hz
            y = (np.arange(n_rows)[:, None] * over - dop_hz / hz_per_row * over
                 + np.zeros((1, n_cols)))
            x = np.arange(n_cols)[None, :] - delay_us * 1e-6 * fs + np.zeros((n_rows, 1))
            mu = scipy.ndimage.map_coordinates(M, [y, x], order=1, mode="constant")
            # Brightness: along delay, independently per limb; across Doppler.
            side_hi = np.arange(n_rows)[:, None] > mid[None, :]
            along = np.where(side_hi, smooth_field(rng, n_cols, 3 * kw["chip_samples"], 0.3)[None, :],
                             smooth_field(rng, n_cols, 3 * kw["chip_samples"], 0.3)[None, :])
            across = smooth_field(rng, n_rows, 20, 0.2)[:, None]
            mu = scale * mu * along * across + N0
            z = rng.standard_normal((n_rows, n_cols)) + 1j * rng.standard_normal((n_rows, n_cols))
            z = scipy.signal.fftconvolve(z, sinc_amp[:, None], mode="same", axes=0)
            z = scipy.signal.fftconvolve(z, tri[None, :], mode="same", axes=1)
            speckle = np.abs(z) ** 2
            speckle /= speckle.mean()
            synth = (0.5 * np.log(mu * speckle)).astype(np.float32)
            res = dea.fit_edge_offsets(
                synth, a["dlt_shifts"], a["delay_values_s"], a["lt_ref"], a["rx_time_s"],
                noise_log_amp=dea.noise_floor_log_amp(synth), **kw)
            d_err = res["delay_s"] * 1e6 - delay_us
            o_err = (res["doppler_hz"] - dop_hz) * 1e3
            rec = {"look": f"{args.station} {hms}", "capture_s": kw["rx_duration_s"],
                   "status": res["status"], "injected_delay_us": delay_us,
                   "injected_doppler_mhz": dop_hz * 1e3,
                   "delay_error_us": d_err, "delay_jackknife_us": res["delay_err_s"] * 1e6,
                   "doppler_error_mhz": o_err, "doppler_jackknife_mhz": res["doppler_err_hz"] * 1e3}
            out.append(rec)
            print(f"{args.station} {hms} ({kw['rx_duration_s']:.0f} s): injected {delay_us:+5.1f} us, "
                  f"{dop_hz * 1e3:+6.1f} mHz -> {res['status']}; error {d_err:+.2f} us "
                  f"({d_err / (res['delay_err_s'] * 1e6):+.1f} jackknife), {o_err:+.2f} mHz "
                  f"({o_err / (res['doppler_err_hz'] * 1e3):+.1f} jackknife)", flush=True)
    path = vc.write_json(f"edge_offsets_{os.path.basename(os.path.normpath(args.run_dir))}_"
                         f"{args.station}.json", {"cases": out, "seed": args.seed})
    print(f"wrote {vc.report_path(path)}")


if __name__ == "__main__":
    main()
