"""
Intra-look Doppler drift experiment.

Splits looks into two half-windows, computes an independent DD image and an
independent rim calibration for each half, and compares the within-look
drift (delta_2 - delta_1) against (a) the rim measurement noise, estimated
from even/odd delay-column splits, and (b) the look-to-look delta scatter
(+/-47 mHz). Uses the 66 s 2025-06-21 looks (longest windows).

Usage (from notebooks/):
    ../.conda/bin/python intra_look_drift.py --n-files 6
"""

import argparse
import csv
import os

import numpy as np
from astropy import units as au

import doppler_equator_alignment as dea


def rim_delta_for(log_A, dlt_shifts, dvs, lt_min, eq, parity=None):
    """Iterative rim calibration; returns delta_dlt or None."""
    lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = eq
    delta = 0.0
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    for _ in range(3):
        r = dea.measure_rim_offset(log_A, dlt_shifts - delta, dvs, lt_min,
                                   lt_min_eq, delay_up, dlt_up, delay_down,
                                   dlt_down, min_samples=15, col_parity=parity)
        if r is None:
            return None
        delta += r["delta_dlt"]
        if abs(r["delta_dlt"]) < 0.5 * ddlt:
            break
    return delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=6)
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), "data.camras.nl/lunar-radar")
    run_dir = os.path.join(os.path.dirname(__file__), "results/REGISTRATION")
    rows = [r for r in csv.DictReader(open(os.path.join(run_dir, "registration_runs_chan1.csv")))
            if "2025_06_21" in r["rx_file"] and float(r["tone_snr"]) >= 15
            and int(r["rim_n"]) > 0]
    rows = rows[:args.n_files]
    print(f"{len(rows)} looks (66 s windows, halves of ~33 s)")

    print(f"{'look':44s} {'d_full':>8s} {'d_h1':>8s} {'d_h2':>8s} "
          f"{'drift':>8s} {'noise':>7s}")
    drifts, noises, fulls = [], [], []
    for r in rows:
        path = os.path.join(root, "2025-06-21", r["rx_file"])
        (rx, tx, fs_q, freq, rx_start, _t, _f) = dea.load_observation(path, root)
        fs = fs_q.to_value(au.Hz)
        f_hz = freq.to_value(au.Hz)
        et0 = dea.et_from_astropy(rx_start)
        tx_abs = et0 + 1.0 + float(r["applied_shift_samples"]) / fs
        df = float(r["applied_df_hz"])
        n = len(rx)
        half = n // 2

        deltas, sigmas = [], []
        for h in (0, 1):
            rx_h = rx[h * half:(h + 1) * half]
            et_h = et0 + h * half / fs
            log_A, dlt_shifts, dvs, lt_min, rate = dea.compute_dd_image(
                rx_h, tx, fs_q, freq, et_h, tx_abs, 0.0, 0.0,
                freq_offset_hz=df)
            T_h = half / fs
            eq = dea.compute_doppler_equator_velocity(
                et_h, n_points=500, rx_duration_s=T_h, dlt_rate_srp=rate)
            d = rim_delta_for(log_A, dlt_shifts, dvs, lt_min, eq)
            d_even = rim_delta_for(log_A, dlt_shifts, dvs, lt_min, eq, parity=0)
            d_odd = rim_delta_for(log_A, dlt_shifts, dvs, lt_min, eq, parity=1)
            if d is None or d_even is None or d_odd is None:
                deltas = None
                break
            deltas.append(-d * f_hz)
            sigmas.append(abs(d_even - d_odd) * f_hz / 2)
        if deltas is None:
            print(f"{r['rx_file'][:44]:44s}  rim too weak in a half-window")
            continue
        d_full = float(r["rim_delta_hz"])
        drift = deltas[1] - deltas[0]
        noise = np.hypot(*sigmas)
        drifts.append(drift)
        noises.append(noise)
        fulls.append(d_full)
        print(f"{r['rx_file'][:44]:44s} {d_full:+8.4f} {deltas[0]:+8.4f} "
              f"{deltas[1]:+8.4f} {drift:+8.4f} {noise:7.4f}")

    drifts = np.array(drifts)
    noises = np.array(noises)
    print(f"\nintra-look drift: median |drift| = {np.median(np.abs(drifts))*1000:.1f} mHz, "
          f"rms = {drifts.std()*1000:.1f} mHz")
    print(f"measurement noise (col-split): median = {np.median(noises)*1000:.1f} mHz")
    excess = np.sqrt(max(drifts.std() ** 2 - np.median(noises) ** 2, 0))
    print(f"noise-corrected intra-look wander over T/2: ~{excess*1000:.1f} mHz")
    print(f"look-to-look scatter (all looks, for scale): 47 mHz")


if __name__ == "__main__":
    main()
