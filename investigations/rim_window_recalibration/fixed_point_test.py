"""Two-fixed-point test on the 06-21 trio (+ healthy controls).

For each look: converge the rim iteration from zero (production start,
solution A), run the coarse sweep to find the best-quality trial, converge
from there (solution B), and compare converged gate-passing counts n.
If the true fixed point wins on n for the trio while A==B on controls,
"converge from both starts, keep the higher-n solution" is a safe rescue.
"""
import csv
import os
import sys

import numpy as np
from astropy import units as au

REPO = "/home/than/code/moon-radar"
sys.path.insert(0, REPO)
os.chdir(REPO)

import doppler_equator_alignment as dea

DATA_ROOT = os.path.join(REPO, "data.camras.nl/lunar-radar")
RUNS = {r["rx_file"]: r for r in csv.DictReader(
    open("results/LOLA_DEM_REGISTRATION/registration_runs_chan1.csv"))}

LOOKS = [  # (stem, label); trend +5.3 mHz
    ("stockert_eme_2025_06_21_10_09_25", "FLAGGED total +70.4"),
    ("stockert_eme_2025_06_21_10_14_34", "FLAGGED total +48.4"),
    ("stockert_eme_2025_06_21_10_17_40", "FLAGGED total -29.5"),
    ("stockert_eme_2025_06_21_10_11_46", "control  total  +9.8"),
    ("stockert_eme_2025_06_21_10_26_29", "control  total  +2.3"),
    ("stockert_eme_2025_06_21_09_58_25", "control  total  +0.6"),
]


def converge(args, dlt_shifts, cap, start):
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    delta, conv = start, None
    for _ in range(12):
        r = dea.measure_rim_offset(args[0], dlt_shifts - delta, *args[1:],
                                   delta_capture_dlt=cap)
        if r is None:
            break
        conv = r
        delta += r["delta_dlt"]
        if abs(r["delta_dlt"]) < 0.5 * ddlt:
            break
    if conv is None:
        return None
    final = dea.measure_rim_offset(args[0], dlt_shifts - delta, *args[1:],
                                   delta_capture_dlt=cap)
    conv = final or conv
    return delta, conv["n_up"] + conv["n_down"], conv["spread_dlt"]


for stem, label in LOOKS:
    base = stem + "_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta"
    r = RUNS[base]
    (rx, tx, sample_rate, frequency, rx_start, _t, _f) = \
        dea.load_observation(os.path.join(DATA_ROOT, "2025-06-21", base), DATA_ROOT)
    fs = sample_rate.to_value(au.Hz)
    f_hz = frequency.to_value(au.Hz)
    rx_duration_s = len(rx) / fs
    log_A, dlt_shifts, dvs, lt_min_image, dlt_rate_srp = dea.compute_dd_image(
        rx, tx, sample_rate, frequency, rx_start, rx_start + 1.0 * au.s,
        float(r["applied_shift_samples"]) / fs, 0.0, "DWINGELOO", "STOCKERT",
        freq_offset_hz=float(r["applied_df_hz"]))
    del rx, tx
    rx_time_s = dea.et_from_astropy(rx_start)
    lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = \
        dea.compute_doppler_equator_velocity(
            rx_time_s, n_points=500, rx_duration_s=rx_duration_s,
            dlt_rate_srp=dlt_rate_srp, tx_name="DWINGELOO", rx_name="STOCKERT")
    args = (log_A, dvs, lt_min_image, lt_min_eq,
            delay_up, dlt_up, delay_down, dlt_down)
    cap = 0.055 / f_hz
    df = float(r["applied_df_hz"]) * 1e3

    A = converge(args, dlt_shifts, cap, 0.0)
    # sweep (trial, n) landscape
    trials = []
    for k in range(1, 13):
        for sgn in (+1, -1):
            t = sgn * k * cap
            m = dea.measure_rim_offset(log_A, dlt_shifts - t, *args[1:],
                                       delta_capture_dlt=cap)
            if m is not None:
                trials.append((t + m["delta_dlt"], m["n_up"] + m["n_down"]))
    B = None
    if trials:
        seed = max(trials, key=lambda x: x[1])[0]
        B = converge(args, dlt_shifts, cap, seed)

    def fmt(sol):
        if sol is None:
            return "None"
        d, n, sp = sol
        return (f"delta {-d*f_hz*1e3:+7.1f} mHz total {df - d*f_hz*1e3:+7.1f} "
                f"n {n:3d} spread {-sp*f_hz*1e3:+6.1f}")
    print(f"\n{stem[13:]}  [{label}]  df {df:+6.1f}")
    print(f"  A (start 0):    {fmt(A)}")
    print(f"  B (seed start): {fmt(B)}")
