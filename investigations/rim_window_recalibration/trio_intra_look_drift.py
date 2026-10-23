"""Targeted intra-look drift: the 06-21 trio vs controls.

Half-window DD images + independent rim calibrations (production's 55 mHz
window anchor, 12-pass cap and coarse-search fallback, but requiring only
15 gate-passing columns per branch where production requires 30).
Large |delta_h2 - delta_h1| on the trio with small values on controls =
bursty intra-look clock drift, which would explain both the anomalous
positive spreads / low n (smeared rims) and the off-trend look totals (real
look-mean offsets).

HISTORICAL (July 2026): the record behind
../rim_window_recalibration_2026-07-03.md, not a check of current
production. It differs from today's calibration (doppler_equator_alignment.
calibrate_rim, as process_file runs it) in:
  - delay reference: the rims are placed from the smooth-sphere SRP light
    time (lt_min); production places them from the terrain under the SRP
    (lt_min - srp_topo_delay_s, extract_srp_elevation);
  - no final re-measure at the applied offset. Each calibration now reports
    its status (printed in the last column): "converged" when its last pass
    moved under half a row, "not_converged" when 12 passes (or a later pass
    failing the gates) ended it first -- its delta is then the sum so far --
    and "failed" when no pass got through the gates;
  - min_samples 15 (production 30), and its own copy of the crawl and seed
    logic;
  - images of half a look (half the integration time).
RUNS is the July run's CSV (see the directory-rename note in the log).
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

LOOKS = [
    ("stockert_eme_2025_06_21_10_09_25", "FLAGGED +70.4"),
    ("stockert_eme_2025_06_21_10_14_34", "FLAGGED +48.4"),
    ("stockert_eme_2025_06_21_10_17_40", "FLAGGED -29.5"),
    ("stockert_eme_2025_06_21_10_11_46", "control  +9.8"),
    ("stockert_eme_2025_06_21_10_26_29", "control  +2.3"),
    ("stockert_eme_2025_06_21_09_58_25", "control  +0.6"),
]


def rim_delta(log_A, dlt_shifts, dvs, lt_min, eq, f_hz, parity=None):
    """(delta_dlt, last passing measurement, status): status "converged"
    (the last pass moved under half a row), "not_converged" (the crawl
    stopped first; delta is the sum so far) or "failed" (no pass got
    through the gates; delta and measurement None)."""
    lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = eq
    args = (dvs, lt_min, lt_min_eq, delay_up, dlt_up, delay_down, dlt_down)
    cap = 0.055 / f_hz
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    delta, rim, seeded, converged = 0.0, None, False, False
    for _ in range(12):
        r = dea.measure_rim_offset(log_A, dlt_shifts - delta, *args,
                                   min_samples=15, col_parity=parity,
                                   delta_capture_dlt=cap)
        if r is None:
            if rim is None and not seeded:
                seeded = True
                seed = dea.rim_seed_search(log_A, dlt_shifts, *args,
                                           delta_capture_dlt=cap,
                                           min_samples=15, col_parity=parity)
                if seed is not None:
                    delta = seed["seed_dlt"]
                    continue
            break
        rim = r
        delta += r["delta_dlt"]
        if abs(r["delta_dlt"]) < 0.5 * ddlt:
            converged = True
            break
    if rim is None:
        return None, None, "failed"
    return delta, rim, "converged" if converged else "not_converged"


print(f"{'look':22s} {'label':14s} {'d_h1':>7s} {'d_h2':>7s} {'drift':>7s} "
      f"{'noise':>6s} {'sp_h1':>6s} {'sp_h2':>6s} {'n_h1':>4s} {'n_h2':>4s}  "
      "calibrations not converged")
for stem, label in LOOKS:
    base = stem + "_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta"
    r = RUNS[base]
    (rx, tx, fs_q, freq, rx_start, _t, _f) = dea.load_observation(
        os.path.join(DATA_ROOT, "2025-06-21", base), DATA_ROOT)
    fs = fs_q.to_value(au.Hz)
    f_hz = freq.to_value(au.Hz)
    et0 = dea.et_from_astropy(rx_start)
    tx_abs = (dea.et_from_astropy(dea.nominal_tx_emit_start(rx_start))
              + float(r["applied_shift_samples"]) / fs)
    df = float(r["applied_df_hz"])
    n = len(rx)
    half = n // 2

    out, status = [], {}
    for h in (0, 1):
        rx_h = rx[h * half:(h + 1) * half]
        et_h = et0 + h * half / fs
        # Stockert looks (Dwingeloo transmitting).
        log_A, dlt_shifts, dvs, lt_min, rate, _overlap = dea.compute_dd_image(
            rx_h, tx, fs_q, freq, et_h, tx_abs, 0.0, 0.0,
            dea.TX_STATION, dea.STATION_RX["stockert"], freq_offset_hz=df)
        eq = dea.compute_doppler_equator_velocity(
            et_h, n_points=500, rx_duration_s=half / fs, dlt_rate_srp=rate,
            tx_name=dea.TX_STATION, rx_name=dea.STATION_RX["stockert"])
        d, rim, status[f"h{h + 1}"] = rim_delta(log_A, dlt_shifts, dvs, lt_min,
                                                eq, f_hz)
        de_, _, status[f"h{h + 1} even"] = rim_delta(log_A, dlt_shifts, dvs, lt_min,
                                                     eq, f_hz, parity=0)
        do_, _, status[f"h{h + 1} odd"] = rim_delta(log_A, dlt_shifts, dvs, lt_min,
                                                    eq, f_hz, parity=1)
        out.append((d, rim, de_, do_))
    del rx, tx

    def mhz(v):
        return "  ---" if v is None else f"{-v * f_hz * 1e3:+7.1f}"
    (d1, r1, e1, o1), (d2, r2, e2, o2) = out
    drift = (None if d1 is None or d2 is None
             else (-d2 * f_hz + d1 * f_hz) * 1e3)
    noise = (None if None in (e1, o1, e2, o2)
             else np.hypot((e1 - o1) * f_hz / 2, (e2 - o2) * f_hz / 2) * 1e3)
    print(f"{stem[13:]:22s} {label:14s} {mhz(d1):>7s} {mhz(d2):>7s} "
          f"{drift if drift is None else format(drift, '+7.1f'):>7} "
          f"{noise if noise is None else format(noise, '6.1f'):>6} "
          f"{'---' if r1 is None else format(-r1['spread_dlt']*f_hz*1e3, '+6.1f'):>6} "
          f"{'---' if r2 is None else format(-r2['spread_dlt']*f_hz*1e3, '+6.1f'):>6} "
          f"{'---' if r1 is None else r1['n_up']+r1['n_down']:>4} "
          f"{'---' if r2 is None else r2['n_up']+r2['n_down']:>4}  "
          + (", ".join(f"{k} {v}" for k, v in status.items() if v != "converged")
             or "none"), flush=True)
