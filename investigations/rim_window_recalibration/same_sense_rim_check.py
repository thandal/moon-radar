"""Same-sense delta inheritance spot-check (DATA_MACHINE_TODO section 3).

Run the rim estimator with relaxed gates on the strongest same-sense (chan0)
looks and compare the measured delta against the inherited opposite-sense value
(REPORT section 3.2 assumption, previously untested).

HISTORICAL (July 2026): the record behind
../rim_window_recalibration_2026-07-03.md, not a check of current
production. It differs from today's calibration (doppler_equator_alignment.
calibrate_rim, as process_file runs it) in:
  - delay reference: the rims are placed from the smooth-sphere SRP light
    time (lt_min_image); production places them from the terrain under the
    SRP (lt_min_image - srp_topo_delay_s, extract_srp_elevation);
  - no final re-measure and no convergence status: a delta is printed when
    12 passes or a later pass failing the gates end the crawl, whether or
    not its last pass moved under half a row; no coarse-search fallback;
  - gates: besides production's (contrast 0.4, 30 columns per branch) it
    tries relaxed ones (0.2 / 10 and 0.1 / 10) on purpose.
CH0 is the archived 2026-06-12 run (see the directory-rename note in the
log).
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
CH0 = {r["rx_file"]: r for r in csv.DictReader(
    open("results/LOLA_DEM_REGISTRATION_FROZEN_0612/registration_runs_chan0.csv"))}

# (session dir, chan0 base, corrected opposite-sense delta mHz from the 2026-07-03
# rimfix end-to-end runs where known, else None -> frozen value stands)
LOOKS = [
    ("2025-09-10", "stockert_radar_2025_09_11_05_53_51", None),
    ("2025-09-10", "stockert_radar_2025_09_11_07_53_44", None),
    ("2025-09-10", "stockert_radar_2025_09_11_08_07_50", None),
    ("2025-06-21", "stockert_eme_2025_06_21_11_18_31", 140.9),
]

for sess, stem, corrected in LOOKS:
    base = stem + "_1299.500MHz_0.25Msps_ci16_le.chan0.sigmf-meta"
    r = CH0[base]
    path = os.path.join(DATA_ROOT, sess, base)
    (rx, tx, sample_rate, frequency, rx_start, _t, _f) = \
        dea.load_observation(path, DATA_ROOT)
    fs = sample_rate.to_value(au.Hz)
    f_hz = frequency.to_value(au.Hz)
    rx_duration_s = len(rx) / fs
    log_A, dlt_shifts, delay_values_s, lt_min_image, dlt_rate_srp, _overlap = \
        dea.compute_dd_image(rx, tx, sample_rate, frequency, rx_start,
                             dea.nominal_tx_emit_start(rx_start),
                             float(r["applied_shift_samples"]) / fs, 0.0,
                             "DWINGELOO", "STOCKERT",
                             freq_offset_hz=float(r["applied_df_hz"]))
    del rx, tx
    rx_time_s = dea.et_from_astropy(rx_start)
    lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = \
        dea.compute_doppler_equator_velocity(
            rx_time_s, n_points=500, rx_duration_s=rx_duration_s,
            dlt_rate_srp=dlt_rate_srp, tx_name="DWINGELOO", rx_name="STOCKERT")
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    cap = 0.055 / f_hz
    results = {}
    for gates in ((0.4, 30), (0.2, 10), (0.1, 10)):
        delta_dlt, rim = 0.0, None
        for _ in range(12):
            m = dea.measure_rim_offset(log_A, dlt_shifts - delta_dlt,
                                       delay_values_s, lt_min_image, lt_min_eq,
                                       delay_up, dlt_up, delay_down, dlt_down,
                                       min_contrast=gates[0],
                                       min_samples=gates[1],
                                       delta_capture_dlt=cap)
            if m is None:
                break
            rim = m
            delta_dlt += m["delta_dlt"]
            if abs(m["delta_dlt"]) < 0.5 * ddlt:
                break
        results[gates] = (None if rim is None else
                          (-delta_dlt * f_hz * 1e3,
                           -rim["spread_dlt"] * f_hz * 1e3,
                           rim["n_up"] + rim["n_down"]))
    inh = float(r["rim_delta_hz"]) * 1e3
    print(f"\n{stem} (pair_snr {float(r['pair_snr']):.0f})")
    print(f"  inherited opposite-sense delta: frozen {inh:+7.1f} mHz"
          + (f", rimfix-corrected {corrected:+7.1f} mHz" if corrected else ""))
    for g, v in results.items():
        if v is None:
            print(f"  gates contrast>{g[0]}, n>={g[1]}: rim not measurable")
        else:
            print(f"  gates contrast>{g[0]}, n>={g[1]}: measured "
                  f"{v[0]:+7.1f} mHz (spread {v[1]:+6.1f}, n {v[2]})")
