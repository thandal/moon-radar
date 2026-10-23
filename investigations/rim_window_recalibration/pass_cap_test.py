"""Pass-cap test on the two crawl-limited looks (cached DD images).

HISTORICAL (July 2026): the record behind
../rim_window_recalibration_2026-07-03.md, not a check of current
production. It differs from today's calibration (doppler_equator_alignment.
calibrate_rim, as process_file runs it) in:
  - delay reference: the rims are placed from the smooth-sphere SRP light
    time (lt_min_image, as dd_cache/ stores it); production places them
    from the terrain under the SRP (lt_min_image - srp_topo_delay_s,
    extract_srp_elevation);
  - no final re-measure and no convergence status: it prints "converged"
    when a pass moves under half a row and nothing when the 12 passes run
    out first; no coarse-search fallback (rim_seed_search) when the first
    pass fails the gates;
  - its own copy of the 55 mHz row scaling (production passes
    delta_capture_dlt = RIM_CAPTURE_HZ / f0 to measure_rim_offset).
It does not make the DD images: it reads those rim_window_variants.py
cached in dd_cache/. FROZEN_CSV is the archived 2026-06-12 run (see the
directory-rename note in the log).
"""
import csv
import os
import sys

import numpy as np

REPO = "/home/than/code/moon-radar"
sys.path.insert(0, REPO)
os.chdir(REPO)
import doppler_equator_alignment as dea

SCRATCH = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRATCH, "dd_cache")
FROZEN_CSV = os.path.join(REPO, "results/LOLA_DEM_REGISTRATION_FROZEN_0612/registration_runs_chan1.csv")
frozen = {r["rx_file"]: r for r in csv.DictReader(open(FROZEN_CSV))}

LOOKS = ["stockert_radar_2025_09_16_13_57_38_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta",
         "stockert_radar_2025_09_11_08_05_44_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta"]

for base in LOOKS:
    d = np.load(os.path.join(CACHE, base + ".npz"))
    f_hz = float(d["f_hz"])
    dlt_shifts = d["dlt_shifts"]
    ddlt = dlt_shifts[1] - dlt_shifts[0]
    pitch = ddlt * f_hz
    s = max(1.0, (0.055 / f_hz) / (10 * ddlt))
    inner = (int(np.ceil(10 * s)), int(np.ceil(50 * s)))
    outer = (int(np.ceil(30 * s)), int(np.ceil(90 * s)))
    args = (d["delay_values_s"], float(d["lt_min_image"]), float(d["lt_min_eq"]),
            d["delay_up"], d["dlt_up"], d["delay_down"], d["dlt_down"])
    applied_df = float(frozen[base]["applied_df_hz"])
    print(f"\n{base[15:34]} pitch {pitch*1e3:.2f} mHz, inner {inner}, outer {outer}, "
          f"applied_df {applied_df*1e3:+.1f} mHz")
    delta_dlt = 0.0
    for it in range(12):
        r = dea.measure_rim_offset(d["log_A"], dlt_shifts - delta_dlt, *args,
                                   inner_off=inner, outer_off=outer)
        if r is None:
            print(f"  pass {it+1}: gated out")
            break
        delta_dlt += r["delta_dlt"]
        print(f"  pass {it+1}: step {-r['delta_dlt']*f_hz*1e3:+7.2f} mHz, "
              f"cum delta {-delta_dlt*f_hz*1e3:+8.2f} mHz, "
              f"total {(applied_df - delta_dlt*f_hz)*1e3:+8.2f} mHz, "
              f"spread {-r['spread_dlt']*f_hz*1e3:+7.2f}, n {r['n_up']}+{r['n_down']}")
        if abs(r["delta_dlt"]) < 0.5 * ddlt:
            print("  converged")
            break
