"""Verify the rim coarse re-acquisition (rim_seed_search).

1. 09-16 14:04:14 — tone deep fade (+270 mHz): expect rescue, delta ~ -290,
   total ~ session trend (-20 mHz).
2. 06-21 10:44:00 — no-signal capture (tone SNR 4): expect rim_status
   "failed" (no spurious lock on noise; process_file then applies nothing
   and leaves rim_delta_hz blank).
3. 09-16 14:06:11 — regression: row must match the 2026-07-03 RIMFIX values.

Written 2026-07-04. The "yesterday" reference is that July run's runs CSV
(results/LOLA_DEM_REGISTRATION/registration_runs_chan1.csv, named before
runs CSVs carried the station; see the directory-rename note in
../rim_window_recalibration_2026-07-03.md). It predates rim_status: there
rim_n 0 meant no rim measured. This run's rows come from today's
registration_stability.process_one. The script prints both; it checks
nothing itself.
"""
import csv
import os
import sys
import time

REPO = "/home/than/code/moon-radar"
sys.path.insert(0, REPO)
os.chdir(REPO)

import registration_stability as rs

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRATCH, "seed_rescue_out")
DATA_ROOT = os.path.join(REPO, "data.camras.nl/lunar-radar")
RIMFIX = {r["rx_file"]: r for r in csv.DictReader(
    open("results/LOLA_DEM_REGISTRATION/registration_runs_chan1.csv"))}

LOOKS = [
    ("2025-09-16", "stockert_radar_2025_09_16_14_04_14", "RESCUE TARGET (trend -19.8 mHz)"),
    ("2025-06-21", "stockert_eme_2025_06_21_10_44_00", "NO-SIGNAL (must stay unmeasured)"),
    ("2025-09-16", "stockert_radar_2025_09_16_14_06_11", "REGRESSION (must match RIMFIX)"),
]

for sess, stem, label in LOOKS:
    base = stem + "_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta"
    prev = RIMFIX[base]
    t0 = time.time()
    # max_shift 40: the batch default (REPORT 8.3); supersample 1: one
    # image cell per map pixel, the projection of these runs (only the rim
    # columns are compared here).
    row = rs.process_one((os.path.join(DATA_ROOT, sess, base), DATA_ROOT,
                          "STOCKERT", 400, OUT_DIR, None, 40, False, 1))
    dt = time.time() - t0
    tot_prev = (float(prev["applied_df_hz"]) + float(prev["rim_delta_hz"])) * 1e3
    print(f"\n{stem}  [{label}]  ({dt:.0f} s)")
    print(f"  yesterday: delta {float(prev['rim_delta_hz'])*1e3:+8.1f} mHz, "
          f"rim_n {prev['rim_n']:>3s}, total {tot_prev:+8.1f} mHz")
    seed = row["rim_seed_hz"]
    seed_text = "none" if seed == "" else f"{float(seed) * 1e3:+.1f} mHz"
    if row["rim_status"] == "failed":
        # No pass got through the gates: process_file applied nothing and
        # left rim_delta_hz blank -- the expected outcome for the no-signal
        # look.
        print(f"  with seed: rim {row['rim_status']} (nothing applied), "
              f"rim_n {row['rim_n']:>3}, coarse-search seed {seed_text}")
        continue
    tot = (float(row["applied_df_hz"]) + float(row["rim_delta_hz"])) * 1e3
    print(f"  with seed: rim {row['rim_status']}, "
          f"delta {float(row['rim_delta_hz'])*1e3:+8.1f} mHz, "
          f"rim_n {row['rim_n']:>3}, total {tot:+8.1f} mHz, "
          f"coarse-search seed {seed_text}, "
          f"spread {float(row['rim_spread_hz'])*1e3 if row['rim_spread_hz'] != '' else float('nan'):+6.1f}, "
          f"resid {float(row['rim_residual_hz'])*1e3 if row['rim_residual_hz'] != '' else float('nan'):+5.2f} mHz")
