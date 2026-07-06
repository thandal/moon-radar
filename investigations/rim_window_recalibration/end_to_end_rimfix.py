"""End-to-end process_one verification of the rim-window fix on 3 looks."""
import csv
import os
import sys

REPO = "/home/than/code/moon-radar"
sys.path.insert(0, REPO)
os.chdir(REPO)

import registration_stability as rs

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRATCH, "e2e_rimfix_out")
DATA_ROOT = os.path.join(REPO, "data.camras.nl/lunar-radar")
FROZEN_CSV = os.path.join(REPO, "results/LOLA_DEM_REGISTRATION/registration_runs_chan1.csv")
frozen = {r["rx_file"]: r for r in csv.DictReader(open(FROZEN_CSV))}

LOOKS = [
    ("2025-09-16", "stockert_radar_2025_09_16_14_06_11_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta"),
    ("2025-09-10", "stockert_radar_2025_09_11_08_05_44_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta"),
    ("2025-06-21", "stockert_eme_2025_06_21_11_18_31_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta"),
]

for sess, base in LOOKS:
    path = os.path.join(DATA_ROOT, sess, base)
    row = rs.process_one((path, DATA_ROOT, "STOCKERT", 400, OUT_DIR, None))
    fr = frozen[base]
    tot_new = float(row["applied_df_hz"]) + float(row["rim_delta_hz"])
    tot_frozen = (float(fr["applied_df_hz"]) + float(fr["rim_delta_hz"]))
    print(f"\n{base[:40]}")
    print(f"  frozen : delta {float(fr['rim_delta_hz'])*1e3:+8.2f} mHz, "
          f"spread {float(fr['rim_spread_hz'] or 'nan')*1e3:+7.2f}, "
          f"rim_n {fr['rim_n']:>4s}, total {tot_frozen*1e3:+8.2f} mHz")
    print(f"  rimfix : delta {float(row['rim_delta_hz'])*1e3:+8.2f} mHz, "
          f"spread {float(row['rim_spread_hz'])*1e3:+7.2f} (first "
          f"{float(row['rim_spread_first_hz'])*1e3:+7.2f}), "
          f"rim_n {row['rim_n']:>4}, total {tot_new*1e3:+8.2f} mHz, "
          f"resid {float(row['rim_residual_hz'])*1e3:+6.2f} mHz")
    print(f"  shift {float(row['shift_refined']):+9.4f} vs frozen "
          f"{float(fr['shift_refined']):+9.4f} samples; "
          f"df {float(row['df_centroid_hz'])*1e3:+8.2f} vs "
          f"{float(fr['df_centroid_hz'])*1e3:+8.2f} mHz")
