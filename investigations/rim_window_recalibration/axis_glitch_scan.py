"""DATA_MACHINE_TODO section 2: legacy SRP-velocity axis glitch scan.

At every recorded look epoch (chan1 runs CSV), compare srp_velocity_analytic
against the legacy 1 s forward difference of the quantized specular zoom
(srp_velocity_fd, refined=False). Flag looks where the legacy axis deviates;
cross-reference their recorded rim spread.
"""
import csv
import os
import sys

import numpy as np
from astropy.time import Time

REPO = "/home/than/code/moon-radar"
sys.path.insert(0, REPO)
os.chdir(REPO)

import doppler_equator as de
import doppler_equator_alignment as dea

CSV = os.path.join(REPO, "results/LOLA_DEM_REGISTRATION_FROZEN_0612/registration_runs_chan1.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axis_glitch_scan.csv")

rows = list(csv.DictReader(open(CSV)))
out = []
for r in rows:
    et = dea.et_from_astropy(Time(r["rx_start_utc"], scale="utc"))
    v_a, ax_a, _ = de.srp_velocity_analytic(et)
    v_l, ax_l, _ = de.srp_velocity_fd(et, dt=1.0, refined=False)
    ang = np.degrees(np.arccos(np.clip(np.dot(ax_a, ax_l), -1, 1)))
    sp = (np.linalg.norm(v_l) / np.linalg.norm(v_a) - 1) * 100
    out.append({
        "rx_file": r["rx_file"], "rx_start_utc": r["rx_start_utc"],
        "axis_angle_deg": round(float(ang), 4),
        "speed_err_pct": round(float(sp), 3),
        "rim_spread_mhz": (round(float(r["rim_spread_hz"]) * 1e3, 2)
                           if r["rim_spread_hz"] else None),
        "rim_n": int(r["rim_n"]),
    })

ang = np.array([o["axis_angle_deg"] for o in out])
spd = np.array([o["speed_err_pct"] for o in out])
print(f"n={len(out)} looks")
print(f"axis angle deg: median {np.median(ang):.4f}, p95 {np.percentile(ang,95):.4f}, "
      f"max {ang.max():.4f}")
print(f"legacy speed err %: median {np.median(np.abs(spd)):.3f}, "
      f"max {np.abs(spd).max():.3f}")
bad = [o for o in out if o["axis_angle_deg"] > 0.5]
print(f"\nlooks with legacy axis error > 0.5 deg: {len(bad)}")
for o in bad:
    print(f"  {o['rx_start_utc']}  angle {o['axis_angle_deg']:7.3f} deg, "
          f"speed {o['speed_err_pct']:+.2f}%, spread "
          f"{o['rim_spread_mhz']} mHz, rim_n {o['rim_n']}")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print("\nwrote", OUT)
