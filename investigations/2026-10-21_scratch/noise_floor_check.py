"""The 2025-09-11 looks whose recorded noise floor sat above most of their own
map (REPORT 3.1: eleven looks at 82-99 %), and the sign of their gain-band
median brightness, as stack_maps then computed it.

Ran 2026-10-21 on results/STOCKERT_2025 as it was before that day's fixes
(the strongest-peak timing rule; median gain). Reproduced here from the
inline check that was run then; it uses the interfaces of that time
(look_brightness's arithmetic is repeated inline) and needs that run.
"""
import csv
import re

import numpy as np
import healpy as hp

import stack_maps as sm

RUN = "results/STOCKERT_2025"
rows = list(csv.DictReader(open(f"{RUN}/registration_runs_stockert_chan1.csv")))
excluded = set(re.findall(r"^  - (\S+): ", open(f"{RUN}/stack_stockert_chan1.log").read(), re.M))
print("time      shift  noise_log  map_med  frac<floor  band_median  in_stack")
cache = {}
for r in rows:
    if not r["rx_file"].startswith("stockert_radar_2025_09_11"):
        continue
    for k in ("map_npy", "mult_npy", "area_npy"):
        r[k] = sm.resolve_npy(r[k], RUN)
    m = np.load(r["map_npy"])
    ok = m != hp.UNSEEN
    nl = float(r["noise_log_amp"])
    frac = (m[ok] < nl).mean()
    mult = np.load(r["mult_npy"])
    area = np.load(r["area_npy"])
    mask = ok & np.isfinite(area) & (area > 0)
    mask &= mult <= 3 * np.median(mult[mult > 0])
    cosi = sm.incidence_cos(sm.dea.csp.str2et(r["rx_start_utc"]), len(m), cache,
                            rx_name="STOCKERT")
    ref = np.median(m[mask])
    noise = np.exp(2 * (nl - ref))
    bright = (np.exp(2 * (m[mask] - ref)) - noise) / area[mask]
    lo, hi = np.cos(np.radians([50, 20]))
    band = (cosi[mask] >= lo) & (cosi[mask] <= hi)
    print(f"{r['rx_start_utc'][11:19]}  {float(r['applied_shift_samples']):6.1f}  {nl:7.2f}  "
          f"{np.median(m[ok]):7.2f}  {frac:8.2f}  {np.median(bright[band]):+.3e}  "
          f"{r['rx_file'] not in excluded}")
