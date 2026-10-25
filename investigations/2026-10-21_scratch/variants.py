"""Plain incoherent opposite-sense stacks (same gating result and weighting as
stack_maps.py, without the scattering law) for subsets of looks."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, "/home/than/code/moon-radar")
import stack_maps as sm

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
OUT = "/tmp/eve_check/blur"
chan = sys.argv[1] if len(sys.argv) > 1 else "chan1"

looks = []
for st in ["stockert", "wsrt"]:
    for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_{chan}.csv")):
        snr = float(r.get("pair_snr") or r["tone_snr"])
        if snr < 15:
            continue
        t = r["rx_start_utc"][11:19]
        dur = 125 if "4x" in r["tx_file"] else 35
        looks.append(dict(st=st, t=t, dur=dur, chip50="fzc50000" in r["tx_file"],
                          map=os.path.join(RUN, os.path.basename(r["map_npy"])),
                          mult=os.path.join(RUN, os.path.basename(r["mult_npy"]))))

late = lambda l: l["t"] >= "12:00:00"
VARIANTS = {
    "S_all": lambda l: l["st"] == "stockert",
    "W_all": lambda l: l["st"] == "wsrt",
    "W_early": lambda l: l["st"] == "wsrt" and not late(l),
    "SW_all": lambda l: True,
    "SW_early": lambda l: not late(l),
    "SW_long": lambda l: l["dur"] == 125,
}
cache = {}
for name, sel in VARIANTS.items():
    chosen = [l for l in looks if sel(l)]
    acc = None
    for l in chosen:
        if l["map"] not in cache:
            cache[l["map"]] = sm.look_linear_intensity(l["map"], l["mult"]).astype(np.float32)
        I = cache[l["map"]].astype(np.float64)
        if acc is None:
            acc = sm.Accumulator(len(I))
        acc.add(I)
    np.save(f"{OUT}/var_{chan}_{name}.npy", acc.stacked_log(3))
    print(f"{name}: {len(chosen)} looks: " + " ".join(f"{l['st'][0]}{l['t'][:5]}" for l in chosen))
