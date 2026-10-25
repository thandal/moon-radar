"""Does map brightness follow the delay-Doppler cell area (~1/distance from
the Doppler-equator plane)? Median-normalized intensity by distance from the
plane, within narrow incidence bands so the scattering law is held fixed."""
import csv, json, os, sys
import numpy as np, healpy as hp
from astropy import time as at
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
from trace_maps_helpers import *

rows = [(st, r) for st in ("stockert", "wsrt")
        for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_chan1.csv"))]
r = [r for st, r in rows if st == "stockert" and r["rx_start_utc"][11:19] == "09:41:25"][0]
meta = json.load(open(os.path.join(DATA, r["rx_file"])))
t0 = de.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
s, w, u, geo = geometry("STOCKERT", t0, float(r["capture_s"]))
dist = np.abs(v_pix @ u) * 1737.4
inc = np.degrees(np.arccos(np.clip(v_pix @ s, -1, 1)))
m = np.load(r["map_npy"]); mu = np.load(r["mult_npy"])
ok = (m != hp.UNSEEN) & (mu <= 3 * np.median(mu[mu > 0]))
I = np.exp(2.0 * (m - np.median(m[ok])))
edges = [25, 50, 100, 200, 400, 800, 1600]
print("single look Stockert 09:41, opposite sense: median intensity (relative to the whole map) by")
print("distance from the Doppler-equator plane, in incidence bands")
print(f"{'incidence':>12s} " + " ".join(f"{f'{a}-{b} km':>11s}" for a, b in zip(edges[:-1], edges[1:])))
for i0, i1 in ((20, 30), (30, 40), (40, 50), (50, 60)):
    cells = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = ok & (inc >= i0) & (inc < i1) & (dist >= a) & (dist < b)
        cells.append(f"{np.median(I[sel]):11.2f}" if sel.sum() > 200 else f"{'':>11s}")
    print(f"{f'{i0}-{i1} deg':>12s} " + " ".join(cells))
print("expected if brightness ~ 1/distance: x2 per column")
