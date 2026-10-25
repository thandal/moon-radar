import csv, json, os, sys
import numpy as np, healpy as hp
from astropy import time as at
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
import reproject_maps as rm
from trace_maps_helpers import *
import doppler_equator_alignment as dea

dea.load_lola_dem()
r = [r for r in csv.DictReader(open(f"{RUN}/registration_runs_stockert_chan1.csv")) if r["rx_start_utc"][11:19] == "09:41:25"][0]
val, mult, area, vf, noise = rm.reproject_row(r, "data/eve2026_circ", "STOCKERT", 4)
apix = 4 * np.pi * 1737.4 ** 2 / hp.nside2npix(400)
ok = val != hp.UNSEEN
print(f"pixel area {apix:.1f} km^2; cell area (both sides) median {np.nanmedian(area[ok]):.2f} km^2, 1/99% {np.nanpercentile(area[ok], 1):.2f}/{np.nanpercentile(area[ok], 99):.1f}")
big = ok & (mult >= 3)
ratio = (area[big] / apix) / mult[big]
print(f"where cells are >= 3 pixels: (area / pixel area) / count-based footprint: median {np.median(ratio):.3f}, 10/90% {np.percentile(ratio, 10):.2f}/{np.percentile(ratio, 90):.2f}  ({big.sum()} pixels)")
print(f"noise floor (log amplitude) {noise:.3f}; CSV map median {np.median(val[ok]):.3f}; noise / median intensity {np.exp(2 * (noise - np.median(val[ok]))):.3f}")

meta = json.load(open(os.path.join(DATA, r["rx_file"])))
t0 = de.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
s, w, u, geo = geometry("STOCKERT", t0, float(r["capture_s"]))
dist = np.abs(v_pix @ u) * 1737.4
inc = np.degrees(np.arccos(np.clip(v_pix @ s, -1, 1)))
mok = ok & (mult <= 3 * np.median(mult[mult > 0]))
I = np.exp(2.0 * val) - np.exp(2.0 * noise)
sig = I / area
sig /= np.median(sig[mok & (inc > 20) & (inc < 60)])
edges = [25, 50, 100, 200, 400, 800, 1600]
print("\nnoise-subtracted brightness per unit area, by distance from the Doppler-equator plane")
print(f"{'incidence':>12s} " + " ".join(f"{f'{a}-{b} km':>11s}" for a, b in zip(edges[:-1], edges[1:])))
for i0, i1 in ((20, 30), (30, 40), (40, 50), (50, 60)):
    cells = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = mok & (inc >= i0) & (inc < i1) & (dist >= a) & (dist < b)
        cells.append(f"{np.median(sig[sel]):11.2f}" if sel.sum() > 200 else f"{'':>11s}")
    print(f"{f'{i0}-{i1} deg':>12s} " + " ".join(cells))
