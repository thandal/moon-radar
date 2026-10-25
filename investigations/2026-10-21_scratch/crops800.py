"""Crater crops of the opposite- + same-sense stacks: nside 400 as the pipeline
makes it (one delay-Doppler cell per pixel), nside 800, and nside 800
averaged back to 400 (four cells per pixel). Cell-sharing mask scaled with
pixel area. usage: crops800.py OUT.png"""
import sys

import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

sys.path.insert(0, "/tmp/eve_check/blur")
sys.path.insert(0, "/home/than/code/moon-radar")
import compare800b as C
import registration_analysis as ra

CRATERS = [("Copernicus", -20.1, 9.6), ("Kepler", -38.0, 8.1),
           ("Bullialdus", -22.2, -20.7), ("Theophilus", 26.4, -11.4)]
HALF, STEP = 6.0, 0.04


def highpass(g, sigma_px):
    mask = np.isfinite(g)
    v = np.where(mask, g, 0.0)
    sv = scipy.ndimage.gaussian_filter(v, sigma_px)
    sw = scipy.ndimage.gaussian_filter(mask.astype(float), sigma_px)
    trend = np.where(sw > 0.05, sv / np.maximum(sw, 1e-9), np.nan)
    return np.where(mask, g - trend, np.nan)


L = {}
for chan in ("chan1", "chan0"):
    for v, run in (("400", C.R4), ("800", C.R8)):
        L.setdefault(v, {}).update({(k[0], k[1], chan): x for k, x in C.looks(run, chan).items()})
L["800->400"] = L["800"]
long_keys = [k for k in L["400"] if k[1] in C.LONG]
all_keys = list(L["400"])
cols = [(f"long looks ({len(long_keys)}), nside 400\n(one cell per pixel, as now)", long_keys, "400"),
        (f"long looks ({len(long_keys)}), nside 800", long_keys, "800"),
        (f"long looks ({len(long_keys)}), nside 800\naveraged to 400", long_keys, "800->400"),
        (f"all looks ({len(all_keys)}), nside 400\n(as now)", all_keys, "400"),
        (f"all looks ({len(all_keys)}), nside 800\naveraged to 400", all_keys, "800->400")]
maps = []
for label, keys, v in cols:
    print("stacking", label.replace("\n", " "), flush=True)
    maps.append((label, C.stack(L[v], keys, v)))

fig, axes = pl.subplots(len(CRATERS), len(cols), figsize=(3.3 * len(cols), 3.5 * len(CRATERS)),
                        constrained_layout=True, squeeze=False)
for i, (cname, lon0, lat0) in enumerate(CRATERS):
    lon = np.arange(lon0 - HALF, lon0 + HALF + STEP / 2, STEP)
    lat = np.arange(lat0 - HALF, lat0 + HALF + STEP / 2, STEP)
    for j, (label, m) in enumerate(maps):
        g = highpass(ra.grid_map(m, lon, lat), 2.0 / STEP)
        ax = axes[i, j]
        val = g[np.isfinite(g)]
        lo, hi = np.percentile(val, [1, 99.5])
        ax.imshow(g, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]],
                  cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"{cname}: {label}" if j == 0 else label, fontsize=8)
        ax.tick_params(labelsize=7)
fig.suptitle("Stockert + WSRT, opposite- + same-sense stacks, 2026-10-11 "
             "(crops 12° square, axes in degrees of longitude / latitude)", fontsize=10)
fig.savefig(sys.argv[1], dpi=80)
print("saved", sys.argv[1])
