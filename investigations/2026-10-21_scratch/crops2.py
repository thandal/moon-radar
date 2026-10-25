"""Crater-centred crops of stacks and single looks, trend removed, no mask erosion.

usage: crops2.py OUT label=path[:mult] label=path[:mult] ...
"""
import sys

import numpy as np
import healpy as hp
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

sys.path.insert(0, "/home/than/code/moon-radar")
import registration_analysis as ra

out = sys.argv[1]
items = []
for a in sys.argv[2:]:
    label, spec = a.split("=", 1)
    path, _, mult = spec.partition(":")
    items.append((label, path, mult or None))

CRATERS = [("Copernicus", -20.1, 9.6), ("Kepler", -38.0, 8.1),
           ("Bullialdus", -22.2, -20.7), ("Theophilus", 26.4, -11.4)]
HALF = 6.0
step = 0.04


def highpass(g, sigma_px):
    mask = np.isfinite(g)
    v = np.where(mask, g, 0.0)
    sv = scipy.ndimage.gaussian_filter(v, sigma_px)
    sw = scipy.ndimage.gaussian_filter(mask.astype(float), sigma_px)
    trend = np.where(sw > 0.05, sv / np.maximum(sw, 1e-9), np.nan)
    return np.where(mask, g - trend, np.nan)


maps = []
for label, path, mult in items:
    m = np.load(path)
    mu = np.load(mult) if mult else None
    maps.append((label, m, mu))

fig, axes = pl.subplots(len(CRATERS), len(maps), figsize=(3.6 * len(maps), 3.6 * len(CRATERS)),
                        constrained_layout=True, squeeze=False)
for i, (cname, lon0, lat0) in enumerate(CRATERS):
    lon = np.arange(lon0 - HALF, lon0 + HALF + step / 2, step)
    lat = np.arange(lat0 - HALF, lat0 + HALF + step / 2, step)
    for j, (label, m, mu) in enumerate(maps):
        g = highpass(ra.grid_map(m, lon, lat, multiplicity=mu), 2.0 / step)
        v = g[np.isfinite(g)]
        ax = axes[i, j]
        if v.size:
            lo, hi = np.percentile(v, [1, 99.5])
            ax.imshow(g, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]],
                      cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"{label}\n{cname}", fontsize=9)
        ax.tick_params(labelsize=7)
fig.savefig(out, dpi=75)
print("saved", out)
