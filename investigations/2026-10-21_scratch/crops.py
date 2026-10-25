"""Zoomed crops of the same lunar regions in several stacks, matched stretch."""
import sys

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

sys.path.insert(0, "/home/than/code/moon-radar")
import registration_analysis as ra

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
names = sys.argv[1].split(",")      # e.g. chan1,wsrt_chan1,stockert_wsrt_chan1
out = sys.argv[2]
REGIONS = [("south-east", (8, 38), (-30, 0)),
           ("north-west", (-40, -10), (0, 30)),
           ("centre", (-15, 15), (-10, 20)),
           ("south-west", (-40, -10), (-35, -5))]
step = 0.05

fig, axes = pl.subplots(len(REGIONS), len(names), figsize=(5.2 * len(names), 5 * len(REGIONS)),
                        constrained_layout=True)
maps = {n: np.load(f"{RUN}/stacked_map_{n}.npy") for n in names}
for i, (rname, (l0, l1), (b0, b1)) in enumerate(REGIONS):
    lon = np.arange(l0, l1 + step / 2, step)
    lat = np.arange(b0, b1 + step / 2, step)
    grids = {n: ra.grid_map(maps[n], lon, lat) for n in names}
    # remove the smooth trend so the same stretch fits every stack
    for n in names:
        g = grids[n]
        sm = ra.bandpass(g, 0.0001, 3.0 / step)  # g minus a 3-deg smooth
        grids[n] = sm
    for j, n in enumerate(names):
        g = grids[n]
        v = g[np.isfinite(g) & (g != 0)]
        lo, hi = np.percentile(v, [1, 99.5])
        ax = axes[i, j]
        ax.imshow(g, origin="lower", extent=[l0, l1, b0, b1], cmap="gray", vmin=lo, vmax=hi,
                  interpolation="nearest")
        ax.set_title(f"{n}: {rname}")
        ax.set_xlabel("lon (deg)")
        ax.set_ylabel("lat (deg)")
fig.savefig(out, dpi=80)
print("saved", out)
