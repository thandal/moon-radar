"""Crater crops, nside 400, long looks (16, co+cross): production, block
method without drift correction, focused. All stacked by stack_maps.py with
equal weights. usage: crops_focus400.py OUT_PNG"""
import sys
import numpy as np, scipy.ndimage, matplotlib
matplotlib.use("Agg"); from matplotlib import pyplot as pl
sys.path.insert(0, "/home/than/code/moon-radar")
import registration_analysis as ra
name = "stacked_map_stockert_wsrt_txmin100s_dual.npy"
cols = [("production", np.load(f"results/EVE2026_CIRC/{name}")),
        ("no drift correction", np.load(f"/var/tmp/focus400/run_block/{name}")),
        ("focused", np.load(f"results/EVE2026_CIRC_FOCUSED/{name}"))]
CR = [("Copernicus", -20.1, 9.6), ("Kepler", -38.0, 8.1), ("Bullialdus", -22.2, -20.7), ("Theophilus", 26.4, -11.4)]
HALF, STEP = 4.0, 0.02
def hpf(g, s):
    m = np.isfinite(g); v = np.where(m, g, 0.0)
    t = scipy.ndimage.gaussian_filter(v, s) / np.maximum(scipy.ndimage.gaussian_filter(m.astype(float), s), 1e-9)
    return np.where(m, g - t, np.nan)
fig, ax = pl.subplots(len(CR), len(cols), figsize=(3.6 * len(cols), 3.7 * len(CR)), constrained_layout=True)
for i, (nm, lo0, la0) in enumerate(CR):
    lon = np.arange(lo0 - HALF, lo0 + HALF + STEP / 2, STEP); lat = np.arange(la0 - HALF, la0 + HALF + STEP / 2, STEP)
    for j, (l, m) in enumerate(cols):
        g = hpf(ra.grid_map(m, lon, lat), 1.5 / STEP); v = g[np.isfinite(g)]
        ax[i, j].imshow(g, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap="gray",
                        vmin=np.percentile(v, 1), vmax=np.percentile(v, 99.5), interpolation="nearest")
        ax[i, j].set_title(f"{nm}: nside 400, {l}", fontsize=8); ax[i, j].tick_params(labelsize=7)
fig.suptitle("Long looks (16, opposite + same sense), 2026-10-11, Stockert + WSRT, equal weights; crops 8° square",
             fontsize=10)
fig.savefig(sys.argv[1], dpi=80); print("saved", sys.argv[1])
