"""Close-ups, nside 400, long-look co+cross stacks: production / no drift
correction / focused, log brightness, one common stretch (from production).
usage: closeup_focus400.py OUT_PNG"""
import sys
import numpy as np, matplotlib
matplotlib.use("Agg"); from matplotlib import pyplot as pl
sys.path.insert(0, "/home/than/code/moon-radar")
import registration_analysis as ra
name = "stacked_map_stockert_wsrt_txmin100s_dual.npy"
cols = [("production", np.load(f"results/EVE2026_CIRC/{name}")),
        ("no drift correction", np.load(f"/var/tmp/focus400/run_block/{name}")),
        ("focused", np.load(f"results/EVE2026_CIRC_FOCUSED/{name}"))]
CR = [("Kepler", -38.0, 8.1, 2.0), ("Copernicus", -20.1, 9.6, 2.5), ("Bullialdus", -22.2, -20.7, 2.0)]
STEP = 0.01
fig, ax = pl.subplots(len(CR), len(cols), figsize=(4.2 * len(cols), 4.3 * len(CR)), constrained_layout=True)
for i, (nm, lo0, la0, half) in enumerate(CR):
    lon = np.arange(lo0 - half, lo0 + half + STEP / 2, STEP); lat = np.arange(la0 - half, la0 + half + STEP / 2, STEP)
    gs = [ra.grid_map(m, lon, lat) for _, m in cols]
    v = gs[0][np.isfinite(gs[0])]
    lo, hi = np.percentile(v, [1, 99.5])
    for j, ((l, _), g) in enumerate(zip(cols, gs)):
        ax[i, j].imshow(g, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap="gray",
                        vmin=lo, vmax=hi, interpolation="nearest")
        ax[i, j].set_title(f"{nm}: {l}", fontsize=9); ax[i, j].tick_params(labelsize=7)
fig.suptitle("nside 400, long looks co+cross, equal weights; log brightness, same stretch in each row", fontsize=10)
fig.savefig(sys.argv[1], dpi=90); print("saved", sys.argv[1])
