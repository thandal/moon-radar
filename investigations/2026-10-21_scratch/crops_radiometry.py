import sys, numpy as np, scipy.ndimage, matplotlib
matplotlib.use("Agg"); from matplotlib import pyplot as pl
sys.path.insert(0, "/home/than/code/moon-radar"); import registration_analysis as ra
OLD = "/var/tmp/moon_backup/EVE2026_CIRC_averaged_equal_weights"; NEW = "results/EVE2026_CIRC"
CR = [("Copernicus", -20.1, 9.6), ("Kepler", -38.0, 8.1), ("Bullialdus", -22.2, -20.7), ("Theophilus", 26.4, -11.4)]
HALF, STEP = 6.0, 0.04
def hpf(g, s):
    m = np.isfinite(g); v = np.where(m, g, 0.0)
    t = scipy.ndimage.gaussian_filter(v, s) / np.maximum(scipy.ndimage.gaussian_filter(m.astype(float), s), 1e-9)
    return np.where(m, g - t, np.nan)
cols = [("co+cross, before", f"{OLD}/stacked_map_stockert_wsrt_dual.npy"),
        ("co+cross, now", f"{NEW}/stacked_map_stockert_wsrt_dual.npy"),
        ("same sense, before", f"{OLD}/stacked_map_stockert_wsrt_chan0.npy"),
        ("same sense, now", f"{NEW}/stacked_map_stockert_wsrt_chan0.npy")]
maps = [(l, np.load(p)) for l, p in cols]
fig, ax = pl.subplots(len(CR), len(cols), figsize=(3.4 * len(cols), 3.6 * len(CR)), constrained_layout=True)
for i, (n, lo0, la0) in enumerate(CR):
    lon = np.arange(lo0 - HALF, lo0 + HALF + STEP / 2, STEP); lat = np.arange(la0 - HALF, la0 + HALF + STEP / 2, STEP)
    for j, (l, m) in enumerate(maps):
        g = hpf(ra.grid_map(m, lon, lat), 2.0 / STEP); v = g[np.isfinite(g)]
        ax[i, j].imshow(g, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap="gray",
                        vmin=np.percentile(v, 1), vmax=np.percentile(v, 99.5), interpolation="nearest")
        ax[i, j].set_title(f"{n}: {l}" if j == 0 else l, fontsize=8); ax[i, j].tick_params(labelsize=7)
fig.suptitle("stacked_map_stockert_wsrt_{dual,chan0}: before = equal weights; now = per unit area, noise removed, weighted", fontsize=10)
fig.savefig(sys.argv[1], dpi=80)
# full disks, side by side, same display rule as stack_maps
import healpy as hp
fig = pl.figure(figsize=(16, 8))
for k, (l, p) in enumerate([("co+cross before", cols[0][1]), ("co+cross now", cols[1][1])]):
    m = np.load(p); ok = m != hp.UNSEEN; lo, hi = np.percentile(m[ok], [5, 99.5])
    hp.orthview(m, sub=(1, 2, k + 1), half_sky=True, flip="geo", min=lo, max=hi, title=l, xsize=1600, fig=fig.number, cbar=False)
fig.savefig(sys.argv[2], dpi=90)
print("saved")
