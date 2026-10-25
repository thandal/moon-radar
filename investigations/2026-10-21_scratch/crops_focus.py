"""Crater crops: long looks, co+cross (16), nside 800, unfocused vs focused,
plus today's nside 400 production long-look stack for reference."""
import json, glob, sys
import numpy as np, healpy as hp, scipy.ndimage, matplotlib
matplotlib.use("Agg"); from matplotlib import pyplot as pl
sys.path.insert(0, "/home/than/code/moon-radar")
import stack_maps as sm, registration_analysis as ra, doppler_equator_alignment as dea
rows = [json.load(open(f)) for f in sorted(glob.glob("/var/tmp/focus/nside800/*/rows.json"))]
npix = hp.nside2npix(800); cache = {}
stk = {}
for ver in ("alpha0", "focused"):
    acc = sm.Accumulator(npix)
    for pair in rows:      # equal weights here: the nside-800 weight estimate is not robust yet
        r = pair[ver]; rx = r["rx_file"].split("_")[0].upper()
        cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), npix, cache, rx_name=rx)
        I, n = sm.look_brightness(r, cosi, max_mult_factor=12.0)
        acc.add(I)
    stk[ver] = acc.stacked_log(3)
prod = np.load("results/EVE2026_CIRC/stacked_map_stockert_wsrt_txmin100s_dual.npy")
CR = [("Copernicus", -20.1, 9.6), ("Kepler", -38.0, 8.1), ("Bullialdus", -22.2, -20.7), ("Theophilus", 26.4, -11.4)]
HALF, STEP = 4.0, 0.02
def hpf(g, s):
    m = np.isfinite(g); v = np.where(m, g, 0.0)
    t = scipy.ndimage.gaussian_filter(v, s) / np.maximum(scipy.ndimage.gaussian_filter(m.astype(float), s), 1e-9)
    return np.where(m, g - t, np.nan)
cols = [("nside 400, production", prod), ("nside 800, not focused", stk["alpha0"]), ("nside 800, focused", stk["focused"])]
fig, ax = pl.subplots(len(CR), len(cols), figsize=(3.6 * len(cols), 3.7 * len(CR)), constrained_layout=True)
for i, (nm, lo0, la0) in enumerate(CR):
    lon = np.arange(lo0 - HALF, lo0 + HALF + STEP / 2, STEP); lat = np.arange(la0 - HALF, la0 + HALF + STEP / 2, STEP)
    for j, (l, m) in enumerate(cols):
        g = hpf(ra.grid_map(m, lon, lat), 1.5 / STEP); v = g[np.isfinite(g)]
        ax[i, j].imshow(g, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]], cmap="gray",
                        vmin=np.percentile(v, 1), vmax=np.percentile(v, 99.5), interpolation="nearest")
        ax[i, j].set_title(f"{nm}: {l}" if j == 0 else l, fontsize=8); ax[i, j].tick_params(labelsize=7)
fig.suptitle("Long looks (16, opposite + same sense), 2026-10-11, Stockert + WSRT; crops 8° square", fontsize=10)
fig.savefig(sys.argv[1], dpi=80); print("saved")
