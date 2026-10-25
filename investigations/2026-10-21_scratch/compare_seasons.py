"""2026-10-11 only vs 2025 only vs all seasons (opposite- + same-sense stacks, equal
weights). Row 1: the disk seen from lon 0 lat 0 (scattering-normalized log
brightness); rows 2-5: crater crops (log brightness, >1.5 deg trend removed).
Columns line up so flip_compare.py can flip across each row.
usage: compare_seasons.py OUT_PNG"""
import sys
import numpy as np, healpy as hp, scipy.ndimage, matplotlib
matplotlib.use("Agg"); from matplotlib import pyplot as pl
sys.path.insert(0, "/home/than/code/moon-radar")
import registration_analysis as ra
R = "/home/than/code/moon-radar/results"
cols = [("2026-10-11 only", f"{R}/EVE2026_CIRC/stacked_map_stockert_wsrt_dual", f"{R}/EVE2026_CIRC/stacked_counts_stockert_wsrt_dual.npy"),
        ("2025 only", f"{R}/STOCKERT_2025/stacked_map_stockert_dual", f"{R}/STOCKERT_2025/stacked_counts_stockert_dual.npy"),
        ("all seasons", f"{R}/ALL_SEASONS/stacked_map_stockert_wsrt_dual", f"{R}/ALL_SEASONS/stacked_counts_stockert_wsrt_dual.npy")]
CR = [("Copernicus", -20.1, 9.6), ("Kepler", -38.0, 8.1), ("Bullialdus", -22.2, -20.7), ("Theophilus", 26.4, -11.4)]
HALF, STEP = 4.0, 0.02


def hpf(g, s):
    m = np.isfinite(g); v = np.where(m, g, 0.0)
    t = scipy.ndimage.gaussian_filter(v, s) / np.maximum(scipy.ndimage.gaussian_filter(m.astype(float), s), 1e-9)
    return np.where(m, g - t, np.nan)


R_KM, PX = 1737.4, 4.0
xs = np.arange(-R_KM, R_KM + PX / 2, PX); X, Y = np.meshgrid(xs, xs)
inside = X ** 2 + Y ** 2 < R_KM ** 2
lat = np.arcsin(np.clip(Y / R_KM, -1, 1)); lon = np.arcsin(np.clip(X / (R_KM * np.maximum(np.cos(lat), 1e-9)), -1, 1))


def disk(hmap):
    mask = hmap != hp.UNSEEN
    th, ph = (np.pi / 2 - lat).ravel(), lon.ravel()
    v = hp.get_interp_val(np.where(mask, hmap, 0.0), th, ph).reshape(X.shape)
    w = hp.get_interp_val(mask.astype(float), th, ph).reshape(X.shape)
    return np.where((w > 0.99) & inside, v / np.maximum(w, 1e-9), np.nan)


fig, ax = pl.subplots(1 + len(CR), len(cols), figsize=(4.2 * len(cols), 4.3 * (1 + len(CR))), constrained_layout=True)
for j, (label, stem, counts) in enumerate(cols):
    n = np.load(counts)
    g = disk(np.load(stem + "_scatnorm.npy")); v = g[np.isfinite(g)]
    ax[0, j].imshow(g, origin="lower", extent=[xs[0], xs[-1], xs[0], xs[-1]], cmap="gray",
                    vmin=np.percentile(v, 2), vmax=np.percentile(v, 99.7), interpolation="nearest")
    ax[0, j].set_title(f"{label}: disk, scattering-normalized\n(median {np.median(n[n >= 3]):.0f} looks per pixel)", fontsize=8)
    ax[0, j].tick_params(labelsize=7)
    m = np.load(stem + ".npy")
    for i, (nm, lo0, la0) in enumerate(CR, start=1):
        lo = np.arange(lo0 - HALF, lo0 + HALF + STEP / 2, STEP); la = np.arange(la0 - HALF, la0 + HALF + STEP / 2, STEP)
        g = hpf(ra.grid_map(m, lo, la), 1.5 / STEP); v = g[np.isfinite(g)]
        ax[i, j].imshow(g, origin="lower", extent=[lo[0], lo[-1], la[0], la[-1]], cmap="gray",
                        vmin=np.percentile(v, 1), vmax=np.percentile(v, 99.5), interpolation="nearest")
        ax[i, j].set_title(f"{nm}: {label}", fontsize=8); ax[i, j].tick_params(labelsize=7)
fig.suptitle("Opposite- + same-sense stacks, nside 400, equal weights; crops 8° square", fontsize=10)
fig.savefig(sys.argv[1], dpi=80); print("saved", sys.argv[1])
