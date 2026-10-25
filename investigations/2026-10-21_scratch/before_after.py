"""Regenerated 2026-10-11 circular-set products: one point per pixel (backup)
vs 4x4 points per pixel (results/EVE2026_CIRC).

1. Half-stack correlation per feature size using ALL looks: the look times
   alternate between half A and half B (both stations of a time go to the
   same half, since same-time looks share some speckle).
2. Crater crops of the stack_maps.py opposite- + same-sense stacks, old vs new.
usage: before_after.py OUT.png
"""
import csv
import os
import sys

import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

sys.path.insert(0, "/tmp/eve_check/blur")
sys.path.insert(0, "/home/than/code/moon-radar")
import tiles as T
import compare800b as C
import registration_analysis as ra

OLD = "/var/tmp/moon_backup/EVE2026_CIRC_one_point_per_pixel"
NEW = "/home/than/code/moon-radar/results/EVE2026_CIRC"
BANDS = [(40, 20), (20, 12), (12, 9)]
C.BANDS = BANDS
C.SEL = [(C.WL <= hi) & (C.WL > lo) for hi, lo in BANDS]


def looks(run, chan):
    out = {}
    for st in ("stockert", "wsrt"):
        for r in csv.DictReader(open(f"{run}/registration_runs_{st}_{chan}.csv")):
            t = r["rx_start_utc"][11:19]
            if float(r["pair_snr"] or r["tone_snr"]) < 15 or t[:2] == "12":
                continue
            out[(st, t)] = (os.path.join(run, os.path.basename(r["map_npy"])),
                            os.path.join(run, os.path.basename(r["mult_npy"])))
    return out


def stack(L, keys):
    acc = None
    for k in keys:
        I = C.sm.look_linear_intensity(*L[k])
        acc = acc or C.sm.Accumulator(len(I))
        acc.add(I)
    return acc.stacked_log(min(3, len(keys)))


for chan in ("chan1", "chan0"):
    L = {"old": looks(OLD, chan), "new": looks(NEW, chan)}
    assert L["old"].keys() == L["new"].keys()
    times = sorted({t for _, t in L["new"]})
    A = [k for k in L["new"] if times.index(k[1]) % 2 == 0]
    B = [k for k in L["new"] if times.index(k[1]) % 2 == 1]
    gA = {v: T.disk_grid(stack(L[v], A)) for v in L}
    gB = {v: T.disk_grid(stack(L[v], B)) for v in L}
    common = C.good_tiles(*gA.values(), *gB.values())
    gBs = {v: np.roll(g, (37, 37), axis=(0, 1)) for v, g in gB.items()}
    common_null = C.good_tiles(*gA.values(), *gBs.values())
    print(f"\n===== {chan} ({'opposite sense' if chan == 'chan1' else 'same sense'}): all looks, "
          f"halves {len(A)} + {len(B)} looks by alternating time, same {len(common)} tiles =====")
    print(f"{'feature size, km':>28s} " + " ".join(f"{f'{hi:g}-{lo:g}':>8s}" for hi, lo in BANDS)
          + "    signal-to-noise, new vs old")
    res = {}
    for v, label in (("old", "one point per pixel (old)"), ("new", "4x4 points per pixel (new)")):
        res[v] = C.band_corr(gA[v], gB[v], common)
        print(f"{label:>28s} " + " ".join(f"{x:8.3f}" for x in res[v]))
    snr = {v: c / (1 - c) for v, c in res.items()}
    print(f"{'':>28s} " + " " * 27 + "    " + " ".join(f"{x:5.2f}x" for x in snr["new"] / snr["old"]))
    for v in ("old", "new"):
        c = C.band_corr(gA[v], gBs[v], common_null)
        print(f"{'chance, ' + v:>28s} " + " ".join(f"{x:8.3f}" for x in c))

# crops of the official stacks
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


cols = [("all looks, old\n(one point per pixel)", f"{OLD}/stacked_map_stockert_wsrt_dual.npy"),
        ("all looks, new\n(4x4 points per pixel)", f"{NEW}/stacked_map_stockert_wsrt_dual.npy"),
        ("long looks only, old", f"{OLD}/stacked_map_stockert_wsrt_txmin100s_dual.npy"),
        ("long looks only, new", f"{NEW}/stacked_map_stockert_wsrt_txmin100s_dual.npy")]
maps = [(label, np.load(p)) for label, p in cols]
fig, axes = pl.subplots(len(CRATERS), len(cols), figsize=(3.4 * len(cols), 3.6 * len(CRATERS)),
                        constrained_layout=True, squeeze=False)
for i, (cname, lon0, lat0) in enumerate(CRATERS):
    lon = np.arange(lon0 - HALF, lon0 + HALF + STEP / 2, STEP)
    lat = np.arange(lat0 - HALF, lat0 + HALF + STEP / 2, STEP)
    for j, (label, m) in enumerate(maps):
        g = highpass(ra.grid_map(m, lon, lat), 2.0 / STEP)
        val = g[np.isfinite(g)]
        lo, hi = np.percentile(val, [1, 99.5])
        ax = axes[i, j]
        ax.imshow(g, origin="lower", extent=[lon[0], lon[-1], lat[0], lat[-1]],
                  cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"{cname}: {label}" if j == 0 else label, fontsize=8)
        ax.tick_params(labelsize=7)
fig.suptitle("stacked_map_stockert_wsrt_dual and _txmin100s_dual (opposite + same sense), "
             "2026-10-11; crops 12° square, degrees of longitude / latitude", fontsize=10)
fig.savefig(sys.argv[1], dpi=80)
print("saved", sys.argv[1])
