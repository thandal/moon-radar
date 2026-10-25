"""Points averaged per pixel (1, 4, 16, 64) at nside 400: detail that repeats
between independent half-stacks of the long looks, per feature size, on the
same tiles for every version; chance level from a 74 km shifted pairing."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, "/tmp/eve_check/blur")
import tiles as T
import compare800b as C

BANDS = [(40, 20), (20, 12), (12, 9)]
C.BANDS = BANDS
C.SEL = [(C.WL <= hi) & (C.WL > lo) for hi, lo in BANDS]
SS = [s for s in (1, 2, 4, 8) if os.path.exists(f"/var/tmp/moon_ss/ss{s}/registration_runs_wsrt_chan0.csv")]


def looks(run, chan):
    out = {}
    for st in ("stockert", "wsrt"):
        for r in csv.DictReader(open(f"{run}/registration_runs_{st}_{chan}.csv")):
            out[(st, r["rx_start_utc"][11:19])] = (r["map_npy"], r["mult_npy"])
    return out


def stack(L, keys):
    acc = None
    for k in keys:
        I = C.sm.look_linear_intensity(*L[k])
        acc = acc or C.sm.Accumulator(len(I))
        acc.add(I)
    return acc.stacked_log(min(3, len(keys)))


for chan in ("chan1", "chan0"):
    L = {s: looks(f"/var/tmp/moon_ss/ss{s}", chan) for s in SS}
    gA = {s: T.disk_grid(stack(L[s], C.HALF_A)) for s in SS}
    gB = {s: T.disk_grid(stack(L[s], C.HALF_B)) for s in SS}
    common = C.good_tiles(*gA.values(), *gB.values())
    gBs = {s: np.roll(g, (37, 37), axis=(0, 1)) for s, g in gB.items()}
    common_null = C.good_tiles(*gA.values(), *gBs.values())
    print(f"\n===== {chan} ({'opposite sense' if chan == 'chan1' else 'same sense'}), "
          f"half-stack correlation, same {len(common)} tiles =====")
    print(f"{'feature size, km':>26s} " + " ".join(f"{f'{hi:g}-{lo:g}':>8s}" for hi, lo in BANDS)
          + "    signal-to-noise vs 1 point, per band")
    base = None
    for s in SS:
        c = C.band_corr(gA[s], gB[s], common)
        snr = c / (1 - c)
        base = snr if base is None else base
        print(f"{f'{s * s} points per pixel':>26s} " + " ".join(f"{x:8.3f}" for x in c)
              + "    " + " ".join(f"{x:5.2f}x" for x in snr / base))
    for s in SS:
        c = C.band_corr(gA[s], gBs[s], common_null)
        print(f"{f'chance, {s * s} points':>26s} " + " ".join(f"{x:8.3f}" for x in c))
