"""Local (tile-by-tile) registration offsets between opposite-sense stacks, on the
disk seen from lon 0 lat 0, in km. Offsets are split into the component
along the Doppler equator (a Doppler error moves features this way) and
across it (a delay error moves features this way, along lines of constant
Doppler).

Compares Stockert vs WSRT over the same 09:24-10:16 span, and, as a noise
control, interleaved halves of each station's looks.
"""
import csv
import os
import sys

import numpy as np
import healpy as hp
import scipy.ndimage

sys.path.insert(0, "/home/than/code/moon-radar")
import registration_analysis as ra
import stack_maps as sm

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
R_KM = 1737.4
PX = 2.0            # km per grid pixel
EXT = 1300.0        # half-width of the disk grid, km
TILE, STEP = 300.0, 150.0
EQ_DEG = -37.5      # Doppler-equator position angle over 09:24-10:16

looks = {}
for st in ["stockert", "wsrt"]:
    for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_chan1.csv")):
        if float(r["tone_snr"]) < 15 or r["rx_start_utc"][11:13] == "12":
            continue
        looks.setdefault(st, []).append(
            (r["rx_start_utc"][11:19], os.path.join(RUN, os.path.basename(r["map_npy"])),
             os.path.join(RUN, os.path.basename(r["mult_npy"]))))

xs = np.arange(-EXT, EXT + PX / 2, PX)
X, Y = np.meshgrid(xs, xs)
inside = X ** 2 + Y ** 2 < (0.98 * R_KM) ** 2
lat = np.arcsin(np.clip(Y / R_KM, -1, 1))
lon = np.arcsin(np.clip(X / (R_KM * np.cos(lat)), -1, 1))
theta, phi = (np.pi / 2 - lat).ravel(), lon.ravel()


def stack_of(items):
    acc = None
    for _, m, mu in items:
        I = sm.look_linear_intensity(m, mu)
        acc = acc or sm.Accumulator(len(I))
        acc.add(I)
    return acc.stacked_log(min(3, len(items)))


def disk_grid(hmap):
    mask = hmap != hp.UNSEEN
    v = hp.get_interp_val(np.where(mask, hmap, 0.0), theta, phi).reshape(X.shape)
    w = hp.get_interp_val(mask.astype(float), theta, phi).reshape(X.shape)
    g = np.where((w > 0.99) & inside, v / np.maximum(w, 1e-9), np.nan)
    # remove the >40 km trend, keep the rest
    ok = np.isfinite(g)
    gv = np.where(ok, g, 0.0)
    s = 20.0 / PX
    trend = scipy.ndimage.gaussian_filter(gv, s) / np.maximum(
        scipy.ndimage.gaussian_filter(ok.astype(float), s), 1e-9)
    return np.where(ok, g - trend, np.nan)


def tile_offsets(a, b):
    n, st = int(TILE / PX), int(STEP / PX)
    taper = np.outer(np.hanning(n), np.hanning(n))
    out = []
    for i0 in range(0, a.shape[0] - n + 1, st):
        for j0 in range(0, a.shape[1] - n + 1, st):
            ta, tb = a[i0:i0 + n, j0:j0 + n], b[i0:i0 + n, j0:j0 + n]
            ok = np.isfinite(ta) & np.isfinite(tb)
            if ok.mean() < 0.85:
                continue
            ta = np.where(ok, ta - np.nanmean(ta), 0.0) * taper
            tb = np.where(ok, tb - np.nanmean(tb), 0.0) * taper
            dy, dx, peak, sig = ra.xcorr_offset(ta, tb, int(20 / PX), int(6 / PX))
            out.append((xs[j0 + n // 2], xs[i0 + n // 2], dx * PX, dy * PX, peak, sig))
    return np.array(out)


def report(name, t):
    good = t[t[:, 5] > 1.3]
    e = np.radians(EQ_DEG)
    along = good[:, 2] * np.cos(e) + good[:, 3] * np.sin(e)
    across = -good[:, 2] * np.sin(e) + good[:, 3] * np.cos(e)
    print(f"{name}: {len(good)}/{len(t)} tiles lock (signif > 1.3); offset km "
          f"along Doppler equator median {np.median(along):+.2f} rms {np.sqrt(np.mean(along**2)):.2f}; "
          f"across median {np.median(across):+.2f} rms {np.sqrt(np.mean(across**2)):.2f}; "
          f"median peak corr {np.median(good[:, 4]):.2f}")
    return good, along, across


S, W = looks["stockert"], looks["wsrt"]
if __name__ == "__main__":
    print("Stockert:", [t for t, _, _ in S]); print("WSRT:", [t for t, _, _ in W])
    gS, gW = disk_grid(stack_of(S)), disk_grid(stack_of(W))
    gSa, gSb = disk_grid(stack_of(S[0::2])), disk_grid(stack_of(S[1::2]))
    gWa, gWb = disk_grid(stack_of(W[0::2])), disk_grid(stack_of(W[1::2]))

    res = {}
    for name, a, b in [("Stockert vs WSRT (all looks each)", gS, gW),
                       ("Stockert half A vs half B", gSa, gSb),
                       ("WSRT half A vs half B", gWa, gWb),
                       ("Stockert half A vs WSRT half A", gSa, gWa)]:
        res[name] = report(name, tile_offsets(a, b))

    good, along, across = res["Stockert vs WSRT (all looks each)"]
    print("\nStockert vs WSRT per tile (disk x, y km -> along, across km, signif):")
    for (x, y, dx, dy, pk, sg), al, ac in zip(good, along, across):
        print(f"  ({x:+6.0f},{y:+6.0f})  along {al:+5.1f}  across {ac:+5.1f}  signif {sg:.2f}")
