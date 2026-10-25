"""nside 400 vs nside 800 maps of the same delay-Doppler images.

1. Autocorrelation widths (as acwidth.py) of single looks and of stacks.
2. Detail that repeats between independent looks: correlation between two
   independent half-stacks of the long looks, per band of feature size
   (tile-averaged cross spectra on the 2 km disk grid). Speckle does not
   repeat between looks; surface detail does. If the finer grid holds real
   detail the 400 grid lost, the correlation at small feature sizes rises.
"""
import csv
import os
import sys

import numpy as np
import healpy as hp
import scipy.ndimage

sys.path.insert(0, "/tmp/eve_check/blur")
import tiles as T

R4 = "/home/than/code/moon-radar/results/EVE2026_CIRC"
R8 = "/home/than/code/moon-radar/results/EVE2026_CIRC_NSIDE800"
LONG = ("09:35:24", "09:41:25", "09:46:36", "10:16:10")


def ac_widths(g):
    ok = np.isfinite(g)
    v = np.where(ok, g - np.nanmean(g), 0.0)
    F = lambda a: np.fft.rfft2(a, s=(2 * a.shape[0], 2 * a.shape[1]))
    ac = np.fft.fftshift(np.fft.irfft2(np.abs(F(v)) ** 2))
    nw = np.fft.fftshift(np.fft.irfft2(np.abs(F(ok.astype(float))) ** 2))
    ac = ac / np.maximum(nw, 1)
    cy, cx = np.array(ac.shape) // 2
    ac /= ac[cy, cx]
    out = []
    for ang in (np.radians(T.EQ_DEG), np.radians(T.EQ_DEG) + np.pi / 2):
        r = np.arange(0, 40, 0.25)
        prof = scipy.ndimage.map_coordinates(ac, [cy + r * np.sin(ang), cx + r * np.cos(ang)], order=1)
        k = np.argmax(prof < 0.5)
        out.append(2 * T.PX * (r[k - 1] + (prof[k - 1] - 0.5) / (prof[k - 1] - prof[k]) * (r[k] - r[k - 1])))
    return out


def looks(run, chan):
    out = {}
    for st in ("stockert", "wsrt"):
        for r in csv.DictReader(open(f"{run}/registration_runs_{st}_{chan}.csv")):
            t = r["rx_start_utc"][11:19]
            if float(r["pair_snr"] or r["tone_snr"]) < 15 or t[:2] == "12":
                continue
            out[(st, t)] = (t, os.path.join(run, os.path.basename(r["map_npy"])),
                            os.path.join(run, os.path.basename(r["mult_npy"])))
    return out


def single(item):
    _, m, mu = item
    m, mu = np.load(m), np.load(mu)
    med = np.median(mu[mu > 0])
    return np.where((m != hp.UNSEEN) & (mu <= 3 * med), m, hp.UNSEEN)


BANDS = [(40, 20), (20, 12), (12, 8), (8, 6), (6, 4.4)]   # feature size (wavelength), km


def band_corr(a, b):
    n, st = int(T.TILE / T.PX), int(T.STEP / T.PX)
    taper = np.outer(np.hanning(n), np.hanning(n))
    f = np.fft.fftfreq(n, d=T.PX)
    kr = np.hypot(*np.meshgrid(f, f))
    with np.errstate(divide="ignore"):
        wl = np.where(kr > 0, 1 / kr, np.inf)
    cross = np.zeros(len(BANDS)); pa = np.zeros(len(BANDS)); pb = np.zeros(len(BANDS))
    ntile = 0
    for i0 in range(0, a.shape[0] - n + 1, st):
        for j0 in range(0, a.shape[1] - n + 1, st):
            ta, tb = a[i0:i0 + n, j0:j0 + n], b[i0:i0 + n, j0:j0 + n]
            ok = np.isfinite(ta) & np.isfinite(tb)
            if ok.mean() < 0.95:
                continue
            ntile += 1
            A = np.fft.fft2(np.where(ok, ta - np.nanmean(ta), 0.0) * taper)
            B = np.fft.fft2(np.where(ok, tb - np.nanmean(tb), 0.0) * taper)
            for k, (hi, lo) in enumerate(BANDS):
                sel = (wl <= hi) & (wl > lo)
                cross[k] += np.real(A[sel] * np.conj(B[sel])).sum()
                pa[k] += (np.abs(A[sel]) ** 2).sum()
                pb[k] += (np.abs(B[sel]) ** 2).sum()
    return cross / np.sqrt(pa * pb), ntile


for chan in ("chan1", "chan0"):
    L4, L8 = looks(R4, chan), looks(R8, chan)
    print(f"\n===== {chan} ({'opposite sense' if chan == 'chan1' else 'same sense'}) =====")
    print(f"{'map':52s} {'nside 400':>18s} {'nside 800':>18s}   width along x across, km")
    cases = [("single look Stockert 09:41 (125 s, 25 kchip/s)", [("stockert", "09:41:25")]),
             ("single look WSRT 10:16 (125 s, 25 kchip/s)", [("wsrt", "10:16:10")]),
             ("single look Stockert 09:28 (35 s, 25 kchip/s)", [("stockert", "09:28:21")]),
             ("single look Stockert 09:49:57 (35 s, 50 kchip/s)", [("stockert", "09:49:57")]),
             ("long looks, both stations (8)", [k for k in L4 if k[1] in LONG]),
             (f"all looks, both stations ({len(L4)})", list(L4))]
    for name, keys in cases:
        if not all(k in L4 for k in keys):
            print(f"{name:52s}  (look not in the gated set)")
            continue
        res = []
        for L in (L4, L8):
            h = single(L[keys[0]]) if len(keys) == 1 else T.stack_of([L[k] for k in keys])
            res.append(ac_widths(T.disk_grid(h)))
        print(f"{name:52s} {res[0][0]:7.1f} x {res[0][1]:5.1f}    {res[1][0]:7.1f} x {res[1][1]:5.1f}")

    print("\ndetail that repeats between independent half-stacks of the long looks")
    print("(half A = 09:35 + 09:46, half B = 09:41 + 10:16, both stations each)")
    print(f"{'feature size, km':>18s} " + " ".join(f"{f'{hi:g}-{lo:g}':>8s}" for hi, lo in BANDS))
    for nside, L in (("400", L4), ("800", L8)):
        hA = T.stack_of([L[(s, t)] for s in ("stockert", "wsrt") for t in ("09:35:24", "09:46:36")])
        hB = T.stack_of([L[(s, t)] for s in ("stockert", "wsrt") for t in ("09:41:25", "10:16:10")])
        c, nt = band_corr(T.disk_grid(hA), T.disk_grid(hB))
        print(f"{'nside ' + nside:>18s} " + " ".join(f"{v:8.3f}" for v in c) + f"   ({nt} tiles)")
    print("control: same-time looks from the two stations (Stockert vs WSRT, all 4 long times each)")
    for nside, L in (("400", L4), ("800", L8)):
        hS = T.stack_of([L[("stockert", t)] for t in LONG])
        hW = T.stack_of([L[("wsrt", t)] for t in LONG])
        c, nt = band_corr(T.disk_grid(hS), T.disk_grid(hW))
        print(f"{'nside ' + nside:>18s} " + " ".join(f"{v:8.3f}" for v in c) + f"   ({nt} tiles)")
