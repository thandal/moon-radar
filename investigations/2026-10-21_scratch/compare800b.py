"""Fair nside 400 vs 800 comparison: the delay-Doppler-cell sharing mask scaled
with pixel area (3 at nside 400 -> 12 at nside 800, i.e. the same ground area
per cell), band correlations on the same tiles for every version, plus a
third version: nside 800 averaged back down to nside 400 (4 cells per pixel
instead of 1)."""
import csv
import os
import sys

import numpy as np
import healpy as hp
import scipy.ndimage

sys.path.insert(0, "/tmp/eve_check/blur")
import tiles as T
import stack_maps as sm

R4 = "/home/than/code/moon-radar/results/EVE2026_CIRC"
R8 = "/home/than/code/moon-radar/results/EVE2026_CIRC_NSIDE800"
LONG = ("09:35:24", "09:41:25", "09:46:36", "10:16:10")
BANDS = [(40, 20), (20, 12), (12, 8), (8, 6), (6, 4.4)]   # feature size (wavelength), km


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


HALF_A = [(s, t) for s in ("stockert", "wsrt") for t in ("09:35:24", "09:46:36")]
HALF_B = [(s, t) for s in ("stockert", "wsrt") for t in ("09:41:25", "10:16:10")]


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


def intensity(item, version):
    m, mu = item
    if version == "400":
        return sm.look_linear_intensity(m, mu, max_mult_factor=3.0)
    I = sm.look_linear_intensity(m, mu, max_mult_factor=12.0)
    if version == "800":
        return I
    # "800->400": mean of the valid nside-800 children of each nside-400 pixel
    # (nside 800 is not a power of 2, so no hp.ud_grade: each nside-400 pixel
    # holds exactly four nside-800 pixels; find the parent from the centre)
    ok = np.isfinite(I)
    s = np.bincount(PARENT[ok], weights=I[ok], minlength=hp.nside2npix(400))
    c = np.bincount(PARENT[ok], minlength=hp.nside2npix(400))
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


PARENT = hp.ang2pix(400, *hp.pix2ang(800, np.arange(hp.nside2npix(800))))
assert np.all(np.bincount(PARENT) == 4)


def stack(L, keys, version):
    if len(keys) == 1:
        I = intensity(L[keys[0]], version)
        return np.where(np.isfinite(I), 0.5 * np.log(I), hp.UNSEEN)
    acc = None
    for k in keys:
        I = intensity(L[k], version)
        acc = acc or sm.Accumulator(len(I))
        acc.add(I)
    return acc.stacked_log(min(3, len(keys)))


n, st = int(T.TILE / T.PX), int(T.STEP / T.PX)
TAPER = np.outer(np.hanning(n), np.hanning(n))
f = np.fft.fftfreq(n, d=T.PX)
kr = np.hypot(*np.meshgrid(f, f))
with np.errstate(divide="ignore"):
    WL = np.where(kr > 0, 1 / kr, np.inf)
SEL = [(WL <= hi) & (WL > lo) for hi, lo in BANDS]
ORIGINS = [(i0, j0) for i0 in range(0, len(T.xs) - n + 1, st) for j0 in range(0, len(T.xs) - n + 1, st)]


def good_tiles(*grids):
    out = set()
    for i0, j0 in ORIGINS:
        if all(np.isfinite(g[i0:i0 + n, j0:j0 + n]).mean() >= 0.95 for g in grids):
            out.add((i0, j0))
    return out


def band_corr(a, b, tiles):
    cross = np.zeros(len(BANDS)); pa = np.zeros(len(BANDS)); pb = np.zeros(len(BANDS))
    for i0, j0 in sorted(tiles):
        ta, tb = a[i0:i0 + n, j0:j0 + n], b[i0:i0 + n, j0:j0 + n]
        ok = np.isfinite(ta) & np.isfinite(tb)
        A = np.fft.fft2(np.where(ok, ta - np.nanmean(ta), 0.0) * TAPER)
        B = np.fft.fft2(np.where(ok, tb - np.nanmean(tb), 0.0) * TAPER)
        for k, s in enumerate(SEL):
            cross[k] += np.real(A[s] * np.conj(B[s])).sum()
            pa[k] += (np.abs(A[s]) ** 2).sum()
            pb[k] += (np.abs(B[s]) ** 2).sum()
    return cross / np.sqrt(pa * pb)


VERSIONS = [("400", "nside 400 (one cell per pixel)"),
            ("800", "nside 800"),
            ("800->400", "nside 800 averaged to 400")]
if __name__ == "__main__":
    for chan in ("chan1", "chan0"):
        L = {"400": looks(R4, chan), "800": looks(R8, chan)}
        L["800->400"] = L["800"]
        print(f"\n===== {chan} ({'opposite sense' if chan == 'chan1' else 'same sense'}) =====")
        print(f"{'width along x across the Doppler equator, km':52s}" + "".join(f"{v[0]:>16s}" for v in VERSIONS))
        for name, keys in [("single look Stockert 09:41 (125 s, 25 kchip/s)", [("stockert", "09:41:25")]),
                           ("single look Stockert 09:28 (35 s, 25 kchip/s)", [("stockert", "09:28:21")]),
                           ("single look Stockert 09:49:57 (35 s, 50 kchip/s)", [("stockert", "09:49:57")]),
                           ("long looks, both stations (8)", [k for k in L["400"] if k[1] in LONG]),
                           (f"all looks, both stations ({len(L['400'])})", list(L["400"]))]:
            cells = []
            for v, _ in VERSIONS:
                w = ac_widths(T.disk_grid(stack(L[v], keys, v)))
                cells.append(f"{w[0]:7.1f} x {w[1]:4.1f}")
            print(f"{name:52s}" + "".join(f"{c:>16s}" for c in cells))

        gA = {v: T.disk_grid(stack(L[v], HALF_A, v)) for v, _ in VERSIONS}
        gB = {v: T.disk_grid(stack(L[v], HALF_B, v)) for v, _ in VERSIONS}
        common = good_tiles(*gA.values(), *gB.values())
        shift = 37  # px (74 km): no real detail lines up -> chance level
        gBs = {v: np.roll(g, (shift, shift), axis=(0, 1)) for v, g in gB.items()}
        common_null = good_tiles(*gA.values(), *gBs.values())
        print(f"\ndetail that repeats between independent half-stacks of the long looks "
              f"(09:35+09:46 vs 09:41+10:16, both stations), same {len(common)} tiles for all")
        print(f"{'feature size, km':>32s} " + " ".join(f"{f'{hi:g}-{lo:g}':>8s}" for hi, lo in BANDS))
        for v, label in VERSIONS:
            c = band_corr(gA[v], gB[v], common)
            print(f"{label:>32s} " + " ".join(f"{x:8.3f}" for x in c))
        for v, label in VERSIONS:
            c = band_corr(gA[v], gBs[v], common_null)
            print(f"{'chance level, ' + v:>32s} " + " ".join(f"{x:8.3f}" for x in c) + f"   ({len(common_null)} tiles)")
