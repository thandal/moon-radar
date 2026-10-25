"""Width of the autocorrelation peak of trend-removed maps on the disk grid,
along and across the Doppler equator, in km (full width at half maximum).
A blurrier map has a wider peak. Region: |disk x|,|disk y| < 1300 km, away
from the Doppler-equator stripe (masked by the per-look ambiguity masks)."""
import csv
import os
import sys

import numpy as np
import healpy as hp
import scipy.ndimage

sys.path.insert(0, "/tmp/eve_check/blur")
sys.argv = sys.argv[:1]
import tiles as T  # reuses the disk grid, stack builder and look lists; runs its report too

RUN = T.RUN
OUT = "/tmp/eve_check/blur"


def ac_widths(g):
    ok = np.isfinite(g)
    v = np.where(ok, g - np.nanmean(g), 0.0)
    w = ok.astype(float)
    F = lambda a: np.fft.rfft2(a, s=(2 * a.shape[0], 2 * a.shape[1]))
    ac = np.fft.fftshift(np.fft.irfft2(np.abs(F(v)) ** 2))
    nw = np.fft.fftshift(np.fft.irfft2(np.abs(F(w)) ** 2))
    ac = ac / np.maximum(nw, 1)
    cy, cx = np.array(ac.shape) // 2
    ac /= ac[cy, cx]
    e = np.radians(T.EQ_DEG)
    out = []
    for ang in (e, e + np.pi / 2):
        r = np.arange(0, 40, 0.25)  # pixels
        prof = scipy.ndimage.map_coordinates(ac, [cy + r * np.sin(ang), cx + r * np.cos(ang)], order=1)
        k = np.argmax(prof < 0.5)
        rk = r[k - 1] + (prof[k - 1] - 0.5) / (prof[k - 1] - prof[k]) * (r[k] - r[k - 1])
        out.append(2 * rk * T.PX)
    return out


def look(st, t):
    for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_chan1.csv")):
        if r["rx_start_utc"][11:19] == t:
            return [(t, os.path.join(RUN, os.path.basename(r["map_npy"])),
                     os.path.join(RUN, os.path.basename(r["mult_npy"])))]


by = {(st, l[0]): l for st, L in (("s", T.S), ("w", T.W)) for l in L}
pick = lambda *keys: [by[k] for k in keys]
S8 = [l for l in T.S if l[0] != "09:49:09"]
cases = [
    ("single look Stockert 09:41 (125 s, 25 kchip/s)", look("stockert", "09:41:25")),
    ("single look Stockert 09:28 (35 s, 25 kchip/s)", look("stockert", "09:28:21")),
    ("single look Stockert 09:49:57 (35 s, 50 kchip/s)", look("stockert", "09:49:57")),
    ("8 looks: Stockert only", S8),
    ("8 looks: WSRT only (same times)", T.W),
    ("8 looks: mixed, same times, alternate stations", pick(("s", "09:24:58"), ("w", "09:27:04"), ("s", "09:28:21"), ("w", "09:35:24"),
                                                       ("s", "09:41:25"), ("w", "09:46:36"), ("s", "09:49:57"), ("w", "10:16:10"))),
    ("8 looks: mixed, the other alternation", pick(("w", "09:24:58"), ("s", "09:27:04"), ("w", "09:28:21"), ("s", "09:35:24"),
                                                   ("w", "09:41:25"), ("s", "09:46:36"), ("w", "09:49:57"), ("s", "10:16:10"))),
    ("16 looks: both stations", S8 + T.W),
    ("6 looks: 125 s only, both stations (3 times)", pick(("s", "09:35:24"), ("s", "09:41:25"), ("s", "09:46:36"),
                                                         ("w", "09:35:24"), ("w", "09:41:25"), ("w", "09:46:36"))),
    ("6 looks: 35 s 25 kchip/s only, both (3 times)", pick(("s", "09:24:58"), ("s", "09:27:04"), ("s", "09:28:21"),
                                                          ("w", "09:24:58"), ("w", "09:27:04"), ("w", "09:28:21"))),
]
print(f"\n{'map':55s} {'width along':>12s} {'width across':>13s}  (km, Doppler equator)")
for name, items in cases:
    if True:
        h = T.stack_of(items) if len(items) > 1 else None
        if h is None:
            m = np.load(items[0][1]); mu = np.load(items[0][2])
            med = np.median(mu[mu > 0])
            h = np.where((m != hp.UNSEEN) & (mu <= 3 * med), m, hp.UNSEEN)
    a, c = ac_widths(T.disk_grid(h))
    print(f"{name:55s} {a:12.1f} {c:13.1f}")
