"""Projection and stacking, measured on the rebuilt 2026-10-11 circular set.

1. Mirror sharing: each delay-Doppler cell holds the echo of two surface
   points, one each side of the Doppler equator; the projection gives the
   cell's full value to both. Correlate each map with itself reflected across
   the look's Doppler-equator plane (detail 4-40 km).
2. Noise floor: mean intensity in the delay columns before the echo starts
   (pure noise) vs the median map intensity.
3. Speckle per look: equivalent number of independent looks (1 / variance of
   intensity about its local mean) -- what inverse-variance weights would use.
"""
import csv
import json
import os
import sys

import numpy as np
import healpy as hp
import scipy.ndimage
from astropy import time as at

sys.path.insert(0, "/tmp/eve_check/blur")
sys.path.insert(0, "/home/than/code/moon-radar")
import doppler_equator as de
import tiles as T
import stack_maps as sm

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
DATA = "/home/than/code/moon-radar/data/eve2026_circ/2026-10-11"
NSIDE = 400
v_pix = np.array(hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE)))).T


def geometry(rx_name, t0, T_s):
    """SRP direction s and the reflection normal u (perpendicular to the
    Doppler-equator plane) at mid-look."""
    tm = t0 + T_s / 2
    s = de.specular_point_bck(tm, "DWINGELOO", rx_name)
    s = s / np.linalg.norm(s)
    v = np.array(hp.pix2vec(64, np.arange(hp.nside2npix(64)))).T
    p = de.moon_surface_points(v)
    sta0 = de.apparent_station_positions(tm - 1, "DWINGELOO", rx_name)
    sta1 = de.apparent_station_positions(tm + 1, "DWINGELOO", rx_name)
    lt = lambda sta: (np.linalg.norm(p - sta[0], axis=1) + np.linalg.norm(p - sta[1], axis=1)) / sta[2]
    l0, l1 = lt(sta0), lt(sta1)
    dlt = (l1 - l0) / 2
    vis = (l0 - l0.min()) < 0.0115
    A = np.column_stack([np.ones(vis.sum()), v[vis]])
    g = np.linalg.lstsq(A, dlt[vis], rcond=None)[0][1:]
    w = g - (g @ s) * s
    w /= np.linalg.norm(w)
    u = np.cross(s, w)
    # check: delay and Doppler of points vs their mirrors
    pm = de.moon_surface_points(v - 2 * (v @ u)[:, None] * u)
    return s, w, u, (p, pm, sta0, sta1, vis)


def mirror_check(geo, f_hz, fs, ddlt_hz):
    p, pm, sta0, sta1, vis = geo
    lt = lambda q, sta: (np.linalg.norm(q - sta[0], axis=1) + np.linalg.norm(q - sta[1], axis=1)) / sta[2]
    d_delay = (lt(pm, sta0) - lt(p, sta0))[vis] * fs
    d_dopp = (((lt(pm, sta1) - lt(pm, sta0)) - (lt(p, sta1) - lt(p, sta0))) / 2)[vis] * f_hz / ddlt_hz
    return np.percentile(np.abs(d_delay), 90), np.percentile(np.abs(d_dopp), 90)


def hp_detail(m):
    """Disk-grid log map with >40 km trend removed (tiles.disk_grid)."""
    return T.disk_grid(m)


rows = []
for st in ("stockert", "wsrt"):
    for chan in ("chan1",):
        for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_{chan}.csv")):
            rows.append((st, r))

print("1. mirror sharing, single looks (correlation of detail 4-40 km between each")
print("   point and its reflection across the Doppler-equator plane)")
print(f"{'look':22s} {'mirror delay/Doppler 90% (samples/rows)':>40s} {'corr, own look':>15s}")
mirror_maps = {}
geo_by = {}
for st, r in rows:
    t = r["rx_start_utc"][11:19]
    if t not in ("09:28:21", "09:41:25", "10:16:10", "12:08:02"):
        continue
    meta = json.load(open(os.path.join(DATA, r["rx_file"])))
    t0 = de.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
    T_s = float(r["capture_s"])
    s, w, u, geo = geometry(st.upper(), t0, T_s)
    dd = np.load(r["dd_npz"])
    ddlt_hz = abs(dd["doppler_hz"][1] - dd["doppler_hz"][0])
    dly, dop = mirror_check(geo, float(r["frequency_hz"]), float(r["sample_rate_hz"]), ddlt_hz)
    mirror_pix = hp.vec2pix(NSIDE, *(v_pix - 2 * (v_pix @ u)[:, None] * u).T)
    geo_by[(st, t)] = mirror_pix
    m = np.load(r["map_npy"]); mu = np.load(r["mult_npy"])
    med = np.median(mu[mu > 0])
    m = np.where((m != hp.UNSEEN) & (mu <= 3 * med), m, hp.UNSEEN)
    mm = np.where(m[mirror_pix] != hp.UNSEEN, m[mirror_pix], hp.UNSEEN)
    a, b = hp_detail(m), hp_detail(mm)
    ok = np.isfinite(a) & np.isfinite(b)
    # exclude the band near the Doppler equator, where a point is its own mirror
    c = np.corrcoef(a[ok], b[ok])[0, 1]
    print(f"{st + ' ' + t:22s} {f'{dly:.1f} / {dop:.1f}':>40s} {c:15.3f}")

# stack vs its own reflection (mean geometry of the 09:24-10:16 looks: use 09:41)
print("\n   stack (all opposite-sense looks, both stations) vs its reflection with the 09:41 geometry:")
L = {}
for st, r in rows:
    if float(r["tone_snr"]) >= 15:
        L[(st, r["rx_start_utc"][11:19])] = (r["map_npy"], r["mult_npy"])
acc = None
for k in L:
    I = sm.look_linear_intensity(*L[k])
    acc = acc or sm.Accumulator(len(I))
    acc.add(I)
stk = acc.stacked_log(3)
mp = geo_by[("stockert", "09:41:25")]
a, b = hp_detail(stk), hp_detail(np.where(stk[mp] != hp.UNSEEN, stk[mp], hp.UNSEEN))
ok = np.isfinite(a) & np.isfinite(b)
print(f"   corr {np.corrcoef(a[ok], b[ok])[0, 1]:.3f}")

print("\n2. noise floor and 3. speckle per look (opposite sense)")
print(f"{'look':22s} {'T s':>4s} {'noise / median map intensity':>29s} {'independent looks per pixel':>28s}")
for st, r in rows:
    dd = np.load(r["dd_npz"])
    I = np.exp(2.0 * dd["log_A"].astype(np.float64))
    noise = I[:, :12].mean()                    # lags before the echo's leading edge
    m = np.load(r["map_npy"]); mu = np.load(r["mult_npy"])
    ok = (m != hp.UNSEEN) & (mu <= 3 * np.median(mu[mu > 0]))
    med_int = np.exp(2.0 * np.median(m[ok]))
    # equivalent looks: intensity about a 20 km local mean on the disk grid
    lin = np.where(ok, np.exp(2.0 * (m - np.median(m[ok]))), hp.UNSEEN)
    g = T.disk_grid(np.where(ok, np.log(np.where(ok, lin, 1.0)) / 2, hp.UNSEEN))
    x = np.exp(2.0 * g[np.isfinite(g)])
    enl = 1.0 / np.var(x / np.mean(x))
    print(f"{st + ' ' + r['rx_start_utc'][11:19]:22s} {float(r['tx_s']):4.0f} {noise / med_int:29.3f} {enl:28.2f}")
