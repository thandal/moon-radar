"""Doppler focus over a look, from the geometry.

1. Common error: the pipeline removes the specular point's Doppler with a
   straight line in frequency (compute_dd_image: doppler_start -> doppler_end,
   each a central difference over +/- T/2). The true Doppler curves (Earth
   rotation). Residual after removing its mean (the rim calibration absorbs
   a constant): peak-to-peak frequency, phase error, and the resulting
   Doppler response vs the ideal 1/T.
2. Differential drift: each surface point's Doppler relative to the specular
   point changes during the look; the projection places the point at its
   window average, so the change over the look smears it across rows.
"""
import csv
import json
import os
import sys

import numpy as np
import healpy as hp
from astropy import time as at

sys.path.insert(0, "/home/than/code/moon-radar")
import doppler_equator as de
import doppler_equator_alignment as dea

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
DATA = "/home/than/code/moon-radar/data/eve2026_circ/2026-10-11"


def response_width(phase, T, n=1 << 16):
    """-3 dB width (Hz) and peak gain of |FT of exp(i phase)| over the look."""
    z = np.exp(1j * phase)
    spec = np.abs(np.fft.fftshift(np.fft.fft(z, n))) ** 2
    spec /= np.abs(np.fft.fft(np.ones(len(z)), n)[0]) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(n, T / len(z)))
    above = f[spec >= spec.max() / 2]
    return above.max() - above.min(), spec.max()


looks = []
for st in ("stockert", "wsrt"):
    for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_chan1.csv")):
        if r["rx_start_utc"][11:19] in ("09:35:24", "09:41:25", "10:16:10", "09:28:21", "12:08:02"):
            looks.append((st, r))

print("\n2. drift of each point's Doppler relative to the specular point during the look,")
print("   near-side points (nside 32), peak-to-peak over the look, in mHz")
de.load_lola_dem()
nside = 32
v = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
p = de.moon_surface_points(v, use_dem=True)
for st, r in looks:
    rx_name = st.upper()
    meta = json.load(open(os.path.join(DATA, r["rx_file"])))
    t0 = de.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
    T = float(r["capture_s"])
    f_hz = float(r["frequency_hz"])
    ts = np.linspace(0, T, 26)
    rows = []
    for tt in ts:
        R_rx, R_tx, cc = de.apparent_station_positions(t0 + tt, "DWINGELOO", rx_name)
        srp = de.specular_point_bck(t0 + tt, "DWINGELOO", rx_name)
        lt_p = (np.linalg.norm(p - R_rx, axis=1) + np.linalg.norm(p - R_tx, axis=1)) / cc
        lt_s = (np.linalg.norm(srp - R_rx) + np.linalg.norm(srp - R_tx)) / cc
        rows.append(lt_p - lt_s)
    dlt = np.array(rows)                       # (time, point) differential light time
    lt_now = dlt[0] + 0                        # visible: within the delay window
    near = (lt_now >= -1e-6) & (lt_now < 2 * 1737.4 / 299792.458)
    # differential Doppler vs time from a quadratic fit per point
    cf = np.polynomial.polynomial.polyfit(ts, dlt[:, near], 3)
    tt = np.linspace(0, T, 200)
    dd = np.polynomial.polynomial.polyval(tt, np.polynomial.polynomial.polyder(cf))  # (point, time)
    fdiff = -dd * f_hz                         # (points, time)
    ptp = np.ptp(fdiff, axis=1) * 1000
    q = np.percentile(ptp, [50, 90, 99, 100])
    print(f"   {st + ' ' + r['rx_start_utc'][11:19]:24s} T {T:4.0f} s (1/T {1000 / T:4.1f} mHz): "
          f"median {q[0]:5.1f}, 90% {q[1]:5.1f}, 99% {q[2]:5.1f}, max {q[3]:5.1f}")
