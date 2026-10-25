"""Copernicus: viewing geometry per look, and the crater's position in each
look measured against the OTHER station's stack (independent noise)."""
import csv
import os
import sys

import numpy as np
import healpy as hp
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

sys.path.insert(0, "/home/than/code/moon-radar")
import doppler_equator_alignment as dea
import registration_analysis as ra
import stack_maps as sm

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
V = "/tmp/eve_check/blur"
LON0, LAT0 = -20.08, 9.62           # Copernicus centre
KM_DEG = 1737.4 * np.pi / 180
step = 0.05
lon = np.arange(LON0 - 4, LON0 + 4 + step / 2, step)
lat = np.arange(LAT0 - 4, LAT0 + 4 + step / 2, step)


def highpass(g, sigma_deg=1.5):
    ok = np.isfinite(g)
    v = np.where(ok, g, 0.0)
    s = sigma_deg / step
    tr = scipy.ndimage.gaussian_filter(v, s) / np.maximum(scipy.ndimage.gaussian_filter(ok.astype(float), s), 1e-9)
    return np.where(ok, g - tr, 0.0)


lo, la = np.radians(LON0), np.radians(LAT0)
n = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
p = 1737.4 * n
east = np.array([-np.sin(lo), np.cos(lo), 0.0])
north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])

ref = {st: highpass(ra.grid_map(np.load(f"{RUN}/stacked_map_{st}_chan1.npy"), lon, lat))
       for st in ("stockert", "wsrt")}
dy, dx, pk, sg = ra.xcorr_offset(ref["stockert"], ref["wsrt"], 12, 3)
print(f"Copernicus, WSRT stack relative to Stockert stack: east {-dx*step*KM_DEG*np.cos(la):+.2f} km, "
      f"north {-dy*step*KM_DEG:+.2f} km (signif {sg:.2f})\n")

print(f"{'look':18s} {'incid':>6s} {'look-from az':>12s} {'bistatic':>8s}   Copernicus position vs other station's stack")
rows_out = []
for st in ("stockert", "wsrt"):
    other = "wsrt" if st == "stockert" else "stockert"
    for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_chan1.csv")):
        et = dea.csp.str2et(r["rx_start_utc"])
        dur = 125 if "4x" in r["tx_file"] else 35
        R_rx, R_tx, _ = dea.apparent_station_positions(et + dur / 2, rx_name=st.upper())
        u = (R_rx - p) / np.linalg.norm(R_rx - p)
        w = (R_tx - p) / np.linalg.norm(R_tx - p)
        b = (u + w) / np.linalg.norm(u + w)
        inc = np.degrees(np.arccos(b @ n))
        az = np.degrees(np.arctan2(b @ east, b @ north)) % 360
        bis = np.degrees(np.arccos(u @ w)) * 3600
        m = np.load(os.path.join(RUN, os.path.basename(r["map_npy"])))
        mu = np.load(os.path.join(RUN, os.path.basename(r["mult_npy"])))
        g = highpass(ra.grid_map(m, lon, lat, multiplicity=mu))
        dy, dx, pk, sg = ra.xcorr_offset(ref[other], g, 12, 3)
        de, dn = -dx * step * KM_DEG * np.cos(la), -dy * step * KM_DEG
        tag = "50k" if "fzc50000" in r["tx_file"] else "25k"
        print(f"{st:8s} {r['rx_start_utc'][11:19]} {inc:6.2f} {az:12.2f} {bis:7.1f}\"   "
              f"east {de:+5.1f} km  north {dn:+5.1f} km  (signif {sg:.2f}, corr {pk:.2f}) {dur} s {tag}")
        rows_out.append((st, r["rx_start_utc"][11:19], inc, az, de, dn, sg))

# Zoomed view of the crater in several stacks
def build(items, min_count):
    acc = None
    for mp, mu in items:
        I = sm.look_linear_intensity(mp, mu)
        acc = acc or sm.Accumulator(len(I))
        acc.add(I)
    return acc.stacked_log(min_count)

late = []
for r in csv.DictReader(open(f"{RUN}/registration_runs_wsrt_chan1.csv")):
    if r["rx_start_utc"][11:13] == "12":
        late.append((os.path.join(RUN, os.path.basename(r["map_npy"])),
                     os.path.join(RUN, os.path.basename(r["mult_npy"]))))
panels = [("Stockert, 9 looks", np.load(f"{RUN}/stacked_map_stockert_chan1.npy")),
          ("WSRT, 8 looks (same times)", np.load(f"{V}/var_chan1_W_early.npy")),
          ("WSRT 12:08 + 12:09 only", build(late, 2)),
          ("Stockert+WSRT, 19 looks", np.load(f"{RUN}/stacked_map_stockert_wsrt_chan1.npy")),
          ("11 sharp looks (125 s or 50k)", np.load(f"{V}/var_chan1_sharp.npy"))]
zs = 0.025
zl = np.arange(LON0 - 3, LON0 + 3 + zs / 2, zs)
zb = np.arange(LAT0 - 3, LAT0 + 3 + zs / 2, zs)
fig, axes = pl.subplots(1, len(panels), figsize=(4 * len(panels), 4.4), constrained_layout=True)
for ax, (name, h) in zip(axes, panels):
    g = ra.grid_map(h, zl, zb)
    v = g[np.isfinite(g)]
    a0, a1 = np.percentile(v, [2, 99.7])
    ax.imshow(g, origin="lower", extent=[zl[0], zl[-1], zb[0], zb[-1]], cmap="gray",
              vmin=a0, vmax=a1, interpolation="nearest")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("lon (deg)")
axes[0].set_ylabel("lat (deg)")
fig.savefig(f"{V}/copernicus_zoom.png", dpi=80)
print("saved", f"{V}/copernicus_zoom.png")
