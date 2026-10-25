"""Per-look orientation of the Doppler equator (the high-ambiguity stripe in
each look's DD-cell multiplicity map), as a position angle on the disk seen
from lon 0 lat 0 (degrees counter-clockwise from the +lon axis)."""
import csv
import os
import sys

import numpy as np
import healpy as hp

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
for st in ["stockert", "wsrt"]:
    for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_chan1.csv")):
        mult = np.load(os.path.join(RUN, os.path.basename(r["mult_npy"])))
        nside = hp.npix2nside(len(mult))
        med = np.median(mult[mult > 0])
        hi = np.where(mult > 3 * med)[0]
        th, ph = hp.pix2ang(nside, hi)
        lat = np.pi / 2 - th
        lon = np.where(ph > np.pi, ph - 2 * np.pi, ph)
        near = np.cos(lat) * np.cos(lon) > 0.5    # front side, within 60 deg of centre
        x = np.cos(lat[near]) * np.sin(lon[near])
        y = np.sin(lat[near])
        c = np.cov(np.vstack([x, y]))
        w, v = np.linalg.eigh(c)
        ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
        ang = (ang + 90) % 180 - 90
        print(f"{st:8s} {r['rx_start_utc'][11:19]}  Doppler-equator angle {ang:+6.1f} deg "
              f"(stripe pixels {near.sum()}, elongation {np.sqrt(w[1] / w[0]):.1f})")
