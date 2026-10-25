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


