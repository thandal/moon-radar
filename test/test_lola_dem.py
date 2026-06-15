"""
Tests for the LOLA DEM surface (REPORT 8.4): PDS loader correctness,
interpolator fidelity at the grid nodes and across the lon seam, agreement
between DEM resolutions, and the topographic light-time shift through both
the SPICE path and the anchored-station field used by lunar_projection.

Requires lola_dem/ldem_16.img and ldem_64.img (run ./fetch_lola_dem.sh).

Run from the repo root:  .conda/bin/python test/test_lola_dem.py
"""

import os
import sys

import numpy as np
import healpy as hp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spice_setup import furnsh_kernels
furnsh_kernels()

import doppler_equator as de

RX_TIME_UTC = "2025-09-16 13:23:26"

failures = []


def check(name, value, bound, unit=""):
    ok = value < bound
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3e} {unit} (bound {bound:.0e})")
    if not ok:
        failures.append(name)


rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. Loader: grid geometry and values against the raw PDS file
# ---------------------------------------------------------------------------
path16 = os.path.join(de.LOLA_DEM_DIR, "ldem_16.img")
de.load_lola_dem(path16)

dn = np.fromfile(path16, dtype="<i2").reshape(2880, 5760)
# Physical range of lunar topography (the label's MINIMUM/MAXIMUM are ~0.1 km
# stale against the file): min ~-9.1 km (Antoniadi), max ~+10.8 km (farside).
check("global min physical", abs(de._LOLA_INTERPOLATOR.values.min() + 9.1),
      0.3, "km")
check("global max physical", abs(de._LOLA_INTERPOLATOR.values.max() - 10.8),
      0.3, "km")

# Interpolation at exact pixel centers must reproduce the raw samples.
i = rng.integers(1, 2879, 300)   # 0-based row (north first), avoid pole rows
j = rng.integers(0, 5760, 300)
lat = np.radians(90.0 - (i + 0.5) / 16.0)
lon = np.radians((j + 0.5) / 16.0)
u = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                     np.sin(lat)])
elev = de.get_lola_elevation(u)
check("node values vs raw DN", np.abs(elev - dn[i, j] * 0.5e-3).max(), 1e-5, "km")

# Magnitudes must not matter (directions, not positions).
elev_scaled = de.get_lola_elevation(u * 1737.4)
check("scale invariance", np.abs(elev_scaled - elev).max(), 1e-9, "km")

# Continuity across the 0/360 lon seam: both sides interpolate the same two
# physical columns, so the residual is just terrain slope x 2*eps.
lat_s = np.radians(rng.uniform(-80, 80, 200))
eps = 1e-6
u_w = np.column_stack([np.cos(lat_s) * np.cos(-eps), np.cos(lat_s) * np.sin(-eps),
                       np.sin(lat_s)])
u_e = np.column_stack([np.cos(lat_s) * np.cos(eps), np.cos(lat_s) * np.sin(eps),
                       np.sin(lat_s)])
check("lon seam continuity", np.abs(de.get_lola_elevation(u_w) -
                                    de.get_lola_elevation(u_e)).max(), 1e-2, "km")

# ---------------------------------------------------------------------------
# 2. Cross-resolution agreement (16 vs 64 ppd are independent gridding runs)
# ---------------------------------------------------------------------------
lat_r = np.arcsin(rng.uniform(-1, 1, 2000))
lon_r = rng.uniform(0, 2 * np.pi, 2000)
u_r = np.column_stack([np.cos(lat_r) * np.cos(lon_r),
                       np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
e16 = de.get_lola_elevation(u_r)
de.load_lola_dem(os.path.join(de.LOLA_DEM_DIR, "ldem_64.img"))
e64 = de.get_lola_elevation(u_r)
check("16 vs 64 ppd median |diff|", np.median(np.abs(e64 - e16)), 0.05, "km")
check("16 vs 64 ppd p95 |diff|", np.quantile(np.abs(e64 - e16), 0.95), 0.5, "km")

# ---------------------------------------------------------------------------
# 3. moon_surface_points: radial displacement and ellipsoid fallback
# ---------------------------------------------------------------------------
p_dem = de.moon_surface_points(u_r, use_dem=True)
check("DEM radius = sphere + elevation",
      np.abs(np.linalg.norm(p_dem, axis=1) - (1737.4 + e64)).max(), 1e-9, "km")
p_ell = de.moon_surface_points(u_r)
check("ellipsoid radius unchanged",
      np.abs(np.linalg.norm(p_ell, axis=1) - 1737.4).max(), 1e-9, "km")
p_one = de.moon_surface_points(u_r[0], use_dem=True)  # single-point shape
check("single-point shape/value", np.abs(p_one - p_dem[0]).max(), 1e-12, "km")

# ---------------------------------------------------------------------------
# 4. Topographic light-time shift: SPICE path vs anchored-station field
# ---------------------------------------------------------------------------
rx_time = de.csp.str2et(RX_TIME_UTC)
v = np.array(hp.pix2vec(32, np.arange(hp.nside2npix(32)))).T
v = v[v[:, 0] > 0.2]  # near side, off the limb
lt_ell = de.moonPointLightTime_BCK(rx_time, v)
lt_dem = de.moonPointLightTime_BCK(rx_time, v, use_dem=True)
R_rx, R_tx, c = de.apparent_station_positions(rx_time)
pe = de.moon_surface_points(v)
pd = de.moon_surface_points(v, use_dem=True)
d_anchor = ((np.linalg.norm(pd - R_rx, axis=1) + np.linalg.norm(pd - R_tx, axis=1))
            - (np.linalg.norm(pe - R_rx, axis=1) + np.linalg.norm(pe - R_tx, axis=1))) / c
check("DEM lt shift: SPICE vs anchored field",
      np.abs((lt_dem - lt_ell) - d_anchor).max(), 1e-9, "s")
# First-order physics: radial displacement h shifts the two-leg light time
# by -h*(cos th_rx + cos th_tx)/c; the residual is the h^2/range curvature
# term (<~1e-9 s for |h| <= 9 km at 380,000 km).
e = de.get_lola_elevation(v)
n = pe / np.linalg.norm(pe, axis=1, keepdims=True)
w_rx = R_rx - pe
w_tx = R_tx - pe
cos_rx = np.einsum("ij,ij->i", n, w_rx) / np.linalg.norm(w_rx, axis=1)
cos_tx = np.einsum("ij,ij->i", n, w_tx) / np.linalg.norm(w_tx, axis=1)
expected = -e * (cos_rx + cos_tx) / c
check("DEM lt shift vs h*(cos+cos)/c",
      np.abs((lt_dem - lt_ell) - expected).max(), 2e-9, "s")

# ---------------------------------------------------------------------------
# 5. SRP elevation extraction (terrain split of per-look timing offsets)
# ---------------------------------------------------------------------------
elev_srp, topo_delay = de.extract_srp_elevation(rx_time)
print(f"      SRP elevation {elev_srp:+.3f} km, topo delay {topo_delay*1e6:+.2f} us")
check("SRP elevation plausible", abs(elev_srp), 9.5, "km")
# At the SRP incidence is near-normal: the actual SPICE lt shift of the
# DEM-displaced SRP must equal -2h/c to <1%.
srp = de.specular_point_bck(rx_time)
d_srp = (de.moonPointLightTime_BCK(rx_time, srp, use_dem=True)
         - de.moonPointLightTime_BCK(rx_time, srp))
check("SRP topo delay vs SPICE", abs(d_srp + topo_delay), 0.01 * abs(topo_delay) + 1e-12, "s")

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    raise SystemExit(1)
print("All LOLA DEM checks passed.")
