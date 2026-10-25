"""_LolaGrid vs scipy RegularGridInterpolator: bit-identical?"""
import os, sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, "/home/than/code/moon-radar")
from spice_setup import furnsh_kernels; furnsh_kernels()
import doppler_equator as de

rng = np.random.default_rng(1)
# 1. ldem_16, whole sphere, plus the poles and the lon seam
de.load_lola_dem(os.path.join(de.LOLA_DEM_DIR, "ldem_16.img"))
G = de._LOLA_INTERPOLATOR
rgi = RegularGridInterpolator(G.grid, G.values, bounds_error=False, fill_value=None)
n = 3_000_000
lat = np.arcsin(rng.uniform(-1, 1, n))
lat[:2000] = np.radians(rng.uniform(89.9, 90.0, 2000)) * rng.choice([-1, 1], 2000)  # polar slivers
lon = rng.uniform(0, 2 * np.pi, n)
lon[2000:4000] = rng.choice([0.0, 1e-9, 2 * np.pi - 1e-9], 2000)                    # seam
xi = np.stack([lat, lon], axis=-1)
a, b = G(xi), rgi(xi)
print(f"ldem_16 global ({n} points): identical {np.array_equal(a, b)}, max diff {np.abs(a - b).max():.2e}")
v = G.values
u = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)[:200000]
e1 = de.get_lola_elevation(u)
print("get_lola_elevation shape", e1.shape, "matches rgi:", np.array_equal(e1, rgi(np.stack([np.arctan2(u[:, 2], np.hypot(u[:, 0], u[:, 1])), np.mod(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)], -1))))

# 2. ldem_64: a cropped block (RGI on the crop gives the full-grid answer for interior points)
de.load_lola_dem(os.path.join(de.LOLA_DEM_DIR, "ldem_64.img"))
G = de._LOLA_INTERPOLATOR
lats, lons = G.grid
r0, r1, c0, c1 = 4000, 7000, 21000, 23042        # incl. the wrapped east edge column
ii, jj = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
block = G._nodes(ii, jj)
rgi = RegularGridInterpolator((lats[r0:r1], lons[c0:c1]), block, bounds_error=False, fill_value=None)
m = 2_000_000
xi = np.stack([rng.uniform(lats[r0 + 1], lats[r1 - 2], m), rng.uniform(lons[c0 + 1], lons[c1 - 1], m)], -1)
a, b = G(xi), rgi(xi)
print(f"ldem_64 block ({m} points): identical {np.array_equal(a, b)}, max diff {np.abs(a - b).max():.2e}")
