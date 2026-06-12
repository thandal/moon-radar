# LOLA DEM Projection Implementation Plan

Based on the analysis of `REPORT.md` and the existing codebase, here is the technical plan to implement the LOLA DEM projection (Item 8.4 in the Open Items list). This will replace the ellipsoid surface mapping and correct the dominant mapping systematic (~7 delay px ≈ 4 km).

## 1. Data Source & Preparation
We will use a global LOLA Digital Elevation Model (DEM) in GeoTIFF format. A suitable choice is the global 118m/pixel (or a decimated version depending on memory constraints) Gridded Data Record (GDR) from the PDS Geosciences node or USGS Astropedia.

**Recommended dependencies:** `rasterio` (for reading the GeoTIFF) and `scipy.interpolate` (for fast evaluation).

## 2. Core Interpolation Module
We need a fast, vectorized lookup for elevation given a surface vector. We can create a new module or add to `doppler_equator.py`.

```python
import rasterio
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import cspyce as csp

_LOLA_INTERPOLATOR = None

def load_lola_dem(dem_path):
    global _LOLA_INTERPOLATOR
    with rasterio.open(dem_path) as src:
        # Assuming a global DEM in Plate Carree (equirectangular)
        dem_data = src.read(1)
        # Construct lat/lon grids based on GeoTIFF transform
        # ...
        # Create interpolator
        _LOLA_INTERPOLATOR = RegularGridInterpolator((lats, lons), dem_data, bounds_error=False, fill_value=0)

def get_lola_elevation(p_moon):
    """Convert Cartesian vectors to lat/lon and sample the DEM."""
    # p_moon is (N, 3)
    rad, lon, lat = csp.reclat_vector(p_moon) # If available, or use numpy arctan2/arcsin
    # Query interpolator
    elevations = _LOLA_INTERPOLATOR((lat, lon))
    return elevations
```

## 3. Modifying `moon_surface_points`
The beauty of the current pipeline is that `lunar_projection` in `doppler_equator_alignment.py` maps healpix vectors `v` to surface points `p`, and computes the two-leg light time `lt_field` directly from `p`. If `p` includes topographic elevation, the delay shifts and differential Doppler effects are **automatically** handled by the existing geometry engine.

Modify `doppler_equator.py`:

```diff
 def moon_surface_points(p_moon, use_dem=True):
     r = moon_radii()
-    return csp.edpnt_vector(p_moon, r[0], r[1], r[2])
+    if not use_dem or _LOLA_INTERPOLATOR is None:
+        return csp.edpnt_vector(p_moon, r[0], r[1], r[2])
+
+    # 1. Normalize p_moon
+    norm = np.linalg.norm(p_moon, axis=-1, keepdims=True)
+    u = p_moon / norm
+    
+    # 2. Get elevations (in km or m, ensure units match r)
+    elevations = get_lola_elevation(u)
+    
+    # 3. Scale unit vectors by (Ellipsoid Radius + Topography)
+    return u * (r[0] + elevations)[..., np.newaxis]
```

## 4. Addressing the SRP & Timing Jitter
The `REPORT.md` states: *"correlate per-look timing offsets with SRP-local elevation to split SDR jitter from terrain"*.

**Crucial Insight for the SRP Solver (`_specular_zoom`)**:
The tangent-plane zoom (`_specular_zoom`) expects a smooth, convex surface (like an ellipsoid). If it evaluates light times over a bumpy DEM, it is highly likely to get trapped in local minima.
*   **Strategy**: Continue using the **ellipsoid** for the `specular_point_bck` calculation to establish the stable geometric anchor and apparent station positions.
*   **Topography Extraction**: Once the ellipsoid SRP is found, we evaluate the LOLA DEM at that specific coordinate.
*   **Correlation**: The expected delay shift is `2 * h / c`. We can subtract this topographic delay from the per-look timing offset (the +35–45 µs mentioned in the report) to isolate the true SDR hardware jitter.

```python
def extract_srp_elevation(rx_time, tx_name, rx_name):
    # Find smooth ellipsoid SRP
    srp_ellipsoid = specular_point_bck(rx_time, tx_name, rx_name)
    # Lookup elevation
    elev = get_lola_elevation(np.array([srp_ellipsoid]))[0]
    # Topographic delay shift
    delay_shift_s = 2 * (elev * 1000) / 299792458.0 
    return elev, delay_shift_s
```

## 5. Next Steps
1. **Acquire the Data**: Determine if the team has a standard LOLA DEM GeoTIFF on hand, or write a fetch script to download one from PDS/Astropedia.
2. **Environment Validation**: Ensure `rasterio` and `scipy` are successfully installed in the target conda environment (current execution environment showed missing bash for subprocesses).
3. **Execution & Verification**: Run the `doppler_equator_alignment.py` batch and verify that the `~7 px` systematic shift is resolved in the newly generated lunar maps.
