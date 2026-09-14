# LOLA terrain projection: implemented

This file replaces the pre-implementation plan. Terrain projection is in
production; results and remaining limitations are in
[REPORT §6](REPORT.md#6-remaining-limits). The original proposal remains in git
history.

The implementation differs from the proposal in several useful details:

- [fetch_lola_dem.sh](fetch_lola_dem.sh) downloads PDS GDR `.img`/`.lbl` pairs
  into `lola_dem/`. The loader selects the highest resolution available;
  GeoTIFF and `rasterio` are not used.
- `load_lola_dem` in [doppler_equator.py](doppler_equator.py) reads the label's
  scale and reference radius, constructs pixel-center coordinates, and wraps
  the longitude seam. Heights are relative to the 1737.4 km sphere.
- `moon_surface_points(use_dem=True)` adds interpolated terrain heights.
  The projection evaluates both delay and Doppler at those surface points.
- The SRP solver and rim curves keep the smooth ellipsoid. The runs CSV
  records `use_dem`, `srp_elevation_km`, and `srp_topo_delay_us`; the last is
  the signed `2h/c` diagnostic, positive for terrain above the reference.

Without a DEM, processing reports an ellipsoid fallback. Terrain correction
is validated by [test_lola_dem.py](test/test_lola_dem.py) and the
[projection comparison](validation/results/lola_dem_projection.json).
Determining how much of the measured timing variation comes from terrain,
SDR settings, or clock drift remains open.
