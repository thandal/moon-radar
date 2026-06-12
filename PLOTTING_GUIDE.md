# Doppler Equator Plotting Guide

Quick reference for plotting doppler equator curves and error analysis visualizations.

## Simple Nominal Plots (No Uncertainties)

### Option 1: Use the standalone script

```bash
python plot_doppler_equator_simple.py
```

Generates: `results/ERRORS/doppler_equator_clean.png`

### Option 2: Use the plotting function directly

```python
import cspyce as csp
from astropy import time as at
from doppler_equator_errors import plot_equator_nominal

# Setup your observation time
obs_time = at.Time("2025-09-16T13:22:02")
rx_time = csp.str2et(obs_time.utc.value)

# Plot
fig, ax = plot_equator_nominal(rx_time)
fig.savefig("my_doppler_equator.png", dpi=150)
```

This gives you clean doppler equator curves showing just the Up-Doppler and Down-Doppler boundaries.

## With Uncertainty Bounds (Optional)

If you want to show ephemeris uncertainties:

```python
from doppler_equator_errors import plot_equator_with_errors

# Uncertainties are TINY (~10^-12), so scale for visibility
fig, ax = plot_equator_with_errors(
    rx_time,
    n_sigma=3,
    include_model_errors=False,  # Only ephemeris uncertainties
    scale_factor=1e6  # Scale by 1 million to see them!
)
```

**Note**: Real ephemeris uncertainties are ~7 parts per trillion. You must scale by ~10⁶ to make them visible on a plot. The plot title will show they are scaled.

## Comparing Methods

To compare the three different computation methods (HEALPix, velocity, terminator):

```python
from doppler_equator import (
    compute_doppler_equator_healpix,
    compute_doppler_equator_velocity,
    compute_doppler_equator_terminator
)
import matplotlib.pyplot as plt

# Compute with all three methods
lt1, delay1, dlt_max1, dlt_min1 = compute_doppler_equator_healpix(rx_time)
lt2, d_up2, dlt_up2, d_down2, dlt_down2 = compute_doppler_equator_velocity(rx_time)
lt3, d_up3, dlt_up3, d_down3, dlt_down3 = compute_doppler_equator_terminator(rx_time)

# Plot all three
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(dlt_max1, delay1, 'b-', label='HEALPix (max)')
ax.plot(dlt_min1, delay1, 'r-', label='HEALPix (min)')
ax.plot(dlt_up2, d_up2, 'b--', label='Velocity (up)')
ax.plot(dlt_down2, d_down2, 'r--', label='Velocity (down)')
ax.plot(dlt_up3, d_up3, 'b:', label='Terminator (up)')
ax.plot(dlt_down3, d_down3, 'r:', label='Terminator (down)')
ax.invert_yaxis()
ax.legend()
ax.set_xlabel('Fractional Doppler Shift')
ax.set_ylabel('Delay (s)')
ax.grid(True, alpha=0.3)
```

## Available Scripts

1. **plot_doppler_equator_simple.py**: Just the nominal curves, no frills
2. **error_visualization_example.py**: Full suite of error analysis plots
3. **test_error_viz.py**: Debug script for testing calculations

## Quick Reference: What the Curves Show

- **Up-Doppler curve** (blue): Maximum Doppler shift at each delay
- **Down-Doppler curve** (red): Minimum Doppler shift at each delay
- These define the boundary of the "doppler equator" in delay-Doppler space
- Points inside this region are visible from Earth at the given observation time
- The curves form a characteristic arc shape

## Parameters You Can Adjust

```python
plot_equator_nominal(
    rx_time,              # Observation time (required)
    tx_name="DWINGELOO",  # Transmitter station
    rx_name="STOCKERT",   # Receiver station
    n_points=500,         # Number of points along curves (higher = smoother)
    figsize=(10, 8)       # Figure size in inches
)
```

## Typical Values

For Earth-Moon bistatic radar:
- Delay range: 0 to ~0.012 seconds (0 to 12 ms)
- Doppler shift range: ~±10⁻⁵ to ±10⁻⁶ (fractional)
- Curve shape: Characteristic arc from SRP to limb

---

## Error Analysis Plots

### Generate All Error Plots

```bash
python_env/bin/python error_visualization_example.py
```

**Outputs** (in `results/ERRORS/`):
- `error_breakdown.png` - Main error source comparison (with pixel scales)
- `velocity_error_breakdown.png` - Velocity components (Moon/Earth orbital, etc.)
- `doppler_equator_nominal.png` - Clean doppler equator (no errors)
- `equator_ephemeris_uncertainty.png` - With uncertainty bands (scaled)
- `equator_with_model_biases.png` - With systematic model errors
- `ephemeris_quality_vs_time.png` - Uncertainty growth over ±100 years
- `sensitivity_analysis.png` - Different uncertainty assumptions

### Quick Error Breakdown Plot

```bash
python_env/bin/python -c "
import cspyce as csp
from astropy import time as at
from doppler_equator_errors import plot_error_breakdown, EphemerisUncertainty, ComputationalErrors
import os
from matplotlib import pyplot as plt

SPICE_KERNEL_DIR = 'spice_kernels'
csp.kclear()
for k in ['naif0012.tls', 'de440s.bsp', 'pck00011.tpc',
           'earth_latest_high_prec.bpc', 'moon_pa_de440_200625.bpc',
           'moon_de440_250416.tf', 'observatories.bsp', 'observatories.tf']:
    csp.furnsh(f'{SPICE_KERNEL_DIR}/{k}')

obs_time = at.Time('2025-09-16T13:22:02')
rx_time = csp.str2et(obs_time.utc.value)

fig, axes = plot_error_breakdown(rx_time, EphemerisUncertainty(), ComputationalErrors())
os.makedirs('results/ERRORS', exist_ok=True)
fig.savefig('results/ERRORS/error_breakdown.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: results/ERRORS/error_breakdown.png')
"
```

### Velocity Error Breakdown

```bash
python_env/bin/python velocity_error_breakdown.py
```

**Output**: `results/ERRORS/velocity_error_breakdown.png`

### Key Results

From standard CAMRAS observations (1299.5 MHz, 0.25 Msps):

**Delay Errors:**
- Ellipsoid approximation: ~4 km (~7 pixels) - DOMINANT
- Clock timing (GPS): ~30 m (~0.05 pixels)
- Other sources: <10 m (<0.02 pixels)

**Doppler Errors:**
- Moon orbital velocity: ~4.3 Hz (~2.5 pixels) - DOMINANT
- Earth orbital velocity: ~2.2 Hz (~1.3 pixels)
- Clock timing (GPS): ~0.4 Hz (~0.25 pixels)
- Total: ~4.85 Hz (~2.8 pixels)

**Lunar Surface Impact:**
- **~2.8 pixels Doppler → ~1.5 km cross-range position uncertainty**
- This is the **fundamental limit** with current DE440 ephemeris

## Documentation

See **../REPORT.md** (error budget, §6) for the complete error source
breakdown, calibration results, and impact on lunar coordinate mapping.
