# Doppler Equator Error Analysis

Comprehensive error quantification and visualization for bistatic lunar radar Doppler-Delay (DD) image calculations and their impact on lunar coordinate mapping.

## Executive Summary

**Key Insight**: Ephemeris errors are **negligible** for modern lunar radar observations. The dominant error sources are **clock timing** and **geometric approximations**.

### Bottom Line

**Ephemeris uncertainties are negligible for lunar radar:**
- **Position**: ~1 cm (LLR) → ~0.002 m range error, ~0.003 Hz Doppler → **sub-pixel**
- **Velocity**: ~10 μm/s (from LLR position constraints) → **0.043 Hz** → **0.025 pixels**
- **Total ephemeris contribution**: ~0.1 Hz → **~0.06 pixels** (completely negligible)

**Actual dominant error sources:**
1. **Clock timing** (GPS): 100 ns → **0.43 Hz** → **0.25 pixels**
2. **Ellipsoid approximation**: ~4 km range (systematic, correctable with LOLA DEM)
3. **Measurement noise** in radar data (not analyzed here)

**For high-precision applications**: Use GPS + atomic clock (10 ns), correct topography with LOLA DEM, understand that **timing is the limiting factor**, NOT ephemeris.

---

## Standard CAMRAS DD Image Resolution

- **Range (Delay)**: 600 m/pixel (0.25 Msps sample rate)
- **Doppler**: 1.73 Hz/pixel @ 1299.5 MHz (3000 bins over ~4×10⁻⁶ DLT)
- **Image size**: 3000 × 2897 pixels (Doppler × Delay)
- **Coverage**: Full Moon disk (delay span = 11.6 ms two-way light time)

---

## Error Magnitude Summary

### Delay (Range) Errors

| Source | Range Error | Pixels @ 600 m/px | Impact |
|--------|-------------|-------------------|--------|
| **Ellipsoid Approximation** | ~4 km | **~7 pixels** | **DOMINANT** - Systematic bias |
| **SRP Averaging** | ~50 m | ~0.08 pixels | Bistatic geometry approx |
| **Clock Timing (GPS)** | ~30 m | ~0.05 pixels | 100 ns uncertainty |
| **Ephemeris Position** | ~2.6 m | ~0.004 pixels | DE440 LLR accuracy |
| **Light-Time Iteration** | ~0.3 mm | negligible | SPICE convergence |

### Doppler Shift Errors

| Source | Doppler Error (Hz) | Pixels @ 1.73 Hz/px | Impact |
|--------|-------------------|---------------------|--------|
| **Clock Timing (GPS)** | ~0.43 Hz | **~0.25 pixels** | **DOMINANT** - 100 ns uncertainty |
| **Ellipsoid Approximation** | ~1.4 Hz | ~0.8 pixels | Systematic bias |
| **SRP Averaging** | ~0.22 Hz | ~0.13 pixels | Bistatic geometry |
| **Moon Orbital Velocity (LLR)** | ~0.043 Hz | ~0.025 pixels | DE440 ephemeris |
| **Earth Orbital Velocity** | ~0.022 Hz | ~0.013 pixels | DE440 ephemeris |
| **Finite Difference** | ~0.02 Hz | ~0.01 pixels | Negligible |
| **Ephemeris Position** | ~0.003 Hz | negligible | DE440 |

**Total ephemeris velocity error**: ~0.048 Hz (RSS) → **~0.028 pixels** (negligible!)
**Dominant error: Clock timing** → **~0.43 Hz** → **~0.25 pixels**

---

## Why Few Pixels Matter for Lunar Mapping

### Transformation: DD Space → Lunar Coordinates

When mapping from Delay-Doppler space to lunar geodetic coordinates (lat/lon):

**1. Range Direction** (Delay → Radial distance):
- **Direct mapping**: 1 pixel ≈ 600 m
- **Dominant error**: Ellipsoid approximation (±4 km systematic bias)
- **Other errors**: Sub-pixel (<60 m)

**2. Cross-Range Direction** (Doppler → Azimuth/Longitude):
- **Geometry-dependent**: At Moon distance (~384,400 km), angular resolution depends on observation geometry
- **Clock timing dominates**: ~0.25 pixels → ~150 m cross-range position uncertainty
- **Ephemeris contribution**: ~0.03 pixels → ~20 m (negligible)
- **Total systematic errors** (timing + measurement noise) typically dominate

### Example: Mapping a Crater at 30° Latitude

- **Doppler uncertainty** (clock timing): ~0.25 pixels → **~150 m cross-range error**
- **Range uncertainty** (excl. ellipsoid): 0.05 pixels → **~30 m radial error**
- **Ellipsoid bias**: **~4 km altitude error** (if not corrected with topography)

**Position uncertainty ellipse on lunar surface**: **~150 m × 30 m**

For comparison:
- Large craters: 10-100 km diameter → ✅✅ **easily detectable**
- Small craters: 1-10 km diameter → ✅ **detectable**
- Large boulders: 100-1000 m → ⚠️ **approaching limit**
- Small boulders: 10-100 m → ❌ **below resolution limit**

---

## Detailed Error Source Breakdown

### 1. Ephemeris Errors (DE440/DE441)

Based on Park et al. 2021 and Lunar Laser Ranging (LLR) data:

**Modern Epoch (within ~50 years of 2020):**
- **Position accuracy**: ~1 cm (LLR weighted RMS residual)
- **Velocity accuracy**: ~10 μm/s (derived from position measurements over time)
- Dominated by LLR measurement uncertainty

**Historical/Future Epochs:**
- Uncertainty grows quadratically with time from present
- DE441 differs from DE440 by ~10 m at ±100 years
- Position error: σ(t) = σ₀ × √(1 + k(t/t_ref)²)

#### Velocity Uncertainty: Why 10 μm/s, Not 1 mm/s?

**The Question**: If LLR position accuracy is ~1 cm, why is velocity uncertainty ~10 μm/s instead of much larger?

**Common Misconception**:
- "Position uncertainty ~1 cm, orbital velocity varies by ~56 m/s → velocity uncertainty ~1 mm/s"
- **Problem**: Confuses orbital variations (which are KNOWN from model) with uncertainties

**Proper Derivation**:
LLR provides position measurements at multiple times. Velocity is derived from how position changes:

```
σ_v ≈ σ_pos × √2 / Δt
```

For ~1 month baseline between measurements:
```
σ_v ≈ 0.01 m × √2 / (30 days × 86400 s/day)
σ_v ≈ 0.0141 m / 2.592×10⁶ s
σ_v ≈ 5.4 × 10⁻⁹ m/s = 5.4 nm/s
```

Conservative estimate including model uncertainties: **~10 μm/s**

#### Velocity Error Components

**Moon Orbital Velocity** (~10 μm/s uncertainty):
- From LLR data constraining Moon's barycentric orbit
- Modern LLR residuals: ~1 cm position accuracy over ~1 month
- **Impact**: 0.043 Hz → 0.025 pixels → **~15 m on lunar surface** (negligible!)

**Earth Orbital Velocity** (~5 μm/s uncertainty):
- Better constrained than Moon (inner planet ranging, spacecraft tracking)
- **Impact**: 0.022 Hz → 0.013 pixels → **~8 m on lunar surface** (negligible!)

**Combined (RSS)**: ~11 μm/s → **~0.048 Hz → ~0.028 pixels** (completely negligible!)

**Minor contributors** (all negligible):
- Observatory position (ITRF2020): ~0.01 mm/s
- Earth rotation (EOP): ~0.0003 mm/s
- Moon libration: ~0.02 mm/s

### 2. Clock Timing Errors

**GPS timing** (typical: 100 ns uncertainty):

**Delay impact**:
- 100 ns timing error = 100 ns delay directly
- Convert to range: 100 ns × c / 2 ≈ 15 m one-way ≈ **30 m two-way**
- In pixels: **~0.05 pixels** (negligible)

**Doppler impact**:
- Timing error causes position error via Moon's motion
- Moon velocity ~1 km/s
- Doppler error: Δ(DLT) ≈ v·Δt/c = (1000 m/s × 100 ns) / c ≈ 3.3×10⁻¹⁰
- At 1299.5 MHz: **0.43 Hz → 0.25 pixels → ~0.15 km**

**Improvement**: GPS-disciplined atomic clock (10 ns) → 10× reduction → **0.015 km**

### 3. Computational Approximations

**Ellipsoid vs. Lunar Topography** (~4 km range, ~1.4 Hz Doppler):
- **Largest single error source** for range measurements
- Lunar topography varies ±8 km peak-to-peak from mean ellipsoid
- **Systematic bias**, not random uncertainty
- **Impact**: ~7 pixels range → several km position error
- **Solution**: Use LOLA (Lunar Orbiter Laser Altimeter) DEM for topographic correction

**SRP (Sub-Radar Point) Averaging** (~50 m range, ~0.22 Hz Doppler):
- Bistatic specular point approximated as average of TX and RX sub-points
- True bistatic reflection point can differ by tens of meters
- Depends on bistatic angle and surface curvature
- **Impact**: 0.08 pixels range, 0.13 pixels Doppler (sub-pixel, minor)

**Finite Difference Velocity** (~0.02 Hz Doppler):
- Numerical differentiation with 1-second timestep
- Error: O(dt × acceleration) where lunar acceleration ≈ 2.6×10⁻⁶ m/s²
- Velocity error: ~2.6 μm/s
- **Impact**: Negligible (~0.01 pixels)

**Light-Time Iteration** (~1 ps delay):
- SPICE iterative light-time corrections converge in 2-3 iterations
- Residual error: sub-millimeter positions
- **Impact**: Negligible (<10⁻⁶ pixels)

---

## Error Budget Table

| Error Source | Range Error | Doppler Error | Lunar Position Impact |
|--------------|-------------|---------------|----------------------|
| **Clock Timing (GPS)** (improvable) | ~30 m | **~0.43 Hz** | **~150 m cross-range** - **DOMINANT** |
| **Ellipsoid Approximation** (correctable) | **~4 km** | ~1.4 Hz | **~4 km radial** |
| **Ephemeris Velocity** (negligible!) | ~2 m | ~0.048 Hz | ~15 m cross-range |
| **Other sources** (negligible) | <10 m | <0.3 Hz | <100 m |

**Total without corrections**: ~4 km range, ~150 m cross-range
**Total with topography + atomic clock (10 ns)**: ~0.03 km range, **~15 m cross-range** (timing-limited, NOT ephemeris!)

---

## Recommendations for High-Precision Lunar Mapping

### Essential Steps

1. **High-precision timing**: GPS + atomic clock - **MOST IMPORTANT!**
   - GPS alone: 100 ns → 0.25 pixel → **~150 m** cross-range
   - GPS + disciplined atomic: 10 ns → 0.025 pixel → **~15 m** cross-range
   - **10× improvement** over GPS alone
   - **This is the limiting factor**, NOT ephemeris!

2. **Topographic correction**: Use LOLA DEM
   - Removes ~4 km ellipsoid bias in range
   - Critical for accurate altitude/depth measurements
   - Reduces range error from 7 pixels to <0.1 pixels

3. **Use DE440 ephemeris** (already excellent!)
   - Modern LLR accuracy: ~1 cm position, ~10 μm/s velocity
   - Ephemeris contribution: **~15 m** cross-range (negligible compared to timing!)
   - No improvement needed - already better than timing allows

### Understanding Actual Limits

**Clock timing is the limiting factor**, NOT ephemeris:
- GPS timing (100 ns) → **~150 m** position uncertainty
- Ephemeris velocity (10 μm/s) → **~15 m** position uncertainty (10× smaller!)
- **Timing dominates** by factor of 10

**Improvement path**:
- ✅ Better timing (atomic clock): can reach **~15 m** accuracy
- ✅ Topographic correction (LOLA): removes systematic 4 km bias
- ❌ Better ephemeris: NOT needed - already 10× better than timing!

**What this means for applications**:
- ✅✅ **Crater identification**: 1-100 km features → easily detectable
- ✅ **Precise crater mapping**: Sub-km features → detectable with good timing
- ✅ **Large boulder detection**: 100-1000 m → approaching limit with atomic clocks
- ⚠️ **Small boulder detection**: 10-100 m → challenging, requires best timing
- ❌ **Fine surface features**: <10 m → below fundamental resolution

---

## Usage Examples

### Basic Error Visualization

```python
from doppler_equator_errors import (
    EphemerisUncertainty,
    ComputationalErrors,
    plot_equator_with_errors,
)

# Create error models
ephem_unc = EphemerisUncertainty()
comp_err = ComputationalErrors()

# Plot doppler equator with ephemeris uncertainty bounds
# NOTE: Uncertainties are TINY (~10^-12), so we scale by 10^6 for visibility
fig, ax = plot_equator_with_errors(
    rx_time,
    ephem_uncertainty=ephem_unc,
    computational_errors=comp_err,
    n_sigma=3,  # Show ±3σ bounds
    include_model_errors=False,  # Only ephemeris uncertainties
    scale_factor=1e6  # Scale by 1 million to make visible!
)

# To show systematic model biases instead:
fig2, ax2 = plot_equator_with_errors(
    rx_time,
    include_model_errors=True,  # Include ellipsoid approx, SRP averaging
    scale_factor=1.0  # No scaling - these biases are large
)
```

### Error Source Breakdown

```python
from doppler_equator_errors import plot_error_breakdown

# Visualize contribution of each error source with standard DD image resolution
fig, axes = plot_error_breakdown(rx_time)
# Shows errors in meters/Hz and pixels (600 m/px range, 1.73 Hz/px Doppler)
```

### Velocity Error Breakdown

```python
# Run detailed velocity error analysis
import os
os.system('python velocity_error_breakdown.py')
# Generates detailed breakdown of Moon orbital, Earth orbital, etc.
```

### Ephemeris Quality Over Time

```python
from doppler_equator_errors import compare_ephemeris_quality
import numpy as np

# Show how uncertainty grows with time
years = np.linspace(-100, 100, 201)
rx_times = reference_et + years * 365.25 * 86400.0

fig, axes = compare_ephemeris_quality(rx_times, reference_et)
```

### Complete Example Suite

```bash
cd notebooks
python error_visualization_example.py
```

This generates all visualizations in `results/ERRORS/`:
1. **error_breakdown.png**: All error sources with dual axes (physical + pixels)
2. **velocity_error_breakdown.png**: Velocity component breakdown
3. **equator_ephemeris_uncertainty.png**: Doppler equator with uncertainty bounds
4. **ephemeris_quality_vs_time.png**: Uncertainty growth over time
5. **sensitivity_analysis.png**: Comparison of uncertainty assumptions

---

## Customizing Error Models

### Custom Ephemeris Uncertainty

```python
# More conservative uncertainty model
ephem_unc = EphemerisUncertainty(
    sigma_pos_modern=0.05,      # 5 cm position uncertainty
    sigma_vel_modern=0.005,     # 5 mm/s velocity uncertainty
    t_ref_years=100,
    sigma_pos_century=10.0      # 10 m at 100 years
)
```

### Custom Timing Source

```python
# High-precision atomic clock
comp_err = ComputationalErrors()
timing_error = comp_err.timing_error(timing_source="atomic")  # 10 ns

# GPS timing
timing_error = comp_err.timing_error(timing_source="GPS")  # 100 ns

# Poor timing (NTP)
timing_error = comp_err.timing_error(timing_source="NTP")  # 1 ms
```

---

## Mathematical Details

### DLT (Fractional Doppler) Sensitivity

The fractional Doppler shift is:
```
DLT = 1 - √[(1-v_rx/c)/(1+v_rx/c)] × √[(1-v_tx/c)/(1+v_tx/c)]
```

For small velocities (v << c):
```
DLT ≈ (v_rx + v_tx)/c
```

Sensitivity to velocity error δv:
```
∂DLT/∂v ≈ 1/c ≈ 3.3×10⁻⁹ s/m
```

### Delay Sensitivity

Two-way light time:
```
τ = (r_tx + r_rx)/c
```

Sensitivity to position error:
```
∂τ/∂r = 2/c
```

For σ_pos = 1 cm:
```
δτ = 2 × 0.01m / 3×10⁸ m/s ≈ 0.067 ns
```

---

## Time Dependence of Ephemeris Uncertainty

| Time from Present | Position Uncertainty | Velocity Uncertainty | Doppler Error |
|-------------------|----------------------|----------------------|---------------|
| 0 years (2020-2025) | ~1 cm | ~1 mm/s | ~4.3 Hz |
| ±50 years | ~2-3 cm | ~1.5 mm/s | ~6.5 Hz |
| ±100 years | ~10 m | ~5 mm/s | ~22 Hz |

Even at ±100 years, Doppler uncertainty (~13 pixels) remains tractable for crater-scale mapping.

---

## References

1. **Park et al. 2021**: "The JPL Planetary and Lunar Ephemerides DE440 and DE441", AJ 161:105
   - https://ssd.jpl.nasa.gov/doc/Park.2021.AJ.DE440.pdf
   - Modern LLR: ~1 cm position residuals
   - DE441-DE440 difference: ~10 m at ±100 years

2. **Murphy et al. 2013**: "Lunar Laser Ranging: The Millimeter Challenge", Rep. Prog. Phys. 76:076901
   - LLR measurement techniques and accuracy

3. **SPICE Toolkit Documentation**: https://naif.jpl.nasa.gov/naif/toolkit.html
   - Complete reference for SPICE functions and conventions

4. **LOLA (Lunar Orbiter Laser Altimeter)**:
   - Topographic data for correcting ellipsoid approximation
   - https://ode.rsl.wustl.edu/moon/

---

## Implementation Files

- **`doppler_equator_errors.py`**: Core error analysis functions
  - `EphemerisUncertainty`: Position/velocity uncertainty model
  - `ComputationalErrors`: Timing, approximation errors
  - `plot_error_breakdown()`: Visualize all error sources
  - `plot_equator_with_errors()`: Doppler equator with uncertainty bounds
  - `compare_ephemeris_quality()`: Uncertainty vs time

- **`velocity_error_breakdown.py`**: Detailed velocity component analysis
  - Breaks down Moon orbital, Earth orbital, station, rotation, libration
  - Propagates to Doppler frequency errors
  - Demonstrates dominance of Moon orbital velocity

- **`error_visualization_example.py`**: Complete example suite
  - Generates all error visualizations
  - Multiple scenarios (nominal, scaled, model errors)

---

## Future Enhancements

Potential improvements to the error analysis:

1. **LOLA Topography Integration**: Incorporate detailed elevation models
2. **Correlated Errors**: Model correlations between position/velocity uncertainties
3. **Station Position Errors**: Include Earth station ITRF uncertainties
4. **Atmospheric Effects**: Model tropospheric/ionospheric delays
5. **Non-Gaussian Statistics**: Use realistic error distributions
6. **Lunar Coordinate Jacobian**: Explicit DD-to-geodetic transformation errors

---

## Contact

For questions or suggestions about the error analysis module, please open an issue on the project repository.
