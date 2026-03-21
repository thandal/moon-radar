# Doppler Equator Error Analysis

Comprehensive error quantification and visualization for bistatic lunar radar Doppler-Delay (DD) image calculations and their impact on lunar coordinate mapping.

## Executive Summary

**Key Insight**: Ephemeris errors and traditional clock timing logic represent **negligible** uncertainties. The true dominant error sources bounding our tracking capabilities are the **SDR hardware limitations** (USRP B210 offset pipeline latency and TCXO stability) and **geometric approximations** (Ellipsoid topography).

### Bottom Line

**Prior claims regarding clock timing delays and Ephemeris are mathematically false:**
- Ephemeris **Position**: ~2 cm (LLR) → ~0.06 ns range error → **negligible**
- Ephemeris **Velocity**: ~10 μm/s (LLR) → **$4.3 \times 10^{-5}$ Hz** → **negligible**
- **Clock Timing**: A 100 ns GPS offset corresponds to interpreting the frequency shift at the wrong time (via orbital acceleration $a$), resolving to a Doppler error of $\sim 10^{-9}$ Hz → **negligible**

**The ACTUAL dominant error sources:**
1. **SDR Pipeline Delay (USRP B210)**: Variable/fixed USB 3.0 processing latency of ~$100\ \mu$s directly creates a **$\sim 15$ km distance measurement error** (~25 pixels). Limit can be bounded only by timestamp-based precise RF timing.
2. **SDR Oscillator Stability (USRP B210 TCXO)**: A standard $\pm 2.0$ ppm base oscillator creates a ~2,600 Hz Doppler shift (~1,500 pixels), which entirely ruins the image. A **GPSDO ($<1$ ppb)** drops this to **~1.3 Hz ($\sim 0.75$ pixels)**, which is tenable. 
3. **Ellipsoid Approximation**: ~4 km range error (systematic, correctable with LOLA DEM).

**For high-precision applications**: You MUST use a GPSDO SDR configuration and compensate for software pipeline processing delays by tying your calculations to the precise time-stamps stamped on individual radio frames by the SDR hardware.

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
| **SDR Pipeline Delay ($100\ \mu$s)** | ~15 km | **~25 pixels** | **DOMINANT** - Highly impactful software latency |
| **Ellipsoid Approximation** | ~4 km | **~7 pixels** | **Systematic bias** |
| **SRP Averaging** | ~50 m | ~0.08 pixels | Bistatic geometry approx |
| **Clock Timing (GPS offset)** | ~15 m | ~0.025 pixels | One-way 100 ns offset resolving to range error |
| **Ephemeris Position** | ~0.02 m | negligible | DE440 ~2cm |

### Doppler Shift Errors

| Source | Doppler Error (Hz) | Pixels @ 1.73 Hz/px | Impact |
|--------|-------------------|---------------------|--------|
| **SDR Base TCXO (2.0 ppm)** | ~2,600 Hz | **~1,500 pixels** | **DOMINANT** - Makes imaging impossible |
| **Ellipsoid Approximation** | ~1.4 Hz | ~0.8 pixels | Systematic bias |
| **SDR GPSDO (<1.0 ppb)** | ~1.3 Hz | ~0.75 pixels | Strongly recommended tolerance |
| **SRP Averaging** | ~0.0006 Hz | negligible | Bistatic geometry |
| **Moon Orbital Velocity (LLR)** | ~$4.3 \times 10^{-5}$ Hz | negligible | Derived from ~10 μm/s |
| **Earth Orbital Velocity** | ~$2.1 \times 10^{-5}$ Hz | negligible | Derived from ~5 μm/s |
| **Clock Timing (GPS offset)** | ~$10^{-9}$ Hz | negligible | Based on kinematic acceleration shift |

**Total ephemeris velocity error**: completely mathematically insignificant relative to the required tolerances of the application. The primary struggle is overcoming **hardware tracking limitations**, not the astrometric tables.

---

## Typical Impact Analysis for Lunar Subsurface Mapping

### Transformation: DD Space → Lunar Coordinates

**1. Range Direction** (Delay → Radial distance):
- The massive 100 $\mu$s penalty of blindly relying on host-time timestamps across a USB 3.0 USRP B210 interface guarantees a shift of the apparent lunar altitude by 15 km.
- Overcoming this requires directly querying the FPGA's intrinsic time-stamped packet headers, pushing this latency error bound toward zero.

**2. Cross-Range Direction** (Doppler → Azimuth/Longitude):
- Doppler measurements represent the instantaneous line-of-sight velocity of the surface. 
- If relying on an internal uncalibrated TCXO on the USRP B210 (2.0 ppm drift), the tracking frequency is shifted by 2,600 Hz. This represents an azimuth mapping error larger than the Moon itself.
- A **GPSDO implementation** drops this to ~1.0 ppb, binding the positional footprint to roughly $\sim 0.75$ pixels (~450 meters cross-range at typical geometry).

### Example: Mapping a Crater at 30° Latitude (with GPSDO)

- **Doppler uncertainty** (GPSDO 1 ppb): ~0.75 pixels → **~450 m cross-range error**
- **Range uncertainty** (Assume corrected FPGA timing bounds): 0.05 pixels → **~30 m radial error**
- **Ellipsoid bias**: **~4 km altitude error** (if not corrected with LOLA DEM topography)

For comparison:
- Large craters: 10-100 km diameter → ✅✅ **easily detectable**
- Small craters: 1-10 km diameter → ✅ **detectable**
- Medium boulders: 1-2 km → ⚠️ **approaching limit**
- Artifacts: <500 m → ❌ **below resolution limit under current SDR setups**

---

## Detailed Error Source Breakdown

### 1. SDR Hardware Errors (Ettus USRP B210)

**Pipeline Latency ($~100\ \mu$s)**
When streaming samples via USB 3.0 to a host machine (i.e., GNU Radio over Linux), there is an indeterminate transfer and buffering delay averaging $~100\ \mu$s.
- **Range Impact**: $100\ \mu$s $\times c \approx 30$ km round-trip distance, leading to ~15 km mapping offset laterally towards radially closer terrains.

**Oscillator Phase/Frequency Stability (TCXO vs GPSDO)**
The USRP B210 operates an onboard TCXO natively rated to $2.0$ ppm. Because the Doppler bin frequency maps strictly linearly across the equator, drift explicitly correlates linearly to target shift.
- **Base (2.0 ppm)**: $1299.5 \text{ MHz} \times 2.0 \times 10^{-6} = 2,599 \text{ Hz}$ frequency shift.
- **GPSDO (<1.0 ppb)**: $1299.5 \text{ MHz} \times 1.0 \times 10^{-9} = 1.3 \text{ Hz}$ frequency shift.

### 2. Ephemeris Errors (DE440/DE441)
Historically grossly overestimated due to velocity-space logic failing to incorporate the scale of the speed of light accurately.

- **Position accuracy (LLR)**: ~2 cm. This resolves to $~0.06$ ns time delay.
- **Velocity accuracy**: Derived as position constraint shift over a baseline month -> bounded to roughly $10\ \mu$s/s. Fractional Doppler equates to $\sim 4.3 \times 10^{-5}$ Hz. **This means that NASA DE440 profiles are "infinitely" strict regarding pixel constraints**.

### 3. The "Clock Timing" False Paradigm
Prior mathematical modeling assumed that a clock timing error of 100 ns cascaded into a wide margin fractional Doppler shift (0.43 Hz). **This is physically impossible.**
Measuring a continuous signal at time $t$ instead of $t+100\text{ns}$ merely evaluates the velocity along the orbital path 100 ns later. The rate of change of the fractional Doppler shift is proportional to the line-of-sight acceleration $a_{\text{LOS}}$. 
Using the Moon's $a \approx 0.0026\text{ m/s}^2$:
$$ \Delta f = f_0 \frac{a \cdot \Delta t}{c} \approx 1300\text{ MHz} \times \frac{0.0026 \times 10^{-7}}{3\cdot 10^8} \approx 10^{-9}\text{ Hz} $$

### 4. Computational Approximations

**Ellipsoid vs. Lunar Topography (~4 km range, ~1.4 Hz Doppler)**
- Remains the largest **systematic** error source impacting pure geometries uncorrected by 3D DEM models.
- Lunar topography limits map variance up to roughly 8 km radially. Correlates to ~7 delay pixels.

---

## Execution Directives

Code has been corrected to appropriately process SPICE arrays.
```bash
python error_visualization_example.py
```
This correctly integrates scaling bounds on:
1. `c` (properly transformed to m/s dynamically across propagation graphs)
2. `HardwareErrors` class inclusion, bounding B210 limitations natively into plots.
3. Accurate scale breakdowns printed systematically in the terminal.

---

## Contact

For questions or suggestions about the error analysis module, please open an issue on the project repository.
