# Doppler Equator Error Analysis

Comprehensive error quantification and visualization for bistatic lunar radar Doppler-Delay (DD) image calculations and their impact on lunar coordinate mapping.

## Executive Summary

**Key Insight**: Ephemeris errors and traditional clock timing assumptions represent **negligible** uncertainties. The true dominant variables bounding our tracking capabilities are the **SDR hardware timing limitations** (USRP B210 AD9361 pipeline delays and PPS sampling ambiguity) and **geometric approximations** (Ellipsoid topography).

### Bottom Line

**Prior claims regarding general clock timing or Ephemeris limits are mathematically false:**
- Ephemeris **Position**: ~2 cm (LLR) → ~0.06 ns range error → **negligible**
- Ephemeris **Velocity**: ~10 μm/s (LLR) → **$4.3 \times 10^{-5}$ Hz** → **negligible**
- **Hardware Oscillator Stability**: Empirical logs confirm that Dwingeloo utilizes a White Rabbit locked Hydrogen Maser ($\sim 10^{-13}$ drift), yielding almost $0$ Hz error relative to the bounds. Stockert utilizes a Rubidium standard ($\sim 5.5 \times 10^{-11}$), producing a negligible Doppler offset of **$\sim 0.07$ Hz** ($\sim 1/25^{\text{th}}$ pixel offset).

**The ACTUAL dominant error sources:**
1. **SDR Pipeline Delay (AD9361/Auto-correlation)**: Variable internal sample transmission offsets within the RF equipment bounds mapping limits. Dwingeloo TX/RX bounds evaluate experimentally to **$\sim 30\ \mu$s**, creating a **$\sim 4.5$ km distance measurement error** (~7.5 pixels). 
2. **PPS Sampling Ambiguity**: A 1-sample ambiguity across the PPS timing pulse at the operating 250 kHz sample rate creates a continuous $\pm 4\ \mu$s jitter. This directly invokes a continuous **~600 m range ambiguity** (~1 pixel width).
3. **Ellipsoid Approximation**: ~4 km range error (systematic, fully correctable only with a 3D LOLA DEM projection matching algorithm).

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
| **Dwingeloo TX/RX Pipeline Delay (~30 $\mu$s)** | ~4.5 km | **~7.5 pixels** | **DOMINANT** - Bounded via auto-correlation tests |
| **Ellipsoid Approximation** | ~4 km | **~7 pixels** | **Systematic bias** |
| **PPS Sampling Jitter (250 kHz)** | ~600 m | ~1 pixel | Sample aliasing mapping bound |
| **Stockert GPS Cable Offset (735 ns)** | ~110 m | ~0.18 pixels | Differential sync delay |
| **SRP Averaging** | ~50 m | ~0.08 pixels | Bistatic geometry approx |

### Doppler Shift Errors

| Source | Doppler Error (Hz) | Pixels @ 1.73 Hz/px | Impact |
|--------|-------------------|---------------------|--------|
| **Ellipsoid Approximation** | ~1.4 Hz | ~0.8 pixels | Systematic bias |
| **Stockert Rubidium Standard ($\sim 5.5 \times 10^{-11}$)** | ~0.07 Hz | ~0.04 pixels | Very low impact |
| **Dwingeloo H-Maser ($\sim 10^{-13}$)** | ~0.0001 Hz | negligible | Effectively exact |
| **Moon Orbital Velocity (LLR)** | ~$4.3 \times 10^{-5}$ Hz | negligible | Derived from ~10 μm/s position tracks over time |

**Conclusion**: The dominant factors constraining any given pixel are entirely separated from classical ephemeris mechanics and bounded essentially uniformly by the **internal tracking pipelines (the 30s $\mu$s USRP B210 offset)** alongside **systematic geographic biases (assuming an ellipsoid topography over true DEM surface rendering)**.

---

## Detailed Error Source Breakdown

### 1. SDR Hardware Errors (Empirical Facility Specifications)

Rather than assuming general parameters for Ettus generic components, strict empirical evaluations bound our true uncertainties.

**Pipeline Latency (30 $\mu$s & 4 $\mu$s Ambiguity)**
The AD9361 RF chipset interacting with the FPGA introduces master-clock-rate delays distinct between TX lines and RX lines. Auto-correlation tests verified at the Dwingeloo facility reveal a **$\sim 30\ \mu$s offset** between TX and RX execution timings.
- **Range Impact**: $30\ \mu$s $\times c / 2 \approx 4.5$ km shift.

Additionally, standard B210 alignment possesses a built-in 1-sample ambiguity upon PPS locking. Operating at 250 kHz, $1 / 250,000 = 4\ \mu$s, providing a baseline jitter tolerance bounding tracking coordinates to $\pm 600$ meters independently of latency.

**Oscillator Phase/Frequency Stability**
Neither Stockert nor Dwingeloo rely purely on baseline uncalibrated TCXO components.
- **Dwingeloo**: Hydrogen Maser clock via White Rabbit offset to roughly $10^{-13}$. This generates a continuous Doppler offset of $< 0.001$ Hz, eliminating timing as a variable.
- **Stockert**: Rubidium local block offset experimentally around $5.5 \times 10^{-11}$. At $1299.5$ MHz, this causes an initial data shift of approximately **0.07 Hz**. Both these values heavily safeguard pixel resolutions (bound by 1.73 Hz boundaries).

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

Code has been corrected to appropriately process SPICE arrays alongside the bespoke hardware mappings across Stockert and Dwingeloo.
```bash
python error_visualization_example.py
```
This correctly integrates scaling bounds on:
1. `c` (properly transformed to m/s dynamically across propagation graphs)
2. `HardwareErrors` class inclusion, bounding AD9361 limits natively into plots.
3. Accurate scale breakdowns printed systematically in the terminal.

---

## Contact

For questions or suggestions about the error analysis module, please open an issue on the project repository.
