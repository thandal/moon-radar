# Plotting the lunar echo boundaries

Run from the repository root with the `.conda` environment and the SPICE
kernels listed in [spice_setup.py](spice_setup.py). These plots show geometry
and assumed error models; measured pipeline limits are in
[REPORT §6](REPORT.md#6-remaining-limits).

## Ready-made plots

```sh
.conda/bin/python plot_doppler_equator_simple.py
.conda/bin/python error_visualization_example.py
.conda/bin/python velocity_error_breakdown.py
```

Outputs go to `results/ERRORS/`. The first script writes
`doppler_equator_clean.png`; the others generate uncertainty, sensitivity,
and error-component figures. `test_error_viz.py` is a separate smoke plotter
that writes under `results/VERIFY/`.

## Plot a chosen epoch

```python
from pathlib import Path
import cspyce as csp
from spice_setup import furnsh_kernels
from doppler_equator_errors import plot_equator_nominal

furnsh_kernels()
rx_time = csp.str2et("2025-09-16T13:22:02")
# The 2025-09-16 session: Dwingeloo transmitting, Stockert receiving.
fig, ax = plot_equator_nominal(rx_time, tx_name="DWINGELOO", rx_name="STOCKERT")
ax.set_xlabel("Light-time derivative, dlt (Doppler = −frequency × dlt)")
Path("results/ERRORS").mkdir(parents=True, exist_ok=True)
fig.savefig("results/ERRORS/doppler_equator_nominal.png", dpi=150)
```

The blue “up” branch approaches the radar and has the **minimum `dlt`**;
the red “down” branch recedes and has the maximum. The plotting helper labels
the horizontal axis “Fractional Doppler Shift”, but its values are `dlt`:
frequency shift has the opposite sign. Delay is relative to the reference
point. The curves bound the modeled echo in DD space; they are not a surface
visibility mask.

The station names are required keywords: there is no default link.

To add uncertainty bands after the setup above:

```python
from doppler_equator_errors import plot_equator_with_errors

fig, ax = plot_equator_with_errors(
    rx_time, tx_name="DWINGELOO", rx_name="STOCKERT",
    oscillator_source="STOCKERT_RUBIDIUM", n_sigma=3, include_model_errors=False
)
ax.set_xlabel("Light-time derivative, dlt (Doppler = −frequency × dlt)")
fig.savefig("results/ERRORS/equator_uncertainty.png", dpi=150)
```

`oscillator_source` is a `HardwareErrors.oscillator_stability` key, also
required. The shaded Doppler band holds the ephemeris terms at the reference
point and that oscillator's stated stability, which dominates it. The
delay band, drawn as the curves shifted by ±n_sigma (dotted), holds the
ephemeris and light-time iteration terms
(about 1 ns: invisible unless `scale_factor` is large), plus the ±4 km
ellipsoid and SRP-averaging terms with `include_model_errors=True`. Fixed
station delays are not drawn: the per-look timing measurement absorbs them.
`scale_factor` multiplies both bands; the title records it and lists what
each band holds. These bands display the error model's assumptions, not a
measured uncertainty on a processed map.

## Comparing geometry methods

[doppler_equator.py](doppler_equator.py) exposes three methods:

| Function | Returns after the minimum light time |
|---|---|
| `compute_doppler_equator` | HEALPix delay bins, `dlt_max`, `dlt_min` |
| `compute_doppler_equator_velocity` | Up delay/`dlt`, then down delay/`dlt`; production analytic-axis method |
| `compute_doppler_equator_terminator` | Up delay/`dlt`, then down delay/`dlt`; limb-sampled method |

For comparison with a processed DD image, pass its `rx_duration_s` and
`dlt_rate_srp` to the velocity or terminator method so the curves use the
same window-averaged Doppler convention. The simple plotting helper shows
instantaneous geometry.
