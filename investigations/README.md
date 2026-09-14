# investigations/ — archived one-off investigation scripts

The `test_*.py` files here are print-only exploration scripts moved out of
`test/`, which holds the maintained regression tests. They are kept as
historical evidence behind [REPORT.md](../REPORT.md). Several encode
conventions REPORT §2 has since superseded —
the legacy per-leg special-relativistic range-rate dlt, the midpoint-subpnt
SRP, and short-baseline (noisy) light-time derivatives — so do not treat
their formulas as current. Run with `.conda/bin/python` from the repo root
(they use cwd-relative
`spice_kernels/`; some need the raw `data.camras.nl/` captures).

The later [rim-window investigation](rim_window_recalibration_2026-07-03.md)
and its `rim_window_recalibration/` scripts document the July calibration
changes. Read its directory-rename note before reusing those scripts; the
paths describe historical runs.

| script | what it explored |
|---|---|
| `test_doppler.py` | dlt/Doppler agreement between J2000 and ITRF93 station frames |
| `test_doppler_rate.py` | SRP Doppler drift rate (Hz/s) over a capture |
| `test_doppler_max.py` | limb-to-limb Doppler extremes: terminator sample vs full-disk sample |
| `test_doppler_curvature.py` | midpoint Doppler error of a linear (chirp-only) rate model |
| `test_doppler_error.py` | image-vs-theory Doppler closure at one epoch (2025-06) |
| `test_doppler_error_09.py` | same closure check on a 2025-09 capture |
| `test_dlt_derivative.py` | numeric vs analytic dlt; short-baseline (dt=1e-3) derivative noise |
| `test_dlt_derivative2.py` | dlt derivative error vs finite-difference baseline dt sweep |
| `test_scaling_v2.py` | differential Doppler-rate (window-average) correction on the terminator |
| `test_scaling_issue.py` | Doppler-axis span/scaling bug hunt in the DD image axes |
| `test_tx_start.py` | true TX start offset vs the `rx_start + 1.0 s` emission convention |
| `test_tx_centroid.py` | TX baseband centroid frequency and resample-stretch shift |
| `test_ab_cor.py` | aberration-correction flavors (LT / LT+S / CN / CN+S) Doppler differences |
| `test_midpoint_fix.py` | window-average `dlt_eff = mean − rate·T/2` model (REPORT §2; ±15 mHz limb term) |
| `test_delay_doppler_error.py` | Doppler error from compensating at delayed vs emit time (rate × τ) |
