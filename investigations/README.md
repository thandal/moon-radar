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

[focused_dd_prototype.py](focused_dd_prototype.py) (2026-10-21) builds the
delay-Doppler image from 10 ms block correlations and focuses each surface
point at its own Doppler drift rate; results in REPORT §6 ("Doppler drift
within a look"). It takes the output root, a run folder, the data root,
nside, supersampling and one capture as arguments, and needs a run made
since 2026-10-21 (it reads that run's `rim_status` and corrections). Its
docstring's numbers were measured with the scratch scripts in
[2026-10-21_scratch/](2026-10-21_scratch/README.md), on products made
before that day's fixes: re-measure before use.

The later [rim-window investigation](rim_window_recalibration_2026-07-03.md)
and its `rim_window_recalibration/` scripts document the July calibration
changes. They are historical (July 2026), not checks of current
production: the rim scripts place the rims from the smooth-sphere SRP light
time (production: the terrain under the SRP), do not record whether a
calibration converged as production's `rim_status` does
(trio_intra_look_drift.py now prints its own status), and some use
non-production gates or pass caps; each script's header says how it
differs. Read the log's directory-rename note before reusing them;
the paths describe historical runs.

The tracked figure `rim_window_recalibration/clock_excursion_structure_function.png`
was made in July by `clock_structure_figure.py` (`clock_excursion_check.py`
writes the same name) and predates the 2026-10-21 fixes to both scripts.
It draws pair differences below 0.2 mHz, zeros included, at 0.2 mHz with
the same marker as the rest, neither marked nor counted; its line at
5.5e-11 of the carrier is labeled "logged Rb ... (71 mHz)", where the script
now calls it a Stockert Rb figure whose attribution is provisional (REPORT
§4); and its title gives a "~5-7 mHz" floor, "flat 30 s - 3 h", where the
script now prints the measured range of the clean RMS per lag bin. It
cannot be redrawn here: both scripts read
`results/LOLA_DEM_REGISTRATION/registration_runs_chan1.csv`, which this
working copy does not have.

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
