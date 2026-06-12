# moon-radar — file index

Bistatic lunar delay-Doppler mapping pipeline. Three documents, three jobs:

- `WALKTHROUGH.md` — how it works: one capture traced end-to-end, with the
  function that does each step (start here)
- `REPORT.md` — system of record: conventions, error budget, measured
  results, open items
- this file — the "what is this file" map

Run everything from the repo root with `.conda/bin/python` (data and
kernel paths are relative).

## Quickstart

```sh
# process a session into per-look DD images + maps + calibration CSV
.conda/bin/python registration_stability.py
# stack registered sessions into deep maps
.conda/bin/python stack_maps.py
# regression-verify the numerical core after any change
.conda/bin/python test/test_pipeline_consistency.py
```

## Core pipeline

| file | role |
|---|---|
| `spice_setup.py` | standard SPICE kernel list + `furnsh_kernels()` (call once at script startup) |
| `doppler_equator.py` | geometry core: light times, Doppler/dlt, SRP solver, window-averaged dlt, apparent station positions, Doppler-equator methods |
| `doppler_equator_alignment.py` | imaging & calibration: DD image, rim calibration, surface projection, degeneracy mask, batch `process_file` |
| `freq_offset_hunt.py` | per-look timing/frequency measurement (product method, tone centroid, sub-sample refinement) |
| `registration_stability.py` | batch driver: per-channel processing → `results/REGISTRATION/registration_runs_{chan0,chan1}.csv` |
| `stack_maps.py` | session-offset solve, scattering normalization, deep stacks |
| `registration_analysis.py` | gridding, band-pass, masked cross-registration helpers (shared by stack_maps) |

## Analysis & diagnostics

| file | role |
|---|---|
| `wander_corrected_batch.py` | A/B batches (uncalibrated vs calibrated) |
| `rim_bias_validation.py` | synthetic-echo validation of the rim δ estimator |
| `recover_railed.py` | one-shot ±40-sample recovery of railed captures (patches CSV) |
| `intra_look_drift.py` | half-window drift measurement (prep for REPORT §8.5) |
| `ata_stockert_crosscheck.py` | independent ATA↔Stockert registration cross-check (2025-09-16) |
| `doppler_equator_errors.py` | error-model classes & plots |
| `velocity_error_breakdown.py` | velocity error component analysis |
| `error_visualization_example.py` | error-plot examples (see `PLOTTING_GUIDE.md`) |
| `plot_doppler_equator_simple.py` | standalone nominal Doppler-equator plotter |
| `test_error_viz.py` | quick smoke plot of equator + uncertainty (output → `results/VERIFY/`) |

## Notebooks

All notebooks are archival — the `.py` pipeline above is the live system.
See `archived_notebooks/README.md`.

## Infrastructure

| file | role |
|---|---|
| `test/` | regression suite; `test/test_pipeline_consistency.py` is the gate for any numerical change |
| `observatories.defs` + `make_observatory_kernels.sh` | station definitions → SPICE kernels (re-run after adding a station) |
| `spice_kernels/` (untracked) | DE440s + lunar/Earth orientation + station kernels |
| `data.camras.nl/` (untracked, 36 GB) | raw sigmf captures — never write here |
| `results/` (untracked) | all generated outputs |
| `PLOTTING_GUIDE.md` | Doppler-equator plotting recipes |
