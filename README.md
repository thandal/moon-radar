# moon-radar — file index

Bistatic lunar delay-Doppler mapping pipeline. Three documents, three jobs:

- `WALKTHROUGH.md` — how it works: one capture traced end-to-end, with the
  function that does each step (start here)
- `REPORT.md` — system of record: conventions, error budget, measured
  results, open items
- this file — the "what is this file" map

Also: `LIBRATION_ANALYSIS.md` — future observation-window planning
(Doppler-equator coverage). (The former `DATA_MACHINE_TODO.md` checklist
was completed and retired 2026-07-04; its results live in REPORT §1–§8 and
`investigations/rim_window_recalibration_2026-07-03.md`, and the remaining
open items are REPORT §8.13/§8.14.)

Run everything from the repo root with `.conda/bin/python` (data and
kernel paths are relative).

## Quickstart

```sh
# one-time: fetch LOLA topography for the surface projection
# (maps fall back to the reference ellipsoid without it)
./fetch_lola_dem.sh
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
| `doppler_equator.py` | geometry core: light times, Doppler/dlt, SRP solver, window-averaged dlt, apparent station positions, Doppler-equator methods, LOLA DEM surface (`load_lola_dem`, `moon_surface_points(use_dem=True)`) |
| `doppler_equator_alignment.py` | imaging & calibration: DD image, rim calibration, surface projection, degeneracy mask, batch `process_file` |
| `freq_offset_hunt.py` | per-look timing/frequency measurement (product method, tone centroid, sub-sample refinement) |
| `registration_stability.py` | batch driver: per-channel processing → `registration_runs_{chan0,chan1}.csv` (current rim-recalibrated LOLA-DEM run in `results/LOLA_DEM_REGISTRATION/`; the frozen 2026-06-12 predecessor is archived as `results/LOLA_DEM_REGISTRATION_FROZEN_0612/`; `results/REGISTRATION/` is the pre-DEM baseline) |
| `stack_maps.py` | session-offset solve, scattering normalization, deep stacks |
| `registration_analysis.py` | gridding, band-pass, masked cross-registration helpers (shared by stack_maps) |

## Analysis & diagnostics

| file | role |
|---|---|
| `wander_corrected_batch.py` | A/B batches (uncalibrated vs calibrated) |
| `predict_libration_opportunities.py` | future observation-window planner (Doppler-equator coverage; see `LIBRATION_ANALYSIS.md`) |
| `validation/scripts/validate_srp_velocity.py` | analytic SRP velocity/Doppler-axis proof vs lattice-free FD reference (kernels-only) |
| `validation/scripts/validate_speckle_floor.py` | variance-vs-N fit: substantiates the structure-limited/speckle-floor claim (data machine) |
| `validation/scripts/validate_absolute_registration.py` | absolute selenolocation tie: stack vs LOLA slope proxy (data machine) |
| `validation/scripts/validate_rim_calibration_stress.py` | synthetic-echo validation of the rim δ estimator |
| `validation/scripts/validate_lola_dem_projection.py` | DEM-vs-ellipsoid displacement field + single-look A/B feature-shift check (REPORT §8.4) |
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
| `test/` | asserting gates only: `test_pipeline_consistency.py` (numerical core), `test_lola_dem.py` (DEM chain), `test_registration_conventions.py` (registration signs) |
| `investigations/` | archived print-only exploration scripts (moved from `test/`; see its README — some encode superseded conventions) |
| `observatories.defs` + `make_observatory_kernels.sh` | station definitions → SPICE kernels (re-run after adding a station) |
| `fetch_lola_dem.sh` | downloads LOLA GDR DEMs (PDS) into `lola_dem/` |
| `lola_dem/` (untracked) | LOLA topography grids (`ldem_<ppd>.img/.lbl`); highest resolution present is used |
| `spice_kernels/` (untracked) | DE440s + lunar/Earth orientation + station kernels |
| `data.camras.nl/` (untracked, 36 GB) | raw sigmf captures — never write here |
| `results/` (untracked) | all generated outputs; `LOLA_DEM_REGISTRATION/` is the current 222-look DEM run (rim-recalibrated, seed-rescued, wander-gated; stacks, per-look maps, runs CSVs), `LOLA_DEM_REGISTRATION_FROZEN_0612/` the frozen 2026-06-12 predecessor it superseded (kept for the §8.4 DEM A/B provenance in its `PRE_DEM_ANALYSIS/`), `REGISTRATION/` the pre-DEM baseline |
| `PLOTTING_GUIDE.md` | Doppler-equator plotting recipes |
