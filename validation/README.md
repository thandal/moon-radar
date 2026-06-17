# Moon-Radar Physics Validation

This folder contains validation code and outputs separate from the production
pipeline and the existing `results/` tree. Scripts read production code/results
as inputs and write only under `validation/results/` and `validation/logs/`.

Run from the repo root with the project conda environment. The default runs
every check at full settings, including the slow registration bootstrap (needs
saved per-look map products) and the GPU/raw-data LOLA DEM A/B feature-shift
check:

```bash
.conda/bin/python validation/scripts/run_validation_suite.py
```

For a fast smoke run while iterating, add `--quick`: smaller settings,
geometry-only LOLA, and no registration bootstrap.

```bash
.conda/bin/python validation/scripts/run_validation_suite.py --quick
```

## Experiments

- `validate_doppler_dem_physics.py`
  - Tests exact SPICE light-time/window-Doppler against the anchored apparent
    station field used by the mapper.
  - Quantifies ellipsoid-vs-DEM topographic delay and Doppler. This directly
    addresses the unresolved report question about DEM Doppler near the limb.
  - Outputs: `doppler_dem_physics.json`.

- `validate_timing_frequency_separability.py`
  - Injects controlled delay and frequency offsets into a synthetic
    constant-modulus echo and runs the product-method estimator.
  - Tests whether delay and reference-frequency calibration are empirically
    separable at the estimator level.
  - Frequency error is SNR-robust and reported over all cases; the delay-error
    headline is gated to reliable detections (`--min-snr`, default 30), with the
    ungated value kept alongside. Below the gate the synthetic's between-bin
    tones hit FFT-scalloping notches that let the delay argmax wander — a test
    artifact, not estimator coupling.
  - Outputs: `timing_frequency_separability.csv/.json`.

- `validate_rim_calibration_stress.py`
  - Self-contained synthetic rim-caustic forward model (built on the real SPICE
    geometry) driven through the production rim estimator.
  - Stresses asymmetry, smear, weak contrast, and scattering-law changes.
  - Outputs: `rim_calibration_stress.csv/.json/.png`.

- `validate_signal_processing.py`
  - Recomputes matched-bandwidth waveform ambiguity metrics for ZC and BPSK.
  - Runs a synthetic nearest-neighbor-vs-bilinear DD sampling experiment to
    estimate quantization artifact scale.
  - Outputs: `waveform_ambiguity_metrics.csv/.png`, `signal_processing.json`.

- `validate_lola_dem_projection.py`
  - Part 1 (geometry): the ellipsoid->DEM delay/Doppler displacement field per
    session -- the mapping systematic the DEM removes -- as delay-pixel and
    surface-km statistics, with a delay-shift map PNG per session.
  - Part 2 (single-look A/B, full run only): processes one real look twice
    (ellipsoid vs DEM) and cross-correlates high-relief ROIs; real map content
    must move by the local mapping Jacobian's prediction. Needs raw SDR data and
    a GPU; skipped under `--quick`.
  - Outputs: `lola_dem_projection.json`,
    `lola_dem_delay_shift_px_<session>.png`, and (with A/B) `lola_dem_tycho_ab.png`.

- `validate_registration_bootstrap.py`
  - Builds random half-stack splits from saved per-look map products and
    measures offset/correlation stability.
  - Tests whether registration conclusions survive speckle and look selection,
    rather than relying on one cross-correlation peak.
  - Outputs: `registration_bootstrap_<channel>.csv/.json`.

## Interpreting Results

The goal is not to produce a single green/red status. These checks quantify
which report conclusions are robust and which need caveats:

- Anchored field residuals should be well below the rim-calibration Doppler
  scale for the approximation to be physically safe.
- DEM topographic Doppler is measured and reported, not assumed negligible.
- Rim calibration bias should remain small compared with the post-calibration
  Doppler error budget under realistic stressors.
- Registration bootstrap offset distributions are more important than one
  nominal cross-correlation value.
- Waveform conclusions should be judged at matched bandwidth/time; bandwidth
  changes are a separate resolution lever.
- The ellipsoid->DEM displacement field is the systematic the projection
  removes; the A/B check should show real map content moving in the direction
  and km-scale the mapping Jacobian predicts, not just the model field.
