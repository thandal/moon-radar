# Pipeline validation

These checks test the geometry, calibration, and map interpretation behind
[REPORT.md](../REPORT.md). Scripts read production inputs and write results
under `validation/results/` and logs under `validation/logs/`. Rerunning a
check replaces its saved outputs; the tracked files are evidence from earlier
runs, principally July 3, 2026.

## Running checks

Run from the repository root with `.conda/bin/python`. Most pipeline modules
load SPICE kernels at import, even for otherwise synthetic checks. Install
the [kernel set](../spice_setup.py) first; terrain checks also need LOLA data.
Raw captures and a working CuPy/CUDA setup are required for real-data imaging.
Registration checks need the saved per-look maps and runs CSVs.

```sh
# Parameter selection: no kernels, GPU, or recordings needed.
.conda/bin/python -m pytest test/test_recommend_zc.py -q

# Geometry, GPU correlation, tone recovery, and real-data closure.
.conda/bin/python test/test_pipeline_consistency.py

# Terrain implementation and synthetic registration-sign checks.
.conda/bin/python test/test_lola_dem.py
.conda/bin/python test/test_registration_conventions.py

# Physics suite, including real-look terrain A/B and both channel bootstraps.
.conda/bin/python validation/scripts/run_validation_suite.py

# Smaller physics run: no real-look A/B or registration bootstrap.
.conda/bin/python validation/scripts/run_validation_suite.py --quick
```

`--quick` still needs the kernels and terrain inputs; it is not an input-free
test. This checkout lacks kernels, terrain, raw recordings, and generated
production maps. The ZC parameter tests were rerun successfully during the
September 14 documentation review; the data-dependent suite was not rerun.

## What the suite measures

Script names below are relative to `validation/scripts/`.

| Script | Question | Main saved result |
|---|---|---|
| `validate_doppler_dem_physics.py` | How closely do the fast surface fields match per-point SPICE, and how much does terrain change delay and Doppler? | [doppler_dem_physics.json](results/doppler_dem_physics.json) |
| `validate_srp_velocity.py` | Does the analytic reference-point velocity agree with a refined numerical reference? | [srp_velocity.json](results/srp_velocity.json) |
| `validate_timing_frequency_separability.py` | Does injecting a timing error disturb frequency recovery, or vice versa? | [timing_frequency_separability.json](results/timing_frequency_separability.json) |
| `validate_rim_calibration_stress.py` | How biased is the rim estimator under smear, speckle, weak contrast, and unequal rim brightness? | [rim_calibration_stress.json](results/rim_calibration_stress.json) |
| `validate_signal_processing.py` | What are the ideal waveform sidelobes and the scale of DD sampling artifacts? | [signal_processing.json](results/signal_processing.json) |
| `validate_lola_dem_projection.py` | Does terrain move real map features as predicted? | [lola_dem_projection.json](results/lola_dem_projection.json) |
| `validate_registration_bootstrap.py` | How stable is alignment when the contributing looks change? | `registration_bootstrap_chan{0,1}.json` |

The timing test reports delay error both above its detection threshold and
without that cut; weak synthetic tones can give unstable delay estimates.
The terrain test's high-correlation regions support the predicted feature
movement, while the Tycho region does not provide a reliable measurement.

## Additional checks

These are separate from the suite runner:

```sh
.conda/bin/python validation/scripts/validate_registration_bootstrap.py \
  --channel chan1 --mode closure
.conda/bin/python validation/scripts/validate_speckle_floor.py --chan chan1
.conda/bin/python validation/scripts/validate_speckle_floor.py --chan chan0
.conda/bin/python validation/scripts/validate_absolute_registration.py
```

The closure bootstrap measures internal agreement, the speckle fit measures
how variance falls with look count, and the absolute-registration test compares
a radar stack with terrain-derived reference images.

The saved speckle fits use 110 looks per channel, before the final production
exclusions. The bootstrap and speckle scripts currently apply SNR and delay
cuts but do not reproduce `stack_maps.py`'s later rim-spread gate. Their results
should not be relabeled as measurements of the final 107-look stacks.

Small closure errors do not establish absolute map accuracy. A constant
variance floor can include stable artifacts as well as lunar features. Ideal
waveform sidelobes do not establish performance through the radio hardware;
use the [signal-check runbook](../SIGNAL_CHECK_PROCESS.md) for that campaign.
