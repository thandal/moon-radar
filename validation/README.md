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

# Physics suite, including real-look terrain A/B and both channel bootstraps
# (on the Stockert looks in --run-dir, made with the 2026-10-21 pipeline).
# Exits non-zero if any step fails to run or fails its criteria.
.conda/bin/python validation/scripts/run_validation_suite.py --run-dir results/STOCKERT_2025

# Smaller physics run: no real-look A/B or registration bootstrap.
.conda/bin/python validation/scripts/run_validation_suite.py --quick
```

`--quick` still needs the kernels and terrain inputs; it is not an input-free
test. Kernels (`spice_kernels/`), terrain (`lola_dem/`), raw recordings
(`data.camras.nl/`) and run directories (`results/`) are present in this
working copy but are not tracked by git (`.gitignore`), so a fresh clone
has none of them. The run directories there, `results/STOCKERT_2025` and
`results/EVE2026_CIRC`, were made before the 2026-10-21 pipeline and will
be regenerated; until then every check below that reads their looks
(`--run-dir`) refuses them. The ZC parameter tests were rerun successfully
during the September 14 documentation review; the data-dependent suite was
not rerun.

## What the suite measures

Script names below are relative to `validation/scripts/`. Only
`validate_srp_velocity.py` tests its results against acceptance criteria
(it exits 1 when they fail). Every other script reports measurements only:
it exits 0 whenever it runs to the end, so "ran" is not "passed", and the
suite labels each step accordingly. Saved results linked below are from the
earlier runs, under their old names; files written since 2026-10-21 name the
receiving station whose geometry or looks they use
(`doppler_dem_physics_stockert.json`, `srp_velocity_stockert.json`,
`rim_calibration_stress_stockert.json`, `lola_dem_projection_stockert.json`,
`registration_bootstrap_stockert_chan1.json`, ...). The suite runs
Stockert's throughout and names each step, and its log, after its results
file.

| Script | Question | Kind | Main saved result |
|---|---|---|---|
| `validate_doppler_dem_physics.py` | How closely do the fast surface fields match per-point SPICE, and how much does terrain change delay and Doppler? | report only | [doppler_dem_physics.json](results/doppler_dem_physics.json) |
| `validate_srp_velocity.py` | Does the analytic reference-point velocity agree with a refined numerical reference? | acceptance criteria | [srp_velocity.json](results/srp_velocity.json) |
| `validate_timing_frequency_separability.py` | Does a frequency offset bias the delay estimate, or a delay the frequency estimate? | report only | [timing_frequency_separability.json](results/timing_frequency_separability.json) |
| `validate_rim_calibration_stress.py` | How biased is the rim estimator under smear, speckle, weak contrast, and unequal rim brightness, and how often does it fail to converge? | report only | [rim_calibration_stress.json](results/rim_calibration_stress.json) |
| `validate_signal_processing.py` | What are the ideal waveform sidelobes and the scale of DD sampling artifacts? | report only | [signal_processing.json](results/signal_processing.json) |
| `validate_lola_dem_projection.py` | Does terrain move real map features as predicted? | report only | [lola_dem_projection.json](results/lola_dem_projection.json) |
| `validate_registration_bootstrap.py` | How stable is alignment when the contributing looks change? | report only | `registration_bootstrap_<stations>_chan{0,1}.json` |

The timing test runs a 30 s synthetic (the shortest production look) with a
chip-structured code at several echo strengths, each named by the tone
strength it gives with no delay and no tone: by default the stacker's
threshold (15), and 40 and 180, which bracket most real co-pol looks so far.
It reports every case with its timing status and whether the stacker's
gates on a timing measurement would accept it, and per strength the
timing-status counts, the worst delay and frequency errors over all cases
and over the accepted ones, and the errors per injected tone and per
injected delay. Its earlier saved result came from a 6 s synthetic at one
strength, with no timing status (the timing rule then took the strongest
peak), and quoted its delay error (3.0 samples at worst) only for cases
above an SNR cut of 30. No basis for that value is recorded, and it sat
just above two of the cases it removed (SNR 29.6 and 29.7). The low SNR
there came from between-bin tones spreading over the search band, not from
weak signal, and the cut kept only two of the five tones (0 and -0.18 Hz).
Of the three it removed, the delay errors reached 9.9 samples at 0.07 Hz
and 10.1 at 0.21 Hz, the coupling the test is meant to find, and 3.1 at
-0.04 Hz.

The terrain test reports every region it measures, with its correlation
significance and whether that is a lock; it has no pass/fail criterion. It
now applies the look's timing and chain-frequency offsets, measured as the
batch measures them, and stops unless that measurement passes the stacker's
timing gates;
gives both passes the DEM pass's Doppler calibration; masks both maps, and
the prediction, to one set of pixels (mapped and not degenerate in both
passes) before band-passing; and leaves pixels without a prediction out of
the region means, counted.

The saved terrain result,
[lola_dem_projection.json](results/lola_dem_projection.json), was made
before all of that and is superseded until the check is rerun: each pass
ran its own rim calibration; both passes applied zero timing and frequency
corrections; the prediction dropped the 5 % of pixels with the smallest
Jacobian determinant |det| (a cut by quantile, not the degeneracy mask);
pixels without a prediction (those and any off the near side) were filled
with zero displacement before the region means; each map was band-passed
on its own mask; and its significances use the old definition (below). In
that run no region reached the lock significance of 1.5 (Tycho 1.22,
Copernicus 1.34, Mare Imbrium 1.33).

The saved absolute-registration and terrain significances use the old
definition: the peak over the strongest correlation outside a fixed 0.5 deg
radius of it. At these smoothing scales that radius lies inside the
correlation's main lobe, so the strongest correlation outside it can be the
main lobe's own flank, which pulls the ratio toward 1. The current
definition (`registration_analysis.xcorr_offset`) leaves out the whole main
lobe, so values under the two are not comparable; the lock level of 1.5
was set under the old one and has not been recalibrated.

## Additional checks

These are separate from the suite runner:

```sh
.conda/bin/python validation/scripts/validate_registration_bootstrap.py \
  --run-dir results/STOCKERT_2025 --stations stockert --channel chan1 --mode closure
.conda/bin/python validation/scripts/validate_speckle_floor.py \
  --run-dir results/STOCKERT_2025 --stations stockert --chan chan1
.conda/bin/python validation/scripts/validate_speckle_floor.py \
  --run-dir results/STOCKERT_2025 --stations stockert --chan chan0
.conda/bin/python validation/scripts/validate_absolute_registration.py \
  --map results/STOCKERT_2025/stacked_linear_stockert_dual_scatnorm.npy
.conda/bin/python validation/scripts/validate_mirror_ghosts.py \
  --run-dir results/EVE2026_CIRC --stations stockert wsrt --chan chan1
.conda/bin/python validation/scripts/validate_edge_offsets.py \
  --run-dir results/EVE2026_CIRC --data-root data/eve2026_circ \
  --station stockert --looks 09:41:25 09:28:21
```

`validate_mirror_ghosts.py` measures how much of the mirror ghosts (each
look's copy of the surface reflected across its Doppler equator) a stack
cancels, per feature size, on a synthetic surface with the stacked looks'
own geometry and masks; a single look keeps all of its ghost power (1).

`validate_edge_offsets.py` checks that the edge fit (`fit_edge_offsets`,
each look's delay and Doppler offsets from its echo edges) recovers offsets
injected into synthetic looks made from the model image, with uneven
brightness along and across the Doppler equator, the image's noise floor
and speckle correlated as the real image's. It needs saved DD images
(`--save-pngs`). It does not test the model itself (the synthetic looks
come from the same model). First run, 2026-10-22, Stockert 09:41:25 (125 s)
and 09:28:21 (35 s), four injections each up to 25 us and 67 mHz: errors
within 0.92 us and 0.49 mHz, within 2 of the fit's own jackknife errors.

None of them has a default run directory or stack; name it on the command line.
The run directory must be made with the 2026-10-21 pipeline
(`registration_stability.py`); `results/STOCKERT_2025` as it stands
predates it and will be regenerated. The stack named in the last example
does not exist yet: `stack_maps.py --run-dir results/STOCKERT_2025
--stations stockert --chans chan1 chan0` writes it once that directory has
been reprocessed.

The closure bootstrap measures internal agreement, the speckle fit measures
how variance falls with look count, and the absolute-registration test compares
a radar stack with terrain-derived reference images. All three report
measurements only.

The bootstrap and speckle scripts now select looks with `stack_maps.py`'s
own gates (`select_looks`) and use its radiometry (`look_brightness`), so
they need run directories made with the 2026-10-21 pipeline; older runs
CSVs are refused. Each records in its results JSON the looks it left out,
with the reason (`looks_left_out`), and the captures the batch did not map
(`captures_not_mapped`). The bootstrap band-passes each stack's mean
brightness as the stacker does (`stack_maps.band_of`); the speckle fit
keeps linear units and the linear band-pass, under which its A/N + C model
stays exact. The saved
speckle fits and bootstraps predate this: they used 110 looks per channel
with only the SNR and delay cuts, log-domain stacks, and (speckle) an
unweighted fit whose coverage rule changed along the ladder.
They should not be relabeled as measurements of the final 107-look stacks.

The saved absolute-registration result printed the correction that would
move the stack onto LOLA, not the stack's offset from it. Its north
components of -3.4 km at the two finest scales (-2.2 and +0.4 km at the
coarser two) therefore mean the stack sat about 3.4 km north of the LOLA
frame there, with significance 1.03-1.15 under the old definition (no
lock). It was made from the stacked log map with the plain band-pass. The
script now reports the offset itself, and reads a stack's mean brightness
(`stacked_linear_*.npy`), band-passed as the stacker band-passes it
(`registration_analysis.log_bandpass`): the log map of a noise-subtracted
stack has no value where the mean is not positive, and a band-pass erodes
its mask around each such hole.

Small closure errors do not establish absolute map accuracy. A constant
variance floor can include stable artifacts as well as lunar features. Ideal
waveform sidelobes do not establish performance through the radio hardware;
use the [signal-check runbook](../SIGNAL_CHECK_PROCESS.md) for that campaign.
