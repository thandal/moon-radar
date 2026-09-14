# Moon radar

Turn lunar radar recordings into maps of the Moon's reflectivity. The active
pipeline calibrates Dwingeloo→Stockert observations, projects them onto lunar
terrain, and aligns and averages the resulting maps.

The latest documented run processed 222 maps and retained 214 after quality
cuts. Internal alignment is below 1 km; absolute map placement remains
unverified. See the [report](REPORT.md) for results and limitations.

- [Walkthrough](WALKTHROUGH.md): follow one capture from recording to map.
- [Report](REPORT.md): measured results, conventions, and remaining work.
- [Observation planning](LIBRATION_ANALYSIS.md): choose complementary views.
- [Signal checks](SIGNAL_CHECK_PROCESS.md): test proposed transmit waveforms.
- [Validation](validation/README.md) and [plotting](PLOTTING_GUIDE.md): developer reference.

## Running the pipeline

Run from the repository root using `.conda/bin/python`, or activate the
project environment with `conda activate .conda/`. Imaging requires a
CUDA-capable GPU and CuPy. You also need these separately supplied inputs:

| Input | Location |
|---|---|
| Raw SigMF recordings and referenced TX files | `data.camras.nl/lunar-radar/` |
| SPICE geometry kernels | `spice_kernels/`; exact list in [spice_setup.py](spice_setup.py) |
| LOLA terrain grids | `lola_dem/`; fetch with `./fetch_lola_dem.sh` |
| Reconstructed September 10 interpolated waveform | `tx_signals/`; see [REPORT §10](REPORT.md#10-data-coverage) |

These inputs and generated `results/` maps are untracked and absent from
this checkout. Without terrain grids the mapper falls back to an ellipsoid;
check `use_dem` in the output CSV before comparing with the reported DEM run.

With those inputs in place, this processes all eligible Stockert captures
into an example output directory. Choose a fresh directory for each rerun:

```sh
# Co-pol first; cross-pol inherits its timing and frequency calibration.
.conda/bin/python registration_stability.py --per-date 0 --chan chan1 \
  --out-dir results/NEW_REGISTRATION
.conda/bin/python registration_stability.py --per-date 0 --chan chan0 \
  --corrections-from results/NEW_REGISTRATION/registration_runs_chan1.csv \
  --out-dir results/NEW_REGISTRATION

# Make each channel's stack, then the combined stack.
.conda/bin/python stack_maps.py --run-dir results/NEW_REGISTRATION --chans chan1
.conda/bin/python stack_maps.py --run-dir results/NEW_REGISTRATION --chans chan0
.conda/bin/python stack_maps.py --run-dir results/NEW_REGISTRATION --chans chan1 chan0
```

Without `--per-date 0`, the batch selects only eight captures per date
folder. September 10's folder includes September 11. Without `--chans`, the
stacker produces only chan1. Batch reruns replace that channel's CSV and
matching map files in the chosen directory.

## File index

| File | Purpose |
|---|---|
| [spice_setup.py](spice_setup.py) | Load the common geometry kernels. |
| [doppler_equator.py](doppler_equator.py) | Light times, Doppler, reference point, and lunar terrain. |
| [doppler_equator_alignment.py](doppler_equator_alignment.py) | DD imaging, rim calibration, and surface projection. |
| [freq_offset_hunt.py](freq_offset_hunt.py) | Measure per-capture delay and frequency offsets. |
| [registration_stability.py](registration_stability.py) | Run the per-channel batch and write calibration CSVs. |
| [registration_analysis.py](registration_analysis.py), [stack_maps.py](stack_maps.py) | Align maps and build session and combined stacks. |
| [recommend_zc.py](recommend_zc.py) | Select candidate ZC waveform lengths and roots. |
| [predict_libration_opportunities.py](predict_libration_opportunities.py) | Search for complementary observing geometries. |
| [ata_stockert_crosscheck.py](ata_stockert_crosscheck.py) | Compare the two receivers' maps. |
| [wander_corrected_batch.py](wander_corrected_batch.py), [intra_look_drift.py](intra_look_drift.py), [recover_railed.py](recover_railed.py) | Calibration experiments and historical recovery tools; check their paths and settings before reuse. |
| [observatories.defs](observatories.defs), [make_observatory_kernels.sh](make_observatory_kernels.sh) | Station definitions and kernel generation. |
| [test/](test/) | Regression tests; commands and dependencies in the [validation guide](validation/README.md). |
| [investigations/](investigations/README.md), [archived_notebooks/](archived_notebooks/README.md) | Historical experiments; some use superseded conventions. |

`results/LOLA_DEM_REGISTRATION/` is the latest documented production run.
`results/LOLA_DEM_REGISTRATION_FROZEN_0612/` preserves its predecessor;
`results/REGISTRATION/` is the pre-terrain baseline. Saved validation evidence
is tracked separately under [validation/results/](validation/results/).
