# Moon radar

Turn lunar radar recordings into maps of the Moon's reflectivity. The active
pipeline calibrates Dwingeloo→Stockert observations, projects them onto lunar
terrain, and aligns and averages the resulting maps.

The July 2025 run processed 222 maps and retained 214 after quality cuts,
with internal alignment below 1 km; absolute map placement remains
unverified. A code review on 2026-10-21 fixed faults that change these
results (REPORT, top), so every product is being regenerated. See the [report](REPORT.md) for results and limitations.

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
| Raw SigMF recordings and referenced TX files | `data.camras.nl/lunar-radar/` (2025), `data.camras.nl/eve2026/` (2026-10-11) |
| Opposite- and same-sense 2026-10-11 captures | `data/eve2026_circ/`, made locally by `make_circular.py` (below) |
| SPICE geometry kernels | `spice_kernels/`; exact list in [spice_setup.py](spice_setup.py) |
| LOLA terrain grids | `lola_dem/`; fetch with `./fetch_lola_dem.sh` |
| Reconstructed September 10 interpolated waveform | `<data root>/tx_signals/`; see [REPORT §10](REPORT.md#10-data-coverage) |

`data.camras.nl/` is a partial copy of https://data.camras.nl with the
server's folder layout, so running, from the repository root,
`wget -r -np -nc -R 'index.html*' https://data.camras.nl/eve2026/2026-10-11/`
(or any other server folder) adds missing files in place. Its one local
addition is the link `eve2026/tx_signals -> tx_signals_eme`, because the
pipeline reads TX files from `<data root>/tx_signals/`.

These inputs and generated `results/` maps are untracked and absent from
this checkout. The mapper requires the terrain grids: without them it stops
(it no longer falls back to a smooth sphere), and each runs-CSV row records
the terrain file (`dem_file`) and a fingerprint of the SPICE kernels
(`spice_kernels`); the stacker refuses to mix looks that differ in either.

With those inputs in place, this processes the 2025 Stockert captures into
an example output directory. Choose a fresh directory for each rerun:

```sh
# The opposite-sense channel (chan1) first; the same-sense channel (chan0)
# takes its timing, frequency, rim and edge-fit calibration (names: REPORT
# §1). --save-pngs keeps each look's DD image (~17 MB), so
# reproject_maps.py can redo the edge fit and projection without the GPU.
.conda/bin/python registration_stability.py --station stockert --chan chan1 \
  --dates 2025-06-21 2025-09-10 2025-09-16 --out-dir results/NEW_REGISTRATION --save-pngs
.conda/bin/python registration_stability.py --station stockert --chan chan0 \
  --dates 2025-06-21 2025-09-10 2025-09-16 --out-dir results/NEW_REGISTRATION --save-pngs \
  --corrections-from results/NEW_REGISTRATION/registration_runs_stockert_chan1.csv

# Make each channel's stack, then the combined stack.
.conda/bin/python stack_maps.py --run-dir results/NEW_REGISTRATION --stations stockert --chans chan1
.conda/bin/python stack_maps.py --run-dir results/NEW_REGISTRATION --stations stockert --chans chan0
.conda/bin/python stack_maps.py --run-dir results/NEW_REGISTRATION --stations stockert --chans chan1 chan0

# The main product: sessions weighted by geometry and noise. Simulate every
# opposite-sense look once (about 20 s a look; the same-sense stacks reuse them),
# with the opposite-sense scattering law, then combine, per channel set as above.
.conda/bin/python combine_sessions.py --run-dir results/NEW_REGISTRATION --stations stockert \
  --chans chan1 --equal-stack-dir results/NEW_REGISTRATION --out-dir results/GEOMETRY_SIM \
  --write-law results/GEOMETRY_SIM/law_stockert_chan1.npz
.conda/bin/python simulate_geometry.py --run-dir results/NEW_REGISTRATION \
  --data-root data.camras.nl/lunar-radar --station stockert --chan chan1 \
  --out-dir results/GEOMETRY_SIM/seed1 --seed 1 --law results/GEOMETRY_SIM/law_stockert_chan1.npz
.conda/bin/python combine_sessions.py --run-dir results/NEW_REGISTRATION --stations stockert \
  --chans chan1 --equal-stack-dir results/NEW_REGISTRATION --sim-dir results/GEOMETRY_SIM/seed1 \
  --out-dir results/NEW_REGISTRATION
```

The batch maps every capture of the station and channel in the date folders
whose transmitted waveform is a Zadoff-Chu code lasting at least
`--min-tx-s` (default 30 s, measured from the waveform file);
`--per-date N` keeps only N per folder. September 10's folder includes
September 11. Every candidate it does not map is listed with the reason in
`registration_skipped_<station>_<chan>.csv` (short capture, missing
waveform file, unreadable metadata, an error, ...), and the batch exits
non-zero if any capture failed with an error. Batch reruns replace that
station and channel's CSV, `registration_runs_<station>_<chan>.csv`, and
matching map files in the chosen directory. Stack outputs are named after
their stations and channels, for example `stacked_map_stockert_chan1.npy`
or `stacked_map_stockert_wsrt_dual_scatnorm.png`. Seasons processed into
separate directories are stacked together by giving `--run-dir` several
directories and an `--out-dir`, for example
`--run-dir results/STOCKERT_2025 results/EVE2026_CIRC --out-dir results/ALL_SEASONS`.

Each map pixel averages the delay–Doppler image over 4×4 points spread
across the pixel (`--supersample`, default 4). Each look also records its
image's noise floor (`noise_log_amp`) and, per pixel, the ground area one
delay–Doppler cell covers (`<capture>_cellarea.npy`, column `area_npy`).
`stack_maps.py` stacks brightness per unit area with the noise floor
removed, each look scaled to a mean of 1 at 20–50° incidence, with equal
weights; run directories made before the 2026-10-21 fixes (including the
current `results/STOCKERT_2025` and `results/EVE2026_CIRC`) lack the columns it needs
and must be reprocessed. A pixel whose mean brightness is not positive has
no log value and shows as missing (grey) in `stacked_map_*`; the mean
itself, negatives included, is in `stacked_linear_*.npy`. Every look read,
used or not, is listed with its reason and gain in
`stacked_looks_<label>.csv`. If the session alignment fails its checks the
stacker stops; `--no-registration` stacks without session shifts (outputs
named `..._unregistered`).

`combine_sessions.py` makes the main stacked product, named
`..._geomweighted`, beside the equal-weight stack it is built from. Looks
from one night share one geometry, so they repeat the same blur and mirror
ghosts, and averaging more of them does not reduce those. It averages each
session (night) on its own, band by band in feature size: a look counts in
a band only where its delay-Doppler cell is small enough to place features
of that size (2, 4, 8 and 16 times the look's median cell area below 10,
20, 40 and 75 km; no cut for broader features), so large cells leave no
holes. It then weights the sessions separately at each place and in each
band by how much error each is expected to carry there: the geometry error,
from passing a synthetic surface through every look's own geometry
(`simulate_geometry.py`), plus the noise measured from the session's own
looks. Where one session sees a place much better (for example 2026 at the
2025 sessions' specular point), it gets most of the weight there. The
weights and each session's mean are kept in `combination_<label>.npz`;
`validation/scripts/validate_session_combination.py` scores the result on
a second synthetic surface and on real data. Each channel and season has
its own scattering law (`stack_maps.law_key`).

The 2026-10-11 receivers recorded two linear feeds, so their captures are
first combined into opposite- and same-sense captures, then processed as above
(WSRT's timing offset, about +45 samples, needs `--max-shift 80`):

```sh
for st in stockert wsrt; do
  .conda/bin/python make_circular.py --data-root data.camras.nl/eve2026 \
    --out-root data/eve2026_circ --date 2026-10-11 --station $st
done
.conda/bin/python registration_stability.py --data-root data/eve2026_circ \
  --station wsrt --chan chan1 --dates 2026-10-11 --max-shift 80 \
  --out-dir results/EVE2026_CIRC
```

When only the projection or the edge fit changes, a run made with `--save-pngs` can be
re-projected from its saved delay–Doppler images without redoing the
correlation, then restacked:

```sh
.conda/bin/python reproject_maps.py --run-dir results/NEW_REGISTRATION \
  --data-root data.camras.nl/lunar-radar \
  --stations stockert --chans chan1 chan0
```

The images take about 18 MB per look. `results/EVE2026_CIRC/` keeps them
(since the 2026-10-22 rerun); `results/STOCKERT_2025/` does not, so that run
must be reprocessed from the recordings instead.

## File index

| File | Purpose |
|---|---|
| [spice_setup.py](spice_setup.py) | Load the common geometry kernels. |
| [doppler_equator.py](doppler_equator.py) | Light times, Doppler, reference point, and lunar terrain. |
| [doppler_equator_alignment.py](doppler_equator_alignment.py) | DD imaging, rim calibration, edge fit (delay and Doppler offsets), and surface projection. |
| [freq_offset_hunt.py](freq_offset_hunt.py) | Measure per-capture delay and frequency offsets. |
| [registration_stability.py](registration_stability.py) | Run the per-channel batch and write calibration CSVs. |
| [reproject_maps.py](reproject_maps.py) | Rebuild a run's per-look maps from its saved delay–Doppler images. |
| [registration_analysis.py](registration_analysis.py), [stack_maps.py](stack_maps.py) | Align maps and build session and combined stacks. |
| [simulate_geometry.py](simulate_geometry.py) | Pass a synthetic surface through each look's own delay–Doppler geometry and back, giving the look's geometry error. |
| [combine_sessions.py](combine_sessions.py) | Second stack: sessions averaged, then weighted per place and feature size by simulated geometry error plus measured noise. |
| [make_circular.py](make_circular.py) | Synthesize opposite- and same-sense captures from two linear feeds (the 2026-10-11 Stockert and WSRT recordings). |
| [plot_dd.py](plot_dd.py) | Plot saved delay–Doppler images as power over the noise floor, the opposite- and same-sense channels side by side. |
| [flip_compare.py](flip_compare.py) | Flip or blink between maps, stacks or images in one window, with manual nudging, to compare them by eye. |
| [recommend_zc.py](recommend_zc.py) | Select candidate ZC waveform lengths and roots. |
| [predict_libration_opportunities.py](predict_libration_opportunities.py) | Search for complementary observing geometries. |
| [ata_stockert_crosscheck.py](ata_stockert_crosscheck.py) | Compare the two receivers' maps. |
| [wander_corrected_batch.py](wander_corrected_batch.py), [intra_look_drift.py](intra_look_drift.py) | Calibration experiments; check their settings before reuse. |
| [recover_railed.py](recover_railed.py) | Retired: re-measuring looks whose timing shift hit the search limit is now `registration_stability.py --max-shift N` into a fresh directory; the file only says so. |
| [observatories.defs](observatories.defs), [make_observatory_kernels.sh](make_observatory_kernels.sh) | Station definitions and kernel generation. |
| [test/](test/) | Regression tests; commands and dependencies in the [validation guide](validation/README.md). |
| [investigations/](investigations/README.md), [archived_notebooks/](archived_notebooks/README.md) | Historical experiments; some use superseded conventions. |

The run directories being regenerated with the 2026-10-21 fixes are
`results/STOCKERT_2025/`, `results/EVE2026_CIRC/` and `results/ALL_SEASONS/`
(REPORT §9). The July production run `results/LOLA_DEM_REGISTRATION/`, its
predecessor `results/LOLA_DEM_REGISTRATION_FROZEN_0612/` and the pre-terrain
baseline `results/REGISTRATION/` are not in this checkout. Saved validation evidence
is tracked separately under [validation/results/](validation/results/).
