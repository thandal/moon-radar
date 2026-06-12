# Archived notebooks

These are exploratory / development snapshots, kept for provenance. They are
**superseded** by the `.py` pipeline at the repo root and by `REPORT.md`,
which is the system of record for all findings, constants, and design
rationale. Any insight in these notebooks that was still load-bearing has been
folded into `REPORT.md` (see §2 for the TX-resampling and numerical-working-point
notes harvested from `test_time_resampling.ipynb` and the ZC notebooks).

| notebook | what it was |
|---|---|
| `Planetary Radar -- Moon.ipynb` | annotated conceptual walkthrough of the geometry/signal model; predates the measured per-look timing calibration (contains the old `TX_START_OFFSET` hacks) |
| `Planetary Radar -- Moon -- extras.ipynb` | extended experiments / debugging variants of the reference pipeline |
| `delay-doppler-dwingeloo-stockert-v4.ipynb` | early operational GPU (CuPy) pipeline, superseded by the `.py` batch pipeline |
| `gen_zadoff_chu.ipynb` | Zadoff-Chu waveform generation; upsampling-method comparison |
| `tx_zc_nathaniel.ipynb` | ZC TX synthesis variant with SigMF metadata |
| `test_time_resampling.ipynb` | phase-vs-magnitude interpolation experiments for TX resampling |
| `test_frequency.ipynb` | stub / abandoned frequency-shift exploration |

No live notebooks remain: the `.py` pipeline at the repo root
(`registration_stability.py` → `stack_maps.py`, verified by
`test/test_pipeline_consistency.py`) is the system of record for processing,
and `REPORT.md` for findings. See `WALKTHROUGH.md` for the end-to-end flow
these notebooks used to narrate.
