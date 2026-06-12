# Archived notebooks

These are exploratory / development snapshots, kept for provenance. They are
**superseded** by the `.py` pipeline under `notebooks/` and by `REPORT.md`,
which is the system of record for all findings, constants, and design
rationale. Any insight in these notebooks that was still load-bearing has been
folded into `REPORT.md` (see §2 for the TX-resampling and numerical-working-point
notes harvested from `test_time_resampling.ipynb` and the ZC notebooks).

| notebook | what it was |
|---|---|
| `Planetary Radar -- Moon -- extras.ipynb` | extended experiments / debugging variants of the reference pipeline |
| `gen_zadoff_chu.ipynb` | Zadoff-Chu waveform generation; upsampling-method comparison |
| `tx_zc_nathaniel.ipynb` | ZC TX synthesis variant with SigMF metadata |
| `test_time_resampling.ipynb` | phase-vs-magnitude interpolation experiments for TX resampling |
| `test_frequency.ipynb` | stub / abandoned frequency-shift exploration |

The two live notebooks remain at the top level: `Planetary Radar -- Moon.ipynb`
(annotated reference pipeline) and `delay-doppler-dwingeloo-stockert-v4.ipynb`
(operational GPU pipeline).
