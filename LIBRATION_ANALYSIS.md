# Choosing complementary lunar views

The most useful new observations show parts of the Moon that earlier views
could not separate clearly. This guide explains how to find those geometries
with [predict_libration_opportunities.py](predict_libration_opportunities.py).
Reviewed 2026-09-14; the saved July search below is historical, and its dates
have passed. No new observing windows were calculated for this review.

## Why geometry matters

A delay–Doppler image cannot distinguish surface points with the same travel
time and frequency shift. Near the **Doppler equator**, this ambiguity stretches
across a stripe of the disk, so stacking masks the most affected pixels.
Changing the observing geometry rotates the stripe and lets other looks fill
those gaps.

The stripe's orientation depends on both lunar libration and the stations'
motion as Earth rotates. It can change substantially within one observing
session. Planning therefore compares each candidate with **all recorded-look
axes**, rather than one representative axis per day. The July analysis found
these axis swings across the actual 2025 recordings:

| Date | Recorded axis swing |
|---|---:|
| June 21 | 5.9° |
| September 10 | 5.5° |
| September 11 | 70° |
| September 16 | 14.4° |

A large change in direction is not sufficient by itself. A small lunar
Doppler span packs the echo into fewer independent frequency-resolution cells.
Elevation, Doppler span, and useful observing duration all matter when
choosing a pass.

## Run a new search

Use the project `.conda` environment from the repository root. Supply the
actual run CSVs and a date range; the script's defaults still search
June 18–September 10, 2026.

```sh
.conda/bin/python predict_libration_opportunities.py \
  --start-date 2026-09-15 --end-date 2027-09-15 \
  --run-dir results/LOLA_DEM_REGISTRATION
```

The default cuts require the Moon above 45° at Dwingeloo and 10° at Stockert,
and a lunar Doppler span of at least 5 Hz. The scan uses hourly steps and
writes `results/ERRORS/libration_opportunities.csv` and
`results/ERRORS/libration_opportunities.png`.

Check two qualifications in the output. If the run CSVs are missing, the tool
approximates old coverage with reconstructed session spans sampled every
15 minutes. Some output labels still call these “midpoints”; the code samples
spans. Beyond the loaded high-precision Earth-orientation kernel's coverage,
it uses an approximate Earth-rotation model while retaining the station
locations. Supply suitable kernels before relying on a future schedule.
Neither kernels nor run CSVs are present in this checkout.

## Reading the ranking

“Displacement” is the smallest angle between a candidate Doppler axis and
any recorded axis, treating opposite axis directions as equivalent. A larger
value means a more distinct stripe orientation. The table also reports the
minimum displacement and axis swing over the pass: a promising peak hour does
not describe a whole observation.

The July 3 search used 111 recorded look epochs. Its three highest peak-hour
results were:

| Peak (UTC, 2026) | Peak displacement | Minimum over pass | Doppler span |
|---|---:|---:|---:|
| July 10, 09:00 | 26.1° | About 0.1° | 13.5 Hz |
| September 4, 04:00 | 26.0° | About 0.1° | 15.3 Hz |
| August 8, 06:00 | 25.4° | About 1.7° | 13.1 Hz |

These are archived predictions, not records of observations. An older 51.2°
claim compared only session midpoints and overstated new coverage. The current
planner also uses the analytic SRP velocity validated in
[REPORT §2](REPORT.md#2-geometry-and-signal-model), replacing a numerical
axis estimate that could occasionally jump between directions.
