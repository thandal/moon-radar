# Lunar Libration Analysis: Optimal Doppler Equator Coverage

To improve the SNR and surface detail of the stacked Moon-radar reflectivity
map, we want to collect new datasets at epochs where the mapping degeneracy is
shifted away from previous runs.

This document details the physical background, the predicted future
observation windows for the Dwingeloo (TX) → Stockert (RX) baseline, and the
predictive planning tool.

*Revised 2026-07-01: the Doppler axis now comes from the validated closed-form
SRP velocity (REPORT §2, `validation/scripts/validate_srp_velocity.py`), and
existing coverage is represented by the axis swept across each session rather
than one midpoint — both changes materially lower the displacement numbers a
previous revision of this document reported (see §5).*

---

## 1. Physical Background & Optimal Geometry

In delay-Doppler radar mapping, the coordinate system degenerates near the
**Doppler equator** (where the contours of constant delay and constant Doppler
are parallel on the Moon's surface). In this zone, a single delay-Doppler cell
maps to a long surface arc (the "bright stripe artifact"), carrying very
little per-pixel selenographic information. These regions are masked out
during stacking.

The Doppler equator is the great circle normal to the **Doppler axis** — the
apparent-rotation axis of the Moon relative to the radar, computed in closed
form from the station states (REPORT §2, `srp_velocity_analytic`).

* **Existing sessions:** in the four 2025 sessions the Doppler axis sat 45–64°
  from the Moon's polar axis (equivalently, the Doppler equator was inclined
  26–45° to the selenographic equator), so the degenerate stripes cut
  diagonally across the disk.
* **The axis is not static.** The apparent rotation is the *sum* of the
  Moon's libration term and the observer's diurnal-parallax term, and the two
  are the same order of magnitude (~0.3–1.7 m/s of SRP drift). Within a single
  session the axis sweeps tens of degrees — measured 58–82° across the 2025
  sessions, and up to ~87° across a candidate 2026 pass. Coverage and novelty
  must therefore be evaluated against the *swept arc* of axes, not a per-
  session snapshot (§5).
* **The opportunity:** observe when the Doppler axis is maximally displaced
  from every axis in the existing swept coverage, moving the degeneracy
  stripe onto previously masked ground.
* **Doppler bandwidth constraint:** a high-quality DD image also needs a
  sufficient limb-to-limb Doppler span (= 2R|g|f₀/c ≈ 4·v_SRP·f₀/c, validated
  to 0.2% against the full-disk dlt field). If the apparent rotation is slow,
  the echo compresses into a narrow band and Doppler resolution collapses —
  and the axis *direction* also becomes ill-conditioned and swings rapidly
  (the 2025-09-11 morning: 0.29 m/s, 5.1 Hz span, ~70° of axis rotation in
  one morning).

---

## 2. Top Predicted Opportunities (TX Elevation ≥ 45°, Doppler Span ≥ 5 Hz)

Scan window **June 18 – September 10, 2026**, hourly steps, gates: Dwingeloo
elevation ≥ 45°, Stockert ≥ 10°, Doppler span ≥ 5 Hz. Displacement = minimum
angle between the candidate's Doppler axis and *any* axis in the existing
coverage (session spans sampled at 15 min; see the caveat below the table).
"Pass disp" is the minimum displacement over the whole visible pass, and
"Swing" the axis rotation across it — a real multi-hour session samples that
entire range, not just the peak hour.

| Rank | Peak (UTC) | Peak disp | Pass disp | Pass swing | Dwingeloo elev | Stockert elev | Doppler span |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **2026-08-06T07:00** | **21.1°** | 2.8°+ | 38° | 53.3° | 54.8° | **16.3 Hz** |
| **2** | 2026-08-07T06:00 | 21.0° | 1.4°+ | 57° | **61.8°** | **64.1°** | 16.1 Hz |
| **3** | 2026-07-11T08:00 | 20.7° | 1.5°+ | 79° | 62.5° | 64.8° | 12.8 Hz |
| **4** | 2026-09-02T05:00 | 20.6° | 2.0°+ | 33° | 51.6° | 53.2° | 17.1 Hz |
| **5** | 2026-09-04T04:00 | 20.5° | 0.4°+ | 77° | 60.7° | 62.7° | 15.3 Hz |
| **6** | 2026-07-10T09:00 | 20.3° | 0.2°+ | 50° | 55.4° | 57.0° | 13.5 Hz |
| **7** | 2026-09-03T04:00 | 19.4° | 2.6°+ | 49° | 60.8° | 63.1° | **17.7 Hz** |
| **8** | 2026-08-08T06:00 | 18.6° | 0.9°+ | 87° | 60.5° | 62.5° | 13.1 Hz |
| **9** | 2026-07-09T09:00 | 16.8° | 1.9°+ | 20° | 45.3° | 46.5° | 14.1 Hz |
| **10** | 2026-07-12T08:00 | 16.6° | 1.2°+ | 68° | 59.4° | 61.2° | 9.5 Hz |

> [!TIP]
> The **August 6–7** and **September 2–4, 2026** early-morning passes are the
> best cluster: peak displacements ~19–21°, elevations 52–64°, spans
> 15–18 Hz. Aug 7 offers the highest elevations at essentially the same
> displacement as Aug 6. But note how compressed the field is: because the
> axis sweeps tens of degrees within any long pass, *every* high-elevation
> candidate covers substantial new axis territory — the practical
> discriminators are elevation, span, and session length, much more than the
> peak-hour displacement.

> [!IMPORTANT]
> These displacements are measured against *reconstructed* session spans
> (approximate start/end times sampled at 15 min). Before committing
> telescope time, re-run on the data machine, where the tool automatically
> uses the per-look epochs from `registration_runs_chan*.csv`
> (see `DATA_MACHINE_TODO.md`).

---

## 3. Doppler Equator Shift Visualization

The tool writes `results/ERRORS/libration_opportunities.png`: per-session
*envelopes* of the Doppler equators swept by the existing looks (shaded), with
the optimal target's equator overlaid. The envelopes make the coverage point
visually obvious — each session's stripe swept a broad band of the disk, not a
single curve.

---

## 4. The Predictive Planning Tool (`predict_libration_opportunities.py`)

### Method
1. **Doppler axis & span:** closed-form `srp_velocity_analytic` (REPORT §2) —
   validated to 0.054° axis / 0.2% span against a lattice-free reference
   (`validation/scripts/validate_srp_velocity.py`). The previous revision
   finite-differenced the specular-zoom solver over 1 s; the solver's ~50 m
   output lattice made that direction estimate rely on error cancellation,
   with a measured 22.7° axis glitch at one epoch adjacent to the 09-10
   session looks.
2. **Existing coverage:** per-look axes from
   `results/LOLA_DEM_REGISTRATION/registration_runs_chan*.csv` when present;
   otherwise reconstructed session spans sampled at 15 min (approximate,
   flagged in the output).
3. **Gating:** Moon above `--min-tx-elevation` (default 45°) at Dwingeloo and
   `--min-rx-elevation` (default 10°) at Stockert; span ≥ `--min-doppler-span`
   (default 5 Hz).
4. **Deep-future fallback:** beyond the high-precision Earth PCK
   (currently 2026-09-11) the tool switches to the analytic IAU_EARTH frame
   but **keeps the topocentric station offsets**. (The previous revision
   dropped to a geocentric model there, which removes the diurnal-parallax
   term from the apparent rotation — a term comparable to the whole signal,
   not a "<1°" refinement. The IAU_EARTH fallback's real error is an
   Earth-rotation-phase error, arcminute-scale over a year or two.)
5. **Pass grouping:** contiguous visible steps (>4 h gap ⇒ new pass); per
   pass the tool reports the displacement peak, the pass-wide displacement
   minimum, and the axis swing across the pass.

### Example commands
```bash
# default search (2026-06-18 .. 2026-09-10)
.conda/bin/python predict_libration_opportunities.py

# search deep into 2027 (IAU_EARTH fallback with stations kept topocentric)
.conda/bin/python predict_libration_opportunities.py --start-date 2026-09-01 --end-date 2027-09-01
```

---

## 5. Why the numbers changed from the previous revision

The previous revision of this document reported a best displacement of
**51.2°** (2026-08-08) against four per-session *midpoint* axes. Two problems:

1. **A midpoint axis is not a session.** The axis rotates 58–82° within the
   2025 sessions (diurnal parallax; worst where the drift speed is small).
   The binding constraint for the Aug-8 candidate — the 2025-09-11 session —
   rotates ~70° within its own morning, so its "midpoint axis" happened to
   sit ~51° from the candidate while other epochs of the *same session* sit
   within ~5°. The 51.2° was an artifact of the snapshot choice, not new
   geometry.
2. **The axis estimator itself** was the quantized finite difference
   (usually fine — median error 0.007° — but with a measured 22.7° glitch
   mode, §4.1).

With both fixed, no candidate in the window exceeds ~21° peak displacement,
and pass-wide minimums are a few degrees. The physical conclusion stands —
new sessions do move the degeneracy stripe onto previously masked ground —
but the margin between candidate windows is far smaller than previously
implied, which *raises* the relative weight of elevation, span, and session
duration in the choice.

---

## Appendix: Metrics for Existing Sessions

Computed with the validated analytic model at one anchor epoch per session
(the axis and span vary through each session — see the swing column):

| Session | Anchor (UTC) | TX elev | RX elev | SRP speed | Doppler span | Axis swing over session |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2025-06-21 | 08:59:29 | 44.7° | 46.0° | 1.04 m/s | 18.1 Hz | (43 min) small |
| 2025-09-10 | 20:35:24 | 14.5° | 14.5° | 0.95 m/s | 16.4 Hz | 58° (evening span) |
| 2025-09-11 | 08:05:44 | 29.6° | 29.8° | 0.29 m/s | 5.1 Hz | 82° (morning span) |
| 2025-09-16 | 13:23:26 | 15.5° | 14.4° | 0.90 m/s | 15.6 Hz | 66° |

The 2025-09-11 row is the regime the REPORT's §3.2 fine-bin caveat describes:
smallest span (1.8 mHz Doppler bins) *and* the fastest axis rotation — slow
apparent rotation degrades both the Doppler resolution and the axis
conditioning together.
