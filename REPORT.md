# Moon radar: results and remaining limits

Reviewed 2026-09-14 against the current code, saved July validation results,
and August waveform-design work. **Revised 2026-10-21:** a code review found
and fixed several faults that changed results -- the timing search locking
onto a later echo peak (§3.1), delays measured from the smooth sphere
instead of the terrain at the specular point (§2), and stacking statistics
that went wrong once the noise floor was removed (§5, §6) -- along with
places where failures were hidden or written out as normal results. Every
product made before then must be regenerated; the July figures quoted below
predate these fixes and are marked where they depend on them. The latest documented full processing run
is July 4; this review does not constitute a new data run. Raw captures,
SPICE kernels, terrain data, and generated maps are not included in this checkout.

The pipeline turns lunar radar echoes transmitted at Dwingeloo and received
at Stockert into reflectivity maps. It corrects timing and frequency offsets,
accounts for lunar terrain, and combines observations from different days.

The July 2025 run (before the 2026-10-21 fixes) contained **222 processed
maps: 111 per polarization channel**.
Quality cuts retain **107 per channel, 214 total**. The maps align internally
to better than one 4.5 km map pixel, but their absolute placement on the Moon
has not been independently established. Adding similar observations now gives
diminishing returns: repeatable surface features and processing artifacts
dominate the remaining random variation (a July estimate whose fit is
superseded until rerun, §5).

For the processing sequence, read the [walkthrough](WALKTHROUGH.md). The
[README](README.md) covers commands and files. Section numbers below retain
the references used in code and investigation notes.

## 1. Terms and resolution

| Term | Meaning |
|---|---|
| Look | One capture processed into a delay–Doppler image and a lunar map. |
| Delay–Doppler (DD) image | Echo strength sorted by travel time and frequency shift. Delay measures path length; Doppler measures its rate of change. |
| Specular reference point (SRP) | The point on a smooth lunar ellipsoid with the shortest transmitter–Moon–receiver path. It anchors the geometry and compensation. SPICE's Moon ellipsoid (`pck00011.tpc`) has all three radii 1737.4 km, so it is a sphere: the one the LOLA heights are measured from. |
| `dlt` | The derivative of that two-leg light time: `Doppler = −frequency × dlt`. |
| Rim | Either extreme-Doppler edge of the lunar echo. “Up” is approaching (`dlt` minimum); “down” is receding (`dlt` maximum). |
| Rim correction, δ | The frequency correction still needed after the initial echo-tone measurement. |
| Ambiguous stripe | Surface region where delay and Doppler cannot distinguish nearby positions. The pipeline masks the most ambiguous pixels. |
| Opposite-sense and same-sense channels | The two circular polarizations, named relative to the transmitted wave as in the planetary radar literature (which abbreviates them OC and SC). A mirror-like reflection reverses a circular wave's sense of rotation, so the strong specular echo arrives in the opposite sense (`chan1`); rocks comparable to the 23 cm wavelength, double bounces and rocks buried in the regolith return some power in the same sense (`chan0`). Same-sense power over opposite-sense power is the circular polarization ratio. Until 2026-10-25 this project called them co-pol and cross-pol; antenna engineers use those words the other way round (co-pol: the transmitting port's own polarization, which picks up the same-sense echo), so they are no longer used. |
| Stack | An equal-weight mean of the looks' brightness per unit area (noise floor removed), after masking, gain normalization, and alignment. |

The Stockert observations use a 1299.5 MHz carrier, 250,000 recorded samples/s,
and a Zadoff–Chu (ZC) waveform at 50,000 chips/s (98 of the 110 2025 looks;
the other 12, and most 2026-10-11 looks, use 25,000). A chip is one element of the
transmitted code. Sampling density and physical resolution are different:

| Quantity | Scale |
|---|---|
| Delay sample spacing | 4 µs, equivalent to 600 m in one-way range |
| Delay resolution from the chip rate | About 20 µs (3 km in one-way range) at 50,000 chips/s; 40 µs (6 km) at 25,000 |
| DD Doppler row spacing | 1.4–6.3 mHz: 3000 rows across the recorded 4.3–18.8 Hz lunar span |
| Doppler resolution from integration time | About `1/T`, or 15–33 mHz for 30–66 s windows |
| Map pixel spacing | 0.147°, about 4.5 km (HEALPix `nside=400`); since 2026-10-20 each pixel averages 4×4 points of the DD image |

Range resolution does not translate into uniform surface resolution: the
mapping stretches it differently across the disk and becomes singular at
the ambiguous stripe.

## 2. Geometry and signal model

[doppler_equator.py](doppler_equator.py) calculates the two-leg light time
using SPICE. The map projection uses station positions evaluated at the
appropriate light-travel epochs, allowing the whole lunar surface to be
calculated in NumPy. In the [saved physics validation](validation/results/doppler_dem_physics.json),
this approximation agrees with per-point SPICE calculations to about 12 ns
in delay and 1.3 mHz at worst in window-averaged Doppler.

The SRP solver searches a smooth ellipsoid. Its velocity and the Doppler-axis
direction come from `srp_velocity_analytic`, avoiding noise from differencing
the solver's discretized positions. The
[41-epoch validation](validation/results/srp_velocity.json) found maximum
axis and speed differences of 0.054° and 0.19% against a refined numerical
reference. A separate July scan found no recorded look affected by the old
axis estimator's occasional large errors.

The [imaging code](doppler_equator_alignment.py) resamples the transmitted
waveform onto the receiving timeline by interpolating its unwrapped phase.
This preserves unit amplitude. It removes the SRP Doppler and its linear
drift before correlation. The surface projection must use the same drift:

```text
dlt_eff = mean(dlt over the window) − dlt_rate_srp × T/2
```

Since 2026-10-24 the correlation (`correlate_blocks`) cuts the capture into
4.7 ms blocks (1179 samples). It correlates each block over the whole delay
window with one 4096-point FFT, then sums the blocks with each row's Doppler
phase (a chirp-z transform) at the same 3000 row frequencies as before. The
phase change within a block is kept as a five-term power series. Plain
block sums would be wrong for these codes. At a wrong delay, the received
signal times the conjugate code is a tone, 45 Hz per chip of delay error for
the 25 kchip/s code, and block sums fold tones near multiples of the block
rate into the image: on a 2026-10-11 look, 15 % of cells changed by more
than 0.1 in log amplitude. The per-row full-length correlation used before
stays available as `method="rows"`. Against it, on two 2026-10-11 Stockert
opposite-sense looks ([investigations/lagsum_dd_image.py](investigations/lagsum_dd_image.py)):

- the rim correction, the noise floor and the image agree to float32
  rounding;
- the edge fit is identical on the 35 s look and 3.5 ns (1 % of its error)
  apart on the 125 s look;
- mapped with the same offsets, the maps differ by at most 0.0043 in log
  amplitude.

Speed, measured 2026-10-24 on the RTX 3050 laptop GPU:

- **Correlation, 125 s look:** the per-row correlation took 165–174 s on
  mains power. The block-sum correlation took 3.9–4.7 s on battery power,
  probably throttled, using at most 0.55 GB of GPU memory. A first version
  that kept the whole capture on the GPU (2.0 GB) took 1.4–1.5 s on mains.
- **Correlation, 35 s look (mains):** 0.67 s instead of 45 s.
- **Whole 125 s look:** 77 s on battery, against about 230–270 s with the
  per-row correlation on mains.

These should be re-measured on mains power. Products made before
2026-10-24 come from the per-row correlation.

Light-time derivatives use wide time intervals because short differences
amplify numerical quantization. Timestamp arithmetic is relative to the
capture start to preserve sub-sample precision.

Transmission is modeled as starting at `rx_start + 1.0 s` (one constant,
`TX_EMIT_DELAY_S`, used by every stage), plus the measured timing
correction. The TX file timestamp records waveform generation, not emission.
The timing search and the image use one shared model of the received
transmission (`compensated_tx`). The image's delay window starts 20 samples
before the predicted SRP echo so the leading edge remains inside it; its
first six columns give the noise floor, which is only valid for codes with
at most 14 samples per chip (the batch refuses slower codes).

The projection uses elevations from the Lunar Orbiter Laser Altimeter (LOLA)
digital elevation model (DEM). Both delay and Doppler are evaluated at these
terrain heights. The SRP and rim curves retain the smooth ellipsoid as a
stable reference. Without a local DEM, processing stops; the runs CSV
records the DEM file and a fingerprint of the SPICE kernels, and the
stacker refuses looks that differ in either.

The timing offset is measured on the specular echo, which comes from the
terrain at the SRP, so the image's lag 0 is that terrain's light time: the
smooth-sphere SRP light time minus 2h/c, with h the LOLA height at the
SRP. Since 2026-10-21 the projection and the rim curves measure delays from
there. Before, they measured from the smooth sphere, so every look was
displaced by 2h/c in delay: −3.2 to +2.3 µs on the 2025 looks (session
means −2.9, −0.1, +1.7 and +0.2 µs on 06-21, 09-10, 09-11 and 09-16), up to
about 1 sample, or roughly 1–2 km on the ground away from the SRP; the
per-session alignment shifts absorbed only its rigid part. The correction
uses one DEM sample at the SRP, while the specular tone comes from a region
tens of kilometres across, so local relief leaves an uncertainty of up to a
few microseconds.

## 3. Per-look calibration

### 3.1 Timing

Multiplying the received signal by the conjugate of the compensated TX
waveform concentrates the specular echo into a narrow tone. The batch tries
delay shifts over ±40 samples and takes the first peak of tone strength
versus shift, refined between samples: the echo's leading edge is where the
profile first reaches the tone strength the stacker requires of a look
(15), independent of any later echo; the peak is the first point after it
that is the highest within half a chip either side (a narrower bump is
noise), and it must lie within two chips of the edge (one chip for the
code's correlation to rise, about one more for the specular echo's own
spread in delay). The specular point is the nearest surface, so nothing
arrives before it. Each look records `timing_status`; only `resolved`
timings are stacked (others: no peak within two chips, an edge already at
the start of the search, or a tone that never reaches 15).

Until 2026-10-21 the batch took the strongest peak instead. Echo from
farther along the Doppler equator also lands near 0 Hz (every delay ring
crosses it), and on sessions with a narrow Doppler spread (2025-09-11: about
5 Hz across the disk, inside the ±2 Hz search band) a later peak can
outgrow a faded specular one: on 06:37:24 the specular peak sat at shift 7
and a stronger one at 35, so the echo's leading edge landed 28 samples
before lag 0, the map was misplaced in delay, and the echo filled the
delay columns that give the noise floor. Eleven 2025-09-11 looks had a
recorded noise floor above 82–99 % of their own map; once stacks removed
the noise floor (2026-10-21) they entered upside down. The three
September 11 "outliers" at +80, +91 and +125 µs that the wider ±40 search
recovered in July were most likely the same false peaks. Each look now also
records the strongest peak (`shift_strongest`). A first attempt at the fix
(2026-10-21) measured the edge at half the strongest peak, which a specular
peak fainter than half a later one would still miss; the edge is therefore
set by the fixed threshold above.

Offsets were reported as usually +35–45 µs; the 2025 rerun will restate
them. The search's lag 0 still sits a few microseconds from where the echo
begins (2026-10-11: median +6.5 µs, spread 6.9 µs); the edge fit measures
and removes that per look (§3.3). A strong-echo measure/apply/re-measure check closes to 0.03 samples
and 12.6 mHz (re-measured after the 2026-10-21 changes by
`test/test_pipeline_consistency.py`: 0.032 samples, 12.6 mHz); weaker
echoes support roughly one-sample precision.

The [operator discussion](email_timing_discussion.txt) identifies about
28 µs of transmitter/receiver processing delay at 250 kHz, a one-sample PPS
timing ambiguity, and a 735 ns station cable-delay difference. It also
identifies different SDR master-clock settings in June. Terrain height and
clock drift since the metadata's `vrt:cal_time` may contribute to the
remaining variation; the observed terrain/timing correlation does not yet
separate these causes.

### 3.2 Frequency and rim correction

The tone centroid gives an initial frequency correction, but echo fading
can move that centroid. The rim estimator refines it by averaging the offsets
of the two predicted echo edges. Their half-difference, recorded as
`rim_spread_hz`, is a diagnostic for unequal edge errors or a smeared echo.

The July recalibration fixed a scan window that missed large corrections,
especially on images with closely spaced Doppler rows. Production now uses
a 55 mHz window anchor, up to 12 refinement passes, and a coarse search when
the initial edge search fails. Each look records how the calibration ended
(`rim_status`): `converged` when the rims pass their gates at the applied
offset and leave under half a Doppler row; `not_converged` when they do not;
`failed` when no pass got through the gates, in which case nothing is
applied (before 2026-10-21 a coarse-search offset was applied even when the
rims failed there, and the quality columns then fell back to first-pass
values); same-sense looks inherit their twin's (`inherited`). The coarse
search's offset and its margin over the next-best trial are recorded too.
Only looks whose opposite-sense calibration converged are stacked. It recovered a badly faded September 16 look
requiring a −286.8 mHz rim correction. The
[investigation log](investigations/rim_window_recalibration_2026-07-03.md)
records the comparisons and individual recoveries.

The July 3 rerun, before that final recovery, found a median absolute rim
correction of 44 mHz and standard deviation of 75 mHz. These describe the
initial centroid error, not final calibration uncertainty or clock drift.
Reported iteration residuals were 0.08 mHz median and 0.61 mHz at the 95th
percentile; this measures convergence, not absolute accuracy.

[Synthetic stress tests](validation/results/rim_calibration_stress.json)
provide a separate bias check: about 0.5 mHz in the baseline case, up to
1.1 mHz across the symmetric variants, and up to 6.6 mHz with an extreme
50% brightness imbalance between the rims. A constant frequency error warps
the surface map differently at different positions, so it must be corrected
before projection; a later map translation cannot repair it. That run did
not check whether each trial's calibration converged; the stress test now
uses production's calibration function itself (`calibrate_rim`), reports
bias over converged trials only, and records how many trials converged,
did not converge or failed.

Stockert `chan1` is treated as the opposite-sense channel (it holds the
strong specular echo) and `chan0` as the same-sense channel (§1).
The weaker same-sense echo inherits timing, frequency, and rim corrections
from its opposite-sense twin when the batch uses `--corrections-from`.
Its quality selection also uses the twin's tone strength. This shares the
clock calibration; it is not an independent quality test of the same-sense capture.

### 3.3 Edge fit: delay and Doppler from the echo edges

At each delay the echo's Doppler extent ends at the two limbs of the
Doppler equator. A delay error moves both edges the same way (inward when
the echo is later than lag 0 says), by the edge's slope in Doppler per unit
delay -- about 5.7 mHz/µs at 10° incidence and 0.3 mHz/µs at 75° on the
2026-10-11 looks (limb Doppler about 11.5 Hz) -- while a Doppler error moves
them in opposite directions. The rim calibration (§3.2) measures only the
second. A remaining delay error therefore leaves the projection reading
pixels along the Doppler equator beyond the echo's edge (a dark band, with
brighter shoulders where the real edge's pile-up lands) or inside it (a
bright one).

`fit_edge_offsets` fits each opposite-sense image near both edges, from 10° to 80°
incidence, as amplitude × model + noise floor, the model shifted in delay
and Doppler. The model (`edge_model_image`) uses only external inputs: the
projection's SPICE light-time field (with each point's Doppler drift over
the look), the LOLA terrain, a generic scattering law (cos^1.5 of
incidence), uniform reflectivity, the code's delay response and the
integration's Doppler response with its sidelobes. The amplitude is free
per limb and per chip of delay (reflectivity along the equator) and the
noise level per limb; the fit maximizes the single-look speckle likelihood
over ±1 chip and ±3 Doppler resolution cells around the rim-calibrated
axes, and errors are delete-one jackknife estimates over ten groups of delay
columns. The projection applies the offsets when the fit is inside its
search (`edge_status` `fitted`); same-sense looks take their twin's; only
fitted looks are stacked. The saved DD images keep the axes before the fit,
so `reproject_maps.py` refits from them.

Checks: on synthetic looks made from the model with uneven brightness along
and across the equator, the noise floor and correlated speckle, offsets
injected up to 25 µs and 67 mHz were recovered within 0.92 µs and 0.49 mHz,
within two of the fit's own jackknife errors
([validate_edge_offsets.py](validation/scripts/validate_edge_offsets.py)).
That test cannot see errors of the model itself; on real looks, swapping
the scattering law (Hagfors plus diffuse, or diffuse only) moved the edge
positions by at most a few mHz. The transmitted waveform's autocorrelation
is the triangle the model assumes (the TX file's spectrum is the square-chip
sinc², its null at the chip rate 44 dB down), and so is the transmission as
received by Dwingeloo (its leakage correlation is within 0.02 of the
triangle, 8.1 samples after the nominal start; a constant delay the timing
search absorbs).

On the 20 opposite-sense looks of 2026-10-11, 18 fitted; the two 09:49:09 looks,
the weakest, did not (best fit beyond the ±1 chip search). The fitted echo
was later than lag 0 by a median +6.5 µs (spread 6.9 µs, −13.8 to +18.6 µs;
jackknife errors 0.35–1.11 µs) and higher in Doppler than the rim
calibration put it by +2.0 to +8.9 mHz (median +6.2; errors 0.31–1.17 mHz).
A fit of the echo's leading edge (all Doppler rows summed)
agrees within 1–3 µs per look when its scattering law puts the echo's peak
where the real one is (Hagfors, 5.7° rms slope, plus diffuse); with a purely
diffuse law, which puts the peak 4 samples late, it gives 3.2 instead of
8.6 µs on Stockert 09:41:25, so the edge positions, which do not depend on
the law, are the measurement. The rim calibration's half-power
rule gives a second reading of the same edges. This was measured with a copy
of the rule run at every delay (production uses only delays beyond 1.5 ms),
taking each column's median crossing over 8 speckled copies of each model,
correlated as the image's speckle. The copy has not been checked against
production's rule. On five looks (Stockert 09:41:25 and 09:49:57, WSRT
09:24:58 and 12:09:19, Stockert 2025-09-16 13:26:14), the real image's
crossings were averaged over both limbs at 10–80° incidence. They lie within
0.1–4.2 mHz of the crossings on the model shifted by the edge fit's offsets.
They lie 8.6–20.4 mHz from those on the model at the timing search's lag 0,
on the four looks whose fitted delay is 9.9 µs or more.
The Doppler part is the rim calibration's own error; what causes it is not
established. `calibrate_rim`, unmodified, was run on model images at lag 0
with no Doppler offset, on Stockert 09:41:25 and WSRT 09:46:36. The delay
axes were in `process_file`'s convention, which reproduces production's
crossing counts and residuals exactly. The model band was wide enough for
the rule's reference strips. Without speckle, it read −2.1 and −1.7 mHz
without terrain and −7.1 and −6.9 mHz with it. With speckle (16
realizations), the readings with terrain were −1.0 and −1.6 mHz (spread
1.5–2.1 mHz). The readings without terrain scattered by 25–33 mHz, so how
the error splits between terrain and the rule itself is not settled. The
edge fit measures the correction directly against the terrain model,
whatever its cause. In the
re-projected 2026 opposite-sense stack (both stations, the same 18 looks), the
brightness within 60 km of the looks' Doppler equators rose by factors of
1.07–1.44 (except at the far end above 50° incidence), at 100–300 km it fell
2–9 %, and the detail repeated between two interleaved half-stacks rose from
0.390 to 0.458 at 20–12 km and from 0.206 to 0.268 at 12–10 km.

Why the timing search leaves lag 0 off is only partly established. A
simulation of the search itself (its tone strength, peak over median within
±2 Hz, and `leading_peak`) on the model with each look's noise level and
speckle puts lag 0 before the echo begins by a median +3.2 µs, predicts the
two negative offsets (09:49:57, the weakest fitted looks) and correlates
+0.48 with the measured offsets per look; it leaves +3.8 µs of the median
and a spread of 6.1 µs. The tone-strength curves themselves (tone strength
against trial shift, real and simulated, on 2026-10-11 and on the 2025
sample) show where the search goes wrong. The real curves come from the
same search, rerun on the captures (it reproduces production's shifts
exactly). Placed at the echo start the edge fit found, a real curve's rising
and falling flanks lie on the simulated ones. Matching the whole curve to
the simulated median puts the echo start +1.4 µs from the edge fit's on 2026
(18 looks, spread 2.2 µs) and −1.8 µs on the 2025 sample (9 of 10 looks,
spread 1.4 µs; 2025-06-21 11:27:05 is off by +24 µs). So the tone and the
edges agree on where the echo starts to about half a sample. But the real
peak is lower, a median 0.48 of the simulated (0.32–0.80) on 2026 and 0.59
on the 2025 sample. It is also flat over several samples, where the
simulated peak is pointed. The search takes the first local maximum on that
plateau, so speckle decides where lag 0 lands. The surface roughness the
simulation assumes does not explain the lower peak (Hagfors C 30 instead of
100: 0.52 on 2026). On many 2026 looks the curves also rise to 1.5–2 times
the noise level up to a chip and a half before the echo starts, which the
model does not produce.

## 4. Clock behavior

The clock comparison is the **sum** of `applied_df_hz` and `rim_delta_hz`
(and, since 2026-10-22, `edge_doppler_hz`, §3.3).
Looking at the rim correction alone confuses echo fading with clock drift.
The July analysis reported:

| Measurement | Result |
|---|---|
| Change between clean looks, over 30 s to 3.3 h | 5–7 mHz RMS, with little dependence on elapsed time |
| Median total offsets: June 21 / September 10 / 11 / 16 | +5 / +54 / −1 / −20 mHz |
| Half-to-half drift within clean looks | 1–8 mHz |
| Drift within three abnormal June 21 looks | About 25 mHz over 66 s |

The three abnormal captures start at 10:09:25, 10:14:34, and 10:17:40 UTC.
Their smeared rims cannot be corrected by one constant offset, so both
polarizations are excluded from stacks. Sub-window frequency correction is
a possible recovery method, not part of production processing.

Dwingeloo's directly received TX leakage was stable to 0.15–0.23 mHz on
these captures, supporting a receiver-side explanation. It does not test a
clock error shared by Dwingeloo's transmitter and receiver. The operator
notes also say that both sites used rubidium references for at least one
dataset. Attribution to Stockert's reference or local oscillator therefore
remains provisional until checked against station logs.

## 5. Registration and stacking

The stacker masks pixels whose DD cells spread over more than three map
pixels' worth of surface (`MAX_MULTIPLICITY`; before 2026-10-21 this read
"three times the median", which with the overlap count floored at 1 was the
same fixed cut). It rejects, in order: transmissions shorter than
`--min-tx-s`; weak echoes (opposite-sense tone SNR below 15, the twin's for
same-sense looks); timing shifts at the search limit (within half a sample of the
look's own `max_shift`, the opposite-sense run's for same-sense looks); an opposite-sense timing
peak that is not resolved (§3.1); a specular tone on the edge of the ±2 Hz
frequency search; opposite-sense rim calibrations that did not converge; opposite-sense
edge fits that did not fit (§3.3); and
abnormal rim spread relative to the session (which rejects the three June
drift captures). A look whose mean brightness at 20–50° incidence is not
three standard errors above zero (from the scatter of ~110 km blocks) -- no
clear signal above its noise floor -- is rejected when it is loaded. Every
decision is written to `stacked_looks_<label>.csv`, with the captures the
batch did not map. The stacker refuses run directories it cannot gate
consistently: old runs CSVs, blank fields, looks made with different
terrain files or kernels, and same-sense runs whose corrections no longer
match their opposite-sense twin's.
Here “tone SNR” is the estimator's peak/median amplitude ratio, not a
calibrated receiver sensitivity measurement.

Each look's brightness per unit area is scaled to a mean of 1 at 20–50°
incidence and looks are averaged with equal weights. An empirical
brightness-versus-incidence curve (the mean in 1° incidence bins, summed
over every pixel of every look) is fitted separately for each channel to
make scattering-normalized products; a bin is used only if its mean is three
standard errors above zero (from the scatter of the looks' own bin means),
and pixels in other bins have no normalized value. Until 2026-10-25 the bins
were 40 equal steps of cos(incidence), so the first spanned 0–12.7°, over
which the curve falls about fivefold (82 at 0.7°, 45 at 5.5°, 16.7 at 10.5°,
2025 + 2026 opposite-sense). Dividing by one value there left every session too
bright within about 6° of its own specular point and too dark in a ring
beyond: within 50 km of the 2025 sessions' mean specular point they came
out 1.5–2.3 times the all-season mean against 2026's 0.86, and 2026 came
out 5.2 times at its own. With 1° bins the 2025 sessions give 0.8–1.1 there
and 2026 1.3; 2026 is still 1.9 times at its own specular point and about
1.2 times elsewhere, so the two seasons' curves may differ in shape.

A second product, `..._geomweighted` (`combine_sessions.py`), stacks the
same looks in two levels. Looks from one night share one geometry and
repeat the same blur and mirror ghosts, which averaging more of them does
not reduce; equal weight per look lets the 99 2025 looks outvote 2026's
18 even where 2026 sees a place far better. Each session is averaged on its
own; the sessions are then weighted separately at each place and in each
feature-size band (below 10, 10–20, 20–40, 40–75, 75–150 and above 150 km)
by their expected errors: geometry error from simulated twins of every look
(`simulate_geometry.py`: a synthetic surface through the look's own
delay–Doppler geometry and back; reading a real image back this way
reproduces the production maps to 0.0004 rms), scaled to the real surface
by the ratio of real to simulated cross-session detail, plus noise measured
from interleaved halves of each session. The weights are zero or positive
(left free, the best combination subtracts the 48-look 2025-09-10/11
session to cancel blur the 2025 sessions share, which leans entirely on the
simulation being exact). Validated 2026-10-25 with the 1° law (weights
from simulations on one synthetic surface, scored on a second;
`validation/results/session_combination_<label>.json`). All seasons,
opposite-sense: the weighted stack keeps 16–20 % less error than equal
weights at features below 40 km over the disk (0.84, 0.81 and 0.84 of
equal's in the <10, 10–20 and 20–40 km bands) and 9–20 % less within 250 km
of the 2025 specular point (0.91, 0.84, 0.80). On the real data, detail
repeated between independent half-stacks there rises from 0.10 to 0.36 at
40–20 km and from 0.45 to 0.76 at 150–75 km. Over the whole disk it falls
(0.81 to 0.71 at 40–20 km, 0.66 to 0.51 at 20–12 km); that measure cannot
tell surface detail from blur and mirror ghosts repeated within a session,
so it does not decide between the two there. The same-sense and combined
stacks gain most: the equal-weight same-sense stack is dominated by noise
(repeated detail 0.14 at 40–20 km over the disk), and the weighted one,
leaning on 2026's 125 s looks, reaches 0.68 (synthetic error 0.54–0.61 of
equal's below 40 km). 2025 alone gains less (synthetic error 0.87–0.94 of
equal's below 40 km): its sessions' specular points are close together, so
none sees the others' spot well. Far from both seasons' specular points the
weighted stack leans almost entirely on 2026 at the finest scales, so it is
sharper there and, in places, noisier than the 117-look mean. Session offsets are
solved from cross-correlations emphasizing features on 0.3°–2.5° scales,
on session stacks averaged over 0.3° in linear brightness before the log
is taken (a pixel whose own mean is not positive still counts towards its
neighbourhood; logging first left most of a few-look stack as holes).
September 10 and 11 are grouped into one registration session. A peak on
the edge of the ±1.5° search is not a measurement and stops the run, as
does a sign check that clearly prefers the opposite sign (a convention
regression). How well the shifts close is recorded in
`stacked_registration_<label>.json`, not used to stop: for three or more
sessions, the probability that pure noise would close as well (the squared
closure ratio follows a Beta distribution under noise). With three sessions
this test has little power -- pure noise closes better than a ratio of 0.16
only 5 % of the time -- and offsets no more consistent than noise may still
be real but small. Two sessions cannot be checked by closure. Before
2026-10-21 the closure test could not fail, and the printed lock
significance excluded only 0.5° around the peak, inside the correlation's
own main lobe, so every pair scored about 1.15–1.18; it is now measured
against the strongest correlation beyond the main lobe (where the
autocorrelation first crosses zero). Stacks are written to a temporary
folder and moved into place only if the run completes.

The July production run (before the 2026-10-21 fixes) reported residual
disagreement after alignment of **0.005° for chan1, 0.027° for chan0, and
0.006° for the combined map**: all below 1 km on the Moon. This is internal consistency. Repeated
random splits of the July data give an estimator scatter around 0.02–0.03°;
the [closure bootstrap](validation/results/registration_closure_bootstrap_chan1.json)
has median 0.025° and 95th percentile 0.040°. That bootstrap predates the
final exclusions and is not a new measurement of the 107-look stacks; it
also stacked log maps with only the signal-strength and search-limit cuts,
not the production brightness or gates, which the script now uses.

An [absolute-position comparison with LOLA terrain](validation/results/absolute_registration.json)
did not meet the test's correlation-significance threshold (1.03–1.15,
under the old significance measure, which excluded only 0.5° inside the
correlation's main lobe; the new measure can come out higher, so "no lock"
needs the rerun). At
the finer scales the stack sat about 3.4 km **north** of the LOLA frame
(2.2 km at 0.7°, −0.4 km at 1.0°), a weak indication. Earlier text said
"southward": the saved numbers are the correction that moves the stack onto
LOLA, the opposite of its offset, and the script now reports the offset.
Neither this test nor the small internal residual establishes an absolute
position-error bound.

The saved variance-versus-look-count fits used **110 looks per channel**,
before the final exclusions. They estimated residual speckle at
[8% of filtered variance in chan1](validation/results/speckle_floor_chan1.json)
and [4% in chan0](validation/results/speckle_floor_chan0.json); the rest is
repeatable structure, which includes both lunar features and stable
artifacts. Both figures are superseded until rerun: the fit was unweighted,
the minimum looks per pixel changed along the look-count ladder, the
variance was taken over the whole grid including empty cells (so it mixed in
coverage area), and log stacks dropped pixels with non-positive means.
Refitting the saved ladders without the two-look point already moves chan0
from 4 % to 1.3 % (chan1 from 8.1 % to 9.0 %), so the chan0 figure was never
robust. The script now uses one coverage rule, weighted fits with residuals
shown, a fixed pixel set and production brightness.

Registration signs are fixed by [synthetic tests](test/test_registration_conventions.py):
positive longitude is east and positive latitude north; `grid_map` preserves
those coordinates; `shift_intensity` moves content in the stated direction;
`xcorr_offset(a, b)` returns the correction that moves `b` onto `a`. The
expected sign when applying solved session offsets is `+1`.

## 6. Remaining limits

| Effect | Evidence and present treatment |
|---|---|
| Terrain distortion | Corrected with the DEM (required since 2026-10-21). Saved ellipsoid/DEM comparisons reach 7.4 delay samples; terrain also changes Doppler by up to 22 mHz. Delays are measured from the terrain's light time at the SRP since 2026-10-21 (§2); before, each look was off by 2h/c there (up to about 1 sample). |
| Timing uncertainty | Corrected per look; demonstrated 0.03-sample closure on strong echoes, roughly one-sample precision on weak ones. Until 2026-10-21 the search could lock onto a later, stronger peak on narrow-Doppler sessions (§3.1). |
| Rim-estimator bias | Synthetic tests give about 0.5–1 mHz for symmetric cases and 6–7 mHz under extreme brightness asymmetry. |
| Frequency drift within a look | Clean looks show 1–8 mHz half-to-half change; three ~25 mHz cases are excluded. No time-varying correction yet. |
| Geometry approximation | Saved full-surface comparison: about 12 ns delay and at most 1.3 mHz Doppler error. |
| DD-cell sampling | Until 2026-10-20 each map pixel took only the DD cell under its centre, ignoring the other cells inside it. Pixels now average 4×4 points. On the 2026-10-11 long looks this raised signal-to-noise for 12–20 km features 3.4× (opposite sense) and 3.1× (same sense), measured as the correlation between independent half-stacks. Maps from earlier runs, including the July 2025 production maps, use one point. Constant-delay rings have not been rechecked; leading-edge delay calibration remains a candidate. Measured on products made before the 2026-10-21 fixes, with the scripts in [investigations/2026-10-21_scratch](investigations/2026-10-21_scratch/README.md); to be re-measured. |
| Single-look ambiguity | Different surface locations can share a delay and Doppler. Masking and geometry diversity help; stacking does not invert this overlap. |
| Map alignment | Small internal residuals; absolute placement remains unverified. |
| Doppler drift within a look | The DD image removes the specular point's Doppler only. Relative to it, each point's Doppler drifts by 28 mHz (median) and up to 57 mHz over a 125 s look -- 3.5-7 times the 8 mHz resolution -- so long looks resolve Doppler about as coarsely as 35 s ones (8-17 mHz drift against 29 mHz). [A prototype](investigations/focused_dd_prototype.py) builds the image from block correlations and focuses each point at its own drift rate; the figures below come from its first version, with plain 10 ms block sums. Without the correction it reproduces today's maps (detail correlation 0.95-0.97) in ~50 s per 125 s look. At nside 400 (eight long looks per channel, equal weights) repeatable detail between independent half-stacks rises 4-14 % (opposite sense) and 6-22 % (same sense) at 9-40 km on linear contrast, mostly along the Doppler direction (1.2x opposite sense and 1.4x same sense at 9-12 km); without the correction the block method matches production, and with the correction's sign flipped detail falls by about as much (0.8x and 0.67x), so the correction does what it should. The gain is a few per cent of the fine-scale texture, most of which is still speckle with eight looks, so it is hard to see: by eye production and focused look alike. Plain block sums fold the Zadoff-Chu code's wrong-delay tones into the image (§2): the strong opposite-sense echo leaked into the delay columns ahead of the echo, so that version read the opposite-sense noise floor about 6 times too high in power (the maps themselves agreed with production within 1 %); the comparison used production's floor. Since 2026-10-24 the prototype uses production's block-sum correlation, which keeps the phase change within a block and so does not fold, its rows carry its own noise floor, and it applies the edge fit's delay and Doppler offsets as production does. An earlier nside 800 test with that wrong floor showed black blotches and is superseded. The two mirror points separate only slightly (0.77 -> 0.70). Not in production. Measured on products made before the 2026-10-21 fixes, with the scripts in [investigations/2026-10-21_scratch](investigations/2026-10-21_scratch/README.md); to be re-measured. |
| Cell area and noise floor | Until 2026-10-21 stacks averaged each look's pixel intensity as it came out of the DD image. That intensity includes the noise floor (on the 2026-10-11 maps about 40 % of the median opposite-sense pixel at 25 kchip/s, 50–63 % of the median same-sense pixel, and 59–90 % on the two 50 kchip/s looks) and grows with the ground area a DD cell covers, which rises as 1/distance from the Doppler equator: at fixed incidence the 2026-10-11 maps were 10 times brighter 100 km from the equator than 800 km from it, and pixels were implicitly weighted toward the looks where they sat closest to it. Stacks now use brightness per unit area with the noise floor removed, averaged with equal weights. The area is counted per cell and spread over the image's power response: along delay the code correlation's triangle, one chip either side, squared (which caps the area within ~140 km of the equator); along Doppler the main lobe of sinc², first null at 1/(integration time) (before 2026-10-21: the amplitude triangle in delay, and a fixed ±3 rows in Doppler, chosen on the 2026-10-11 looks, which is ±5 to ±21 mHz depending on the session's row spacing). The pixel's area is the plain mean over its 4×4 points, like its intensity; it was a harmonic mean, which overstated brightness where the area changes across a pixel (up to +12 % for a 1:2 change, about +4 % for a smooth ramp). Each look is scaled to a mean of 1 at 20–50° incidence, and rejected if that mean is not three standard errors above zero; it was the median, which for single-look speckle sits well below the mean (about 0.65× at 50°) and, with the noise removed, goes negative where the echo is weak -- that is how the eleven mis-timed 2025-09-11 looks (§3.1) entered the stacks upside down. A stacked pixel whose mean brightness is not positive has no log value and is left empty in the log maps (before 2026-10-21 it was set to 1/1000 of the median, which showed black and fed the session alignment a made-up edge); the mean itself, negatives included, is saved as `stacked_linear_*`. On the 2026-10-11 stacks the per-area brightness removed the brightness ramp and the bright lines beside the stripe. Repeatable detail between independent half-stacks (linear contrast) was about 12 % lower than the old stack's at 12–40 km and the same at 9–12 km; that measure also counts anything that repeats between halves, including the old stack's lines beside the stripe and ghosts shared by similar geometry, so it is not a clean loss. These figures predate the 2026-10-21 fixes and will be restated after the rerun; the scripts are kept in [investigations/2026-10-21_scratch](investigations/2026-10-21_scratch/README.md). |
| Scattering-law normalization | The `_scatnorm` stacks divide each look by the channel's mean brightness at its incidence (1° bins since 2026-10-25; before, 40 bins equal in cos(incidence), which left a glow at each session's specular point, §5). Until 2026-10-21 this was the median of single-look samples, and once the noise floor was removed the median went negative beyond ~80° incidence (51–64 % of single-look pixels there are at or below zero; the mean stays positive). A clamp to 1e-12 then blew the limb up to values near 13 in log amplitude. The mean is now summed over every pixel (no subsampling, which also removed aliasing onto 25 meridians); a bin is used only if its mean is three standard errors above zero (from the scatter of the looks' own bin means), so a bin at noise level no longer divides noise by a near-zero mean; pixels in unused bins get no normalized value rather than a clamped one. Within a used bin the curve is interpolated only towards neighbouring used bins (a first version left the outer half of the end bins empty, a 9° hole at the specular point). Measured with `law_check.py` (investigations/2026-10-21_scratch). The 51–64 % figures were measured on the 2025 looks before the 2026-10-21 fixes. |
| Stack weights and mirror ghosts | A main purpose of stacking many looks is to cancel mirror ghosts: each look puts a copy of every point at its reflection across that look's Doppler equator, and the copies land elsewhere as the equator turns. Weights from how well each look agreed with the rest of its session (inverse variance, tried 2026-10-21) read ghost disagreement as noise. Simultaneous Stockert/WSRT looks share their ghosts; the disagreement of any other pair beyond their speckle grew with the turn between their Doppler equators (zero below 0.3°, about +0.35 beyond 4°) and followed the other looks' stack at the two mirror points. So the weights marked down the looks turned farthest from the session's majority (10:16 and 12:08/12:09, turned 3–10°) -- the ones that cancel the others' ghosts. On a synthetic surface with the session's geometry and masks, the weights cut speckle 16–22 % but left ghosts of 75–150 km features 8–10 % stronger. Stacks use equal weights since. Within one session the turn (about 2° by 09:50, 4° by 10:16, 11.5° by 12:08) separates ghosts of small features well and large ones poorly; other sessions' geometry cancels both. Re-measured 2026-10-22 on the reprocessed opposite-sense stacks with [validate_mirror_ghosts.py](validation/scripts/validate_mirror_ghosts.py) (fraction of one look's ghost power the stack keeps, synthetic surface, the stacks' own looks and masks): 2026-10-11 (19 looks) 0.34 at 75–150 km, 0.10 at 12–20 km, 0.08 at 10–12 km; 2025 (103 looks, three sessions) 0.10, 0.03, 0.03; all four sessions (122 looks) 0.08, 0.03, 0.02. The earlier figures (1/3 and 1/12) came from a measurement that smoothed the ghosts but not the surface, and reported bands finer than an nside-400 map resolves (below about 9 km). |
| Brightness calibration | Residual seams and polarization leakage limit quantitative interpretation of brightness and channel ratios. |

The [DEM projection check](validation/results/lola_dem_projection.json)
compared one look mapped with and without terrain in three regions; it has
no pass/fail criterion, and none of the three cross-correlations is a lock
by the old 1.5 significance rule (1.22–1.34). Mare Imbrium (flat; its
predicted shift comes from the mare's height relative to the sphere) moved
4.6 km east against 4.3 km predicted. Copernicus moved 1.2 km east against
2.0 km predicted. Tycho moved −0.3 km against +3.7 km predicted, with a
weak correlation (0.20). So the check supports the direction and rough size
of the terrain shift in one flat region and not, so far, in the two with
relief. Terrain corrections improve the maps without necessarily improving
session alignment, because much of the distortion is shared across
sessions.

## 7. Verification and reproducibility

Use the project `.conda` environment from the repository root. The
[validation guide](validation/README.md) lists dependencies, saved outputs,
and commands. The numerical-core test covers geometry, GPU correlation (the
block-sum method against the per-row one, on a Zadoff-Chu code), tone
recovery, and a real-data calibration closure. Separate tests cover terrain
projection, registration signs, and ZC parameter selection.

Saved physics-validation logs are dated July 3. Final production counts and
alignment figures above come from the documented July 4 processing work;
the untracked run directories are required to recalculate them. After the
2026-10-21 fixes the numerical-core test (including the real-data closure,
0.03 samples and 12.6 mHz), the terrain test and the registration-convention
tests pass; the core test's synthetic timing check now uses a chip-structured
waveform like the real code, which the first-strong-peak rule needs. The
data-dependent validation scripts were fixed but not yet rerun. The September
documentation review checked the ZC recommendation and its two unit tests
locally, but could not repeat the data-dependent pipeline without the data,
kernels, DEM, and GPU setup.

## 8. Completed work and next steps

The identifiers below are retained for older references.

| Item | Status or next action |
|---|---|
| 8.1 Error-model units and pixel scales | Fixed. Old plotting notes that claimed multi-Hz ephemeris uncertainty are obsolete. |
| 8.2 Rim estimator | Synthetic bias tests complete; July scan-window correction and coarse recovery are in production. |
| 8.3 Delay-search recovery | ±40 samples is the batch default; since 2026-10-21 the offset is the first strong peak, not the strongest (§3.1). Restate the July "outliers" after the rerun. |
| 8.4 Terrain projection | Implemented and validated with LOLA; timing-scatter attribution remains open. |
| 8.5 Frequency drift within a look | Fit and apply sub-window corrections to recover the three excluded captures. The targeted July investigation uses the current rim window; `intra_look_drift.py` takes its run directory as an argument but still uses the older rim window. |
| 8.6 Delay-axis refinement | Pixel averaging of the DD image and the terrain delay reference at the SRP are in production (§2, §6). Rerun every session with the 2026-10-21 fixes (the 2025 recordings are on data.camras.nl); recheck the ring artifacts; replace the single DEM sample at the SRP with the region the specular tone comes from. |
| 8.7 Surface inversion | Fit one surface model to the DD images from multiple geometries. Requires further calibration and validation work. |
| 8.8 Polarization-ratio map | Possible from registered channels; characterize leakage and low-signal bias first. |
| 8.9 Brightness matching | Improve normalization between sessions for quantitative maps. |
| 8.10 More data | ATA comparison is implemented; Dwingeloo self-receive and March experimental data still need dedicated batch handling (§10). |
| 8.11 Feed characterization | Quantify how much of the specular echo leaks into the same-sense channel before interpreting channel ratios. |
| 8.13 Operator cross-checks | Obtain per-session reference-frequency logs; use `vrt:cal_time` and SDR clock settings to reassess timing scatter. |
| 8.14 SPICE light-time convergence | The code still uses `LT`; switching to `CN`/`XCN` is a small numerical refinement requiring the core tests and a real-look closure. |
| 8.16 Lock label | `registration_analysis.LOCK_SIGNIFICANCE` (1.5) was chosen for the old significance measure and is not recalibrated for the main-lobe one; nothing in production uses it, but the absolute-registration headline and the ATA control scale select data by it. Calibrate it on map pairs known to be unrelated. |
| 8.15 Observation planning | Rerun the planner for new dates with the recorded-look CSVs and suitable Earth-orientation kernels; see [planning notes](LIBRATION_ANALYSIS.md). |

### 8.12 Waveform design

The August addition, [recommend_zc.py](recommend_zc.py), selects ZC lengths
and roots for a chosen duration, chip rate, Doppler span, and coupling limit.
Since 2026-10-21 a root must also pass a ghost check: a Doppler offset of j
whole frequency bins moves the full-strength correlation peak by j × q⁻¹
(mod N) chips, and for every j within the span that lag must fall beyond
the Moon's 11.6 ms delay depth. (Before, PASS held by construction.) The
production `N=1500007, q=1201` passes the ghost check (nearest ghost
25.0 ms); it does not meet the 0.05-cell ridge limit (0.47 cell, below).
Its ideal calculation gives **0.50 µs/Hz**, or **0.47 delay cell across 19 Hz**,
for the existing `N=1500007, q=1201` at 50 kchip/s. The older report's
0.03 µs/Hz figure should not be used for this production sequence.

Those are uncompensated design estimates. They do not establish how much
delay motion remains after production compensation and correlation. The
[signal-check runbook](SIGNAL_CHECK_PROCESS.md) defines the needed numerical,
RF-chain, and lunar A/B tests; completed results for that campaign are not
present here. The earlier claim that compensation removes coupling exactly
was stronger than the available validation supports.

The saved [ideal waveform comparison](validation/results/signal_processing.json)
found integrated periodic-autocorrelation sidelobes of exactly zero for ZC
(the computed −153 dB is floating-point round-off), −51 dB for an
m-sequence, and 0 dB for random BPSK. These are ideal
code metrics, not achieved lunar-image dynamic ranges. They support retaining
ZC as the reference waveform while testing its parameters through the actual
transmit/receive chain.

Increasing chip rate from 50 to 250 kchip/s could improve nominal range
resolution from 3 km to 600 m at the same recorded sample rate. Usable filter
bandwidth and map reconstruction must support it. For 30 s, a 19 Hz span,
and a 0.05-cell coupling limit, the tool selects `N=7500013, q=11401`
(nearest ghost 15.8 ms). This is a candidate for testing, not a validated
replacement waveform.

## 9. Code and product locations

The [README file index](README.md#file-index) identifies the active modules.
Run directories hold runs CSVs (`registration_runs_<station>_<chan>.csv`),
the list of captures not mapped (`registration_skipped_<station>_<chan>.csv`),
per-look maps, ambiguity counts and cell areas, and the stacks with their
decision lists (`stacked_looks_<label>.csv`). The planned directories are
`results/STOCKERT_2025/` (the 2025 Stockert sessions), `results/EVE2026_CIRC/`
(2026-10-11, Stockert and WSRT) and `results/ALL_SEASONS/` (stacks of both);
all three must be regenerated with the 2026-10-21 fixes. The July production
directory `results/LOLA_DEM_REGISTRATION/` is not in this checkout; files
written before the station was always named lack the `stockert_` part.

`results/LOLA_DEM_REGISTRATION_FROZEN_0612/` preserves the previous DEM run
and its `PRE_DEM_ANALYSIS/` comparison. `results/REGISTRATION/` is the
pre-DEM baseline. These generated directories are untracked; old investigation
scripts may still use their former names.

## 10. Data coverage

The July Stockert processing counts, before the 2026-10-21 fixes (the 2025
rerun will restate them; 11 of the 2025-09-11 looks then carried timing
shifts of 20–35 samples from the old strongest-peak rule), were:

| Observing date | Processed looks per channel | Retained per channel |
|---|---:|---:|
| 2025-06-21 | 34 | 30 |
| 2025-09-10 | 12 | 12 |
| 2025-09-11 | 44 | 44 |
| 2025-09-16 | 21 | 21 |
| Total | 111 | 107 |

The totals include the September 10 `interp_zadoff-chu` capture. Its missing
TX waveform was reconstructed from the published sequence parameters and
FFT upsampling identified in Dwingeloo's TX leakage. Both Stockert
polarizations were recovered. Reproducing that look requires the reconstructed
file under `tx_signals/`, which is not included in this checkout.

ATA recorded 11 dual-channel captures on September 16. Its independent
receiver comparison gave a broad correlation peak near a ≤0.3° offset
(correlation +0.29, versus +0.80 for a Stockert split-half control). That
comparison is superseded until rerun: it chose the best smoothing scale
separately for ATA and the control, used the old significance measure and
brightness, and compared look counts that did not match. It was weak
supporting evidence for alignment at best, not an independent positional
certification. ATA products remain separate from the primary Stockert maps.

About 110 Dwingeloo self-receive captures remain candidates for processing
with TX-leakage handling. Short March 2025 experiments were imaged for a
waveform comparison, but production selection and the 20 s duration guard
exclude them. A July inventory also noted unconverted VRT captures; their
availability and usability need checking before adding them to a new batch.
