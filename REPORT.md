# Moon radar: results and remaining limits

Reviewed 2026-09-14 against the current code, saved July validation results,
and August waveform-design work. The latest documented full processing run
is July 4; this review does not constitute a new data run. Raw captures,
SPICE kernels, terrain data, and generated maps are not included in this checkout.

The pipeline turns lunar radar echoes transmitted at Dwingeloo and received
at Stockert into reflectivity maps. It corrects timing and frequency offsets,
accounts for lunar terrain, and combines observations from different days.

The current run contains **222 processed maps: 111 per polarization channel**.
Quality cuts retain **107 per channel, 214 total**. The maps align internally
to better than one 4.5 km map pixel, but their absolute placement on the Moon
has not been independently established. Adding similar observations now gives
diminishing returns: repeatable surface features and processing artifacts
dominate the remaining random variation.

For the processing sequence, read the [walkthrough](WALKTHROUGH.md). The
[README](README.md) covers commands and files. Section numbers below retain
the references used in code and investigation notes.

## 1. Terms and resolution

| Term | Meaning |
|---|---|
| Look | One capture processed into a delay–Doppler image and a lunar map. |
| Delay–Doppler (DD) image | Echo strength sorted by travel time and frequency shift. Delay measures path length; Doppler measures its rate of change. |
| Specular reference point (SRP) | The point on a smooth lunar ellipsoid with the shortest transmitter–Moon–receiver path. It anchors the geometry and compensation. |
| `dlt` | The derivative of that two-leg light time: `Doppler = −frequency × dlt`. |
| Rim | Either extreme-Doppler edge of the lunar echo. “Up” is approaching (`dlt` minimum); “down” is receding (`dlt` maximum). |
| Rim correction, δ | The frequency correction still needed after the initial echo-tone measurement. |
| Ambiguous stripe | Surface region where delay and Doppler cannot distinguish nearby positions. The pipeline masks the most ambiguous pixels. |
| Stack | An average of maps in linear intensity, after masking, gain normalization, and alignment. |

The Stockert observations use a 1299.5 MHz carrier, 250,000 recorded samples/s,
and a Zadoff–Chu (ZC) waveform at 50,000 chips/s. A chip is one element of the
transmitted code. Sampling density and physical resolution are different:

| Quantity | Scale |
|---|---|
| Delay sample spacing | 4 µs, equivalent to 600 m in one-way range |
| Delay resolution from the chip rate | About 20 µs, or 3 km in one-way range |
| DD Doppler row spacing | 1.4–6.3 mHz: 3000 rows across the recorded 4.3–18.8 Hz lunar span |
| Doppler resolution from integration time | About `1/T`, or 15–33 mHz for 30–66 s windows |
| Map pixel spacing | 0.147°, about 4.5 km (HEALPix `nside=400`) |

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

Light-time derivatives use wide time intervals because short differences
amplify numerical quantization. Timestamp arithmetic is relative to the
capture start to preserve sub-sample precision.

Transmission is modeled as starting at `rx_start + 1.0 s`, plus the measured
timing correction. The TX file timestamp records waveform generation, not
emission. The image's delay window starts 20 samples before the predicted SRP
echo so the leading edge remains inside it.

The projection uses elevations from the Lunar Orbiter Laser Altimeter (LOLA)
digital elevation model (DEM). Both delay and Doppler are evaluated at these
terrain heights. The SRP and rim curves retain the smooth ellipsoid as a
stable reference. Without a local DEM, processing reports a fallback to the
ellipsoid; the runs CSV records `use_dem`.

## 3. Per-look calibration

### 3.1 Timing

Multiplying the received signal by the conjugate of the compensated TX
waveform concentrates the specular echo into a narrow tone. Searching delay
shifts for the strongest tone, then refining between samples, estimates the
timing offset. The batch searches ±40 samples.

Reported offsets are usually +35–45 µs, with three September 11 outliers at
+80, +91, and +125 µs. These outliers exceeded the former ±20-sample search
and are now recovered. A strong-echo measure/apply/re-measure check closed to
0.03 samples; weaker echoes support roughly one-sample precision.

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
the initial edge search fails. It recovered a badly faded September 16 look
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
before projection; a later map translation cannot repair it.

Stockert `chan1` is treated as co-polarized and `chan0` as cross-polarized.
The weaker cross-polarized echo inherits timing, frequency, and rim corrections
from its co-polarized twin when the batch uses `--corrections-from`.
Its quality selection also uses the twin's tone strength. This shares the
clock calibration; it is not an independent quality test of the cross-pol capture.

## 4. Clock behavior

The clock comparison is the **sum** of `applied_df_hz` and `rim_delta_hz`.
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

The stacker masks pixels sharing a DD cell with more than three times the
median number of surface pixels. It rejects weak echoes (paired co-pol tone
SNR below 15), delay estimates at the search limit (`|shift| ≥ 39.5` samples),
and abnormal rim spread relative to the session. The latter cut rejects the
three June drift captures. Here “tone SNR” is the estimator's peak/median
amplitude ratio, not a calibrated receiver sensitivity measurement.

Maps are averaged in linear intensity. An empirical brightness-versus-incidence
curve is fitted separately for each channel to make scattering-normalized
products. Session offsets are solved from cross-correlations emphasizing
features on 0.3°–2.5° scales. September 10 and 11 are grouped into one
registration session, giving three session stacks from four observing dates.

The latest documented production run reports residual disagreement after
alignment of **0.005° for chan1, 0.027° for chan0, and 0.006° for the combined
map**: all below 1 km on the Moon. This is internal consistency. Repeated
random splits of the July data give an estimator scatter around 0.02–0.03°;
the [closure bootstrap](validation/results/registration_closure_bootstrap_chan1.json)
has median 0.025° and 95th percentile 0.040°. That bootstrap predates the
final exclusions and is not a new measurement of the 107-look stacks.

An [absolute-position comparison with LOLA terrain](validation/results/absolute_registration.json)
did not meet the test's correlation-significance threshold. A roughly 3 km
southward offset appeared at finer scales, but remains a weak indication.
Neither this test nor the small internal residual establishes an absolute
position-error bound.

The saved variance-versus-look-count fits used **110 looks per channel**,
before the final exclusions. They estimate residual speckle at
[8% of filtered variance in chan1](validation/results/speckle_floor_chan1.json)
and [4% in chan0](validation/results/speckle_floor_chan0.json). The rest is
repeatable structure, which includes both lunar features and stable artifacts.
These results support improving the mapping and observing geometry before
simply collecting more of the same views.

Registration signs are fixed by [synthetic tests](test/test_registration_conventions.py):
positive longitude is east and positive latitude north; `grid_map` preserves
those coordinates; `shift_intensity` moves content in the stated direction;
`xcorr_offset(a, b)` returns the correction that moves `b` onto `a`. The
expected sign when applying solved session offsets is `+1`.

## 6. Remaining limits

| Effect | Evidence and present treatment |
|---|---|
| Terrain distortion | Corrected when the DEM is loaded. Saved ellipsoid/DEM comparisons reach 7.4 delay samples; terrain also changes Doppler by up to 22 mHz. |
| Timing uncertainty | Corrected per look; demonstrated 0.03-sample closure on strong echoes, roughly one-sample precision on weak ones. |
| Rim-estimator bias | Synthetic tests give about 0.5–1 mHz for symmetric cases and 6–7 mHz under extreme brightness asymmetry. |
| Frequency drift within a look | Clean looks show 1–8 mHz half-to-half change; three ~25 mHz cases are excluded. No time-varying correction yet. |
| Geometry approximation | Saved full-surface comparison: about 12 ns delay and at most 1.3 mHz Doppler error. |
| DD-cell sampling | Nearest-neighbor projection leaves visible constant-delay rings. Bilinear sampling and leading-edge delay calibration remain candidates. |
| Single-look ambiguity | Different surface locations can share a delay and Doppler. Masking and geometry diversity help; stacking does not invert this overlap. |
| Map alignment | Small internal residuals; absolute placement remains unverified. |
| Brightness calibration | Residual seams and polarization leakage limit quantitative interpretation of brightness and channel ratios. |

The [DEM projection check](validation/results/lola_dem_projection.json)
found the expected direction and kilometre-scale movement in well-correlated
features: Mare Imbrium moved 4.6 km east versus 4.3 km predicted. The Tycho
comparison was too weakly correlated to validate a displacement. Terrain
corrections improve the maps without necessarily improving session alignment,
because much of the distortion is shared across sessions.

## 7. Verification and reproducibility

Use the project `.conda` environment from the repository root. The
[validation guide](validation/README.md) lists dependencies, saved outputs,
and commands. The numerical-core test covers geometry, GPU correlation, tone
recovery, and a real-data calibration closure. Separate tests cover terrain
projection, registration signs, and ZC parameter selection.

Saved physics-validation logs are dated July 3. Final production counts and
alignment figures above come from the documented July 4 processing work;
the untracked run directories are required to recalculate them. The September
documentation review checked the ZC recommendation and its two unit tests
locally, but could not repeat the data-dependent pipeline without the data,
kernels, DEM, and GPU setup.

## 8. Completed work and next steps

The identifiers below are retained for older references.

| Item | Status or next action |
|---|---|
| 8.1 Error-model units and pixel scales | Fixed. Old plotting notes that claimed multi-Hz ephemeris uncertainty are obsolete. |
| 8.2 Rim estimator | Synthetic bias tests complete; July scan-window correction and coarse recovery are in production. |
| 8.3 Delay-search recovery | Complete; ±40 samples is now the batch default. |
| 8.4 Terrain projection | Implemented and validated with LOLA; timing-scatter attribution remains open. |
| 8.5 Frequency drift within a look | Fit and apply sub-window corrections to recover the three excluded captures. The targeted July investigation uses the current rim window; `intra_look_drift.py` still uses the older window and baseline directory. |
| 8.6 Delay-axis refinement | Test bilinear DD sampling and a leading-edge timing calibration to reduce ring artifacts. |
| 8.7 Surface inversion | Fit one surface model to the DD images from multiple geometries. Requires further calibration and validation work. |
| 8.8 Polarization-ratio map | Possible from registered channels; characterize leakage and low-signal bias first. |
| 8.9 Brightness matching | Improve normalization between sessions for quantitative maps. |
| 8.10 More data | ATA comparison is implemented; Dwingeloo self-receive and March experimental data still need dedicated batch handling (§10). |
| 8.11 Feed characterization | Quantify the cross-pol specular leakage before interpreting channel ratios. |
| 8.13 Operator cross-checks | Obtain per-session reference-frequency logs; use `vrt:cal_time` and SDR clock settings to reassess timing scatter. |
| 8.14 SPICE light-time convergence | The code still uses `LT`; switching to `CN`/`XCN` is a small numerical refinement requiring the core tests and a real-look closure. |
| 8.15 Observation planning | Rerun the planner for new dates with the recorded-look CSVs and suitable Earth-orientation kernels; see [planning notes](LIBRATION_ANALYSIS.md). |

### 8.12 Waveform design

The August addition, [recommend_zc.py](recommend_zc.py), selects ZC lengths
and roots for a chosen duration, chip rate, Doppler span, and coupling limit.
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
found integrated periodic-autocorrelation sidelobes of approximately −153 dB
for ZC, −51 dB for an m-sequence, and 0 dB for random BPSK. These are ideal
code metrics, not achieved lunar-image dynamic ranges. They support retaining
ZC as the reference waveform while testing its parameters through the actual
transmit/receive chain.

Increasing chip rate from 50 to 250 kchip/s could improve nominal range
resolution from 3 km to 600 m at the same recorded sample rate. Usable filter
bandwidth and map reconstruction must support it. For 30 s, a 19 Hz span,
and a 0.05-cell coupling limit, the tool selects `N=7500013, q=11401`.
This is a candidate for testing, not a validated replacement waveform.

## 9. Code and product locations

The [README file index](README.md#file-index) identifies the active modules.
The current production directory is `results/LOLA_DEM_REGISTRATION/`: runs
CSVs, per-look maps and ambiguity counts, session stacks, scattering curves,
and `stacked_map_{chan1,chan0,dual}{,_scatnorm}.npy/.png`.

`results/LOLA_DEM_REGISTRATION_FROZEN_0612/` preserves the previous DEM run
and its `PRE_DEM_ANALYSIS/` comparison. `results/REGISTRATION/` is the
pre-DEM baseline. These generated directories are untracked; old investigation
scripts may still use their former names.

## 10. Data coverage

The latest documented Stockert processing counts are:

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
(correlation +0.29, versus +0.80 for a Stockert split-half control). This is
weak supporting evidence for alignment, not an independent positional
certification. ATA products remain separate from the primary Stockert maps.

About 110 Dwingeloo self-receive captures remain candidates for processing
with TX-leakage handling. Short March 2025 experiments were imaged for a
waveform comparison, but production selection and the 20 s duration guard
exclude them. A July inventory also noted unconverted VRT captures; their
availability and usability need checking before adding them to a new batch.
