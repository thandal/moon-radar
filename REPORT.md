# Moon-Radar Pipeline: Fixes, Findings, Drift Characterization, Error Budget

*Status report, 2026-06-12. Covers the physics review and subsequent work in the
`moon-radar-claude` checkout: per-look calibration, multi-session registration,
deep stacking, and polarimetric products. All quantitative claims below were
measured on the 2025-06-21 / 2025-09-10–11 / 2025-09-16 Dwingeloo→Stockert
datasets (1299.5 MHz, 0.25 Msps, Zadoff-Chu 30/60 s).*

---

## 1. Pipeline fixes

### Correctness

| Fix | Impact | Verification |
|---|---|---|
| **TX timestamps are waveform-generation times**, not emission epochs (off by 25 h on 06-21, 88 days on 09-16). Emission epoch must be `rx_start + 1.0 s` + measured offset. | Months-scale geometry error if trusted | `test/test_tx_start.py` |
| **`et_from_astropy` quantized epochs to 1 ms** (astropy default ISO precision through `str2et`). Now rendered at ns precision. | Silently nulled µs-scale timing corrections passed as `Time` objects | closure test failed before fix, closes to 0.03 samples after |
| **SPICE light times are quantized at ~2×10⁻¹¹ s** (~6 mm) in this kernel chain. Any Doppler from short-baseline finite differences inherits ~0.25 Hz noise (at dt=0.05 s). All derivative baselines widened to T/2; surface Doppler uses the exact window-average identity `mean(dlt) = (lt(t₀+T)−lt(t₀))/T`. | ±0.25 Hz → ~1 mHz Doppler noise | slope-vs-baseline measurement; `test_pipeline_consistency.py` |
| **One Doppler convention everywhere**: finite-difference d(lt)/dt replaces the per-leg special-relativistic range-rate product. (The SR product actually agrees to 0.5 mHz — the 0.118 Hz "systematic" in `test_dlt_derivative.py` was granularity noise in its numeric reference. The β² relativistic terms do not apply to a ground-to-ground bounce, but they are sub-mHz.) | consistency; mHz-level | measured vs wide-stencil reference; ITRF93 vs J2000 confirmed exactly equivalent |
| **Differential Doppler-rate correction**: the image compensates the SRP rate, so each surface point's energy lands at its *window-averaged* Doppler: `dlt_eff = mean(dlt) − rate_srp·T/2`. (Lands the physics from the old `patch_scaling.py`; the midpoint variant over-shifts by `rate_srp·T/2` and remains wrong.) | ±15 mHz at the limbs (66 s) | rim-spread statistic: −6 ± 17 mHz across 106 looks |
| **Delay window leads the SRP by 20 samples** — with corrected timing, the echo's leading edge sat exactly on the window boundary and was clipped. | leading-edge photometry, center-of-disk mapping | A/B render |
| **Doppler-equator branch conventions unified** (up = approaching = dlt-min in all three methods; the terminator method was mirrored — the source of the old "flipped" plot hacks). | sign-error trap removed | cross-method agreement test |
| **`edterm` radii** moved from PINPOINT-generated `observatories.tf` (regeneration would silently drop them) to `spice_kernels/observatory_radii.tpc`. | fragility | kernel load test |

### Performance (RTX 3080, 24-core)

| Stage | Before | After | How |
|---|---|---|---|
| Lunar projection (nside 400) | 127 s | 0.5 s | SRP-anchored station positions + numpy distance fields (12 ns / <1 mHz vs exact per-point SPICE) |
| Tone measurement (41 delay shifts) | ~35 s | 0.34 s | GPU block-sum decimation to 5 kHz before FFT (same bin width) |
| DD correlation (3000 rows) | 13 s | ~8 s | batched 2-D FFTs, float32 ramps (6×10⁻⁷ relative vs reference) |
| Full A/B batch (7 files × 2 passes, nside 400) | 37 min | 2.5 min | + 3-worker spawn multiprocessing, OOM serial retry |

**Regression suite**: `test/test_pipeline_consistency.py` — 10 checks, each
optimization vs its exact counterpart, plus a real-data closure test
(measure → apply corrections → residual 0.03 samples / 13 mHz). Run it after
any pipeline change.

---

## 2. Calibration findings

### Timing (delay)
- Per-capture timing offsets: **+35–45 µs** (TX starts late relative to the
  `rx_start + 1.0 s` convention), with **±9–12 µs scatter between captures**
  on all three sessions — matches the historical `TX_START_OFFSET = 1.00005`
  hack and explains why no constant ever worked.
- Measured per file by the product method (rx · conj(model TX)) at sub-sample
  resolution; closure 0.03 samples on strong echoes, ~1 sample resolution on
  weak ones. Three 09-11 files rail at the +20-sample (80 µs) search limit —
  recoverable with a wider search.
- The old ±10 µs alignment grid search did not contain the true offset; the
  direct measurement supersedes it.

### Frequency (Doppler)
- **No constant TX/RX chain offset** (session means −0.2…+0.1 Hz, ≲1.6×10⁻¹⁰
  fractional). The ~800 Hz offsets of the 2025-03 era are gone.
- The specular line is **fading-broadened to ~0.2 Hz RMS** (multi-lobed,
  libration fading), so the line *centroid* calibrates a look only to
  ±30–80 mHz. The within-file "segment flips" seen early on were fading-lobe
  hopping, not clock jumps.
- **Rim calibration** (the decisive instrument): a signed Doppler residual
  shifts both horseshoe rims identically, so the mean of the up/down rim
  offsets measures δ per look using ~340 delay samples. Results across 106
  co-pol looks: **unbiased (mean +7 mHz), scatter ±47 mHz, range ±110 mHz**,
  correlated with line width (r = 0.26 → fading bias of the centroid);
  iterative closure to **0.13 mHz median**; measurement noise ~1 mHz
  (even/odd column split). Cross-pol looks inherit their co-pol twin's δ
  (same clocks, same geometry, same applied compensation).
- **Wedge mechanism** (the visible symptom): at the mapping-degeneracy stripe
  the transverse Doppler gradient vanishes, so a residual δ displaces the
  sampled locus as √(δ/curvature) — tens of mHz become tens of km, and the
  sign of δ selects which arm samples beyond the rim (dark wedge) vs inside
  it (fanned bin-boundary seams). Verified by deliberate ∓80 mHz perturbation
  flipping the wedge between arms.

### Whole-disk effect of δ (why calibration ≠ masking)
Surface displacement = δ / |∇doppler|, measured across the disk (stripe
excluded):

| δ | median displacement | 90th pct |
|---|---|---|
| 10 mHz (post-calibration) | 2.2 km (0.5 px) | 4 km |
| 47 mHz (typical pre-calibration) | 10.3 km (2.3 px) | 19 km |
| 110 mHz (worst look) | 24 km (5.4 px) | 45 km |

This is a spatially varying **warp** (not a shift): it cannot be removed by
image registration after the fact. The stripe mask handles the divergent
locus; the calibration fixes the coordinate system everywhere else.

### Scattering laws (fitted empirically from the stacks)
- **Co-pol (chan1)**: classic lunar quasi-specular curve — +24 dB at 9°
  incidence, steep fall to ~+7 dB at 30°, diffuse tail to −4.5 dB at 88°.
- **Cross-pol (chan0)**: 9 dB weaker specular peak, much shallower slope,
  crossover at ~58°, flatter limb tail — diffuse/volume scattering, as
  expected. The residual cross-pol specular peak suggests modest polarization
  leakage / imperfect feed circularity.

---

## 3. Drift characterization (clock/reference behavior)

Measured with the rim instrument (noise ~1 mHz per half-window):

| timescale | frequency wander | fractional |
|---|---|---|
| intra-look (33 s half-windows) | **10.6 mHz rms** (median \|drift\| 12 mHz) | 8×10⁻¹² |
| look-to-look (minutes–hours) | **±47 mHz** | 3.6×10⁻¹¹ |
| session-to-session (days–months) | no constant offset detected | ≲1.6×10⁻¹⁰ |

The √τ scaling between 33 s and the inter-look spacing is consistent with
random-walk FM — continuous reference wander (GPSDO-class), not discrete
jumps. Timing side: ±9–12 µs per-capture jitter (SDR timestamping), stable
~40 µs mean per session.

Uncorrected, intra-look wander costs ±0.25–0.5 px of Doppler-axis blur and
2–4 bins of spectral smear per look (the look's *mean* δ is already
calibrated). Correction is feasible with margin: the drift is measurable at
SNR ~10 in half-windows (SNR ≥3 expected in quarter-windows).

Incidentally: the half-window rim measurement is a clock-comparison
instrument at 8×10⁻¹² in 33 s, operating through a Moon bounce.

---

## 4. Registration and stacking results

- **Registration** (band-passed feature cross-correlation, degeneracy-masked):
  intra-session sub-pixel on all sessions (43 min, 3.5 h, and a 12 h
  overnight session); cross-session closed-loop residual after rim
  calibration: **0.005° (chan1) / 0.006° (dual) ≈ 150–200 m ≈ 1/30 pixel**
  (was 0.033° pre-rim). The cross-pol independent solve agrees within 0.05°.
- **Stacks** (106 looks/channel, 212 dual; median 105 looks/pixel;
  stripe-free by mask rotation): band-passed floor follows speckle ∝ 1/√N
  plus a constant structure floor — the maps are **structure-limited**
  (residual speckle ~13% of variance at 106 looks). Rim calibration *raised*
  the band-passed floor by 11% — concentrated feature contrast, the expected
  signature of sharper alignment at structure-limited depth.
- **Products** (`notebooks/results/REGISTRATION/`):
  `stacked_map_{chan1,chan0,dual}{,_scatnorm}.npy/.png`, per-pixel look
  counts, per-session stacks, scattering-law plots, and per-look maps +
  degeneracy masks + renders for all 220 looks, with the full calibration
  audit trail in `registration_runs_{chan1,chan0}.csv`.

---

## 5. Current error budget

### Per look
| term | size | status |
|---|---|---|
| Delay: per-capture timing | ±9–12 µs raw | measured & corrected to ≲1 sample (4 µs); 0.03-sample closure on strong echoes |
| Delay: model (SPICE, field approx, granularity) | ≤ 20 ns | negligible (0.005 sample) |
| Doppler: chain offset | ±47 mHz raw | rim-calibrated; closure 0.13 mHz; residual systematic est. ≲5 mHz (edge-shape bias, unquantified) |
| Doppler: intra-look wander | ±5–10 mHz | **uncorrected** → ±0.25–0.5 px blur + 2–4 bin smear |
| Doppler: rate/curvature model | rim spread −6 ± 17 mHz | bounded; window-average model validated |
| Doppler: SPICE/geometry model | ≲1 mHz | negligible |
| Mapping: nearest-neighbor DD sampling | ~1 delay sample | **iso-delay ring artifact — current leading artifact** |
| Mapping: N–S ambiguity fold | inherent | un-deconvolved; partially decorrelates in cross-session stacks |
| Mapping: degenerate stripe | masked (mult > 3× median) | needed mask width shrank ~√7 with calibration |

### Stack level
| term | size | status |
|---|---|---|
| Session registration | 0.005–0.006° (0.15–0.2 km) | solved + closed-loop verified |
| Speckle residual | ~13% of structure variance (106 looks) | structure-limited |
| Photometric session seams | few % along stripe-fill zones | cosmetic; needs per-look photometric matching |
| Incidence-angle normalization | empirical law, terrain-independent | adequate for display; refine for photometry |

### Resolution accounting (per look)
Delay sample 4 µs ≈ 600 m range; Doppler bin 5.3 mHz (oversampled vs the
1/T Rayleigh limit of 15 mHz at 66 s); surface pixel 0.147° ≈ 4.5 km
(nside 400). The Doppler axis currently delivers ~2–4 bins of effective
width due to intra-look wander; delay axis is sampling-limited.

---

## 6. Next steps

1. **Intra-look δ(t) correction** *(designed, feasibility proven)* —
   2–4 sub-window rim fits per look → smooth δ(t) → fold into the
   compensation ramp → one corrected full-window image. ~2× per-look
   compute; recovers Doppler resolution to the 1/T limit. Prerequisite for
   inversion-grade coherence.
2. **Delay-axis refinement** — bilinear (not nearest) DD-cell sampling in
   `lunar_projection` and a leading-edge delay calibration (analogue of the
   rim method) to remove the iso-delay ring artifact, now the leading
   stack artifact.
3. **Forward-model inversion (super-resolution path)** — solve for the
   surface map that predicts all 212 DD images, using the per-look
   projection operators (already computed; multiplicity = footprint). This
   is what converts cross-look sampling diversity into resolution; plain
   stacking cannot.
4. **Photometric session matching** — per-look gain/law matching in overlap
   regions to remove the few-percent stripe-fill seams.
5. **CPR product** — the chan0/chan1 ratio map (circular polarization
   ratio), a standard roughness proxy; both inputs exist and are registered.
6. **Data recovery** — wider shift search for the three railed 09-11 files;
   the missing `interp_zadoff-chu` TX waveform; consider the 2025-03-11
   1 Msps sets and Dwingeloo self-receive (needs leakage handling).
7. **Feed characterization** — quantify the cross-pol specular leakage seen
   in the chan0 scattering law.
8. **Housekeeping** — port the timing/Doppler/window fixes back into
   `Planetary Radar -- Moon.ipynb` where still relevant, and reconcile this
   checkout with the main (Codex) working tree before merging.

---

## Tool inventory (all under `notebooks/`)

| tool | purpose |
|---|---|
| `doppler_equator_alignment.py` | core library: SPICE geometry, DD image, equator methods, rim calibration, projection, batch `process_file` |
| `freq_offset_hunt.py` | per-file timing/frequency measurement (product method) |
| `registration_stability.py` | parallel batch driver (per channel, paired corrections) |
| `stack_maps.py` | session-offset solve + raw/scatnorm/dual stacking |
| `registration_analysis.py` | gridding, band-pass, masked cross-registration |
| `wander_corrected_batch.py` | A/B (baseline vs corrected) comparison batches |
| `intra_look_drift.py` | half-window drift experiment |
| `../test/test_pipeline_consistency.py` | regression suite (run after any change) |
