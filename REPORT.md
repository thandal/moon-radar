# Moon-Radar Pipeline: State of the System

*2026-06-12. Bistatic lunar delay-Doppler mapping, Dwingeloo (TX) → Stockert
(RX), 1299.5 MHz, 0.25 Msps Zadoff-Chu waveforms. Datasets: 2025-06-21,
2025-09-10/11, 2025-09-16 (110 usable captures per polarization channel).
All quantitative claims were measured on these data; the regression suite
(`notebooks/test/test_pipeline_consistency.py`) re-verifies the numerical
core after any change.*

---

## 1. Terminology

| term | definition |
|---|---|
| **look** | one capture processed into one DD image and one lunar map |
| **DD image** | delay-Doppler correlation image: 3000 Doppler rows × ~2920 delay columns |
| **dlt** | delta-light-time, d(lt)/dt of the two-leg TX→Moon→RX path; Doppler = −dlt·f₀. The Doppler axis is labeled by start-anchored, window-averaged dlt |
| **SRP** | the bistatic specular point: minimum two-leg light-time point on the ellipsoid |
| **horseshoe / rim** | the bright echo outline in the DD image; the **rims** are its extreme-Doppler edges — **up-rim** = approaching (dlt-min), **down-rim** = receding (dlt-max) |
| **δ** | per-look Doppler residual: signed offset between the predicted Doppler axis and where the echo actually sits, after nominal compensation (physically: the TX/RX reference-frequency offset at that epoch) |
| **rim calibration** | per-look measurement of δ as the mean of the up-rim and down-rim offsets; their half-difference (**rim spread**) diagnoses Doppler-rate/curvature model error |
| **stripe** | the surface zone where the DD→surface mapping degenerates (∇delay ∥ ∇Doppler), running through the SRP along the apparent-velocity direction |
| **degeneracy mask** | per-look mask of pixels whose DD cell is shared by many surface pixels (multiplicity > 3× median); covers the stripe and the SRP bloom |
| **wedge / fan** | single-look artifacts on the stripe's two arms caused by uncalibrated δ: predictions past the rim sample empty image cells (dark **wedge**); predictions inside it sample the rim through discrete Doppler bins (**fan** of seams). A balanced stripe zone ⇔ δ calibrated |
| **session** | one observing day; **stack** = incoherent mean of looks in linear intensity, after per-look gain normalization |

Resolution scales used throughout: delay sample 4 µs (600 m one-way range);
Doppler bin 1.8–8 mHz depending on epoch (the axis spans the limb-to-limb
terminator dlt range — measured 5–24 Hz across the sessions — over 3000
rows; the 1/T Rayleigh limit is 15–33 mHz, so rows oversample); map pixel
0.147° ≈ 4.5 km (HEALPix nside 400).

---

## 2. Geometry and signal model

- **Doppler convention.** All Doppler is the derivative of the total
  two-leg light time (exact for coordinate clocks; the geoid stations'
  proper-clock rate difference is ~10⁻¹³). The legacy per-leg
  special-relativistic range-rate product agrees to ~0.5 mHz; its β² terms
  do not apply to a ground-to-ground bounce.
- **Light-time granularity.** SPICE light times from this kernel chain are
  quantized at ~2×10⁻¹¹ s (~6 mm of path). Derivatives therefore use wide
  stencils (half the integration window) or the exact window-average
  identity `mean(dlt) = (lt(t₀+T) − lt(t₀))/T`; a short baseline
  (dt = 0.05 s) would carry ~0.25 Hz of Doppler noise at L-band.
- **Window-averaged mapping.** The DD image compensates the SRP Doppler
  rate, so a surface point's correlation energy lands at its window-averaged
  Doppler: `dlt_eff = mean(dlt) − rate_SRP·T/2`. The differential term
  reaches ±15 mHz at the limbs for 66 s windows; the rim-spread statistic
  (−6 ± 17 mHz over the co-pol looks) bounds the model's residual error.
- **Surface fields.** Full-disk delay/Doppler fields are evaluated in numpy
  from light-time-consistent ("apparent") station positions anchored at the
  SRP — verified against exact per-point SPICE to 12 ns in delay (0.003
  samples) and <1 mHz in window-averaged Doppler, and ~600× faster.
- **SRP solver.** Vectorized tangent-plane zoom; agrees with a Nelder-Mead
  reference to 14 m / 3×10⁻¹³ s.
- **Conventions.** "Up" branches are the approaching (dlt-min) side in
  every Doppler-equator method. The TX emission epoch is
  `rx_start + 1.0 s` plus the measured per-look timing offset (the TX file
  timestamps are waveform-generation times and are not used). The delay
  window leads the SRP delay by 20 samples so the echo onset sits inside it.
- **Ellipsoid approximation.** The surface is currently the reference
  ellipsoid; true topography (±4 km radial) is the dominant mapping
  systematic — see the error budget.
- **Model-error bounds** (why neither ephemeris nor epoch errors matter):
  DE440 is LLR-constrained to ~2 cm in position (→ 0.06 ns of delay) and
  ~10 µm/s in velocity (→ ~4×10⁻⁵ Hz of Doppler). An epoch error Δt couples
  into Doppler only through the line-of-sight acceleration,
  Δf = f₀·a_LOS·Δt/c ≈ 10⁻⁹ Hz per 100 ns — so timing errors live entirely
  in the delay axis, and reference-frequency errors entirely in the Doppler
  axis; the two calibrations are independent.
- **TX resampling.** The TX is a constant-modulus Zadoff-Chu signal, so the
  Doppler/timing resample interpolates the *unwrapped phase* (linear) and
  reconstructs `exp(i·φ)`, rather than interpolating the complex samples —
  this preserves |tx| = 1 and the CAZAC autocorrelation. Of the upsampling
  options tried: linear interpolation of the complex samples lowers the
  autocorrelation peak and spreads it over ~20 samples; `scipy.signal.resample`
  preserves the peak height but introduces anti-peaks and can exceed unit
  modulus; integer-repeat and phase interpolation preserve the property. The
  ZC root index is chosen large enough to stay robust over the band's max
  Doppler (`q ≳ 4·f_Dmax·N_zc / B_zc`).
- **Numerical working point.** All timestamp and resampling arithmetic is done
  relative to the capture start (near zero), never in absolute ET (~8×10⁸ s),
  where float64 spacing (~10⁻⁷ s) would swamp the sub-sample time variations
  being resolved.

---

## 3. Per-look calibration

Every look is calibrated independently; all values are recorded in
`results/REGISTRATION/registration_runs_{chan1,chan0}.csv`.

### 3.1 Timing
Measured by the product method (`rx · conj(model TX)`, which collapses the
specular echo to a tone), with sub-sample delay refinement:

- **Per-capture offsets: +35–45 µs typical, scatter ±9–12 µs**, consistent
  across all three sessions, with three 09-11 outliers at +80/+91/+125 µs.
  Closure after applying the correction: 0.03 samples on strong echoes;
  ~1 sample resolution on weak ones.
- **Decomposition against facility measurements**: ~30 µs is the USRP
  B210/AD9361 TX–RX pipeline latency (Dwingeloo autocorrelation tests);
  ±4 µs/station PPS one-sample ambiguity; 735 ns Stockert GPS cable delay.
  The remaining scatter may be partly lunar topography at the SRP (terrain
  height h shifts the echo onset by 2h/c ≈ 6.7 µs/km) — testable by
  correlating per-look offsets with LOLA elevation at each SRP.
- The three outlier captures originally railed a ±20-sample search; a
  ±40-sample search (`recover_railed.py`, now the batch default) resolves
  them cleanly (tone SNR 68–94) and they pass into the stacks.

### 3.2 Frequency (δ)
Two instruments, applied in sequence:

1. **Specular-line centroid** (coarse): the tone is fading-broadened to
   ~0.2 Hz RMS and multi-lobed, so its centroid estimates δ only to
   ±30–80 mHz. Unbiased on average.
2. **Rim calibration** (fine): δ = mean(up-rim, down-rim offset), measured
   over ~340 delay samples per look and iterated to convergence. Across the
   co-pol looks the post-centroid residual is **±47 mHz (range ±110 mHz),
   mean +7 mHz**; closure 0.13 mHz median; measurement noise ~1 mHz
   (even/odd column split). Cross-pol looks inherit δ from their co-pol
   twin (same clocks, geometry, and applied compensation; the cross-pol rim
   is too diffuse to measure).

   The estimator is **validated end-to-end against synthetic echoes**
   (`rim_bias_validation.py`: analytic rim-caustic forward model on the
   real SPICE geometry, sinc² resolution kernel, correlated speckle, the
   exact iterative calibration loop): linear in δ to ±80 mHz with bias
   < 0.5 mHz under symmetric conditions — including resolution smear,
   weak contrast, and scattering-law shape — and ≤ 3.4 mHz worst case
   under an extreme ±50% up/down rim-brightness asymmetry (sub-mHz to
   ~1 mHz at realistic asymmetries). A symmetric edge-shape displacement
   cancels into the rim *spread* by construction, as designed.

   One regime caveat: epochs with a small limb-to-limb Doppler span and
   short windows (the 09-11 morning looks: 1.8 mHz bins, 1/T smear ~16
   rows) can starve the contrast gates — the rim then goes uncertified
   (δ = 0 from the rim; the centroid still applies). Verified harmless on
   the one affected look (08:05:44: relaxed-gate residual +2.4 mHz).

**Why δ matters everywhere, not just at the stripe.** A constant δ
displaces each pixel's assigned surface position by δ/|∇Doppler| — a
spatially varying warp that no post-hoc registration can undo:

| δ | median displacement | 90th pct |
|---|---|---|
| 10 mHz (post-calibration) | 2.2 km (0.5 px) | 4 km |
| 47 mHz (typical pre-calibration) | 10.3 km (2.3 px) | 19 km |
| 110 mHz (worst look) | 24 km (5.4 px) | 45 km |

At the stripe the same δ is amplified as √(δ/curvature) into the
wedge/fan asymmetry (verified by perturbing δ by ∓80 mHz, which flips the
wedge between the stripe's arms). Calibration balances the stripe zone and
removes the whole-disk warp; the degeneracy mask handles the divergent
locus itself, whose required width shrinks as √δ (~7× post-calibration).

---

## 4. Clock and reference characterization

Measured with the rim instrument (~1 mHz noise per half-window):

| timescale | wander | fractional |
|---|---|---|
| intra-look (33 s half-windows) | 10.6 mHz rms | 8×10⁻¹² |
| look-to-look (minutes–hours) | ±47 mHz | 3.6×10⁻¹¹ |
| session-to-session (days–months) | no constant offset | ≲1.6×10⁻¹⁰ |

The √τ scaling is consistent with continuous random-walk FM.
**Attribution**: Dwingeloo runs a White-Rabbit-locked H-maser (~10⁻¹³), so
the wander is essentially all **Stockert's rubidium standard** — its
empirically logged ~5.5×10⁻¹¹ (≈0.07 Hz at f₀) matches the measured
per-look δ statistics quantitatively. The rim calibration is, in effect, a
per-look Stockert-Rb corrector, and the half-window rim measurement is a
clock comparison at 8×10⁻¹² in 33 s, operating through a Moon bounce.

Uncorrected intra-look wander costs ±0.25–0.5 px of Doppler-axis blur and
2–4 bins of spectral smear per look (the look *mean* is calibrated).
Correction is feasible with margin — the drift is measured at SNR ~10 per
half-window: fit δ(t) from 2–4 sub-window rim fits, fold it into the
compensation ramp, recompute once (~2× per-look compute).

---

## 5. Registration and stacking

- **Method**: per-look maps are degeneracy-masked, gain-normalized,
  converted to linear intensity, and stacked per session;
  session-to-session offsets are solved by least squares from band-passed
  (0.3°–2.5°) masked cross-correlation and applied with a closed-loop
  residual check. Raw single-look cross-correlation is speckle-limited
  (~1% peaks) — stack first, then register.
- **Registration**: intra-session sub-pixel on all sessions (43 min, 3.5 h,
  and a 12 h overnight session). Cross-session closed-loop residual:
  **0.005° (chan1 and dual) ≈ 150 m ≈ 1/30 pixel**. The cross-pol solve,
  done independently, agrees within 0.05° (closure 0.032°).
- **Stacks** (109 looks/channel, 218 dual; median 108/216 looks/pixel;
  stripe-free because the masked zone rotates between looks): band-passed
  variance follows speckle ∝ 1/√N plus a constant structure floor — the
  maps are **structure-limited**, with residual speckle ~13% of variance.
- **Scattering laws** (fitted empirically per channel): co-pol shows the
  classic quasi-specular lunar curve (+24 dB at 9° incidence → −4.5 dB at
  88°); cross-pol is 9 dB weaker at the peak, flatter, crossing over near
  58° — diffuse/volume scattering, with a residual specular peak indicating
  modest polarization leakage in the feeds.
- **Products** (`notebooks/results/REGISTRATION/`):
  `stacked_map_{chan1,chan0,dual}{,_scatnorm}.npy/.png`, per-pixel look
  counts, per-session stacks, scattering-law plots, and per-look maps,
  degeneracy masks, and renders for all 220 looks.

---

## 6. Error budget

### Per look
| term | size | status |
|---|---|---|
| Delay: per-capture timing | +35–45 µs typical (outliers to +125 µs), ±9–12 µs | measured & corrected to ≲1 sample; 0.03-sample closure (strong echoes) |
| Delay: lunar topography vs ellipsoid | ±4 km radial → up to ~±13 µs onset shift, ~7 px mapping systematic | **uncorrected — dominant mapping systematic**; needs LOLA DEM |
| Delay: geometry model (anchored field, granularity) | ≤20 ns | negligible (0.005 sample) |
| Doppler: δ (Stockert Rb) | ±47 mHz raw | rim-calibrated; closure 0.13 mHz; edge-shape bias validated synthetically: < 0.5 mHz symmetric, ≤ 3.4 mHz at extreme rim-brightness asymmetry (§3.2) |
| Doppler: intra-look wander | ±5–10 mHz | uncorrected → ±0.25–0.5 px blur, 2–4 bin smear |
| Doppler: rate/curvature model | rim spread −6 ± 17 mHz | bounded; window-average model validated |
| Doppler: topography | est. ~tens of mHz near the limb | uncorrected; needs proper derivation (with the DEM work, §8.4) |
| Doppler: ephemeris/SPICE | ≲1 mHz | negligible (DE440: ~2 cm position, ~10 µm/s velocity → ~4×10⁻⁵ Hz) |
| Mapping: nearest-neighbor DD sampling | ~1 delay sample | iso-delay ring artifact — leading stack artifact |
| Mapping: N–S ambiguity fold | inherent to a single look | un-deconvolved; partially decorrelates across sessions |

### Stack level
| term | size | status |
|---|---|---|
| Session registration | 0.005–0.006° (0.15–0.2 km) | solved, closed-loop verified |
| Residual speckle | ~13% of structure variance (109 looks) | structure-limited |
| Photometric session seams | few % along stripe-fill zones | cosmetic; needs per-look photometric matching |
| Incidence normalization | empirical law, terrain-independent | adequate for display; refine for photometry |

### Resolution accounting (per look)
The delay axis is sampling-limited (4 µs). The Doppler axis is
wander-limited at 2–4 bins effective width (vs the 1.8–8 mHz bin) until
intra-look δ(t) correction is implemented. The map pixel (4.5 km) is
matched to current registration and feature SNR, not to the DD cell, which
is finer.

---

## 7. Performance and verification

- Throughput (RTX 3080, 24-core, 3 workers): ~25 s per look for the full
  pipeline (image + calibration + projection); a 110-capture channel batch
  ≈ 20 min; stacking a channel ≈ 5 min.
- `notebooks/test/test_pipeline_consistency.py`: 10 checks — every
  numerical optimization against its exact counterpart (SRP solver,
  anchored field, batched GPU correlation, decimated tone measurement),
  synthetic ground-truth recovery, and the real-data closure of the
  measure→correct loop (residual 0.03 samples / 13 mHz). All pass.

---

## 8. Open items and next steps

Prioritized by error contribution per unit effort. "Error contribution" is
what the item currently costs (i.e., what completing it removes or gains),
in the units of the §6 budget.

### P0 — completed 2026-06-12
| item | outcome |
|---|---|
| **8.1 Error-model pixel-scale fix** — `doppler_equator_errors.py` used 1.73 Hz/Doppler-pixel, conflating the dlt *magnitude* (~4×10⁻⁶) with the axis *span* (measured 5–24 Hz across epochs) | fixed (bin default 6 mHz, ~290× finer). Also found & fixed a 1000× units bug in `velocity_error_breakdown.py` (`clight()` returns km/s, velocities were m/s). Both tools now match this report: ephemeris Doppler ≈ 4×10⁻⁵ Hz (negligible); the Stockert Rb (0.07 Hz ≈ 12 bins) is the look-to-look δ that rim calibration corrects |
| **8.2 Rim edge-shape bias validation** — was the one unquantified term in the δ chain (bounded ≲5 mHz) | quantified by synthetic-echo injection (§3.2, `rim_bias_validation.py`): bias < 0.5 mHz symmetric, ≤ 3.4 mHz at extreme ±50% rim-brightness asymmetry, linear in δ to ±80 mHz. The δ chain has no open systematics |
| **8.3 Railed-capture recovery** — three 09-11 captures railed the ±20-sample shift search | recovered with ±40-sample search (`recover_railed.py`; offsets +80/+91/+125 µs, the dataset's largest); batch default and stack gate updated. Stacks now **109 looks/channel, 218 dual**; chan1/dual closure 0.005° |

### P1 — dominant error reductions
| item | error contribution / payoff | effort |
|---|---|---|
| **8.4 LOLA DEM projection** — replace the ellipsoid surface in the mapping; correlate per-look timing offsets with SRP-local elevation to split SDR jitter from terrain | removes the **dominant mapping systematic** (~7 delay px ≈ 4 km); likely explains part of the ±9–12 µs timing scatter | days (DEM plumbing into surface points, SRP solver, rim curves + re-verification) |
| **8.5 Intra-look δ(t) correction** — 2–4 sub-window rim fits → δ(t) into the compensation ramp; designed and feasibility-proven (§4) | Doppler axis from 2–4 bins effective width to the 1/T limit; −0.25–0.5 px per-look blur; **prerequisite for inversion-grade coherence** | ~1 day; ~2× per-look compute |
| **8.6 Delay-axis refinement** — bilinear DD-cell sampling + leading-edge delay calibration (delay analogue of the rim method) | removes the iso-delay ring pattern (**leading stack artifact**, ~1 delay sample) | hours (bilinear) + ~1 day (edge calibration) |

### P2 — products and expansion (mostly gated on P1)
| item | error contribution / payoff | effort |
|---|---|---|
| **8.7 Forward-model inversion** — solve for the surface map predicting all 218 DD images; per-look operators and calibration audit trail exist | step-change: resolution toward the DD cell (vs 4.5 km pixel) and N–S ambiguity removal via geometry diversity; stacking cannot do either | 1–2 weeks (sparse operator + GPU LSQR + regularization + validation); needs 8.4/8.5/8.6 first |
| **8.8 CPR product** — chan0/chan1 ratio map (roughness proxy); inputs registered | new science product; no error reduction | hours (ratio + low-SNR bias care) |
| **8.9 Photometric session matching** — per-look gain matching in overlaps | removes few-% stripe-fill seams; needed for photometric use | hours |
| **8.10 Data expansion** — full inventory and coverage in §10. Remaining unused good data: Dwingeloo monostatic self-receive (~110 looks, leakage handling); the 2025-03-04 & 2025-03-11 1 Msps experimental sets (loader/window changes). The 3 captures formerly blocked by the unpublished `interp_zadoff-chu` TX waveform are now **unblocked** — the waveform was reconstructed from its published twin's parameters and the upsampling method (`scipy.signal.resample`/FFT) recovered from the Dwingeloo TX-leakage (§10). The **ATA second bistatic RX (2025-09-16, 11 dual-pol looks)** is now wired in (kernels + batch + stacking, §10) and serves as an independent registration cross-check (§10: consistent at ≤0.3°, but too few looks to certify sub-degree) rather than a contributing map | a same-epoch second bistatic geometry for cross-validation; not a noise contributor (8 co-pol looks are speckle-limited) | ATA done; others ~1 day each |
| **8.11 Feed characterization** — quantify the cross-pol specular leakage seen in the chan0 scattering law | polarimetric purity of CPR / cross-pol law; no geometric effect | moderate; may need hardware info |

---

## 9. Tool inventory (under `notebooks/`)

| tool | purpose |
|---|---|
| `doppler_equator.py` | geometry module: light times, Doppler, SRP, window-averaged dlt, apparent station positions, Doppler-equator methods |
| `doppler_equator_alignment.py` | imaging & calibration: DD image, rim calibration, projection, degeneracy mask, batch `process_file` |
| `freq_offset_hunt.py` | per-look timing/frequency measurement (product method) |
| `registration_stability.py` | parallel batch driver (per channel; cross-pol inherits co-pol calibration) |
| `stack_maps.py` | session-offset solve; raw/scattering-normalized/dual stacks |
| `registration_analysis.py` | gridding, band-pass, masked cross-registration |
| `wander_corrected_batch.py` | A/B (uncalibrated vs calibrated) comparison batches |
| `intra_look_drift.py` | half-window drift measurement |
| `rim_bias_validation.py` | synthetic-echo validation of the rim estimator (§3.2; results in `results/RIM_BIAS/`) |
| `recover_railed.py` | one-shot recovery of the railed 09-11 captures (±40-sample search; patches the runs CSVs) |
| `doppler_equator_errors.py`, `velocity_error_breakdown.py`, `error_visualization_example.py`, `plot_doppler_equator_simple.py` | error-model & visualization tools (pixel scale and `clight` units fixed per §8.1) |
| `test/test_pipeline_consistency.py` | regression suite — run after any change |
| `make_observatory_kernels.sh` | regenerate `spice_kernels/observatories.{bsp,tf}` + radii from the tracked `observatories.defs` (PINPOINT; run after adding a station) |

---

## 10. Data inventory and coverage

Source: `data.camras.nl/lunar-radar` (mirror; verified file-for-file against
the live site — the site's reorganization only re-exposed the June session
as duplicate copies under `thomas/sdr-eme/{rx,tx}-2025-06-21/`, which the
canonical `lunar-radar/` tree already holds). "Good" = the pipeline gate:
zadoff-chu, 30 s or 60 s, not CW/pulsed (`candidate_files`).

Each session was recorded by multiple receivers. Good captures per
receiver/session (co-pol count; ATA and Stockert are dual-pol):

| receiver | role | 06-21 | 09-10/11 | 09-16 | used in stacks? |
|---|---|---|---|---|---|
| **Stockert** | bistatic RX (primary) | 35 | 57 | 22 | **yes** — 111 co-pol looks (110 + the recovered `interp_zadoff-chu` look, §10) |
| **ATA** (Hat Creek) | 2nd bistatic RX | — | — | 11 (×2 pol) | **registration cross-check** — consistent ≤0.3°, speckle-limited (§8.10) |
| **Dwingeloo** | monostatic self-receive | 34 | 53 | 23 | no |

Of the 114 good Stockert co-pol captures, **110 were in the stacks**; the
gap was 3 captures (2025-09-10 20:35:24, Dwingeloo + Stockert chan0/chan1)
referencing the **`interp_zadoff-chu` TX waveform never published to the
site**, plus one marginal drop. That waveform is **now reconstructed** (below),
so the co-pol stack is **111 looks** (the recovered chan1 look: SNR 45,
solidly mid-pack; map delta negligible — rms 0.005 dB — as expected for a
structure-limited stack). (A corrupt-metadata pair, `stockert_radar …13_53_30`,
is a CW calibration capture, not science data — harmless, skipped automatically.)

**Reconstructing the `interp_zadoff-chu` TX waveform.** The base Zadoff-Chu
sequence is fully determined by the filename parameters (`l1500007`, `q1201`),
and its non-`interp` twin (`zadoff-chu-…-fzc50000-l1500007-q1201-30sec-1x`) is
published — so only the ×5 upsampling (50→250 ksps) was unknown. The Dwingeloo
monostatic capture's heavy TX-leakage is effectively a clean recording of the
transmitted signal; correlating three candidate upsamplings against it
disambiguates the method: `scipy.signal.resample` (FFT interpolation) wins
(|corrcoef| 0.936 / −9.1 dB vs linear 0.914 and repeat/ZOH 0.874), the leakage
peaking at exactly the `rx_start + 1.0 s` emit time. The reconstructed file
(`tx_signals/interp_zadoff-chu-…-1x.sigmf-*`, peak-normalized int16 — the
correlator uses TX phase only) is resolved automatically by `dea.load_observation`
with no code changes; verified by the Dwingeloo leakage peak (1.000 s) and the
Stockert lunar echo (3.387 s = 1.0 s emit + 2.39 s two-way bistatic light-time).
The same file also unblocks the Dwingeloo 20:35:24 capture for the monostatic
work below.

**Unused good data, by priority:**

- **ATA bistatic, 2025-09-16 (11 dual-pol looks)** — a second receiver
  (Allen Telescope Array) viewing the same bounces as Stockert. Now a
  first-class station: `ATA` added to `observatories.defs` (NAIF 399997) and
  the kernels regenerated; `registration_stability.py --station ata` and
  `stack_maps.py --run-prefix registration_runs_ata --rx-name ATA` ingest it.
  Geometry verified — at the bounce epoch the Moon is +61° at ATA, +15°/+14°
  at Dwingeloo/Stockert, and the SRP solver converges (two-leg light time
  2494.7 ms). The observatory's own `correlated/combined_ata_radar_…` image
  confirms real echoes. **As a standalone map ATA does not contribute:** 8
  co-pol looks from a single ~12-min pass over one bistatic geometry, at lower
  G/T than Stockert (below), leave the stack dominated by the central specular
  glint and the Doppler-equator degeneracy stripe with no surface detail. Its
  value is instead as an independent **registration cross-check** — a second
  receiver, with its own SDR/clock/pointing, viewing the same bounces.
  - *Cross-check (`ata_stockert_crosscheck.py`, 2025-09-16 co-pol):* build a
    single-session stack per receiver, band-pass both, and cross-correlate.
    Speckle is receiver-specific (different site → different realization) and
    correctly decorrelates, so agreement is sought in the larger-scale
    reflectivity by sweeping the low-pass scale. Result: residual offset
    **≤0.3°** (no gross misregistration) with positive correlation **+0.29**.
    Calibrated against a same-receiver control — Stockert's own looks split
    even/odd into two ~10-look sub-stacks agree at **+0.80** — ATA recovers
    **37%** of the achievable agreement. **Verdict: consistent at the gross
    level (sub-0.3° pointing, positive correlation), but the agreement is too
    diffuse for a high-significance lock.** ATA confirms the registration is not
    grossly wrong; it cannot independently certify it to sub-degree. Figure:
    `results/REGISTRATION/ata_stockert_crosscheck.png`.
  - *Why ATA falls short of the control — not geometry, not atmosphere, not
    look count.* At the bounce epoch ATA was at **+61°** elevation (nearly
    overhead) while Dwingeloo (TX) and Stockert sat at **+15°/+14°**. So ATA is
    *not* the horizon-scraper; the European stations are. The two-way slant
    path is shorter for ATA (TX 3.9 + RX 1.1 = 5.1 airmasses) than for Stockert
    (3.9 + 4.2 = 8.1), and the +15° Dwingeloo uplink is common-mode to both
    receivers — so atmosphere actually favors ATA and cannot explain its
    deficit. The bistatic angle at the surface is negligible (~1°; Earth is a
    point from the Moon). Geometry contributes only a minor penalty (apparent
    SRP rotation rate ~20% lower → ~20% narrower Doppler bandwidth; degeneracy
    stripe ~2× wider, 4.9% vs 2.1% of the disk). The split-half control reaches
    +0.80 with only ~10 looks, so ATA's 8 looks are *not* the limiter. The
    residual gap is **ATA's lower G/T** — per-look tone SNR 25–38 vs Stockert
    32–88 *despite* the cleaner slant path, i.e. smaller effective aperture /
    higher system temperature — plus the genuinely independent systematics the
    cross-check is designed to expose.
  - *Empirical check that elevation/atmosphere is not the driver (Stockert
    co-pol tone SNR by session vs Dwingeloo TX elevation):*

    | Session | n | Stockert SNR (med, range) | Dwingeloo TX elev | Stockert elev |
    |---|---|---|---|---|
    | 2025-06-21 | 34 | 58 (4–148) | +45° | +46° |
    | 2025-09-10 | 11 | 51 (40–60) | +14° | +14° |
    | 2025-09-11 | 44 | **87** (60–185) | +30° | +30° |
    | 2025-09-16 | 21 | 55 (32–88) | +15° | +14° |

    Stockert's SNR does not track elevation: the best session (09-11) is at
    *mid* elevation +30°, beating the +45° session, and the low-TX-elevation
    09-16 session (median 55) matches the high-elevation 06-21 (58). At 1.3 GHz
    the absorption difference between 14° and 45° is only ~0.1 dB, so session
    spread is set by TX power / pointing / conditions, not airmass. Since the
    Dwingeloo uplink is common-mode, the low 09-16 TX elevation did not depress
    *either* receiver — confirming ATA's deficit is per-receiver G/T, not the
    shared geometry. (ATA's exact array configuration — number of phased 6.1 m
    dishes — is not in the sigmf metadata, which carries only SDR-level
    `vrt:rx_gain`/`reference`/`pps` fields; it would have to come from Thomas.)
- **Dwingeloo monostatic self-receive (~110 good looks, all 3 sessions)** —
  different geometry (Dwingeloo→Moon→Dwingeloo) with TX-leakage handling;
  genuine future work.
- **2025-03-04 + 2025-03-11 (`thomas/sdr-eme/`, ~446 captures)** — earlier
  monostatic campaign at **1 Msps** with experimental waveforms (rnd-phase,
  bpsk, 2-tone, `--dt-trace`); needs loader/window changes. Lower value.
- **3× `interp_zadoff-chu` captures (2025-09-10)** — **resolved.** TX waveform
  reconstructed from the published twin + Dwingeloo TX-leakage (above); the
  Stockert chan1 look is now in the co-pol stack (110→111), and chan0/Dwingeloo
  are recoverable via the same file for the cross-pol/monostatic stacks.
