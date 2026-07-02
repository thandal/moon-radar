# Pipeline walkthrough: one look, end to end

The narrative the old notebooks used to provide: what happens to a single
capture, in order, with the function that does it. `REPORT.md` is the
reference for conventions, measured numbers, and the error budget; this
document is the thread that ties the code together. File references are
`module.py:function`.

## 0. Setup

`spice_setup.furnsh_kernels()` — clears and furnishes the standard kernel set
(DE440s, lunar PA frame, Earth orientation, station ephemerides + radii).
`doppler_equator.py` deliberately has no import-time SPICE side effects.

## 1. Load a capture

`doppler_equator_alignment.load_observation` — reads the RX sigmf file and
the TX sigmf it names in `core:description`. Yields complex64 samples,
sample rate (0.25 Msps), carrier (1299.5 MHz), and the RX start epoch.

**Timing convention** (the single most important convention in the system):
the TX file's `core:datetime` is the waveform *generation* time, not the
emission epoch — off by hours-to-months. Emission is defined as
`rx_start + 1.0 s`, plus the measured per-look correction from step 2.
(REPORT §3.1.)

## 2. Measure the per-look chain offsets

`freq_offset_hunt.measure_offset` — the **product method**: form
`y(t) = rx(t) · conj(tx_compensated(t))`. With perfect geometry and clocks the
specular echo collapses to DC; what remains is

- a **delay offset** (typ. +35–45 µs: SDR pipeline latency + PPS ambiguity +
  cable delays), found by searching integer sample shifts (±40), and
- a **frequency offset** (the specular line centroid, typ. tens of mHz:
  TX/RX reference-clock rate difference — physically the Stockert rubidium
  vs the Dwingeloo H-maser, REPORT §4).

RFI is spread over the 50 kHz ZC bandwidth by `conj(tx)`, so the echo is the
only narrowband feature — this is why the measurement is robust.

## 3. Geometry for this epoch

`doppler_equator.py` — all SPICE geometry:

- `specular_point_bck/fwd` — the SRP, the minimum two-leg light-time point
  on the ellipsoid (vectorized tangent-plane zoom).
- `moonSRP_DLT_FWD/BCK` — SRP light time and dlt (= d lt/dt; Doppler =
  −dlt·f₀). Derivatives use **wide stencils** (half the window) because SPICE
  light times are quantized at ~2×10⁻¹¹ s — a short baseline would inject
  ~0.25 Hz of Doppler noise. (REPORT §2.)
- `apparent_station_positions` — light-time-consistent station positions
  anchored at the SRP; lets full-disk delay/Doppler fields be evaluated in
  numpy ~600× faster than per-point SPICE (verified to 12 ns / <1 mHz).

## 4. Build the delay-Doppler image

`doppler_equator_alignment.compute_dd_image`:

1. **TX resampling** — map TX sample times onto the RX timeline through the
   forward light times, then interpolate the *unwrapped phase* linearly and
   reconstruct `exp(i·φ)`. Phase interpolation (not complex interpolation)
   preserves |tx| = 1 and the ZC autocorrelation. All time arithmetic is done
   relative to capture start, not in absolute ET (~8×10⁸ s), where float64
   spacing would swamp sub-sample shifts (one absolute-ET difference per
   look, `tx_start − rx_start`, is absorbed by the timing offset). (REPORT §2.)
2. **Doppler compensation** — remove the SRP dlt and its linear rate (chirp),
   plus the measured chain offset from step 2, so image rows stay labeled by
   geometric dlt.
3. **Correlation** — chunked GPU (CuPy) correlation over 3000 Doppler rows ×
   ~2920 delay columns; the delay window leads the SRP delay by 20 samples so
   the echo onset sits inside it. Returns `log_A` plus the axes and the
   compensated SRP rate (`dlt_rate_srp` — passed onward so the projection's
   rate correction matches the compensation exactly).

## 5. Rim calibration (per-look δ)

`doppler_equator_alignment.measure_rim_offset` — measures the signed offset
between the predicted Doppler axis and where the echo actually sits, as the
mean of the up-rim and down-rim offsets, iterated to convergence. The
half-difference (**rim spread**) diagnoses Doppler-rate/curvature model error.
Post-calibration residual ±47 mHz; estimator validated synthetically by
`validation/scripts/validate_rim_calibration_stress.py` (bias <0.5 mHz nominal). Cross-pol looks inherit δ
from their co-pol twin (same clocks, same geometry). (REPORT §3.2.)

## 6. Project to the lunar surface

`doppler_equator_alignment.lunar_projection` — for every HEALPix pixel
(nside 400, 0.147° ≈ 4.5 km), evaluate its window-averaged (delay, Doppler)
from the anchored fields and sample the DD image there. A surface point's
energy lands at `dlt_eff = mean(dlt) − rate_SRP·T/2` (REPORT §2). Pixels
whose DD cell is shared by many surface pixels (multiplicity > 3× median) are
masked — the **degeneracy mask** covering the stripe and SRP bloom. The
surface points carry LOLA topography (`moon_surface_points(use_dem=True)`,
sphere + `lola_dem/` elevation; `./fetch_lola_dem.sh` once to enable, falls
back to the ellipsoid without it) — this removes the dominant ±7-delay-px
mapping systematic (REPORT §8.4, validated by
`validation/scripts/validate_lola_dem_projection.py`). The
SRP solver and the rim/equator curves stay on the smooth ellipsoid: the
minimum-light-time zoom needs a convex surface, and the rim calibration is
differential in Doppler where terrain is second-order.

## 7. Batch over all captures

`registration_stability.py` — the driver: per channel, runs steps 1–6 for
every capture (`process_file`), recording all per-look corrections, SRP
parameters (including `srp_elevation_km`/`srp_topo_delay_us` from the DEM),
and quality metrics in `registration_runs_{chan0,chan1}.csv`. The current
220-look LOLA-DEM run lives in `results/LOLA_DEM_REGISTRATION/`
(`results/REGISTRATION/` is the pre-DEM baseline, REPORT §8.4).
`recover_railed.py` patches the CSV for captures whose delay offset exceeded
the standard search window.

## 8. Stack

`stack_maps.py` — gates looks (tone SNR ≥ 15 co-pol, non-railed shifts),
divides out an empirical scattering law (median gain-normalized intensity vs
cos incidence), solves least-squares session offsets from band-passed
(0.3°–2.5°) masked cross-correlations (`registration_analysis.xcorr_offset`),
and forms incoherent linear-intensity stacks per channel and dual-channel.
Closed-loop registration residual: 0.009–0.025° per channel (≲1 km,
sub-pixel; unchanged by the DEM — terrain parallax is common-mode across
sessions). (REPORT §5, §8.4.)

## 9. Verify

`test/test_pipeline_consistency.py` — the gate for the
geometry/correlation/tone core (9 checks): SRP solver vs Nelder-Mead
reference, anchored fields vs exact SPICE, chunked GPU correlation vs loop,
synthetic tone recovery, and measure→correct closure on real data (residuals
≤0.03 samples / ~13 mHz). If you touch that math, run it; if it is green,
the optimization agrees with the reference. It does not exercise rim
calibration, `lunar_projection`, or stacking — the registration sign
conventions are pinned by `test/test_registration_conventions.py` (below)
and the physics by the validation suite (`validation/`). (REPORT §7.)

`test/test_registration_conventions.py` — pins the registration sign
conventions (REPORT §5): `xcorr_offset`, `grid_map`, and `shift_intensity` are
each validated in isolation against synthetic ground truth, plus an end-to-end
check that the closed-loop global sign is `+1`. Run it if you touch any of
those three — a unit-level flip fails here, whereas the production `±` check
would silently absorb an even-parity double flip.
