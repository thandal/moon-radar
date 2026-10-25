# Scratch analyses, 2026-10-20/21

The working scripts behind figures quoted in REPORT §3.1, §6 and §8, kept
so that those figures have a source. They are historical:

- They ran on products made **before the 2026-10-21 fixes** (the timing
  search that could lock onto a later echo peak, delays measured from the
  smooth sphere, median-based stacking statistics, the harmonic-mean cell
  area); every figure they produced must be re-measured on regenerated
  products before it is relied on.
- They use interfaces that have since changed (for example
  `stack_maps.look_linear_intensity`, `registration_analysis.nanstack`,
  `xcorr_offset` with an exclusion radius, the old `rim_seed_search` result)
  and hard-coded paths under `/tmp/eve_check` and `/var/tmp`. They will not
  run as they are.

| REPORT figure | Script |
|---|---|
| §3.1: 06:37:24 specular peak at shift 7 vs strongest at 35 (and 06:38:25's peak 1.2 chips after the edge, cited in `freq_offset_hunt.leading_peak`) | `delay_diag.py` |
| §3.1: eleven 2025-09-11 looks with a noise floor above 82–99 % of their own map, entering the stacks inverted | `noise_floor_check.py` (reproduced from the inline check run that day) |
| §6 scattering law: single-look medians negative beyond ~80° (51–64 % of pixels ≤ 0), mean positive | `law_check.py` |
| §6 DD-cell sampling: 4×4 points per pixel, 3.4× / 3.1× signal-to-noise at 12–20 km | `compare_ss.py`, `before_after.py` |
| §6 Doppler drift within a look: 28 / 57 mHz drift; focusing prototype comparisons (0.95–0.97; 4–22 %; sign flip 0.8× / 0.67×; mirror points 0.77 → 0.70) | `trace_doppler.py`, `drift.py`, `focus_proto.py`, `focus_proto_flipped.py`, `make_focused_run.py`, `compare_focus400.py`, `power_focus400.py`, `closeup_sign400.py`, `validate_focus.py` |
| §6 cell area and noise floor: brightness ramp with distance from the Doppler equator; half-stack comparison, 12 % lower at 12–40 km | `trace_area.py`, `check_area.py`, `compare_radiometry.py`, `radiometry_parts.py`, `trace_refine.py` |
| §6 stack weights and mirror ghosts: speckle-only disagreement, ghost leakage on a synthetic surface | `ghost_weights.py`, `trace_maps_helpers.py`, `tiles.py` |
| §8.12 / waveform notes: roots passing the ghost and coupling rules | `zc_both.py` |

The other scripts are intermediate checks from the same two days (map
widths, crops, registration offsets between stations, DEM loader checks).
