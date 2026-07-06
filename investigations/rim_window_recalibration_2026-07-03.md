# Rim scan-window recalibration (2026-07-03, data machine)

> [!NOTE]
> **Directory rename (2026-07-06):** the batch this note produced,
> originally written to `results/LOLA_DEM_REGISTRATION_RIMFIX/`, was blessed
> as the canonical run and renamed to **`results/LOLA_DEM_REGISTRATION/`**;
> the previous 2026-06-12 run it superseded is archived as
> **`results/LOLA_DEM_REGISTRATION_FROZEN_0612/`**. Path strings below and in
> the archival scripts under `rim_window_recalibration/` predate that rename
> (`_RIMFIX/` → `LOLA_DEM_REGISTRATION/`; the bare `LOLA_DEM_REGISTRATION/`
> those scripts read as the *frozen* baseline is now `_FROZEN_0612/`).

## What happened

`DATA_MACHINE_TODO.md` §1.3 — the one-look end-to-end sanity check after
the 2026-07-01 review fixes (that checklist was completed and retired
2026-07-04; see git history, results absorbed into REPORT) — predicted the
adaptive rim window would move δ "only at the few-mHz level on normal-bin
looks". The check **failed**: on the strong
09-16 14:06:11 look, δ moved from the frozen +64.2 mHz to +19.6 mHz
(−44.5 mHz). Everything upstream reproduced exactly (shift, tone centroid,
SRP topo delay, map valid sets), and re-measuring the same DD image with the
legacy fixed window reproduced the frozen δ (+64.4), so the change was
entirely the scan-window scaling.

Root cause of the wrong expectation: the DD image Doppler axis is a fixed
3000 rows across the disk, so the row pitch is limb-span/3000 ≈ **1.7–8 mHz
on every look** (5.5 mHz on 09-16, ~6 on 06-21, 1.8 on the 09-11 morning) —
not the ~25 mHz 1/T resolution. The 80 mHz capture target therefore rescaled
the scan **and the reference strips** on essentially all looks (×1.44 at
5.5 mHz pitch), and on real profiles (unlike the synthetic caustic, where
both windows are sub-0.1 mHz unbiased) strip placement moves the half-power
threshold by tens of mHz.

## Arbiter

The applied tone centroid `applied_df_hz` and the rim δ measure the same
chain frequency offset, so their sum (the total chain offset) must drift
smoothly across consecutive looks; look-to-look scatter of the sum is
estimator error. Two consecutive blocks were A/B'd on identical DD images
(`rim_window_variants.py` / `.csv`, `rim_window_ab_*.csv` here):

| window | block | fails | total-offset range | look-to-look RMS |
|---|---|---|---|---|
| legacy (10,50)/(30,90) rows | 09-16 (5.5 mHz pitch, 6 looks) | 0 | 10.8 mHz | 5.4 mHz |
| legacy | 09-11 morning (1.8 mHz, 10 looks) | 1 | 64.6 mHz | 31.6 mHz |
| adaptive 80 mHz (2026-07-01 fix) | 09-16 | 0 | 52.0 mHz | 32.3 mHz |
| adaptive 80 mHz | 09-11 morning | 0 | 25.2 mHz | 7.6 mHz |
| **55 mHz anchor + 12-pass cap** | 09-16 | 0 | **9.4 mHz** | **4.5 mHz** |
| **55 mHz anchor + 12-pass cap** | 09-11 morning | 0 | **8.6 mHz** | **4.6 mHz** |

(09-16 stats exclude 13:57:38, a badly faded tone centroid, −244 mHz — see
below.)

The winning geometry keeps the reference strips at their *physical* Doppler
placement (55…275 / 165…495 mHz from the rim — exactly the legacy row
geometry at its native 5.5 mHz pitch) and scales rows only at finer pitch.
Algebraically this is the existing production scaling rule with the capture
constant changed from 0.080 to **0.055 Hz**. Inward reach beyond the
single-pass capture comes from raising the convergence-loop cap 3 → **12**
(measured crawl ~15–30 mHz/pass; the early break keeps normal looks at 1–3
passes):

- 09-11 08:05:44 (formerly uncertified, rim_n 0): converges at pass 8,
  δ = +127.7 mHz, total +3.9 mHz vs block trend ~−5 mHz.
- 09-11 08:12:02 (formerly uncertified): δ = −47.9 mHz, total on trend.
- 09-11 08:08:32: legacy had censored a ~60 mHz error (frozen total
  −65 mHz); recovered to −4.8 mHz, on trend.
- 09-16 13:57:38 (tone centroid −244 mHz, true total ≈ −20): crawls from
  −229 to −24 mHz by pass 12 (frozen/legacy sat at −201, i.e. this look's
  frozen δ was always wrong; now nearly fully recovered).

## Changes applied

- `doppler_equator_alignment.py`: `rim_capture_dlt = 0.055 / f_hz`
  (was 0.080), iteration cap `range(12)` (was 3), comments updated.
- `validation/scripts/validate_rim_calibration_stress.py`: `recover_delta`
  mirrors production (0.055 / 12 passes); `finebin_inward` variant capture
  0.055.

## Consequences

- Frozen `results/LOLA_DEM_REGISTRATION` CSVs carry the legacy-window δ.
  On normal-pitch looks the new δ agrees at the few-mHz level; on the
  09-11 fine-bin morning looks δ changes by up to ~60 mHz (censoring
  removed) and the two formerly uncertified looks now calibrate.

## 222-look batch re-run (same day, `results/LOLA_DEM_REGISTRATION/` — then `_RIMFIX`)

- Chain-offset (applied_df + δ) health per session: >20 mHz outliers
  25 → 5 (chan1); rim failures 6 → 2; per-session total-offset MAD now
  2.7–3.9 mHz everywhere (09-10 was 11.1). The uncensored δ distribution:
  std 75 mHz, median |δ| 44 mHz, tails to ±225 mHz (16 looks >100 mHz) —
  matching Stockert's logged Rb 5.5e-11 better than the censored 47 mHz.
- Registration improves: every cross-session correlation lock rises
  (0.486→0.506, 0.439→0.461, 0.483→0.509); stack closed-loop
  0.009/0.011/0.025 → 0.007/0.007/0.026 (chan1/dual/chan0).
- Remaining known stragglers: 06-21 10:44:00 (tone SNR 4 — no-signal
  capture, excluded from stacks by the SNR gate) and 09-16 14:04:14
  (rim never measures, total +270 mHz — enters stacks with uncorrected
  tone-fade δ; candidate for a rim_n>0 stacking gate). 06-21 10:09:25 and
  10:14:34 improve but stay ~+50…+70 mHz off trend (partially corrected).
- Full validation suite green with the recalibrated parameters
  ("All validation steps completed", 2026-07-03 logs).

## Straggler resolution (2026-07-04)

- **09-16 14:04:14 rescued** by a coarse re-acquisition sweep
  (`rim_seed_search` in `doppler_equator_alignment.py`): when the gates
  fail at zero offset, trial offsets to ±12× the capture are scanned and
  the loop is seeded from the trial with the most gate-passing columns.
  Result: δ = −286.8 mHz, rim_n 407, chain total −16.8 vs trend −19.8 mHz,
  converged. No-signal captures stay unmeasured (gates reject noise on
  10:44:00); a healthy control look is bit-identical with the seed path in
  place (`seed_rescue_test.py` here).
- **The 06-21 trio (10:09:25 / 10:14:34 / 10:17:40) is NOT clock
  excursions and NOT a fixable estimator error — it is bursty intra-look
  frequency wander.** Evidence chain (scripts here): (1) the structure
  function of the chain offset over clean rim-verified looks is essentially
  FLAT at 5–7 mHz RMS from 30 s to 3.3 h (4–5×10⁻¹²) — a stable clock
  cannot produce ±50–70 mHz look-to-look excursions
  (`clock_excursion_structure_function.png`); (2) converging the estimator
  from a coarse-seeded start gives inconsistent second fixed points with
  low quality (`fixed_point_test.py`); (3) half-window rim fits show
  ~25 mHz drift WITHIN the trio's 66 s windows vs 1–8 mHz on interleaved
  controls at 0.8–2 mHz measurement noise (`trio_intra_look_drift.py`).
  The smeared rims leave the estimator's fingerprints: sign-flipped
  converged spread (+34…+42 vs session median −16) and depressed rim_n.
- **Session-relative rim-spread anomaly gate added to `stack_maps.py`**
  (threshold max(3×MADσ, 20 mHz) around the session median, keyed by
  capture stem so both polarization channels gate together): across all
  109 measured looks it flags exactly the trio. Stacks are now 107
  looks/channel (111 − 1 no-signal − 3 wander). The recovery path for
  gated looks is the REPORT §3 sub-window δ(t) recompensation.
- **Clock re-characterization** (REPORT §4 rewritten): per-look δ scatter
  (75 mHz std) is the tone centroid's fading error, not clock wander;
  clean chain stability is ~4–5×10⁻¹² flat from 30 s to hours; the logged
  Rb 5.5×10⁻¹¹ matches the session set-points (+5/+54/−1/−20 mHz), not
  the wander; the earlier "intra-look 10.6 mHz, √τ random walk" story was
  measurement noise plus the three burst looks.
- **Burst localized to the Stockert side** (2026-07-04,
  `leakage_wander.py` / `dwingeloo_leakage_wander.png` here): the
  Dwingeloo self-receive TX-leakage line (TX exciter vs Dwingeloo's own
  maser-referenced RX chain; no Moon, no path, no Stockert) is stable to
  **0.15–0.23 mHz RMS over the full 60 s** on the exact burst looks
  (10:09:25 / 10:14:34 / 10:17:40) — >100× below the ~25 mHz bistatic
  wander. This exonerates the TX exciter/synthesizer; a common-mode maser/
  White-Rabbit excursion is the only TX-side channel the test cannot see,
  and 2×10⁻¹¹ is ~100× that reference's class. With ionosphere and
  multipath already disfavored (mid-latitude midday TEC rates two orders
  short; 43° elevation, no on/off structure available from geometry), the
  burst is attributed to **Stockert's Rb reference or its disciplining /
  LO chain**. Independent confirmation available from operator logs:
  look for a Stockert reference steering/settling event ~10:09–10:18 UTC
  2025-06-21 (REPORT §8.13). Method note: the transmission does
  not loop the waveform file seamlessly, so the leakage analysis stays
  within one playback (60 s file for these looks; 30 s at 10:26:29).
- The legacy SRP-velocity axis glitch scan (also in this directory,
  `axis_glitch_scan.{py,csv}`) cleared all 111 look epochs: max
  analytic-vs-legacy axis deviation 0.124°, median 0.043°, speed ≤0.28% —
  no recorded look was affected; the analytic-SRP switch is not a factor in
  any δ change.
