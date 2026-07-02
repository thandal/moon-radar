# Data-machine checklist

Work that needs the full untracked assets (`results/`, `data.camras.nl/`,
`lola_dem/`) and could not be executed in the 2026-07-01 review/fix
session. That session DID run, on downloaded kernels + a local GPU: the
SRP-velocity proof (`validation/logs/srp_velocity.log`, PASS), the
registration-conventions test (7/7), the rim-calibration stress test
including the new fine-bin inward cases (`validation/logs/
rim_calibration_stress.log` — adaptive window bias ≤0.06 mHz; asymmetry
worst case ≈4 mHz), the waveform ISLR regeneration
(`validation/logs/signal_processing.log` — reproduces REPORT §8.12 exactly),
and the reworked libration scan. Everything below re-verifies the changes
against real data or fills the remaining evidence gaps.

## 1. Re-verify after the code changes (do these first)

- [ ] `test/test_pipeline_consistency.py` — bounds were tightened
      (closure 0.5 samples / 25 mHz). If green with margin, pin closer to the
      measured 0.03 samples / 13 mHz.
- [ ] `test/test_lola_dem.py` and `test/test_registration_conventions.py`.
- [ ] One-look end-to-end sanity: `process_file` on a strong 09-16 capture;
      confirm δ, timing offset, and map match the frozen CSV row (the rim
      scan-window and converged-spread changes should move δ only at the
      few-mHz level on normal-bin looks; `rim_spread_hz` is now the
      converged-pass value, old value preserved as `rim_spread_first_hz`).
- [ ] Re-run the 2025-09-11 fine-bin morning looks specifically: the adaptive
      inward rim window (±80 mHz at 1.8 mHz bins ≈ 42 rows vs the old 10) is
      designed for exactly these; check the formerly uncertified 08:05:44
      look and REPORT §3.2's caveat.
- [ ] Full validation suite: `validation/scripts/run_validation_suite.py`
      (now includes `validate_srp_velocity.py`, both bootstrap channels, and
      the fine-bin inward rim-stress cases — the latter also demonstrate the
      old window's censoring via the `finebin_inward_legacywin` variant).
- [ ] Commit the small `validation/results/*.json/csv` summaries so REPORT
      numbers are traceable in git (root cause of most evidence-chain
      findings in the review).

## 2. Libration planning (before requesting telescope time)

- [ ] Re-run `predict_libration_opportunities.py` — with `results/…/
      registration_runs_chan*.csv` present it automatically switches from
      reconstructed session spans to per-look coverage axes. Update the
      LIBRATION_ANALYSIS.md table if the ranking moves.
- [ ] Check whether any *recorded look* was hit by the legacy axis glitch:
      compare `srp_velocity_analytic` vs the legacy 1 s FD at every look
      epoch in the CSVs (the proof found a 22.7° glitch at 2025-09-10T18:05,
      between looks; a hit look would show an inflated rim spread).
- [ ] Validate the 4v/λ span estimator against the per-look *measured*
      limb-to-limb spans (REPORT §1 quotes 5–24 Hz measured; the estimator
      max at session anchors is ~18 Hz — find where 24 Hz occurs and confirm
      the estimator tracks it).

## 3. Evidence gaps from the review (new scripts, ready to run)

- [ ] `validation/scripts/validate_speckle_floor.py` — substantiates (or
      corrects) the "structure-limited, residual speckle ~13%" claim
      (REPORT §5/§6), and probes structure-vs-artifact floor via the
      split-half correlation and `--session` geometry-freeze mode.
- [ ] `validation/scripts/validate_absolute_registration.py` — the absolute
      selenolocation tie (stack vs LOLA slope proxy). If it locks
      (signif ≥ 1.5), add the absolute offset to REPORT §5; if not, try the
      dual scatnorm map / a shaded-relief reference.
- [ ] Bootstrap the cross-session closure (extend
      `validate_registration_bootstrap.py`): resample looks within each
      session, re-run the session solve, and put an error bar on the
      0.009–0.025° closures (the review estimated the noise floor at
      ~0.02–0.03° median from the existing half-stack bootstrap).
- [ ] Cross-pol δ inheritance spot-check: run the rim estimator with relaxed
      gates on the strongest cross-pol looks and compare against the
      inherited co-pol δ (REPORT §3.2 assumption, currently untested).
- [ ] Look-count recount: one authoritative count from the current CSVs
      (REPORT §1/§5/§8.3/§10 and README currently say 110/220 with the
      111th interp look noted; replace the bracketed note in REPORT §1).

## 4. Clock/timing items (need operator data)

- [ ] Obtain Thomas's logged per-site frequency offsets (both sites were on
      Rb for ≥1 dataset, per `email_timing_discussion.txt`) and compare with
      the rim-measured per-look δ by session — a genuinely independent
      validation of the whole δ chain, and a check of REPORT §4's
      "all Stockert Rb" attribution.
- [ ] Incorporate `vrt:cal_time` (PPS-sync epoch) into the timing model:
      ~5.5e-11 sample-clock drift after sync ⇒ ~2.4 µs over the 12 h
      session — then re-test the §8.4 SRP-terrain correlation (+0.50,
      session-driven) against the June single-vs-dual-channel clock-rate
      change as the competing explanation.

## 5. Optional / deferred

- [ ] `AB_COR "LT" → "CN"/"XCN"` in `doppler_equator.py`: strictly better
      light-time convergence, ~10–30 ns effect, absorbed by timing
      calibration. Deferred in the fix session to avoid perturbing numerics
      without the consistency suite; flip it, run the suite + one-look
      closure, keep or revert.
- [ ] `intra_look_drift.py`: consider passing the new adaptive rim capture
      window (`delta_capture_dlt`) — it currently keeps the legacy fixed
      window (harmless: its looks are normal-bin), and re-measure the
      §4 intra-look wander over all 06-21 looks rather than the default 6.
- [ ] Kernel note: this session populated `spice_kernels/` from NAIF
      (2026-07-01 EOP: high-precision coverage to 2026-09-11) and rebuilt the
      observatory kernels with PINPOINT 3.3.0. The data machine's existing
      kernels are unaffected; refresh `earth_latest_high_prec.bpc`
      periodically for planning runs.
