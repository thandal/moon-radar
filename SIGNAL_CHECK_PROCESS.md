# Testing a proposed Zadoff–Chu waveform

Use this runbook before changing the transmit waveform. It tests whether
delay motion, peak loss, broadening, or sidelobes remain after the production
Doppler compensation and matched filtering. The August 2026 work added the
parameter-selection tool and this procedure; completed RF and lunar A/B
results are not recorded here.

## Select candidates

From the repository root, use the project environment:

```sh
.conda/bin/python recommend_zc.py \
  --chip-rate 250000 --look-duration 30 \
  --doppler-span-hz 19 --max-coupling-cell 0.05
```

The tool chooses prime code lengths `N` near the requested duration and roots
`q` coprime with `N`. For the odd-length convention
`x[n] = exp(−jπ q n(n+1)/N)`, its local ambiguity-ridge estimate is:

```text
delay motion in chips = N × Doppler interval / (q × chip rate)
```

This estimates uncompensated delay–Doppler coupling: frequency offset can
move the apparent correlation peak in delay. It is a design screen, not a
measurement of the compensated pipeline.

Compare at least these candidates:

| Purpose | Chip rate | N | q |
|---|---:|---:|---:|
| Existing reference | 50 kchip/s | 1,500,007 | 1,201 |
| Lower coupling at the same rate | 50 kchip/s | 1,500,007 | 11,401 |
| Proposed full-rate waveform | 250 kchip/s | 7,500,013 | 11,401 |

The final row is the tool's verified output for the command above. Recalculate
parameters if duration, span, or tolerance changes. The existing reference's
ideal motion is about 0.47 delay cell across 19 Hz; both candidate roots
reduce that estimate to about 0.05 cell.

Use the actual residual Doppler interval across the Moon after reference-point
compensation. Record its minimum, maximum, and the reference's position within
it. A 19 Hz span means −9.5 to +9.5 Hz only when centered; include margin
beyond the actual endpoints.

## 1. Numerical test

For each candidate, generate the signal at the intended chip rate and apply
the proposed interpolation, pulse shaping, and recording rate. Inject trial
frequency shifts across the residual lunar interval:

```text
y_f[n] = x[n] × exp(j 2π f n / sample_rate)
```

Use the production correlation functions and compare results before and after
production bulk/rate compensation. Include zero and both endpoints; a 1 Hz
step is a starting point, with finer sampling around unexpected behavior.
Estimate peak delay between samples and fit its slope against frequency.

Record delay displacement, peak loss, main-lobe width, peak sidelobe ratio
(PSLR), and integrated sidelobe ratio (ISLR) at each frequency. Plot both
uncompensated and compensated results. The ideal calculation in
[quant_waveform_scratch.py](quant_waveform_scratch.py) is a cross-check; it
does not exercise the full processing path.

## 2. Radio-chain test

Record each candidate through the intended transmit/receive chain using an
attenuated cabled loopback, a low-level monitor output, or strong TX leakage.
For a cabled connection, establish attenuation, DC blocking, and receiver
power limits first; never connect a transmitter output directly to a receiver.

Use the intended sample and chip rates, master-clock rate, filters, gains,
frequency, and waveform scaling. Check for clipping, dropped samples, timing
slips, spectral regrowth, and attenuation near the band edges. Full-rate
250 kchip/s at 250 ksamples/s especially needs a filter-response check because
it leaves little transition-band margin.

Remove the bulk carrier offset and repeat the numerical frequency-shift scan
on the recording. Where possible, compare matching against the ideal waveform
and against a clean recorded reference:

| Result | What to investigate |
|---|---|
| Ideal reference poor, recorded reference good | Filter/RF distortion; a calibrated reference may help. |
| Both show root-dependent delay motion | Residual ZC coupling. |
| Both broaden or develop sidelobes | Clipping, filtering, clocking, discontinuities, or sample-rate mismatch. |
| Neither shows significant compensated residuals | Ideal uncompensated coupling may not limit this setup. |

## 3. Lunar A/B test

Alternate the existing and candidate signals over several repetitions, for
example A/B/A/B with 30 s per look. Hold transmit power, receiver gain,
pointing, clock settings, and processing fixed. Document any rate or filter
changes needed by the candidate.

Process all looks with the same calibration, imaging, projection, and quality
cuts. Compare calibration strength, rim sharpness and spread, delay motion,
map detail, registration, and rejection rate. Summarize several looks with
uncertainty: a single bright or faint echo can simply be speckle.

## Acceptance and record keeping

Suggested starting thresholds, to be evaluated through the actual production
path, are:

- Residual delay motion ≤0.05–0.10 cell across the planned interval.
- Main-lobe broadening <5% and peak loss <0.2 dB.
- Sidelobes and calibration strength no worse than the existing signal.
- No new clipping, discontinuities, filter-edge problems, or increase in
  rejected looks; no degradation in lunar registration.

These are engineering targets, not established science requirements. Record
any changed threshold and its demonstrated effect on the maps.

Retain exact waveform parameters and generation command, hardware and SDR
settings, code revision and processing arguments, predicted Doppler interval,
raw capture identifiers, metric tables, plots, and the acceptance decision.
Keep the captures and metadata as well as summary plots so later processing
changes can be tested against the same inputs.
