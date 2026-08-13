# Zadoff–Chu Signal Check Process

This runbook validates a proposed Zadoff–Chu sequence before it is used for a
lunar observing session. It covers three progressively stronger checks:

1. numerical ambiguity testing;
2. end-to-end RF loopback or transmitter-leakage testing; and
3. an interleaved lunar A/B observation.

The central question is not merely whether an ideal ZC has analytic
delay–Doppler coupling. It is whether any significant delay displacement,
peak loss, broadening, or sidelobe increase remains **after applying the same
Doppler compensation and matched filtering as the production pipeline**.

## Candidate selection

Use `recommend_zc.py` to obtain prime lengths and coprime roots near the desired
look duration:

```bash
conda activate .conda/
./recommend_zc.py \
  --chip-rate 250000 \
  --look-duration 30 \
  --doppler-span-hz 19 \
  --max-coupling-cell 0.05
```

At minimum, compare these configurations:

| purpose | chip rate | N | q |
|---|---:|---:|---:|
| existing production reference | 50 kchip/s | 1,500,007 | 1,201 |
| lower-coupling root at the existing rate | 50 kchip/s | 1,500,007 | approximately 11,401 |
| proposed full-rate signal | 250 kchip/s | 7,500,013 | approximately 11,401 |

Always use the exact output of `recommend_zc.py`, rather than copying an
approximate value from this document. A valid pair must satisfy
`gcd(N, q) = 1`.

For the odd-length convention

```text
x[n] = exp(-j pi q n(n+1)/N),
```

the local ambiguity-ridge displacement across a Doppler interval `delta_f` is

```text
delta_chips = N delta_f / (q chip_rate).
```

This is a useful design estimate, but it is an uncompensated, ideal
discrete-time result. It is not by itself an operational pass/fail result.

## Establish the relevant Doppler interval

Use the residual Doppler range across the lunar disk after compensating the
reference point, not blindly the absolute carrier Doppler or a fixed symmetric
range.

For each planned geometry, record:

- minimum residual lunar Doppler;
- maximum residual lunar Doppler;
- peak-to-peak span; and
- location of the compensated reference within that span.

If the residual interval is truly 19 Hz and centered on the reference, test
approximately -9.5 to +9.5 Hz. If it is asymmetric, test the actual asymmetric
endpoints. Include some margin beyond both endpoints.

## Stage 1: numerical ambiguity test

### Procedure

For every candidate `(N, q)`:

1. Generate the ZC at the intended chip rate.
2. Apply the intended interpolation, pulse shaping, and recording sample rate.
3. Apply a grid of artificial frequency shifts spanning the residual lunar
   Doppler interval:

   ```text
   y_f[n] = x[n] exp(j 2 pi f n / sample_rate).
   ```

4. Correlate each shifted signal with the unshifted reference using the same
   conventions as the production matched filter.
5. Estimate correlation-peak delay with sub-sample interpolation.
6. Record peak position, peak amplitude, main-lobe width, PSLR, and ISLR at
   every Doppler.
7. Repeat the test after applying the production bulk and Doppler-rate
   compensation.
8. Fit peak delay against Doppler and compare its slope with the analytic
   prediction.

Prefer importing the production correlation and compensation functions over
building a separate approximation. This catches sign, normalization,
resampling, windowing, and boundary-handling differences.

The Doppler grid should include both endpoints and zero. A 1 Hz step is a
reasonable first scan; refine around unexpected motion, loss, or sidelobes.

### Record

For each candidate, save:

- maximum delay displacement in chips and delay cells;
- fitted coupling slope in chips/Hz and microseconds/Hz;
- correlation peak loss relative to zero Doppler;
- main-lobe broadening relative to zero Doppler;
- peak sidelobe ratio (PSLR);
- integrated sidelobe ratio (ISLR);
- uncompensated results;
- compensated production-path results; and
- plots of peak delay, loss, and width versus Doppler.

The existing analytic calculation in `quant_waveform_scratch.py` is a useful
cross-check, but it does not replace the production-path numerical test.

## Stage 2: RF loopback or TX-leakage test

This stage measures distortion introduced by the DAC, SDR filters, analog RF
chain, ADC, receiver filters, clocks, and resampling.

### Safe acquisition

Preferred sources, in order:

1. a cabled loopback with suitable attenuation and DC blocking;
2. a dedicated low-level monitor output; or
3. a strong transmitter-leakage recording such as the Dwingeloo leakage path
   previously used in this project.

Never connect a transmitter output directly to a receiver input. Establish the
required attenuation and power limits from the hardware specifications before
making a cabled connection.

Use the exact intended production settings, including:

- sample and chip rates;
- SDR master-clock rate;
- interpolation and decimation;
- analog and digital filter bandwidths;
- transmit amplitude;
- center frequency; and
- waveform file representation and scaling.

### Procedure

For each candidate:

1. Transmit and record the waveform through the full proposed chain.
2. Check for clipping, dropped samples, discontinuities, and timing slips.
3. Estimate and remove the bulk carrier offset.
4. Correlate the recording with the original ideal waveform.
5. Artificially add the planned residual lunar Doppler offsets to the
   recording and repeat the Stage 1 scan.
6. If practical, derive a clean recorded reference and repeat the scan using
   both ideal and recorded references.
7. Run the production compensation and matched-filter path on the recording.

### Interpret ideal-versus-recorded matching

| result | likely interpretation |
|---|---|
| ideal reference poor, recorded reference good | RF/filter response dominates; consider a calibrated reference or predistortion |
| both show the same root-dependent delay motion | intrinsic ZC delay–Doppler coupling dominates |
| neither shows significant compensated residual | analytic uncompensated coupling is not an operational limitation |
| both show broadening or elevated sidelobes | investigate clipping, filtering, clocking, discontinuities, or sample-rate mismatch |

Also inspect:

- occupied spectrum and spectral regrowth;
- passband amplitude and group-delay flatness;
- band-edge attenuation;
- correlation main-lobe shape;
- PSLR and ISLR;
- fractional-delay stability; and
- sensitivity to realistic carrier and sample-clock errors.

This stage is especially important when chip rate equals sample rate, because
the waveform then relies on the usable response near the full complex-Nyquist
band and leaves little transition-band margin.

## Stage 3: interleaved lunar A/B observation

The final check uses actual lunar echoes. Interleave the current and candidate
signals so changing elevation, libration, propagation, and equipment drift do
not masquerade as a waveform effect. For example:

```text
A: current N/q, 30 seconds
B: candidate N/q, 30 seconds
A: current N/q, 30 seconds
B: candidate N/q, 30 seconds
```

Obtain several repetitions rather than relying on one pair. Keep transmit
power, receiver gain, sample rate, master-clock rate, filter settings, pointing,
and processing identical except for the waveform under test.

Process every look through the same calibration, DD imaging, projection, and
quality gates. Compare:

- product-tone and rim-calibration SNR;
- delay–Doppler lunar-rim sharpness;
- delay position versus residual Doppler;
- residual rim spread after calibration;
- peak loss and main-lobe width;
- projected-map sharpness and artifacts;
- registration offsets and closure;
- sensitivity to intra-look clock wander; and
- rejected-look rate.

Because individual looks contain different speckle realizations, summarize
several A and B looks and report uncertainty. Do not interpret a single bright
or faint cell as a waveform difference.

## Suggested acceptance criteria

A candidate is ready for production when all of the following hold through the
actual production processing path:

- residual delay motion is at most 0.05–0.10 delay cell across the planned
  lunar Doppler interval;
- main-lobe broadening is below 5%;
- correlation peak loss is below 0.2 dB;
- RF-chain sidelobes are not meaningfully worse than the current signal;
- calibration SNR is no worse than the current root;
- no new clipping, discontinuity, or filter-edge behavior appears; and
- the lunar A/B test shows no degradation in registration or rejection rate.

These are engineering starting points, not immutable science requirements.
Relaxing a threshold should be tied to a demonstrated map- or inversion-level
tradeoff.

## Results record

For reproducibility, retain the following with every test:

- date, operator, stations, and hardware path;
- exact `N`, `q`, chip rate, sample rate, and look duration;
- waveform-generation command and code revision;
- SDR master-clock, filter, frequency, gain, and amplitude settings;
- raw or referenced capture filenames;
- predicted residual lunar Doppler interval;
- software and RF test tables and plots;
- lunar A/B look identifiers;
- production pipeline revision and arguments; and
- final pass/fail decision with any waived criteria.

Do not replace raw captures or generated waveform metadata with summary plots;
the full chain may need to be re-evaluated after a processing change.
