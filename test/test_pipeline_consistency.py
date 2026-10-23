"""
Regression tests for the accelerated pipeline: every optimization must agree
with its exact/reference counterpart within stated bounds, and the
measure->correct loop must close on real data.

Run from the repo root:  .conda/bin/python test/test_pipeline_consistency.py
"""

import os
import sys

import numpy as np
import scipy.optimize
import cupy
import healpy as hp
from astropy import units as au

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import doppler_equator_alignment as dea
import freq_offset_hunt as foh

RX_TIME_UTC = "2025-09-16 13:23:26"
# The test data are Stockert captures of Dwingeloo's transmission.
TX, RX = "DWINGELOO", "STOCKERT"
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..",
                         "data.camras.nl/lunar-radar")
RX_FILE = (DATA_ROOT + "/2025-09-16/stockert_radar_2025_09_16_13_23_26"
           "_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta")

failures = []


def check(name, value, bound, unit=""):
    ok = value < bound
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value:.3e} {unit} (bound {bound:.0e})")
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------------------
# 1. Specular zoom search vs Nelder-Mead reference
# ---------------------------------------------------------------------------
rx_time = dea.csp.str2et(RX_TIME_UTC)
p_zoom = dea.specular_point_bck(rx_time, TX, RX)
x0 = dea.subpoint_average_guess(rx_time, TX, RX)
res = scipy.optimize.minimize(
    lambda x: float(dea.moonPointLightTime_BCK(rx_time, np.asarray(x), TX, RX)), x0,
    method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 250})
p_nm = dea.moon_surface_points(res.x)
lt_zoom = dea.moonPointLightTime_BCK(rx_time, p_zoom, TX, RX)
lt_nm = dea.moonPointLightTime_BCK(rx_time, p_nm, TX, RX)
check("specular zoom vs NM light time", abs(lt_zoom - lt_nm), 1e-11, "s")
check("specular zoom vs NM distance", np.linalg.norm(p_zoom - p_nm), 0.1, "km")

# ---------------------------------------------------------------------------
# 2. Anchored station field vs exact per-point SPICE (near side, incl. limb)
# ---------------------------------------------------------------------------
T = 30.0
v = np.array(hp.pix2vec(64, np.arange(hp.nside2npix(64)))).T
v = v[v[:, 0] > 0]
p = dea.moon_surface_points(v)
lt_exact = dea.moonPointLightTime_BCK(rx_time, p, TX, RX)
lt_end_exact = dea.moonPointLightTime_BCK(rx_time + T, p, TX, RX)
R_rx, R_tx, c = dea.apparent_station_positions(rx_time, TX, RX)
R_rx1, R_tx1, _ = dea.apparent_station_positions(rx_time + T, TX, RX)
lt_f = (np.linalg.norm(p - R_rx, axis=1) + np.linalg.norm(p - R_tx, axis=1)) / c
lt_end_f = (np.linalg.norm(p - R_rx1, axis=1) + np.linalg.norm(p - R_tx1, axis=1)) / c
check("anchored field lt error", np.abs(lt_f - lt_exact).max(), 5e-8, "s")
check("anchored field window-avg dlt error",
      np.abs((lt_end_f - lt_f) / T - (lt_end_exact - lt_exact) / T).max(), 2e-12)

# ---------------------------------------------------------------------------
# 3. Chunked GPU correlation vs naive per-row loop (synthetic)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(1)
n = 200000
rx_s = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype("complex64")
tx_s = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype("complex64")
f_shifts = np.linspace(-5, 5, 37)
fs_val = 250000.0
rx_g, tx_g = cupy.asarray(rx_s), cupy.asarray(tx_s)
fft_rx = cupy.fft.fft(rx_g)
tx_range = 1j * 2 * cupy.pi / fs_val * cupy.arange(n)
ref = np.zeros((len(f_shifts), 500))
for i, f in enumerate(f_shifts):
    cor = cupy.fft.ifft(fft_rx * cupy.conj(cupy.fft.fft(
        (tx_g * cupy.exp(f * tx_range)).astype("complex64"))))
    ref[i] = cupy.asnumpy(cupy.abs(cor[:500]))
t_norm = cupy.arange(n, dtype=cupy.float32) * cupy.float32(2 * np.pi / fs_val)
new = np.zeros_like(ref)
for i0 in range(0, len(f_shifts), 4):
    fc = cupy.asarray(f_shifts[i0:i0 + 4], dtype=cupy.float32)
    blk = tx_g[None, :] * cupy.exp(1j * fc[:, None] * t_norm[None, :])
    cor = cupy.fft.ifft(fft_rx[None, :] * cupy.conj(cupy.fft.fft(blk, axis=1)), axis=1)
    new[i0:i0 + len(fc)] = cupy.asnumpy(cupy.abs(cor[:, :500]))
check("chunked correlation vs loop", np.abs(new - ref).max() / ref.max(), 1e-5)

# ---------------------------------------------------------------------------
# 4. Tone measurement: synthetic ground truth (sign conventions included)
# ---------------------------------------------------------------------------
fs_val = 250000.0
n = int(30 * fs_val)
t = np.arange(n) / fs_val
# Constant-modulus signal chipped like the 50 kHz ZC waveform at 250 ksps (a
# random phase held for 5 samples: a correlation triangle one chip either
# side, as the real code's) and a WEAK echo in noise: in the strong-signal
# regime the median-normalized SNR is nearly flat versus shift and the test
# would not probe delay recovery at all.
CHIP = 5
phase = np.repeat(rng.uniform(0.0, 2 * np.pi, n // CHIP + 1), CHIP)[:n]
tx_synth = np.exp(1j * phase).astype("complex64")
TRUE_SHIFT, TRUE_DF = 9, 0.123
ramp = np.exp(1j * 2 * np.pi * TRUE_DF * t)
echo = np.roll(tx_synth, TRUE_SHIFT) * ramp
rx_synth = (0.1 * echo + (rng.normal(size=n) + 1j * rng.normal(size=n))).astype("complex64")
m = foh.measure_offset(rx_synth, tx_synth, fs_val, CHIP, max_shift=20)
del rx_synth
# Delay resolution for a weak echo is ~1 sample (SNR-vs-shift argmax on a
# ~chip-wide bump); this check guards sign/rough magnitude. The sub-sample
# validation is the real-data closure below (high-SNR specular line).
check("synthetic tone shift recovery", abs(m["shift_refined"] - TRUE_SHIFT), 1.5, "samples")
check("synthetic tone freq recovery", abs(m["f_centroid"] - TRUE_DF), 0.01, "Hz")

# 4b. The reason for the timing rule (freq_offset_hunt.leading_peak): a weak
# first echo -- the specular one, the nearest surface -- followed 4 chips
# later by an echo of three times its amplitude at the same tone (later delay
# rings also cross the Doppler equator). The timing must be the first
# echo's, resolved, although the later echo gives the strongest tone. With
# unit-variance noise per component, an aligned on-bin echo of amplitude a
# gives a tone strength of about a * sqrt(n / (2 ln 2)) = 2300 a at 30 s:
# ~45 for the first echo here and ~140 for the later one. The 0.123 Hz tone
# falls between bins, and its spread over the search band raises the median
# the strength is measured against: on four noise draws of this case (run
# on the CPU, 2026-10-21) the two measured ~33 and ~65, the first still
# twice the 15 threshold.
LATE_SHIFT = TRUE_SHIFT + 4 * CHIP
rx_two = (0.02 * echo + 0.06 * np.roll(tx_synth, LATE_SHIFT) * ramp
          + (rng.normal(size=n) + 1j * rng.normal(size=n))).astype("complex64")
del echo, ramp
m2 = foh.measure_offset(rx_two, tx_synth, fs_val, CHIP, max_shift=40)
del rx_two
print(f"      two echoes at {TRUE_SHIFT} and {LATE_SHIFT} samples: timing "
      f"{m2['shift_refined']:+.2f} ({m2['timing_status']}), strongest at "
      f"{m2['shift_strongest']:+d}, tone snr {m2['snr']:.0f}")
# The strongest tone must be the later echo (within its one-chip
# correlation), or this case would not test the rule at all.
check("two-echo synthetic: strongest tone at the later echo",
      abs(m2["shift_strongest"] - LATE_SHIFT), CHIP, "samples")
check("two-echo synthetic: timing at the first echo",
      abs(m2["shift_refined"] - TRUE_SHIFT), 1.5, "samples")
resolved = m2["timing_status"] == "resolved"
print(f"{'PASS' if resolved else 'FAIL'}  two-echo synthetic timing status: "
      f"{m2['timing_status']} (must be resolved)")
if not resolved:
    failures.append("two-echo synthetic timing status")
del t, phase, tx_synth

# ---------------------------------------------------------------------------
# 5. Closure on real data: measure -> apply corrections -> residual ~ 0
# ---------------------------------------------------------------------------
(rx, tx, fs_q, freq, rx_start, _t, _f) = dea.load_observation(RX_FILE, DATA_ROOT)
fs = fs_q.to_value(au.Hz)
tx_emit = dea.nominal_tx_emit_start(rx_start)
tx_comp = foh.compensated_tx(rx, tx, fs_q, freq, rx_start, tx_emit, tx_name=TX, rx_name=RX)
m0 = foh.measure_offset(rx, tx_comp, fs, dea.chip_samples_of(_f, fs))
# Apply the measured corrections the same way compute_dd_image does: delay as
# a float ET offset on the TX emission epoch (NOT baked into an astropy Time,
# which only carries ms in its default string form), frequency as an extra
# compensation ramp.
tx_emit_corr_et = dea.et_from_astropy(tx_emit) + m0["shift_refined"] / fs
t_s = np.arange(len(rx)) / fs
tx_comp_corr = (foh.compensated_tx(rx, tx, fs_q, freq, rx_start, tx_emit_corr_et,
                                   tx_name=TX, rx_name=RX)
                * np.exp(1j * 2 * np.pi * m0["f_centroid"] * t_s)).astype("complex64")
m1 = foh.measure_offset(rx, tx_comp_corr, fs, dea.chip_samples_of(_f, fs))
print(f"      closure: shift {m0['shift_refined']:+.2f} -> {m1['shift_refined']:+.2f} samples, "
      f"centroid {m0['f_centroid']:+.3f} -> {m1['f_centroid']:+.3f} Hz")
# Measured closure: 0.0315 samples / 12.55 mHz, by this test on 2026-10-21
# under the leading-peak timing rule (freq_offset_hunt.leading_peak); the
# 2026-07-03 measurement, under the old strongest-peak rule, was 0.031 /
# 12.6 mHz. Bounds pinned just above the measured values; a real regression
# historically moved these by tenths of a sample / tens of mHz.
check("closure residual shift", abs(m1["shift_refined"]), 0.05, "samples")
check("closure residual centroid", abs(m1["f_centroid"]), 0.015, "Hz")
# Both measurements must be usable ones (resolved timing peak, tone over the
# threshold, inside the search, not on the band edge): an unresolved timing
# is not refined and could pass the shift bound at an integer 0.
for name, m in (("first", m0), ("closure", m1)):
    problems = foh.measurement_problems(m, 20)
    print(f"{'PASS' if not problems else 'FAIL'}  {name} measurement usable: "
          f"{'; '.join(problems) or m['timing_status']}")
    if problems:
        failures.append(f"{name} measurement usable")

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    raise SystemExit(1)
print("All pipeline consistency checks passed.")
