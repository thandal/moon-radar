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
p_zoom = dea.specular_point_bck(rx_time)
x0 = dea.subpoint_average_guess(rx_time)
res = scipy.optimize.minimize(
    lambda x: float(dea.moonPointLightTime_BCK(rx_time, np.asarray(x))), x0,
    method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 250})
p_nm = dea.moon_surface_points(res.x)
lt_zoom = dea.moonPointLightTime_BCK(rx_time, p_zoom)
lt_nm = dea.moonPointLightTime_BCK(rx_time, p_nm)
check("specular zoom vs NM light time", abs(lt_zoom - lt_nm), 1e-11, "s")
check("specular zoom vs NM distance", np.linalg.norm(p_zoom - p_nm), 0.1, "km")

# ---------------------------------------------------------------------------
# 2. Anchored station field vs exact per-point SPICE (near side, incl. limb)
# ---------------------------------------------------------------------------
T = 30.0
v = np.array(hp.pix2vec(64, np.arange(hp.nside2npix(64)))).T
v = v[v[:, 0] > 0]
p = dea.moon_surface_points(v)
lt_exact = dea.moonPointLightTime_BCK(rx_time, p)
lt_end_exact = dea.moonPointLightTime_BCK(rx_time + T, p)
R_rx, R_tx, c = dea.apparent_station_positions(rx_time)
R_rx1, R_tx1, _ = dea.apparent_station_positions(rx_time + T)
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
# Constant-modulus signal with a ~5-sample coherence length (like the 50 kHz
# ZC waveform at 250 ksps) and a WEAK echo in noise: in the strong-signal
# regime the median-normalized SNR is nearly flat versus shift and the test
# would not probe delay recovery at all.
phase = np.cumsum(rng.normal(0.0, 0.53, n))
tx_synth = np.exp(1j * phase).astype("complex64")
TRUE_SHIFT, TRUE_DF = 9, 0.123
echo = np.roll(tx_synth, TRUE_SHIFT) * np.exp(1j * 2 * np.pi * TRUE_DF * t)
rx_synth = (0.1 * echo + (rng.normal(size=n) + 1j * rng.normal(size=n))).astype("complex64")
m = foh.measure_offset(rx_synth, tx_synth, fs_val, max_shift=20)
# Delay resolution for a weak echo is ~1 sample (SNR-vs-shift argmax on a
# ~chip-wide bump); this check guards sign/rough magnitude. The sub-sample
# validation is the real-data closure below (high-SNR specular line).
check("synthetic tone shift recovery", abs(m["shift_refined"] - TRUE_SHIFT), 1.5, "samples")
check("synthetic tone freq recovery", abs(m["f_centroid"] - TRUE_DF), 0.01, "Hz")

# ---------------------------------------------------------------------------
# 5. Closure on real data: measure -> apply corrections -> residual ~ 0
# ---------------------------------------------------------------------------
(rx, tx, fs_q, freq, rx_start, _t, _f) = dea.load_observation(RX_FILE, DATA_ROOT)
fs = fs_q.to_value(au.Hz)
tx_emit = rx_start + 1.0 * au.s
tx_comp = foh.compensated_tx(rx, tx, fs_q, freq, rx_start, tx_emit)
m0 = foh.measure_offset(rx, tx_comp, fs)
# Apply the measured corrections the same way compute_dd_image does: delay as
# a float ET offset on the TX emission epoch (NOT baked into an astropy Time,
# which only carries ms in its default string form), frequency as an extra
# compensation ramp.
tx_emit_corr_et = dea.et_from_astropy(rx_start) + 1.0 + m0["shift_refined"] / fs
t_s = np.arange(len(rx)) / fs
tx_comp_corr = (foh.compensated_tx(rx, tx, fs_q, freq, rx_start, tx_emit_corr_et)
                * np.exp(1j * 2 * np.pi * m0["f_centroid"] * t_s)).astype("complex64")
m1 = foh.measure_offset(rx, tx_comp_corr, fs)
print(f"      closure: shift {m0['shift_refined']:+.2f} -> {m1['shift_refined']:+.2f} samples, "
      f"centroid {m0['f_centroid']:+.3f} -> {m1['f_centroid']:+.3f} Hz")
# Measured closure is 0.03 samples / ~13 mHz (REPORT section 7); bounds sit
# well above that but tight enough to catch a real regression. Pin tighter
# after re-running on the data machine.
check("closure residual shift", abs(m1["shift_refined"]), 0.5, "samples")
check("closure residual centroid", abs(m1["f_centroid"]), 0.025, "Hz")

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    raise SystemExit(1)
print("All pipeline consistency checks passed.")
