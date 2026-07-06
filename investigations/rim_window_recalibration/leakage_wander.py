"""Dwingeloo TX-leakage frequency wander across the 06-21 burst window.

The self-receive leakage line measures the TX exciter against Dwingeloo's
own RX chain (common H-maser reference; no Moon, no Stockert, no path).
If the 10:09-10:18 burst is TX-side, delta_f_leak(t) wanders ~25 mHz within
those looks; if it is clean (sub-mHz), the burst is Stockert-side.

Method per look: integer-align RX to the TX waveform by correlation (TX
starts ~1 s into the record by convention), mix rx*conj(tx) (tx tiled if
the record outruns the file), remove the constant product tone (fractional
-sample alignment: ~200 Hz/sample for this ZC), block-average to 20 Hz,
then measure delta_f(t) by phase-slope fits in sliding windows.
"""
import os
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as pl
from astropy import units as au

REPO = "/home/than/code/moon-radar"
sys.path.insert(0, REPO)
os.chdir(REPO)

import doppler_equator_alignment as dea

DATA_ROOT = os.path.join(REPO, "data.camras.nl/lunar-radar")
OUT_PNG = "investigations/rim_window_recalibration/dwingeloo_leakage_wander.png"

LOOKS = [
    ("10_09_25", "BURST"),
    ("10_14_34", "BURST"),
    ("10_17_40", "BURST"),
    ("10_06_39", "ctrl"),
    ("10_11_46", "ctrl"),
    ("10_26_29", "ctrl"),
    ("09_58_25", "ctrl"),
]


def leak_df(stem):
    base = (f"dwingeloo_eme_2025_06_21_{stem}_1299.500MHz_0.25Msps_ci16_le"
            ".sigmf-meta")
    (rx, tx, fs_q, freq, rx_start, _t, _f) = dea.load_observation(
        os.path.join(DATA_ROOT, "2025-06-21", base), DATA_ROOT)
    fs = fs_q.to_value(au.Hz)
    i0 = int(round(1.0 * fs))

    # Integer alignment by FFT correlation on a 2 s slice.
    seg = rx[i0:i0 + int(2 * fs)]
    ref = tx[:len(seg)]
    n = len(seg)
    C = np.fft.ifft(np.fft.fft(seg) * np.conj(np.fft.fft(ref, n)))
    lags = np.arange(n)
    lags[lags > n // 2] -= n
    k = int(np.argmax(np.abs(C)))
    lag = int(lags[k])
    snr_align = float(np.abs(C[k]) / np.median(np.abs(C)))

    start = i0 + lag
    # No tiling: the on-air transmission does not loop the waveform file
    # seamlessly (verified on the 36 s capture / 30 s file at 10:26:29),
    # so stay within one waveform playback.
    n_use = min(len(rx) - start, len(tx))
    mix = rx[start:start + n_use] * np.conj(tx[:n_use])

    # Coarse product-tone frequency from a 2 s FFT, then mix down.
    m0 = mix[:int(2 * fs)]
    F = np.fft.fftshift(np.fft.fft(m0))
    fax = np.fft.fftshift(np.fft.fftfreq(len(m0), 1 / fs))
    f0 = float(fax[int(np.argmax(np.abs(F)))])
    t = np.arange(n_use) / fs
    mix = mix * np.exp(-1j * 2 * np.pi * f0 * t)

    # Block-average to 20 Hz (coherent for residuals < ~10 Hz).
    blk = int(fs / 20)
    nb = n_use // blk
    z = mix[:nb * blk].reshape(nb, blk).mean(axis=1)
    tz = (np.arange(nb) + 0.5) * blk / fs
    # refine residual constant slope once so unwrap is safe
    ph = np.unwrap(np.angle(z))
    slope = np.polyfit(tz, ph, 1)[0] / (2 * np.pi)
    z = z * np.exp(-1j * 2 * np.pi * slope * tz)
    ph = np.unwrap(np.angle(z))

    # delta_f(t): phase-slope in sliding 5 s windows, 1 s step.
    win, step = 100, 20  # blocks of 50 ms
    ts, dfs = [], []
    for a in range(0, nb - win, step):
        s = slice(a, a + win)
        dfs.append(np.polyfit(tz[s], ph[s], 1)[0] / (2 * np.pi) * 1e3)
        ts.append(tz[a + win // 2])
    dfs = np.array(dfs)
    leak_snr = float(np.mean(np.abs(z)) / np.std(np.abs(z) - np.mean(np.abs(z))))
    return (np.array(ts), dfs, f0 + slope, snr_align, lag,
            len(tx) / fs, n_use / fs)


fig, ax = pl.subplots(figsize=(9, 6))
print(f"{'look':10s} {'class':6s} {'tone Hz':>9s} {'lag':>6s} {'alignSNR':>8s} "
      f"{'span s':>7s} {'df p2p mHz':>10s} {'df RMS mHz':>10s}")
for i, (stem, label) in enumerate(LOOKS):
    ts, dfs, f0, snr_align, lag, tx_s, used_s = leak_df(stem)
    dfr = dfs - np.median(dfs)
    print(f"{stem:10s} {label:6s} {f0:+9.2f} {lag:6d} {snr_align:8.0f} "
          f"{used_s:7.1f} {np.ptp(dfr):10.3f} {np.std(dfr):10.3f}", flush=True)
    color = "C3" if label == "BURST" else "C0"
    ax.plot(ts, dfr + 10 * i, color=color, lw=1)
    ax.text(ts[-1] + 1, 10 * i, f"{stem} ({label})", fontsize=8,
            color=color, va="center")
ax.set_xlabel("time in look (s)")
ax.set_ylabel("leakage delta_f wander (mHz, offset 10/look)")
ax.set_title("Dwingeloo self-receive TX-leakage frequency wander, 2025-06-21\n"
             "(TX exciter vs own RX chain; burst looks would show ~25 mHz if TX-side)")
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print("wrote", OUT_PNG)
