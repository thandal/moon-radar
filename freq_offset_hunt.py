"""
TX/RX chain frequency-offset hunt.

For each observation: resample + SPICE-compensate the TX exactly as the DD
pipeline does, then form y(t) = rx(t) * conj(tx_compensated(t)). If the
geometry/clock model were perfect, the specular echo would collapse to DC;
any residual tone is the TX/RX chain frequency offset (reference clock-rate
offset between the stations), plus the SPICE prediction error (~mHz).

The product form is robust: RFI tones in rx get spread by conj(tx) over the
ZC bandwidth (50 kHz), so the only narrowband feature in Y(f) is the echo.
Frequency resolution is 1/T (~0.033 Hz for 30 s), refined by parabolic
interpolation; small integer sample shifts of rx are tried because the
product method needs delay alignment within a chip (~5 samples at 250 ksps).

Usage (from the repo root):
    .conda/bin/python freq_offset_hunt.py --date 2025-09-16 --station stockert --limit 12
"""

import argparse
import os

import numpy as np
import scipy.interpolate
import cupy
import sigmf
from astropy import units as au

import doppler_equator_alignment as dea


def compensated_tx(rx_samples, tx_samples, sample_rate, frequency,
                   rx_start_astrotime, tx_start_astrotime,
                   tx_name="DWINGELOO", rx_name="STOCKERT"):
    """Mirror of the resample+compensate stages of dea.compute_dd_image."""
    rx_duration = len(rx_samples) / sample_rate
    tx_duration = len(tx_samples) / sample_rate
    rx_duration_s = rx_duration.to(au.s).value

    rx_start_time_s = dea.et_from_astropy(rx_start_astrotime)
    rx_end_time_s = rx_start_time_s + rx_duration_s
    tx_start_time_s = dea.et_from_astropy(tx_start_astrotime)
    tx_end_time_s = tx_start_time_s + tx_duration.to(au.s).value

    lt_tx_start, _ = dea.moonSRP_DLT_FWD(tx_start_time_s, tx_name, rx_name)
    lt_tx_end, _ = dea.moonSRP_DLT_FWD(tx_end_time_s, tx_name, rx_name)

    tx_rx_offset = tx_start_time_s - rx_start_time_s
    rx_sample_times0 = np.arange(len(rx_samples)) / sample_rate.to(au.Hz).value
    adjusted_tx_times0 = np.linspace(
        tx_rx_offset + lt_tx_start,
        tx_rx_offset + tx_duration.to(au.s).value + lt_tx_end,
        len(tx_samples), endpoint=False)
    tx_phase_interp = scipy.interpolate.interp1d(
        adjusted_tx_times0, np.unwrap(np.angle(tx_samples)),
        kind='linear', fill_value=np.nan, bounds_error=False)
    tx_resampled = np.exp(1j * tx_phase_interp(rx_sample_times0))
    np.nan_to_num(tx_resampled, copy=False)

    lt_rx_start, dlt_rx_start = dea.moonSRP_DLT_BCK(rx_start_time_s, tx_name, rx_name,
                                                    dt=rx_duration_s / 2)
    lt_rx_end, dlt_rx_end = dea.moonSRP_DLT_BCK(rx_end_time_s, tx_name, rx_name,
                                                dt=rx_duration_s / 2)
    doppler_start = -dlt_rx_start * frequency
    doppler_end = -dlt_rx_end * frequency
    doppler_rate = (doppler_end - doppler_start) / rx_duration

    t_s = np.arange(len(rx_samples)) / sample_rate
    phi_Hz = -(doppler_start + doppler_rate * t_s / 2)
    tx_comp = tx_resampled * np.exp(-1j * 2 * np.pi * (phi_Hz * t_s).value)
    return tx_comp.astype("complex64")


def line_centroid_width(Y, freqs, f_peak, half_width=0.75):
    """Noise-floor-subtracted centroid and RMS width of the line around f_peak.

    The integrated DD image responds to the power-weighted center of the
    specular line, not its instantaneous peak, so the centroid is the right
    per-file offset estimate; the width diagnoses how much of the apparent
    per-file "wander" is just fading-broadened line structure.
    """
    sel = np.abs(freqs - f_peak) <= half_width
    f = freqs[sel]
    p = Y[sel] ** 2
    floor = np.median(p)
    w = np.clip(p - floor, 0, None)
    if w.sum() <= 0:
        return float(f_peak), float("nan")
    centroid = float(np.sum(f * w) / np.sum(w))
    width = float(np.sqrt(np.sum((f - centroid) ** 2 * w) / np.sum(w)))
    return centroid, width


def peak_in_band(Y, freqs, band):
    """Peak magnitude in |freqs|<=band with parabolic interpolation."""
    sel = np.abs(freqs) <= band
    idx_band = np.where(sel)[0]
    mags = Y[idx_band]
    k = int(np.argmax(mags))
    i = idx_band[k]
    # Parabolic interpolation on log magnitude (neighbors in the full array)
    if 0 < i < len(Y) - 1:
        a, b, c = np.log(Y[i - 1]), np.log(Y[i]), np.log(Y[i + 1])
        delta = 0.5 * (a - c) / (a - 2 * b + c)
    else:
        delta = 0.0
    df = freqs[1] - freqs[0]
    f_peak = freqs[i] + delta * df
    snr = Y[i] / np.median(mags)
    return float(f_peak), float(snr)


def measure_offset(rx_samples, tx_comp, sample_rate_hz, band=2.0,
                   wide_band=1000.0, max_shift=20, n_segments=3):
    """Measure the residual tone of rx * conj(tx_comp).

    Tries small integer sample shifts of rx (delay alignment within a chip),
    keeps the shift with the strongest narrow-band peak. Also reports the
    strongest peak in a wide band (catches large chain offsets like the
    ~800 Hz seen on 2025-03 Stockert data) and per-segment values (drift).

    The product is block-sum decimated to ~5 kHz on the GPU before the FFT:
    everything we search for lives within +/-1 kHz of DC, and the boxcar
    response is ~1 there, so this shrinks the per-shift FFTs and the
    GPU->CPU transfers by ~50x without changing the frequency resolution
    (bin width stays sample_rate/n).
    """
    decim = max(1, int(sample_rate_hz // 5000))
    n = (len(tx_comp) // decim) * decim
    n_dec = n // decim
    freqs = np.fft.fftfreq(n_dec, decim / sample_rate_hz)
    tx_conj = cupy.conj(cupy.asarray(tx_comp[:n]))
    rx_gpu = cupy.asarray(rx_samples[:len(tx_comp)], dtype=cupy.complex64)
    best = None
    shifts = list(range(-max_shift, max_shift + 1))
    snr_profile = np.zeros(len(shifts))
    for k, shift in enumerate(shifts):
        y_gpu = (cupy.roll(rx_gpu, -shift)[:n] * tx_conj).reshape(-1, decim).sum(axis=1)
        Y = cupy.asnumpy(cupy.abs(cupy.fft.fft(y_gpu)))
        f_peak, snr = peak_in_band(Y, freqs, band)
        snr_profile[k] = snr
        if best is None or snr > best["snr"]:
            best = {"shift": shift, "f_peak": f_peak, "snr": snr, "Y": Y,
                    "y_gpu": y_gpu}
    # Sub-sample delay: parabolic vertex of the SNR-vs-shift profile around
    # its maximum (the integer argmax alone jitters by +/-1-2 samples on the
    # ~chip-wide coherence bump).
    k = int(np.argmax(snr_profile))
    if 0 < k < len(shifts) - 1:
        a, b, cc = snr_profile[k - 1], snr_profile[k], snr_profile[k + 1]
        denom = a - 2 * b + cc
        delta = 0.5 * (a - cc) / denom if denom < 0 else 0.0
    else:
        delta = 0.0
    shift_refined = shifts[k] + float(np.clip(delta, -1, 1))
    f_wide, snr_wide = peak_in_band(best["Y"], freqs, wide_band)
    centroid, width = line_centroid_width(best["Y"], freqs, best["f_peak"])

    # Per-segment peaks (drift within the file)
    seg_peaks = []
    seg_len = n_dec // n_segments
    for s in range(n_segments):
        seg = best["y_gpu"][s * seg_len:(s + 1) * seg_len]
        Yseg = cupy.asnumpy(cupy.abs(cupy.fft.fft(seg)))
        fseg = np.fft.fftfreq(seg_len, decim / sample_rate_hz)
        fp, _ = peak_in_band(Yseg, fseg, band)
        seg_peaks.append(fp)

    # Keep the line-region spectrum for diagnostics/plots
    sel = np.abs(freqs) <= band
    order = np.argsort(freqs[sel])
    return {"shift": best["shift"], "shift_refined": shift_refined,
            "f_peak": best["f_peak"], "snr": best["snr"],
            "f_centroid": centroid, "line_width": width,
            "f_wide": f_wide, "snr_wide": snr_wide, "segments": seg_peaks,
            "spectrum_f": freqs[sel][order], "spectrum_mag": best["Y"][sel][order]}


def candidate_files(data_root, date, station, limit=None, chan=None):
    data_dir = os.path.join(data_root, date)
    files = []
    for name in sorted(os.listdir(data_dir)):
        if not name.startswith(station) or not name.endswith(".sigmf-meta"):
            continue
        if "_1970_" in name:
            continue
        if chan and chan not in name:
            continue
        path = os.path.join(data_dir, name)
        try:
            info = sigmf.sigmffile.fromfile(path, skip_checksum=True).get_global_info()
        except Exception as exc:
            print(f"Skipping unreadable metadata {path}: {exc}")
            continue
        desc = info.get("core:description", "")
        if "zadoff-chu" not in desc or "cw-" in desc or "pulsed" in desc:
            continue
        if "30sec" not in desc and "60sec" not in desc:
            continue
        files.append(path)
    if limit and len(files) > limit:
        idx = np.linspace(0, len(files) - 1, limit).round().astype(int)
        files = [files[i] for i in idx]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__),
                                                            "data.camras.nl/lunar-radar"))
    parser.add_argument("--date", default="2025-09-16")
    parser.add_argument("--station", default="stockert",
                        help="RX file prefix: stockert (bistatic) or dwingeloo (monostatic)")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--chan", default=None,
                        help="filename substring filter, e.g. chan1 (co-pol)")
    parser.add_argument("--max-shift", type=int, default=20)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__),
                                                          "results/FREQ_OFFSET"))
    args = parser.parse_args()

    rx_name = "STOCKERT" if args.station == "stockert" else "DWINGELOO"

    files = candidate_files(args.data_root, args.date, args.station, args.limit,
                            chan=args.chan)
    print(f"{len(files)} candidate files for {args.date}/{args.station} (chan={args.chan})")

    rows = []
    for path in files:
        base = os.path.basename(path)
        try:
            (rx_samples, tx_samples, sample_rate, frequency,
             rx_start, _tx_file_time, tx_filename) = dea.load_observation(path, args.data_root)
            rx_duration = len(rx_samples) / sample_rate
            if rx_duration < 20 * au.s:
                print(f"  {base}: skipped (short rx {rx_duration})")
                continue
            tx_emit_start = rx_start + 1.0 * au.s
            tx_comp = compensated_tx(rx_samples, tx_samples, sample_rate, frequency,
                                     rx_start, tx_emit_start,
                                     tx_name="DWINGELOO", rx_name=rx_name)
            res = measure_offset(rx_samples, tx_comp, sample_rate.to_value(au.Hz),
                                 max_shift=args.max_shift)
            drift = (res["segments"][-1] - res["segments"][0]) / \
                    (rx_duration.to_value(au.s) * (len(res["segments"]) - 1) / len(res["segments"]))
            row = {
                "rx_file": base,
                "rx_start_utc": rx_start.utc.value,
                "frequency_hz": frequency.to_value(au.Hz),
                "delta_f_hz": res["f_peak"],
                "delta_f_centroid_hz": res["f_centroid"],
                "line_width_hz": res["line_width"],
                "fractional": res["f_peak"] / frequency.to_value(au.Hz),
                "peak_snr": res["snr"],
                "best_shift_samples": res["shift"],
                "wide_peak_hz": res["f_wide"],
                "wide_peak_snr": res["snr_wide"],
                "seg_peaks_hz": ";".join(f"{p:.4f}" for p in res["segments"]),
                "drift_hz_per_s": drift,
            }
            rows.append(row)
            print(f"  {base}: df={res['f_peak']:+.4f} Hz (snr {res['snr']:.0f}, "
                  f"shift {res['shift']}), wide={res['f_wide']:+.3f} Hz, "
                  f"segs={row['seg_peaks_hz']}")
        except Exception as exc:
            print(f"  {base}: ERROR {exc}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"freq_offsets_{args.date}_{args.station}.csv")
    if rows:
        keys = list(rows[0].keys())
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")
        dfs = np.array([r["delta_f_hz"] for r in rows])
        print(f"\nWrote {len(rows)} rows to {out_csv}")
        print(f"delta_f: mean {dfs.mean():+.4f} Hz, std {dfs.std():.4f} Hz, "
              f"min {dfs.min():+.4f}, max {dfs.max():+.4f}")
        print(f"fractional (mean): {dfs.mean() / rows[0]['frequency_hz']:.3e}")


if __name__ == "__main__":
    main()
