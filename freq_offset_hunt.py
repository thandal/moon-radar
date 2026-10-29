"""
TX/RX chain timing and frequency offsets, and the batch's file selection.

For each observation: resample + SPICE-compensate the TX exactly as the DD
pipeline does (dea.compensated_tx, the one shared model), then form
y(t) = rx(t + shift) * conj(tx_compensated(t)). If the geometry/clock model
were perfect, the specular echo would collapse to DC; any residual tone is
the TX/RX chain frequency offset (reference clock-rate offset between the
stations), plus the SPICE prediction error (~mHz).

The product form is robust: RFI tones in rx get spread by conj(tx) over the
ZC bandwidth (50 kHz), so the only narrowband features in Y(f) are echo.
Frequency resolution is 1/T (~0.033 Hz for 30 s), refined by parabolic
interpolation. Integer sample shifts of rx are tried because the product
method needs delay alignment within a chip (~5 samples at 250 ksps); the
timing offset is the first strong peak of tone strength vs shift (see
leading_peak).

Usage (from the repo root):
    .conda/bin/python freq_offset_hunt.py --date 2025-09-16 --station stockert \
        --chan chan1 --limit 12
"""

import argparse
import csv
import json
import os
import re

import numpy as np
import cupy
import sigmf
from astropy import units as au

import doppler_equator_alignment as dea


def compensated_tx(rx_samples, tx_samples, sample_rate, frequency,
                   rx_start_astrotime, tx_start_astrotime, *, tx_name, rx_name):
    """dea.compensated_tx for a capture: the TX model the timing search
    correlates against (no frequency offset applied)."""
    tx_comp, _ = dea.compensated_tx(
        len(rx_samples), tx_samples, sample_rate, frequency,
        dea.et_from_astropy(rx_start_astrotime), dea.et_from_astropy(tx_start_astrotime),
        tx_name=tx_name, rx_name=rx_name)
    return tx_comp


def line_centroid_width(Y, freqs, f_peak, half_width=0.75):
    """Noise-floor-subtracted centroid and RMS width of the line around f_peak.

    The integrated DD image responds to the power-weighted center of the
    specular line, not its instantaneous peak, so the centroid is the right
    per-file offset estimate; the width diagnoses how much of the apparent
    per-file "wander" is just fading-broadened line structure. It is only
    the starting point: process_file's rim calibration measures the
    remaining Doppler offset from the echo edges and records it.
    """
    sel = np.abs(freqs - f_peak) <= half_width
    f = freqs[sel]
    p = Y[sel] ** 2
    floor = np.median(p)
    w = np.clip(p - floor, 0, None)
    if not w.sum() > 0:
        raise ValueError(f"no line above the noise within {half_width} Hz of {f_peak:+.3f} Hz")
    centroid = float(np.sum(f * w) / np.sum(w))
    width = float(np.sqrt(np.sum((f - centroid) ** 2 * w) / np.sum(w)))
    return centroid, width


def peak_in_band(Y, freqs, band):
    """Strongest bin of |Y| in |freqs| <= band, its frequency refined by a
    parabola (on log magnitude) through it and its two neighbours -- taken
    circularly, as the FFT is periodic, so the 0 Hz bin is refined too.

    Returns (f_peak, snr, at_edge). snr: peak over the band's median.
    at_edge: the peak is the band's outermost bin; the line may then lie
    outside the band (a chain offset beyond it), and no refinement is done.
    """
    idx_band = np.where(np.abs(freqs) <= band)[0]
    mags = Y[idx_band]
    i = idx_band[int(np.argmax(mags))]
    band_f = freqs[idx_band]
    at_edge = bool(freqs[i] == band_f.max() or freqs[i] == band_f.min())
    delta = 0.0
    if not at_edge:
        n = len(Y)
        a, b, c = np.log(Y[(i - 1) % n]), np.log(Y[i]), np.log(Y[(i + 1) % n])
        den = a - 2 * b + c
        # Inside the band the strongest bin is a local maximum, so den <= 0
        # (0 only for three equal bins, whose centre is the peak).
        delta = 0.5 * (a - c) / den if den < 0 else 0.0
    df = freqs[1] - freqs[0]
    return float(freqs[i] + delta * df), float(Y[i] / np.median(mags)), at_edge


# Tone strength (peak over median of |Y| in the +-2 Hz band) the stacker
# requires of a look (stack_maps.py --min-snr); also where the timing search
# takes the echo's leading edge to begin (leading_peak). On the 2025 looks the
# profile reads 3-8 at shifts with no echo.
MIN_TONE_SNR = 15.0

# How far past the edge the specular peak may lie beyond the code's own
# one-chip rise (leading_peak): the specular echo's spread in delay (terrain
# and curvature near the SRP), a property of the Moon, not of the code.
# 20 us (3 km of range depth) covers the 125 kchip/s looks of 2026-10-24
# (peak 16-20 us after the edge on Stockert 18:30:42 and 19:27:32, with a
# one-chip rise of 8 us), and is no wider than the two-chip reach the
# 25 and 50 kchip/s looks already had (40-80 us).
SPECULAR_SPREAD_S = 20e-6


def leading_peak(profile, chip_samples, threshold, spread_samples=0.0):
    """Index and status of the timing offset in a tone-strength-vs-shift
    profile: the first peak after the profile reaches threshold.

    The tone rises out of the noise where the TX lines up with the echo's
    leading edge -- the specular point, the nearest surface, so nothing
    arrives earlier -- and peaks soon after. Later shifts line up the TX
    with echo from farther away, and some of that echo is also at ~0 Hz:
    every delay ring crosses the Doppler equator. On sessions with a narrow
    Doppler spread (2025-09-11: ~5 Hz across the disk, inside the +-2 Hz
    search band) a later peak can outgrow a faded specular peak (06:37:24:
    specular peak at shift 7, a stronger one at 35); taking the strongest
    put the echo's leading edge 28 samples before lag 0 there.

    The edge is where the profile first reaches threshold, the tone strength
    a look must reach to be stacked at all (MIN_TONE_SNR) -- independent of
    how strong any later echo is. A peak is a point that is the highest
    within half a chip either side -- the half-width at half maximum of the
    code's correlation triangle (one chip either side): a real echo peak
    dominates at least that neighbourhood, so a narrower bump is noise, not
    echo, while a full chip would let a neighbouring echo hide a real peak.
    The first such peak after the edge must lie within reach of it: the
    correlation rises over one chip, and the specular echo's own spread in
    delay (terrain and curvature near the SRP) adds up to about one more chip
    at 25-50 kchip/s (on 2025-09-11 06:38:25 the peak sits 1.2 chips after
    the edge; investigations/2026-10-21_scratch/delay_diag.py). That spread
    is in delay, not in chips: at 125 kchip/s (2 samples per chip) the peak
    sits 4-5 samples after the edge, beyond two chips. So the reach is two
    chips, or one chip plus spread_samples (SPECULAR_SPREAD_S, from
    measure_offset) if that is longer; with spread_samples 0, two chips.

    Status: "resolved" (such a peak exists); "peak_not_resolved" (no peak
    within two chips of the edge -- the specular echo is not resolved from
    what follows -- or the search ends first); "edge_at_search_start" (the
    edge is within half a chip of the first shift: it may lie outside the
    search, and a peak there cannot be checked against its neighbourhood);
    "below_threshold" (the tone never reaches threshold). The index is the peak, else the strongest point within two
    chips of the edge, else the strongest point. Only "resolved" timings are
    stacked.
    """
    p = np.asarray(profile, dtype=float)
    above = np.nonzero(p >= threshold)[0]
    if len(above) == 0:
        return int(np.argmax(p)), "below_threshold"
    k_edge = int(above[0])
    h = int(np.ceil(chip_samples / 2))
    reach = int(np.ceil(max(2 * chip_samples, chip_samples + spread_samples)))
    if k_edge < h:
        return k_edge + int(np.argmax(p[k_edge:k_edge + reach + 1])), "edge_at_search_start"
    last = min(k_edge + reach, len(p) - 1 - h)
    for k in range(k_edge, last + 1):
        if p[k] == p[k - h:k + h + 1].max():
            return k, "resolved"
    return k_edge + int(np.argmax(p[k_edge:k_edge + reach + 1])), "peak_not_resolved"


def measurement_problems(res, max_shift, min_snr=MIN_TONE_SNR):
    """Why a fresh measure_offset result (searched over +-max_shift
    samples) is not a usable timing/frequency measurement: the checks
    stack_maps.select_looks applies to the recorded opposite-sense rows (tone
    strength, a shift at the search limit, an unresolved timing peak, a
    tone on the band's edge). An empty list if it is usable."""
    reasons = []
    if not res["snr"] >= min_snr:
        reasons.append(f"tone snr {res['snr']:.1f} < {min_snr:g}")
    if not abs(res["shift_refined"]) < max_shift - 0.5:
        reasons.append(f"timing shift {res['shift_refined']:+.1f} at the search limit "
                       f"+-{max_shift}")
    if res["timing_status"] != "resolved":
        reasons.append(f"timing peak {res['timing_status']}")
    if res["tone_at_band_edge"]:
        reasons.append(f"specular tone at the edge of the frequency search band "
                       f"({res['f_peak']:+.3f} Hz)")
    return reasons


def _product(rx_gpu, tx_conj, shift, n, decim):
    """rx(t + shift) * conj(tx(t)), block-summed by decim (on the GPU)."""
    return (cupy.roll(rx_gpu, -shift)[:n] * tx_conj).reshape(-1, decim).sum(axis=1)


def measure_offset(rx_samples, tx_comp, sample_rate_hz, chip_samples,
                   threshold=MIN_TONE_SNR, band=2.0, wide_band=1000.0,
                   max_shift=20, n_segments=3):
    """Timing and frequency offset from the residual tone of rx * conj(tx_comp).

    Tries integer sample shifts of rx in +-max_shift; the timing offset is
    the leading peak of tone strength vs shift (leading_peak, with its
    status; chip_samples is the code's chip length in samples), refined to a
    fraction of a sample by a parabola through the profile when that peak is
    resolved (a local maximum, so the vertex lies within half a sample). The
    frequency comes from that shift's spectrum. Also reports the strongest
    shift (to show when it differs), the strongest peak in a wide band (a
    chain offset beyond +-band shows up there) and per-segment peaks (drift)
    with whether each sat on the band's edge.

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

    shifts = np.arange(-max_shift, max_shift + 1)
    snr_profile = np.zeros(len(shifts))
    for j, shift in enumerate(shifts):
        Y = cupy.asnumpy(cupy.abs(cupy.fft.fft(_product(rx_gpu, tx_conj, int(shift), n, decim))))
        snr_profile[j] = peak_in_band(Y, freqs, band)[1]
    k, timing_status = leading_peak(snr_profile, chip_samples, threshold,
                                    spread_samples=SPECULAR_SPREAD_S * sample_rate_hz)
    k_strong = int(np.argmax(snr_profile))
    # Sub-sample delay: parabolic vertex of the profile around a resolved
    # peak (the integer peak alone jitters by +/-1-2 samples on the
    # ~chip-wide coherence bump). A resolved peak is a local maximum, so
    # den <= 0 and the vertex is within half a sample (den = 0: three equal
    # points, the middle one is the peak). Unresolved timings are not
    # refined, and are not stacked.
    delta = 0.0
    if timing_status == "resolved":
        a, b, cc = snr_profile[k - 1], snr_profile[k], snr_profile[k + 1]
        denom = a - 2 * b + cc
        if denom < 0:
            delta = 0.5 * (a - cc) / denom
    shift_refined = shifts[k] + delta

    y_gpu = _product(rx_gpu, tx_conj, int(shifts[k]), n, decim)
    Y = cupy.asnumpy(cupy.abs(cupy.fft.fft(y_gpu)))
    f_peak, snr, at_edge = peak_in_band(Y, freqs, band)
    f_wide, snr_wide, _ = peak_in_band(Y, freqs, wide_band)
    centroid, width = line_centroid_width(Y, freqs, f_peak)

    # Per-segment peaks (drift within the file)
    seg_peaks, seg_edge = [], []
    seg_len = n_dec // n_segments
    for s in range(n_segments):
        seg = y_gpu[s * seg_len:(s + 1) * seg_len]
        Yseg = cupy.asnumpy(cupy.abs(cupy.fft.fft(seg)))
        fseg = np.fft.fftfreq(seg_len, decim / sample_rate_hz)
        f_seg, _, edge_seg = peak_in_band(Yseg, fseg, band)
        seg_peaks.append(f_seg)
        seg_edge.append(edge_seg)
    del tx_conj, rx_gpu, y_gpu
    cupy.get_default_memory_pool().free_all_blocks()

    # Keep the line-region spectrum for diagnostics/plots
    sel = np.abs(freqs) <= band
    order = np.argsort(freqs[sel])
    return {"shift": int(shifts[k]), "shift_refined": float(shift_refined),
            "timing_status": timing_status,
            "shift_strongest": int(shifts[k_strong]),
            "snr_strongest": float(snr_profile[k_strong]),
            "f_peak": f_peak, "snr": snr, "tone_at_band_edge": at_edge,
            "f_centroid": centroid, "line_width": width,
            "f_wide": f_wide, "snr_wide": snr_wide, "segments": seg_peaks,
            "segments_at_band_edge": seg_edge,
            "snr_profile": snr_profile, "shifts": shifts,
            "spectrum_f": freqs[sel][order], "spectrum_mag": Y[sel][order]}


# sigmf datatype -> bytes per complex sample
_SAMPLE_BYTES = {"ci16_le": 4, "ci16": 4, "cf32_le": 8, "cf32": 8}


def tx_waveform_seconds(data_root, tx_meta_name):
    """Length (s) of a TX waveform in data_root/tx_signals, from its data
    file's size; None if the waveform's files are not there."""
    meta = os.path.join(data_root, "tx_signals", tx_meta_name)
    data = meta.replace(".sigmf-meta", ".sigmf-data")
    if not (os.path.exists(meta) and os.path.exists(data)):
        return None
    with open(meta, encoding="utf-8") as f:
        info = json.load(f)["global"]
    dtype = info["core:datatype"]
    if dtype not in _SAMPLE_BYTES:
        raise ValueError(f"{meta}: unsupported datatype {dtype!r}")
    return os.path.getsize(data) / _SAMPLE_BYTES[dtype] / info["core:sample_rate"]


def candidate_files(data_root, date, station, chan, min_tx_s, limit=None,
                    require_data=True):
    """The station's captures of one channel in data_root/date, split into
    those to process and those not, with a reason for each of the latter.

    A capture is processed if its TX waveform (the first field of its
    core:description) is a Zadoff-Chu code -- a name starting "zadoff-chu"
    or "interp_zadoff-chu" -- lasting at least min_tx_s seconds, measured
    from the waveform file itself, and (require_data) its own data file is
    there. Captures of the station with no .chanN. tag in their name are
    listed as skipped too. limit, if given, keeps that many captures evenly
    spaced in time. Returns (files, skipped): .sigmf-meta paths, and
    (path, reason) pairs.
    """
    data_dir = os.path.join(data_root, date)
    files, skipped = [], []
    for name in sorted(os.listdir(data_dir)):
        if not name.startswith(station) or not name.endswith(".sigmf-meta"):
            continue
        path = os.path.join(data_dir, name)
        if not re.search(r"\.chan\d+\.", name):
            skipped.append((path, "no .chanN. channel tag in the file name"))
            continue
        if f".{chan}." not in name:
            continue
        if require_data and not os.path.exists(path.replace(".sigmf-meta", ".sigmf-data")):
            skipped.append((path, "RX data file missing"))
            continue
        if "_1970_" in name:
            skipped.append((path, "capture time stamped 1970 (receiver clock unset)"))
            continue
        try:
            info = sigmf.sigmffile.fromfile(path, skip_checksum=True).get_global_info()
        except Exception as exc:
            skipped.append((path, f"unreadable metadata: {exc}"))
            continue
        tx_name = info.get("core:description", "").split(";")[0]
        if not re.match(r"(interp_)?zadoff-chu-", tx_name):
            skipped.append((path, f"TX waveform is not a Zadoff-Chu code: {tx_name!r}"))
            continue
        tx_s = tx_waveform_seconds(data_root, tx_name)
        if tx_s is None:
            skipped.append((path, f"TX waveform file missing: tx_signals/{tx_name}"))
            continue
        if tx_s < min_tx_s:
            skipped.append((path, f"TX waveform {tx_s:.1f} s < {min_tx_s:g} s"))
            continue
        files.append(path)
    if limit and len(files) > limit:
        keep = set(np.linspace(0, len(files) - 1, limit).round().astype(int))
        skipped += [(p, f"not among the {limit} evenly spaced captures kept")
                    for i, p in enumerate(files) if i not in keep]
        files = [p for i, p in enumerate(files) if i in keep]
    return files, skipped


def write_rows(path, rows):
    """Write dicts as CSV (csv.writer quoting, header from the first row)."""
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__),
                                                            "data.camras.nl/lunar-radar"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--station", required=True, choices=sorted(dea.STATION_RX),
                        help="RX file prefix")
    parser.add_argument("--chan", required=True, help="chan1 (opposite sense) or chan0")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--min-tx-s", type=float, default=30.0)
    parser.add_argument("--max-shift", type=int, default=40)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__),
                                                          "results/FREQ_OFFSET"))
    args = parser.parse_args()

    rx_name = dea.STATION_RX[args.station]
    files, skipped = candidate_files(args.data_root, args.date, args.station,
                                     args.chan, args.min_tx_s, args.limit)
    print(f"{len(files)} candidate files for {args.date}/{args.station} "
          f"({args.chan}), {len(skipped)} skipped")

    rows = []
    for path in files:
        base = os.path.basename(path)
        try:
            (rx_samples, tx_samples, sample_rate, frequency,
             rx_start, _tx_file_time, tx_filename) = dea.load_observation(path, args.data_root)
            rx_duration_s = len(rx_samples) / sample_rate.to_value(au.Hz)
            if rx_duration_s < dea.MIN_CAPTURE_S:
                skipped.append((path, f"capture {rx_duration_s:.1f} s < {dea.MIN_CAPTURE_S:g} s"))
                continue
            tx_comp = compensated_tx(rx_samples, tx_samples, sample_rate, frequency,
                                     rx_start, dea.nominal_tx_emit_start(rx_start),
                                     tx_name=dea.TX_STATION, rx_name=rx_name)
            fs = sample_rate.to_value(au.Hz)
            res = measure_offset(rx_samples, tx_comp, fs, dea.chip_samples_of(tx_filename, fs),
                                 max_shift=args.max_shift)
            # A segment peak on the band's edge is not a measurement.
            drift = (float("nan") if any(res["segments_at_band_edge"]) else
                     (res["segments"][-1] - res["segments"][0]) /
                     (rx_duration_s * (len(res["segments"]) - 1) / len(res["segments"])))
            row = {
                "rx_file": base,
                "rx_start_utc": rx_start.utc.value,
                "frequency_hz": frequency.to_value(au.Hz),
                "delta_f_hz": res["f_peak"],
                "delta_f_centroid_hz": res["f_centroid"],
                "line_width_hz": res["line_width"],
                "fractional": res["f_peak"] / frequency.to_value(au.Hz),
                "peak_snr": res["snr"],
                "tone_at_band_edge": res["tone_at_band_edge"],
                "best_shift_samples": res["shift"],
                "timing_status": res["timing_status"],
                "segments_at_band_edge": ";".join(str(e) for e in res["segments_at_band_edge"]),
                "shift_refined": res["shift_refined"],
                "shift_strongest": res["shift_strongest"],
                "max_shift": args.max_shift,
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
            skipped.append((path, f"error: {exc}"))
            print(f"  {base}: ERROR {exc}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"freq_offsets_{args.date}_{args.station}_{args.chan}"
    out_csv = os.path.join(args.out_dir, f"{stem}.csv")
    skip_csv = os.path.join(args.out_dir, f"{stem}_skipped.csv")
    if skipped:
        write_rows(skip_csv, [{"rx_file": os.path.basename(p), "reason": r}
                              for p, r in skipped])
        print(f"{len(skipped)} captures skipped, listed in {skip_csv}")
    elif os.path.exists(skip_csv):
        os.remove(skip_csv)
    if not rows:
        if os.path.exists(out_csv):
            os.remove(out_csv)
        raise SystemExit(f"no capture measured; no {out_csv}")
    write_rows(out_csv, rows)
    dfs = np.array([r["delta_f_hz"] for r in rows])
    print(f"\nWrote {len(rows)} rows to {out_csv}")
    errors = sum(r.startswith("error:") for _, r in skipped)
    print(f"delta_f: mean {dfs.mean():+.4f} Hz, std {dfs.std():.4f} Hz, "
          f"min {dfs.min():+.4f}, max {dfs.max():+.4f}")
    print(f"fractional (mean): {dfs.mean() / rows[0]['frequency_hz']:.3e}")
    if errors:
        raise SystemExit(f"{errors} captures failed with errors")


if __name__ == "__main__":
    main()
