"""
Synthesize opposite- and same-sense (circular) captures from two linear feeds.

The 2026-10-11 receivers (Stockert, WSRT) recorded two linear feeds: both
channels carry the circularly polarized echo at equal power, about 90 deg
apart in phase. The pipeline instead expects chan1 = opposite-sense (the specular,
polarized echo) and chan0 = same-sense. For each capture this measures the
channel-1 / channel-0 phase and amplitude of the specular tone, then writes

    chan1 = (ch0 + r exp(-j theta) ch1) / sqrt(2)   opposite-sense: tone adds coherently
    chan0 = (ch0 - r exp(-j theta) ch1) / sqrt(2)   same-sense: tone cancels

with theta = arg(sum Y1 conj(Y0)) and r = sqrt(P0 / P1) over the tone bins of
the despread product rx * conj(tx_compensated) (as in freq_offset_hunt), the
noise cross-spectrum from nearby bins removed. The output keeps the input
file names under a new data root, so registration_stability.py runs on it
unchanged (chan1 first, chan0 with --corrections-from).

Assumption, checked per capture: the echo is circular and the feeds' gains
are equal, so the tone's amplitude ratio r equals the feeds' noise amplitude
ratio. Both are written to the metadata (circ:amp_ratio,
circ:noise_amp_ratio) with the tone's phase coherence between the feeds
(circ:tone_coherence; process_file copies these into each look's runs-CSV
row). Note that the same-sense output cancels the measured tone by
construction: a missing specular echo in that output is not evidence of
polarization purity.

Captures without a measurable tone in both feeds are skipped and listed,
with the reason, in <out-root>/<date>/make_circular_skipped_<station>.csv.

Usage (from the repo root):
    .conda/bin/python make_circular.py --data-root data.camras.nl/eve2026 \\
        --out-root data/eve2026_circ --date 2026-10-11 --station stockert
"""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cupy
from astropy import units as au

import doppler_equator_alignment as dea
import freq_offset_hunt as foh


class CircSkip(Exception):
    """A capture this script does not convert, for a recorded reason."""


def despread_spectra(rx_list, tx_comp, fs, shift, decim):
    """FFT of the block-summed rx * conj(tx_comp) for each channel (the
    timing search's own product, freq_offset_hunt._product)."""
    n = (len(tx_comp) // decim) * decim
    tx_conj = cupy.conj(cupy.asarray(tx_comp[:n]))
    out = []
    for rx in rx_list:
        rx_gpu = cupy.asarray(rx[:len(tx_comp)], dtype=cupy.complex64)
        y = foh._product(rx_gpu, tx_conj, shift, n, decim)
        out.append(cupy.asnumpy(cupy.fft.fft(y)))
        del rx_gpu, y
    del tx_conj
    cupy.get_default_memory_pool().free_all_blocks()
    return out, np.fft.fftfreq(n // decim, decim / fs)


def tone_snr(Y, freqs, f_peak, band=2.0):
    """Peak power over the median in |f - f_peak| <= band (as foh.peak_in_band)."""
    sel = np.abs(freqs - f_peak) <= band
    mag = np.abs(Y[sel])
    return float(mag.max() / np.median(mag))


def polarization(Y0, Y1, freqs, f_peak, line_hw=0.75, noise_lo=3.0, noise_hi=30.0):
    """Feed phase theta, tone amplitude ratio r, tone coherence and the
    feeds' noise amplitude ratio, from the tone bins (|f - f_peak| <=
    line_hw) with the noise level of nearby bins (noise_lo..noise_hi Hz
    away) removed. Raises CircSkip if either feed has no tone power above
    its noise."""
    line = np.abs(freqs - f_peak) <= line_hw
    off = (np.abs(freqs - f_peak) >= noise_lo) & (np.abs(freqs - f_peak) <= noise_hi)
    nl = line.sum()
    n0, n1 = np.mean(np.abs(Y0[off]) ** 2), np.mean(np.abs(Y1[off]) ** 2)
    c = np.sum(Y1[line] * np.conj(Y0[line])) - nl * np.mean(Y1[off] * np.conj(Y0[off]))
    p0 = np.sum(np.abs(Y0[line]) ** 2) - nl * n0
    p1 = np.sum(np.abs(Y1[line]) ** 2) - nl * n1
    if not (p0 > 0 and p1 > 0):
        raise CircSkip(f"no tone power above the noise (ch0 {p0:.3g}, ch1 {p1:.3g})")
    return (float(np.angle(c)), float(np.sqrt(p0 / p1)),
            float(abs(c) / np.sqrt(p0 * p1)), float(np.sqrt(n0 / n1)))


def to_ci16(z):
    """Interleaved int16 I/Q of z, as the input files store it (sigmf's
    read_samples divides ci16 by 2**15; this undoes that, with the
    1/sqrt(2) of the combination). Raises CircSkip if any sample would
    clip or is not finite."""
    scale = 32768 / np.sqrt(2.0)
    iq = np.empty(2 * len(z), dtype=np.float32)
    iq[0::2] = z.real * scale
    iq[1::2] = z.imag * scale
    if not np.all(np.isfinite(iq)):
        raise CircSkip("combined samples are not finite")
    peak = float(np.abs(iq).max())
    if peak > 32767:
        raise CircSkip(f"combined samples clip ({peak:.0f})")
    return np.rint(iq).astype("<i2"), peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="the two-feed recordings")
    parser.add_argument("--out-root", required=True, help="where the circular captures go")
    parser.add_argument("--date", required=True)
    parser.add_argument("--station", required=True, choices=sorted(dea.STATION_RX))
    parser.add_argument("--min-tx-s", type=float, default=30.0,
                        help="shortest TX waveform to convert, seconds (default 30)")
    parser.add_argument("--max-shift", type=int, default=80)
    parser.add_argument("--min-snr", type=float, default=8.0,
                        help="skip captures whose chan0 tone is weaker than this "
                             "(default 8: one linear feed carries half the echo, and "
                             "the combined opposite-sense look is checked against the "
                             "stacker's own 15 later; also the timing-edge threshold "
                             "on this single feed)")
    args = parser.parse_args()
    rx_name = dea.STATION_RX[args.station]

    out_dir = os.path.join(args.out_root, args.date)
    os.makedirs(out_dir, exist_ok=True)
    tx_link = os.path.join(args.out_root, "tx_signals")
    if not os.path.lexists(tx_link):
        os.symlink(os.path.relpath(os.path.join(args.data_root, "tx_signals"),
                                   args.out_root), tx_link)

    files0, skipped = foh.candidate_files(args.data_root, args.date, args.station,
                                          "chan0", args.min_tx_s)
    # Each capture is converted in a fresh process. A process keeps, until
    # it exits, host shared memory as large as the most GPU memory it has
    # used (measured 2026-10-22: 0.9 GB after the first 125 s capture,
    # beside up to 1.9 GB of its own); one process for all captures went
    # past a 3 GB cap on the fifth.
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        for meta0 in files0:
            name = os.path.basename(meta0).split("_1299")[0]
            try:
                pool.submit(convert, meta0, args, rx_name, out_dir).result()
            except CircSkip as exc:
                skipped.append((meta0, str(exc)))
                print(f"{name}: skipped, {exc}", flush=True)
                # An earlier run's outputs for this capture must not stay behind
                # under normal names for registration_stability.py to map.
                for chan in (".chan0.", ".chan1."):
                    base = os.path.basename(meta0).replace(".chan0.", chan)
                    for ext in (".sigmf-meta", ".sigmf-data"):
                        old = os.path.join(out_dir, base.replace(".sigmf-meta", ext))
                        if os.path.exists(old):
                            os.remove(old)
                            print(f"  removed the earlier {os.path.basename(old)}", flush=True)
    skip_csv = os.path.join(out_dir, f"make_circular_skipped_{args.station}.csv")
    if skipped:
        foh.write_rows(skip_csv, [{"rx_file": os.path.basename(p), "reason": r}
                                  for p, r in skipped])
        print(f"{len(skipped)} captures skipped, listed in {skip_csv}")
    elif os.path.exists(skip_csv):
        os.remove(skip_csv)


def convert(meta0, args, rx_name, out_dir):
    """Write the opposite- and same-sense captures for one two-feed capture."""
    meta1 = meta0.replace(".chan0.", ".chan1.")
    for src in (meta0, meta1):
        with open(src, encoding="utf-8") as f:
            dtype = json.load(f)["global"]["core:datatype"]
        if dtype != "ci16_le":
            raise ValueError(f"{src}: datatype {dtype}; to_ci16 writes ci16_le like the input")
    (rx0, tx, sample_rate, frequency, rx_start, _, tx_file) = dea.load_observation(
        meta0, args.data_root)
    fs = sample_rate.to_value(au.Hz)
    tx_comp = foh.compensated_tx(rx0, tx, sample_rate, frequency, rx_start,
                                 dea.nominal_tx_emit_start(rx_start),
                                 tx_name=dea.TX_STATION, rx_name=rx_name)
    del tx
    # The second feed is loaded only now: on a 125 s capture each feed is
    # 250 MB, and the TX preparation above is this script's memory peak.
    rx1 = dea.load_observation(meta1, args.data_root)[0]
    # The tone vs shift from chan0 alone: summing the feeds would partly
    # cancel the tone where their echo phases differ by more than 90 deg
    # (WSRT). The feed phase is measured at the shift where the tone is
    # strongest: it is the same for all the polarized echo, at any delay, so
    # it does not need the echo's leading edge. (Until 2026-10-26 it needed
    # a resolved leading peak, freq_offset_hunt.leading_peak; on the
    # 2026-10-24 captures the receivers' filters put a skirt about a tenth
    # of the peak's height some 40-50 us ahead of the echo, which crosses
    # this script's low single-feed threshold early: 50 kchip/s captures
    # failed as unresolved, and at 125 kchip/s a bump in the skirt passed as
    # the peak, so the phase was measured on a tone of strength 9 instead of
    # ~150.) The batch measures the timing itself on the combined captures.
    res = foh.measure_offset(rx0, tx_comp, fs, dea.chip_samples_of(tx_file, fs),
                             threshold=args.min_snr, max_shift=args.max_shift)
    if res["snr_strongest"] < args.min_snr:
        raise CircSkip(f"no specular tone (strongest {res['snr_strongest']:.1f} < "
                       f"{args.min_snr:g}) to measure the feed phase")
    decim = max(1, int(fs // 5000))
    (Y0, Y1), freqs = despread_spectra([rx0, rx1], tx_comp, fs, res["shift_strongest"], decim)
    del tx_comp
    f_tone = foh.peak_in_band(np.abs(Y0), freqs, 2.0)[0]
    theta, r, coh, r_noise = polarization(Y0, Y1, freqs, f_tone)
    w = r * np.exp(-1j * theta)
    snr = [tone_snr(Y, freqs, f_tone) for Y in
           (Y0, Y1, Y0 + w * Y1, Y0 - w * Y1)]

    w = np.complex64(w)
    outputs = []
    for src, sign, pol in [(meta1, 1, "opposite-sense"), (meta0, -1, "same-sense")]:
        iq, peak = to_ci16(rx0 + np.complex64(sign) * w * rx1)
        meta = json.load(open(src))
        meta["global"].pop("core:sha512", None)
        meta["global"].update({
            "circ:pol": pol,
            "circ:theta_deg": np.degrees(theta),
            "circ:amp_ratio": r,
            "circ:noise_amp_ratio": r_noise,
            "circ:tone_coherence": coh,
            "circ:sources": [os.path.basename(meta0), os.path.basename(meta1)],
            "circ:combination": "(ch0 %s amp_ratio*exp(-j*theta)*ch1)/sqrt(2)"
                                % ("+" if sign > 0 else "-"),
        })
        base = os.path.basename(src)[:-len(".sigmf-meta")]
        outputs.append((os.path.join(out_dir, base), iq, meta, peak))
    del rx0, rx1
    # Both checked before either is written.
    for path, iq, meta, _ in outputs:
        iq.tofile(path + ".sigmf-data")
        with open(path + ".sigmf-meta", "w") as f:
            json.dump(meta, f, indent=4)
    print(f"{os.path.basename(meta0).split('_1299')[0]}: theta {np.degrees(theta):+6.1f} deg, "
          f"amp ratio {r:.3f} (noise {r_noise:.3f}), tone coherence {coh:.3f}; tone strength "
          f"at shift {res['shift_strongest']:+d} "
          f"ch0 {snr[0]:.0f} ch1 {snr[1]:.0f} -> opposite-sense {snr[2]:.0f} same-sense {snr[3]:.0f}; "
          f"peak |I/Q| {max(o[3] for o in outputs):.0f}", flush=True)


if __name__ == "__main__":
    main()
