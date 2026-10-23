"""Does a frequency offset bias the delay estimate, or a delay the frequency?

The report's error budget assumes delay timing errors live on the delay axis
and reference-frequency errors on the Doppler axis. This synthetic test
injects known delays and tones into a constant-modulus waveform, runs the
production estimator (freq_offset_hunt.measure_offset, whose shift_refined
and f_centroid process_file applies), and records the error in each.

The synthetic has the shape of the shortest production look: 30 s at
250 ksps, and a chip-structured code -- a random phase held for each chip,
5 samples at 50 kchip/s (the Zadoff-Chu code's own phase sequence is not
needed to test the estimator) -- with one echo delayed by a fraction of a
sample, shifted in frequency, in white noise. The test tones fall between
FFT bins (1/30 s = 33 mHz apart), as real offsets do. There the unwindowed
spectrum spreads the tone over the +-2 Hz search band, which lowers the
estimator's peak-over-median "SNR" without any loss of signal; that is a
property of the production estimator, so every case is reported with its
SNR and none is left out.

Every case runs at several echo strengths (--tone-snrs), each named by the
tone strength the stacker measures (peak over median of |Y| in the +-2 Hz
band) that it gives an echo with no delay and no tone. Its amplitude a per
sample follows from that: aligned and on an FFT bin, the echo's peak is
a*n over n samples, and noise of unit variance per component gives spectral
magnitudes with median sqrt(2 n ln 2) (Rayleigh), so a = S sqrt(2 ln 2 / n)
for a tone strength S. The case with no delay and no tone checks the rule:
its measured tone strength is recorded beside S (the band's median, over
~120 bins at 30 s, scatters by about 7 %). The default strengths are the
stacker's threshold (freq_offset_hunt.MIN_TONE_SNR), and 40 and 180, which
bracket most real co-pol looks so far (the 130 in the runs CSVs under
results/ on 2026-10-21: 10th percentile 41, median 66, largest 185).

Each case records the timing status (freq_offset_hunt.leading_peak) and
whether the stacker's gates on a timing measurement would accept it
(freq_offset_hunt.measurement_problems: tone strength, shift at the search
limit, unresolved timing peak, tone at the band edge). Errors are reported
over all cases and over the accepted ones.

Measurements only: there is no pass/fail criterion. The JSON gives, per
tone strength, the timing-status counts, the worst cases and the errors per
injected tone and per injected delay; the CSV has every case.
"""

from __future__ import annotations

import argparse
import collections
import csv

import numpy as np

import validation_common as vc

import freq_offset_hunt as foh


SHIFTS = [-12.0, -4.5, 0.0, 7.25, 15.0]       # injected delay, samples
FREQS = [-0.18, -0.04, 0.0, 0.07, 0.21]       # injected tone, Hz


def chip_waveform(n: int, chip_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Constant-modulus code: a random phase, held for each chip."""
    n_chips = -(-n // chip_samples)
    chips = np.exp(1j * rng.uniform(0.0, 2 * np.pi, n_chips))
    return np.repeat(chips, chip_samples)[:n].astype(np.complex64)


def inject(tx: np.ndarray, fs: float, shift_samples: float, freq_hz: float,
           amp: float, rng: np.random.Generator) -> np.ndarray:
    """Echo of tx arriving shift_samples late (linear interpolation between
    samples; zero before the code starts), shifted by freq_hz and scaled by
    amp, in complex white noise of unit variance per component."""
    n = len(tx)
    k = np.arange(n)
    src = k - shift_samples
    echo = (np.interp(src, k, tx.real, left=0.0, right=0.0)
            + 1j * np.interp(src, k, tx.imag, left=0.0, right=0.0))
    del src
    echo *= np.exp(2j * np.pi * freq_hz * k / fs)
    echo *= amp
    echo += rng.standard_normal(n)
    echo += 1j * rng.standard_normal(n)
    return echo.astype(np.complex64)


def amplitude_for(tone_snr: float, n: int) -> float:
    """Echo amplitude per sample (noise of unit variance per component)
    whose tone, aligned and on an FFT bin, has strength tone_snr after n
    samples (see the module docstring)."""
    return tone_snr * np.sqrt(2.0 * np.log(2.0) / n)


def worst(rows: list[dict], key: str):
    """The case with the largest |rows[key]| among finite values, or None."""
    finite = [r for r in rows if np.isfinite(r[key])]
    if not finite:
        return None
    r = max(finite, key=lambda r: abs(r[key]))
    return {"value": r[key], "shift_true_samples": r["shift_true_samples"],
            "freq_true_hz": r["freq_true_hz"], "snr": r["snr"],
            "timing_status": r["timing_status"], "accepted": r["accepted"]}


def error_stats(rows: list[dict]) -> dict:
    """Shift and frequency error statistics, over all rows and over those
    the stacker's gates accept, with the counts."""
    acc = [r for r in rows if r["accepted"]]
    return {
        "n": len(rows),
        "n_unresolved": sum(r["timing_status"] != "resolved" for r in rows),
        "n_accepted": len(acc),
        "shift_error_samples": vc.finite_stats([r["shift_error_samples"] for r in rows]),
        "freq_error_hz": vc.finite_stats([r["freq_error_hz"] for r in rows]),
        "shift_error_samples_accepted": vc.finite_stats(
            [r["shift_error_samples"] for r in acc]),
        "freq_error_hz_accepted": vc.finite_stats([r["freq_error_hz"] for r in acc]),
        "snr": vc.finite_stats([r["snr"] for r in rows]),
    }


def stats_by(rows: list[dict], key: str) -> dict:
    """error_stats for each value of rows[key]."""
    return {f"{value:+g}": error_stats([r for r in rows if r[key] == value])
            for value in dict.fromkeys(r[key] for r in rows)}


def summarize(rows: list[dict], tone_snr: float, amp: float) -> dict:
    """One tone strength's results for the JSON."""
    on_bin = [r for r in rows if r["shift_true_samples"] == 0.0 and r["freq_true_hz"] == 0.0]
    return {
        "tone_snr_target": tone_snr,
        "snr_amp": amp,
        "on_bin_aligned_snr_measured": on_bin[0]["snr"] if on_bin else None,
        "timing_status_counts": dict(collections.Counter(r["timing_status"] for r in rows)),
        "n_tone_at_band_edge": sum(r["tone_at_band_edge"] for r in rows),
        "n_nonfinite": sum(not (np.isfinite(r["shift_error_samples"])
                                and np.isfinite(r["freq_error_hz"])) for r in rows),
        "all_cases": error_stats(rows),
        "worst_shift_error": worst(rows, "shift_error_samples"),
        "worst_freq_error": worst(rows, "freq_error_hz"),
        "worst_shift_error_accepted": worst([r for r in rows if r["accepted"]],
                                            "shift_error_samples"),
        "worst_freq_error_accepted": worst([r for r in rows if r["accepted"]],
                                           "freq_error_hz"),
        # Coupling: how the delay error depends on the injected tone, and the
        # frequency error on the injected delay.
        "by_injected_tone_hz": stats_by(rows, "freq_true_hz"),
        "by_injected_shift_samples": stats_by(rows, "shift_true_samples"),
    }


def describe_worst(label: str, w, scale: float, unit: str) -> str:
    if w is None:
        return f"{label}: no case"
    return (f"{label} {scale * w['value']:+.3f} {unit} (shift {w['shift_true_samples']:+g}, "
            f"tone {w['freq_true_hz']:+g} Hz, snr {w['snr']:.0f}, {w['timing_status']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--duration-s", type=float, default=30.0,
                        help="synthetic length; default the shortest production look")
    parser.add_argument("--sample-rate", type=float, default=250_000.0)
    parser.add_argument("--chip-rate", type=float, default=50_000.0,
                        help="chips per second of the synthetic code")
    parser.add_argument("--max-shift", type=int, default=40,
                        help="timing search half-width, samples "
                             "(registration_stability.py's default)")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--tone-snrs", type=float, nargs="+",
                        default=[foh.MIN_TONE_SNR, 40.0, 180.0],
                        help="echo strengths to run, each as the tone strength it "
                             "gives with no delay and no tone (default: the "
                             "stacker's threshold, 40 and 180)")
    args = parser.parse_args()

    chip_samples = args.sample_rate / args.chip_rate
    if chip_samples != int(chip_samples):
        raise SystemExit(f"--sample-rate / --chip-rate = {chip_samples:g}: the "
                         "synthetic holds each chip for a whole number of samples")
    chip_samples = int(chip_samples)
    # The timing search must reach past every injected delay by more than a
    # chip, or the leading peak can be cut off at the edge of the search.
    if max(abs(s) for s in SHIFTS) + chip_samples >= args.max_shift:
        raise SystemExit(f"--max-shift {args.max_shift} does not cover the injected "
                         f"delays (up to {max(abs(s) for s in SHIFTS):g} samples) "
                         f"plus a chip ({chip_samples} samples)")

    vc.ensure_dirs()
    rng = np.random.default_rng(args.seed)
    n = int(round(args.duration_s * args.sample_rate))
    tx = chip_waveform(n, chip_samples, rng)

    rows, by_strength = [], {}
    for tone_snr in args.tone_snrs:
        amp = amplitude_for(tone_snr, n)
        print(f"tone strength {tone_snr:g} (echo amplitude {amp:.4g} per sample):")
        sub = []
        for shift in SHIFTS:
            for freq in FREQS:
                rx = inject(tx, args.sample_rate, shift, freq, amp, rng)
                m = foh.measure_offset(rx, tx, args.sample_rate, chip_samples,
                                       max_shift=args.max_shift)
                del rx
                reasons = foh.measurement_problems(m, args.max_shift, foh.MIN_TONE_SNR)
                sub.append({
                    "tone_snr_target": tone_snr,
                    "snr_amp": amp,
                    "shift_true_samples": shift,
                    "freq_true_hz": freq,
                    "shift_est_samples": m["shift_refined"],
                    "shift_strongest_samples": m["shift_strongest"],
                    "timing_status": m["timing_status"],
                    "freq_est_hz": m["f_centroid"],
                    "shift_error_samples": m["shift_refined"] - shift,
                    "freq_error_hz": m["f_centroid"] - freq,
                    "snr": m["snr"],
                    "tone_at_band_edge": m["tone_at_band_edge"],
                    "accepted": not reasons,
                    "rejected_because": "; ".join(reasons),
                })
                r = sub[-1]
                print(f"  shift {shift:+6.2f} tone {freq:+.3f} Hz: shift error "
                      f"{r['shift_error_samples']:+.3f} samples, freq error "
                      f"{1e3 * r['freq_error_hz']:+.2f} mHz, snr {r['snr']:.0f}, "
                      f"timing {r['timing_status']}"
                      + (f"  REJECTED: {r['rejected_because']}" if reasons else ""))
        by_strength[f"{tone_snr:g}"] = summarize(sub, tone_snr, amp)
        rows += sub

    csv_path = vc.RESULTS_DIR / "timing_frequency_separability.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "purpose": "controlled delay/frequency injection through the production "
                   "estimator (freq_offset_hunt.measure_offset)",
        "criterion": "none: measurements only",
        "duration_s": args.duration_s,
        "sample_rate_hz": args.sample_rate,
        "chip_rate_hz": args.chip_rate,
        "chip_samples": chip_samples,
        "max_shift_samples": args.max_shift,
        "min_tone_snr": foh.MIN_TONE_SNR,
        "amplitude_rule": "amp = tone_snr * sqrt(2 ln 2 / n): the tone strength of an "
                          "aligned, on-bin echo over unit-variance noise",
        "accepted_means": "passes the stacker's gates on a timing measurement "
                          "(freq_offset_hunt.measurement_problems)",
        "shifts_samples": SHIFTS,
        "freqs_hz": FREQS,
        "n_rows": len(rows),
        "n_unresolved": sum(r["timing_status"] != "resolved" for r in rows),
        "n_accepted": sum(r["accepted"] for r in rows),
        "by_tone_snr": by_strength,
    }
    json_path = vc.write_json("timing_frequency_separability.json", payload)
    print(f"wrote {vc.report_path(csv_path)}")
    print(f"wrote {vc.report_path(json_path)}")

    for key, s in by_strength.items():
        a = s["all_cases"]
        print(f"\ntone strength {key}: measured {s['on_bin_aligned_snr_measured']:.1f} "
              f"with no delay and no tone; timing {s['timing_status_counts']}; "
              f"{a['n_accepted']} of {a['n']} cases pass the stacker's gates")
        if s["n_nonfinite"]:
            print(f"  {s['n_nonfinite']} cases gave no finite estimate (see the CSV); "
                  "the statistics are over the others")
        print("  " + describe_worst("worst shift error, all cases",
                                    s["worst_shift_error"], 1.0, "samples"))
        print("  " + describe_worst("worst shift error, accepted cases",
                                    s["worst_shift_error_accepted"], 1.0, "samples"))
        print("  " + describe_worst("worst freq error, all cases",
                                    s["worst_freq_error"], 1e3, "mHz"))
        print("  " + describe_worst("worst freq error, accepted cases",
                                    s["worst_freq_error_accepted"], 1e3, "mHz"))
        for tone, t in s["by_injected_tone_hz"].items():
            e = t["shift_error_samples"]
            if e["n"] == 0:
                print(f"  tone {tone} Hz: no finite shift estimate")
                continue
            print(f"  tone {tone} Hz: shift error max |{e['max_abs']:.3f}| "
                  f"mean {e['mean']:+.3f} samples ({e['n']} delays, "
                  f"{t['n_unresolved']} unresolved, {t['n_accepted']} accepted)")
    print(f"\n{payload['n_unresolved']} of {len(rows)} timings unresolved, "
          f"{payload['n_accepted']} accepted by the stacker's gates")
    print("measurements only: no pass/fail criterion")


if __name__ == "__main__":
    main()
