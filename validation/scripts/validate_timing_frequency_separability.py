"""Validate timing/frequency offset separability in the product estimator.

The report's error budget assumes delay timing errors live on the delay axis
and reference-frequency errors live on the Doppler axis. This synthetic test
injects controlled shifts and tones into a constant-modulus pseudo-ZC-like
waveform, runs the same product-method estimator used by the pipeline, and
measures cross-coupling.

Frequency recovery is SNR-robust, so its error is summarized over every case.
The delay estimate is the argmax of a broad coherence bump and is only
trustworthy above an SNR floor (``--min-snr``); below it the synthetic's
between-bin tones fall in FFT-scalloping notches that depress the tone SNR and
let the argmax wander. The headline shift-error statistic is therefore gated to
reliable detections, with the ungated value kept alongside it for transparency.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

import validation_common as vc

import freq_offset_hunt as foh


def synth_waveform(n: int, rng: np.random.Generator) -> np.ndarray:
    # Constant modulus with roughly chip-scale phase coherence. This targets
    # estimator behavior, not the exact production ZC root.
    phase = np.cumsum(rng.normal(0.0, 0.53, n))
    return np.exp(1j * phase).astype("complex64")


def inject(tx: np.ndarray, fs: float, shift_samples: float, freq_hz: float,
           snr_amp: float, rng: np.random.Generator) -> np.ndarray:
    n = len(tx)
    t = np.arange(n) / fs
    idx = np.arange(n) - shift_samples
    real = np.interp(idx, np.arange(n), tx.real, left=0.0, right=0.0)
    imag = np.interp(idx, np.arange(n), tx.imag, left=0.0, right=0.0)
    echo = (real + 1j * imag) * np.exp(1j * 2 * np.pi * freq_hz * t)
    noise = rng.normal(size=n) + 1j * rng.normal(size=n)
    return (snr_amp * echo + noise).astype("complex64")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--sample-rate", type=float, default=250_000.0)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--snr-amp", type=float, default=0.2)
    parser.add_argument("--min-snr", type=float, default=30.0,
                        help="reliability gate for the delay estimate: rows whose "
                             "narrow-band tone SNR is below this are excluded from "
                             "the headline shift-error statistic. Empirically the "
                             "delay argmax on the broad-autocorrelation synthetic "
                             "is stable above ~30 and wanders below it (the low-SNR "
                             "rows are FFT-scalloping notches of the between-bin "
                             "test tones, not production cases). Frequency error is "
                             "SNR-robust and always reported over all rows.")
    args = parser.parse_args()

    vc.ensure_dirs()
    rng = np.random.default_rng(args.seed)
    n = int(args.duration_s * args.sample_rate)
    tx = synth_waveform(n, rng)

    shifts = [-12.0, -4.5, 0.0, 7.25, 15.0]
    freqs = [-0.18, -0.04, 0.0, 0.07, 0.21]
    rows = []
    for shift in shifts:
        for freq in freqs:
            rx = inject(tx, args.sample_rate, shift, freq, args.snr_amp, rng)
            m = foh.measure_offset(rx, tx, args.sample_rate, max_shift=25)
            rows.append({
                "shift_true_samples": shift,
                "freq_true_hz": freq,
                "shift_est_samples": m["shift_refined"],
                "freq_est_hz": m["f_centroid"],
                "shift_error_samples": m["shift_refined"] - shift,
                "freq_error_hz": m["f_centroid"] - freq,
                "snr": m["snr"],
                "reliable": bool(m["snr"] >= args.min_snr),
            })

    csv_path = vc.RESULTS_DIR / "timing_frequency_separability.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reliable = [r for r in rows if r["reliable"]]
    shift_err_all = [r["shift_error_samples"] for r in rows]
    shift_err_rel = [r["shift_error_samples"] for r in reliable]
    payload = {
        "purpose": "controlled delay/frequency injection through product estimator",
        "duration_s": args.duration_s,
        "sample_rate_hz": args.sample_rate,
        "min_snr_gate": args.min_snr,
        "n_rows": len(rows),
        "n_reliable": len(reliable),
        # Frequency error is SNR-robust, so it is summarized over all rows. The
        # delay estimate is only trustworthy above the SNR gate, so its headline
        # is the reliable subset; the ungated value is kept alongside so nothing
        # is hidden. Reporting only the ungated max conflated estimator coupling
        # with detections the pipeline would never use.
        "freq_error_hz": vc.finite_stats([r["freq_error_hz"] for r in rows]),
        "shift_error_samples_all": vc.finite_stats(shift_err_all),
        "shift_error_samples_reliable": vc.finite_stats(shift_err_rel),
        "max_abs_freq_error_hz": max(abs(r["freq_error_hz"]) for r in rows),
        "max_abs_shift_error_samples_all": max(abs(e) for e in shift_err_all),
        "max_abs_shift_error_samples_reliable":
            max((abs(e) for e in shift_err_rel), default=float("nan")),
    }
    json_path = vc.write_json("timing_frequency_separability.json", payload)
    print(f"wrote {vc.report_path(csv_path)}")
    print(f"wrote {vc.report_path(json_path)}")
    print(f"reliability gate snr >= {args.min_snr:g}: "
          f"{len(reliable)}/{len(rows)} rows pass")
    print(f"max shift error {payload['max_abs_shift_error_samples_reliable']:.3f} "
          f"samples (reliable); {payload['max_abs_shift_error_samples_all']:.3f} "
          f"(all rows)")
    print(f"max freq error {payload['max_abs_freq_error_hz']:.4f} Hz "
          f"(SNR-robust, all rows)")


if __name__ == "__main__":
    main()
