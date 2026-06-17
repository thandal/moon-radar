"""Validate waveform and DD sampling signal-processing claims."""

from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage

import validation_common as vc


def mseq(deg: int, taps: list[int]) -> np.ndarray:
    s = [1] * deg
    out = np.empty(2 ** deg - 1)
    for i in range(2 ** deg - 1):
        out[i] = s[-1]
        fb = 0
        for tap in taps:
            fb ^= s[tap - 1]
        s = [fb] + s[:-1]
    return out * 2 - 1


def autocorr_metrics(x: np.ndarray) -> dict:
    X = np.fft.fft(x)
    ac = np.abs(np.fft.ifft(X * np.conj(X)))
    ac /= ac.max()
    sidelobe = ac.copy()
    sidelobe[0] = 0.0
    return {
        "pslr_db": float(20 * np.log10(sidelobe.max() + 1e-20)),
        "islr_db": float(10 * np.log10((sidelobe ** 2).sum() + 1e-20)),
    }


def bilinear_sample(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return scipy.ndimage.map_coordinates(img, [y, x], order=1, mode="constant",
                                         cval=np.nan)


def sampling_experiment(rng: np.random.Generator, n: int = 20000) -> dict:
    # Synthetic DD image: a smooth bright rim-like field plus a one-pixel-scale
    # delay ripple. Subpixel sample coordinates emulate projection coordinates.
    yy, xx = np.mgrid[0:512, 0:512]
    img = np.exp(-((yy - 260 - 35 * np.sin(xx / 70)) ** 2) / (2 * 16 ** 2))
    img += 0.25 * np.exp(-((xx - 260) ** 2 + (yy - 230) ** 2) / (2 * 45 ** 2))
    img += 0.04 * np.sin(xx * 2 * np.pi / 5.0)

    y = rng.uniform(2, img.shape[0] - 3, n)
    x = rng.uniform(2, img.shape[1] - 3, n)
    truth = bilinear_sample(img, y, x)
    nearest = img[np.rint(y).astype(int), np.rint(x).astype(int)]
    # Compare nearest neighbor to bilinear interpolation as a proxy for
    # projection quantization artifacts.
    err = nearest - truth
    return {
        "nearest_minus_bilinear": vc.finite_stats(err),
        "relative_rms": float(np.std(err) / np.std(truth)),
        "relative_p95_abs": float(np.quantile(np.abs(err), 0.95) / np.std(truth)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--code-length", type=int, default=100003)
    args = parser.parse_args()

    vc.ensure_dirs()
    rng = np.random.default_rng(args.seed)
    n = args.code_length
    q = 1301
    k = np.arange(n)
    zc = np.exp(-1j * np.pi * q * k * (k + 1) / n)
    bpsk_random = (rng.integers(0, 2, n) * 2 - 1).astype(complex)
    bpsk_mseq = mseq(17, [17, 3]).astype(complex)

    wave_rows = []
    for name, code in [
        ("ZC", zc),
        ("BPSK_mseq", bpsk_mseq),
        ("BPSK_random", bpsk_random),
    ]:
        row = {"waveform": name}
        row.update(autocorr_metrics(code))
        wave_rows.append(row)

    csv_path = vc.RESULTS_DIR / "waveform_ambiguity_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(wave_rows[0]))
        writer.writeheader()
        writer.writerows(wave_rows)

    sampling = sampling_experiment(rng)
    payload = {
        "purpose": "waveform ambiguity and DD nearest-neighbor sampling artifact scale",
        "waveform_metrics": wave_rows,
        "sampling_experiment": sampling,
        "interpretation_note": (
            "Waveform metrics isolate self-clutter at matched bandwidth/time. "
            "Sampling experiment quantifies the scale of nearest-neighbor DD "
            "quantization relative to a smooth bilinear reference."
        ),
    }
    json_path = vc.write_json("signal_processing.json", payload)

    fig, ax = plt.subplots(figsize=(6, 4))
    names = [r["waveform"] for r in wave_rows]
    islr = [r["islr_db"] for r in wave_rows]
    ax.bar(names, islr)
    ax.set_ylabel("delay ISLR (dB)")
    ax.set_title("Matched-bandwidth self-clutter")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png_path = vc.RESULTS_DIR / "waveform_ambiguity_metrics.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    print(f"wrote {vc.report_path(csv_path)}")
    print(f"wrote {vc.report_path(json_path)}")
    print(f"wrote {vc.report_path(png_path)}")
    print(f"nearest-vs-bilinear relative RMS {sampling['relative_rms']:.3f}")


if __name__ == "__main__":
    main()
