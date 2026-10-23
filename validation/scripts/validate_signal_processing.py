"""Validate waveform and DD sampling signal-processing claims.

Waveform metrics are the one-period periodic autocorrelation of each code:
peak sidelobe (PSLR) and integrated sidelobe (ISLR) levels relative to the
main peak. For a Zadoff-Chu code (odd length N, root q coprime to N) every
nonzero-lag sidelobe is exactly zero in theory; what the computation gives
is floating-point round-off, reported as the numerical floor, not as a
property of the code. For a +-1 m-sequence every sidelobe is exactly -1/N.
Measurements only: no pass/fail criterion.
"""

from __future__ import annotations

import argparse
import csv
import math

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


def db(power_ratio: float, scale: float = 10.0):
    """scale * log10 of a ratio; None for a ratio of exactly zero, which has
    no dB value."""
    return float(scale * np.log10(power_ratio)) if power_ratio > 0 else None


def autocorr_metrics(x: np.ndarray) -> dict:
    """Peak and integrated sidelobe levels of the one-period periodic
    autocorrelation, relative to its zero-lag peak (linear and dB)."""
    X = np.fft.fft(x)
    ac = np.abs(np.fft.ifft(X * np.conj(X)))
    ac /= ac[0]
    sidelobe = ac[1:]
    pslr = float(sidelobe.max())
    islr = float((sidelobe ** 2).sum())
    return {"pslr": pslr, "pslr_db": db(pslr, 20.0),
            "islr": islr, "islr_db": db(islr)}


def bilinear_sample(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return scipy.ndimage.map_coordinates(img, [y, x], order=1, mode="constant",
                                         cval=np.nan)


def sampling_experiment(rng: np.random.Generator, n: int = 20000,
                        pixel_cells: float = 3.0, supersample: int = 4) -> dict:
    # supersample: production's doppler_equator_alignment.SUPERSAMPLE (4),
    # written out to keep this script free of the CuPy/SPICE that module
    # loads; pass the current value if it changes.
    """How well a map pixel represents the DD image over its footprint.

    Synthetic DD image: a smooth bright rim-like field plus a one-cell-scale
    delay ripple. A map pixel covers pixel_cells x pixel_cells image cells
    (a 4.4 km pixel spans roughly 1-4 independent cells on the 2026-10-11
    looks; 3 is a middle value). Reference: the pixel's area average of the
    bilinear field (16 x 16 samples). Compared: the cell under the pixel
    centre (the projection before 2026-10-20) and the mean of the cells
    under supersample x supersample points across the pixel (production,
    doppler_equator_alignment.SUPERSAMPLE)."""
    yy, xx = np.mgrid[0:512, 0:512]
    img = np.exp(-((yy - 260 - 35 * np.sin(xx / 70)) ** 2) / (2 * 16 ** 2))
    img += 0.25 * np.exp(-((xx - 260) ** 2 + (yy - 230) ** 2) / (2 * 45 ** 2))
    img += 0.04 * np.sin(xx * 2 * np.pi / 5.0)

    half = pixel_cells / 2
    cy = rng.uniform(half + 2, img.shape[0] - half - 3, n)
    cx = rng.uniform(half + 2, img.shape[1] - half - 3, n)

    def offsets(k):
        # centres of k equal sub-intervals across the pixel
        return (np.arange(k) + 0.5) / k * pixel_cells - half

    def area_mean(k, sampler):
        o = offsets(k)
        oy, ox = np.meshgrid(o, o, indexing="ij")
        ys = cy[:, None] + oy.ravel()[None, :]
        xs = cx[:, None] + ox.ravel()[None, :]
        return sampler(ys, xs).mean(axis=1)

    def nearest(y, x):
        return img[np.rint(y).astype(int), np.rint(x).astype(int)]

    truth = area_mean(16, lambda y, x: bilinear_sample(img, y.ravel(), x.ravel()).reshape(y.shape))
    centre = nearest(cy, cx)
    production = area_mean(supersample, nearest)
    out = {"pixel_cells": pixel_cells, "supersample": supersample}
    for name, est in (("centre_cell", centre), ("production_mean", production)):
        err = est - truth
        out[name] = {"error": vc.finite_stats(err),
                     "relative_rms": float(np.std(err) / np.std(truth)),
                     "relative_p95_abs": float(np.quantile(np.abs(err), 0.95) / np.std(truth))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--code-length", type=int, default=100003)
    args = parser.parse_args()

    vc.ensure_dirs()
    rng = np.random.default_rng(args.seed)
    n = args.code_length
    q = 1301
    # The exactly-zero sidelobes below hold for this form of the code only
    # with an odd length and a root coprime to it.
    if n % 2 == 0 or math.gcd(q, n) != 1:
        raise SystemExit(f"Zadoff-Chu length {n}, root {q}: the length must be odd "
                         "and the root coprime to it")
    k = np.arange(n)
    zc = np.exp(-1j * np.pi * q * k * (k + 1) / n)
    bpsk_random = (rng.integers(0, 2, n) * 2 - 1).astype(complex)
    n_mseq = 2 ** 17 - 1
    bpsk_mseq = mseq(17, [17, 3]).astype(complex)

    # Theory, where it is exact: Zadoff-Chu sidelobes are all zero (no dB
    # value); an m-sequence's are all -1/N. Random BPSK has only an
    # expectation (ISLR about (N-1)/N, 0 dB), so no exact value.
    theory = {
        "ZC": {"pslr": 0.0, "islr": 0.0},
        "BPSK_mseq": {"pslr": 1.0 / n_mseq, "islr": (n_mseq - 1) / n_mseq ** 2},
        "BPSK_random": None,
    }
    wave_rows = []
    for name, code in [
        ("ZC", zc),
        ("BPSK_mseq", bpsk_mseq),
        ("BPSK_random", bpsk_random),
    ]:
        computed = autocorr_metrics(code)
        row = {"waveform": name, "length": len(code)}
        t = theory[name]
        if t is None:
            row.update(computed)
            row["theory"] = "no exact value (random code)"
        else:
            row.update({"theory_pslr": t["pslr"], "theory_pslr_db": db(t["pslr"], 20.0),
                        "theory_islr": t["islr"], "theory_islr_db": db(t["islr"])})
            if name == "ZC":
                # Exactly zero in theory: the computed values are the
                # floating-point floor of this calculation, nothing more.
                row.update({f"numerical_floor_{key}": v for key, v in computed.items()})
            else:
                row.update(computed)
        wave_rows.append(row)

    csv_path = vc.RESULTS_DIR / "waveform_ambiguity_metrics.csv"
    fields = list(dict.fromkeys(k for r in wave_rows for k in r))
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(wave_rows)

    sampling = sampling_experiment(rng)
    payload = {
        "purpose": "waveform ambiguity, and how well a map pixel represents the DD image over its footprint",
        "waveform_metrics": wave_rows,
        "sampling_experiment": sampling,
        "interpretation_note": (
            "Waveform metrics isolate self-clutter at matched bandwidth/time. "
            "Zadoff-Chu sidelobes are exactly zero in theory; its computed "
            "values are the floating-point floor of the calculation. "
            "Sampling experiment compares the cell under the pixel centre "
            "(before 2026-10-20) and production's mean over 4x4 points with "
            "the pixel's area average of a smooth bilinear field."
        ),
        "criterion": "none: measurements only",
    }
    json_path = vc.write_json("signal_processing.json", payload)

    # Zadoff-Chu has no bar: its ISLR is zero (no dB value) in theory, and
    # its computed value is round-off, not a level worth drawing.
    fig, ax = plt.subplots(figsize=(6, 4))
    shown = [r for r in wave_rows if r["waveform"] != "ZC"]
    ax.bar([r["waveform"] for r in shown], [r["islr_db"] for r in shown])
    ax.set_ylabel("delay ISLR (dB)")
    ax.set_title("Self-clutter (Zadoff-Chu: exactly zero, not drawn)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png_path = vc.RESULTS_DIR / "waveform_ambiguity_metrics.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)

    print(f"wrote {vc.report_path(csv_path)}")
    print(f"wrote {vc.report_path(json_path)}")
    print(f"wrote {vc.report_path(png_path)}")

    def fmt(v):
        return "exactly zero (no dB value)" if v is None else f"{v:.1f} dB"

    for r in wave_rows:
        if r["waveform"] == "ZC":
            print(f"ZC: sidelobes exactly zero in theory; numerical floor of this "
                  f"calculation: ISLR {fmt(r['numerical_floor_islr_db'])}")
        else:
            print(f"{r['waveform']}: ISLR {fmt(r['islr_db'])}, PSLR {fmt(r['pslr_db'])}")
    print(f"map pixel vs its area average, relative RMS: centre cell "
          f"{sampling['centre_cell']['relative_rms']:.3f}, production 4x4 mean "
          f"{sampling['production_mean']['relative_rms']:.3f}")


if __name__ == "__main__":
    main()
