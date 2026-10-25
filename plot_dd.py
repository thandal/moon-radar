"""
Plot saved DD images (process_file's *_dd.npz, written with --save-pngs) as
power over the noise floor (signal plus noise, over noise: 0 dB is
noise alone, 3 dB signal equal to noise), the opposite- and same-sense channels side by side.

The noise floor is the pipeline's own (doppler_equator_alignment
.noise_floor_log_amp: the mean power in the leading delay columns, which lead
the echo), so the dB scale matches what stack_maps.py subtracts.
Power is averaged over blocks of --bin-doppler x --bin-delay cells before
converting to dB; the binning is shown in the title. Colour scale, axes and
orientation follow the CAMRAS plots (0-40 dB, Doppler decreasing to the
right, delay since the start of transmission upward) for side-by-side
comparison.

Usage (from the repo root):
    .conda/bin/python plot_dd.py --run-dir results/EVE2026_CIRC \\
        --capture stockert_25m_moon_2026_10_11_09_41_25 --out dd_0941.png
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

from doppler_equator_alignment import noise_floor_log_amp


def snr_db(npz_path, bin_dop, bin_del):
    d = np.load(npz_path)
    log_A = d["log_A"].astype(np.float64)
    power = np.exp(2.0 * log_A)
    noise = np.exp(2.0 * noise_floor_log_amp(log_A))
    nd, nl = (power.shape[0] // bin_dop) * bin_dop, (power.shape[1] // bin_del) * bin_del
    p = power[:nd, :nl].reshape(nd // bin_dop, bin_dop, nl // bin_del, bin_del).mean(axis=(1, 3))
    dop = d["doppler_hz"][:nd].reshape(-1, bin_dop).mean(axis=1)
    dly = d["delay_s"][:nl].reshape(-1, bin_del).mean(axis=1)
    return 10 * np.log10(p / noise), dop, dly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--capture", required=True,
                        help="file-name stem before _1299, e.g. stockert_25m_moon_2026_10_11_09_41_25")
    parser.add_argument("--bin-doppler", type=int, default=4)
    parser.add_argument("--bin-delay", type=int, default=4)
    parser.add_argument("--vmax", type=float, default=40.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    panels = []
    for chan, label in [("chan1", "opposite sense"), ("chan0", "same sense")]:
        hits = glob.glob(os.path.join(args.run_dir, f"{args.capture}_*.{chan}.sigmf-meta_dd.npz"))
        if len(hits) != 1:
            raise SystemExit(f"expected one {chan} DD file for {args.capture}, found {hits}")
        panels.append((label, *snr_db(hits[0], args.bin_doppler, args.bin_delay)))

    fig, axes = pl.subplots(1, 2, figsize=(14, 6), sharey=True, constrained_layout=True)
    for ax, (label, img, dop, dly) in zip(axes, panels):
        im = ax.imshow(img.T, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=args.vmax, interpolation="nearest",
                       extent=[dop[0], dop[-1], dly[0], dly[-1]])
        ax.set_xlim(max(dop[0], dop[-1]), min(dop[0], dop[-1]))
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("Doppler relative to the specular point (Hz)", fontsize=11)
    axes[0].set_ylabel("Delay since start of transmission (s)", fontsize=11)
    fig.colorbar(im, ax=axes, label="power over the noise floor (dB; 0 dB = noise alone)", fraction=0.025)
    fig.suptitle(f"{args.capture}  (power averaged over {args.bin_doppler} Doppler x "
                 f"{args.bin_delay} delay cells)", fontsize=13)
    fig.savefig(args.out, dpi=100)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
