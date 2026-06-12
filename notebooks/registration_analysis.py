"""
Cross-look registration analysis (speckle-aware).

Single-look lunar radar maps are speckle-dominated at full resolution: the
fine structure is an interference pattern that decorrelates between looks
even when registration is perfect, so cross-correlating two raw looks
measures nothing (peak correlations ~1%). Stable features (albedo and
roughness contrast) survive incoherent averaging; speckle does not.

So this tool:
  1. grids each healpix map onto a common selenographic lon/lat grid,
  2. forms incoherent stacks: per-session half-stacks (early half vs late
     half) and full session stacks,
  3. band-passes them (smooth at ~0.3 deg to average speckle cells, remove
     >2.5 deg background so the specular envelope cannot dominate),
  4. measures offsets by masked FFT cross-correlation, sub-pixel peak:
       - intra-session: half-stack vs half-stack (registration stability
         over minutes-hours),
       - cross-session: session stack vs session stack (stability over
         weeks-months, large libration/geometry changes).

Registration is "stable" if offsets are small versus the map pixel
(nside 400 ~ 0.147 deg ~ 4.5 km) and the correlation peaks are clearly
above the speckle floor.

Usage (from notebooks/):
    ../.conda/bin/python registration_analysis.py
"""

import argparse
import csv
import os
import re

import numpy as np
import healpy as hp
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

KM_PER_DEG = 2 * np.pi * 1737.4 / 360.0  # ~30.3 km

SESSION_OF = {"2025_06_21": "2025-06-21", "2025_09_10": "2025-09-10/11",
              "2025_09_11": "2025-09-10/11", "2025_09_16": "2025-09-16"}


def grid_map(m, lon_axis, lat_axis, multiplicity=None, max_mult_factor=3.0):
    """Sample a healpix map (UNSEEN-masked) onto a lon/lat grid (NaN outside).

    If a DD-cell multiplicity map is given, pixels in degenerate mapping
    regions (multiplicity > max_mult_factor * median) are masked: these are
    the Doppler-equator stripe and the sub-radar blow-up, where one bright
    image cell smears along a surface arc. They carry no per-pixel surface
    information and would otherwise dominate any cross-look correlation.
    """
    mask = (m != hp.UNSEEN) & np.isfinite(m)
    if multiplicity is not None:
        med = np.median(multiplicity[multiplicity > 0])
        mask &= multiplicity <= max_mult_factor * med
    lon_g, lat_g = np.meshgrid(lon_axis, lat_axis)
    theta = np.radians(90.0 - lat_g).ravel()
    phi = np.radians(lon_g).ravel()
    vals = np.where(mask, m, 0.0)
    g_v = hp.get_interp_val(vals, theta, phi).reshape(lon_g.shape)
    g_w = hp.get_interp_val(mask.astype(float), theta, phi).reshape(lon_g.shape)
    return np.where(g_w > 0.99, g_v / np.maximum(g_w, 1e-9), np.nan)


def nanstack(grids):
    """Incoherent mean of looks; valid where at least two looks contribute."""
    arr = np.array(grids)
    count = np.isfinite(arr).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(arr, axis=0)
    return np.where(count >= min(2, len(grids)), mean, np.nan)


def bandpass(grid, lo_sigma_px, hi_sigma_px):
    """Masked band-pass: speckle-average at lo_sigma, remove >hi_sigma trend."""
    mask = np.isfinite(grid)
    v = np.where(mask, grid, 0.0)
    w = mask.astype(float)

    def nsmooth(sigma):
        sv = scipy.ndimage.gaussian_filter(v, sigma)
        sw = scipy.ndimage.gaussian_filter(w, sigma)
        return np.where(sw > 0.05, sv / np.maximum(sw, 1e-9), 0.0)

    band = nsmooth(lo_sigma_px) - nsmooth(hi_sigma_px)
    band = np.where(mask, band, 0.0)
    w_eroded = scipy.ndimage.binary_erosion(mask, iterations=int(2 * hi_sigma_px))
    return band * scipy.ndimage.gaussian_filter(w_eroded.astype(float), hi_sigma_px / 4)


def xcorr_offset(a, b, search_px, exclude_px):
    """Offset (dy, dx) of b vs a by FFT cross-correlation, sub-pixel peak.

    Also returns a significance measure: the ratio of the main peak to the
    strongest correlation outside an exclude_px radius around it (within the
    search window). Near 1 means the "offset" is indistinguishable from the
    noise floor; >~1.5 indicates a genuine feature lock.
    """
    cc = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)),
                                       s=a.shape))
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    win = cc[cy - search_px:cy + search_px + 1, cx - search_px:cx + search_px + 1]
    k = np.unravel_index(np.argmax(win), win.shape)
    dy, dx = k[0] - search_px, k[1] - search_px

    def refine(i, j, axis):
        lo = win[i - 1, j] if axis == 0 else win[i, j - 1]
        hi = win[i + 1, j] if axis == 0 else win[i, j + 1]
        c = win[i, j]
        den = lo - 2 * c + hi
        return 0.5 * (lo - hi) / den if den < 0 else 0.0

    if 0 < k[0] < win.shape[0] - 1 and 0 < k[1] < win.shape[1] - 1:
        dy += refine(k[0], k[1], 0)
        dx += refine(k[0], k[1], 1)
    peak = win.max() / np.sqrt((a ** 2).sum() * (b ** 2).sum())

    yy, xx = np.mgrid[:win.shape[0], :win.shape[1]]
    outside = (yy - k[0]) ** 2 + (xx - k[1]) ** 2 > exclude_px ** 2
    sidelobe = win[outside].max() if outside.any() else np.nan
    significance = win.max() / sidelobe if sidelobe > 0 else np.inf
    return dy, dx, peak, significance


def report(label, a, b, search_px, exclude_px, step):
    dy, dx, peak, signif = xcorr_offset(a, b, search_px, exclude_px)
    dlat, dlon = dy * step, dx * step
    print(f"  {label:48s} dlon {dlon:+.3f} deg dlat {dlat:+.3f} deg "
          f"({dlon*KM_PER_DEG:+.1f}, {dlat*KM_PER_DEG:+.1f}) km   "
          f"corr {peak:.3f}  signif {signif:.2f}")
    return {"pair": label, "dlon_deg": dlon, "dlat_deg": dlat,
            "dlon_km": dlon * KM_PER_DEG, "dlat_km": dlat * KM_PER_DEG,
            "peak_corr": peak, "significance": signif}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=os.path.join(os.path.dirname(__file__),
                                                          "results/REGISTRATION"))
    parser.add_argument("--step-deg", type=float, default=0.075)
    parser.add_argument("--extent-deg", type=float, default=55.0)
    parser.add_argument("--lo-sigma-deg", type=float, default=0.3)
    parser.add_argument("--hi-sigma-deg", type=float, default=2.5)
    parser.add_argument("--search-deg", type=float, default=1.5)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.run_dir, "registration_runs.csv"))))
    rows.sort(key=lambda r: r["rx_start_utc"])
    step = args.step_deg
    lon_axis = np.arange(-args.extent_deg, args.extent_deg + step / 2, step)
    lat_axis = np.arange(-args.extent_deg, args.extent_deg + step / 2, step)
    lo_px, hi_px = args.lo_sigma_deg / step, args.hi_sigma_deg / step
    search_px = int(args.search_deg / step)

    by_session = {}
    for r in rows:
        date_key = re.search(r"(\d{4}_\d{2}_\d{2})", r["rx_file"]).group(1)
        sess = SESSION_OF[date_key]
        mult = np.load(r["mult_npy"]) if r.get("mult_npy") and os.path.exists(r["mult_npy"]) else None
        by_session.setdefault(sess, []).append(
            grid_map(np.load(r["map_npy"]), lon_axis, lat_axis, multiplicity=mult))
        print(f"  gridded {r['rx_file']} -> {sess}"
              + ("" if mult is not None else "  (no degeneracy mask!)"))

    exclude_px = int(0.5 / step)
    results = []
    print("\nIntra-session stability (early-half stack vs late-half stack):")
    session_bands = {}
    for sess, grids in sorted(by_session.items()):
        h = len(grids) // 2
        a = bandpass(nanstack(grids[:h]), lo_px, hi_px)
        b = bandpass(nanstack(grids[h:]), lo_px, hi_px)
        results.append(report(f"{sess}  ({h} vs {len(grids)-h} looks)",
                              a, b, search_px, exclude_px, step))
        session_bands[sess] = bandpass(nanstack(grids), lo_px, hi_px)

    print("\nCross-session stability (full session stacks):")
    names = sorted(session_bands)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            results.append(report(f"{names[i]} vs {names[j]}",
                                  session_bands[names[i]], session_bands[names[j]],
                                  search_px, exclude_px, step))

    out_csv = os.path.join(args.run_dir, "registration_offsets.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        keys = list(results[0].keys())
        f.write(",".join(keys) + "\n")
        for r in results:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\nWrote {out_csv}")

    # Visual: the three band-passed session stacks side by side
    fig, axes = pl.subplots(1, len(names), figsize=(7 * len(names), 7))
    for ax, name in zip(np.atleast_1d(axes), names):
        band = session_bands[name]
        s = np.nanstd(band[band != 0])
        ax.imshow(band, origin="lower", vmin=-3 * s, vmax=3 * s, cmap="gray",
                  extent=[lon_axis[0], lon_axis[-1], lat_axis[0], lat_axis[-1]])
        ax.set_title(name)
        ax.set_xlabel("selenographic lon (deg)")
        ax.set_ylabel("lat (deg)")
    pl.tight_layout()
    out_png = os.path.join(args.run_dir, "session_stacks_bandpassed.png")
    pl.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
