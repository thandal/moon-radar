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

Usage (from the repo root):
    .conda/bin/python registration_analysis.py --run-dir results/STOCKERT_2025 \\
        --out-dir results/STOCKERT_2025 --stations stockert --chan chan1
"""

import argparse
import collections
import csv
import os

import numpy as np
import healpy as hp
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

KM_PER_DEG = 2 * np.pi * 1737.4 / 360.0  # ~30.3 km

# Map pixels whose DD cells spread over more than this many map pixels'
# worth of surface (doppler_equator_alignment.lunar_projection's
# multiplicity) are masked everywhere downstream: the Doppler-equator
# stripe and the sub-radar blow-up, where one image cell smears along a
# surface arc. Counted in pixels of the map's nside (4.4 km at nside 400).
MAX_MULTIPLICITY = 3.0


def grid_map(m, lon_axis, lat_axis, multiplicity=None):
    """Sample a healpix map (UNSEEN-masked) onto a lon/lat grid (NaN outside).

    Orientation (REPORT §5; enforced by test/test_registration_conventions.py):
    identity in selenographic coordinates -- a feature at (lon, lat) lands at
    grid column lon_axis ~ lon, row lat_axis ~ lat; +x is east (+lon), +y is
    north (+lat).

    If a DD-cell multiplicity map is given, pixels in degenerate mapping
    regions (multiplicity > MAX_MULTIPLICITY) are
    masked: these are the Doppler-equator stripe and the sub-radar blow-up,
    where one bright image cell smears along a surface arc. They carry no
    per-pixel surface information and would otherwise dominate any
    cross-look correlation.
    """
    mask = (m != hp.UNSEEN) & np.isfinite(m)
    if multiplicity is not None:
        mask &= multiplicity <= MAX_MULTIPLICITY
    lon_g, lat_g = np.meshgrid(lon_axis, lat_axis)
    theta = np.radians(90.0 - lat_g).ravel()
    phi = np.radians(lon_g).ravel()
    vals = np.where(mask, m, 0.0)
    g_v = hp.get_interp_val(vals, theta, phi).reshape(lon_g.shape)
    g_w = hp.get_interp_val(mask.astype(float), theta, phi).reshape(lon_g.shape)
    return np.where(g_w > 0.99, g_v / np.maximum(g_w, 1e-9), np.nan)


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


def _masked_smooth(values, mask, sigma):
    """Gaussian smooth of values over mask, normalized by the smoothed mask;
    NaN where less than 5 % of the kernel's weight falls on the mask."""
    sv = scipy.ndimage.gaussian_filter(np.where(mask, values, 0.0), sigma)
    sw = scipy.ndimage.gaussian_filter(mask.astype(float), sigma)
    out = np.full(values.shape, np.nan)
    ok = sw > 0.05
    out[ok] = sv[ok] / sw[ok]
    return out


def log_bandpass(grid, lo_sigma_px, hi_sigma_px):
    """Band-pass of a grid of linear brightness (stack_maps' mean, which can
    be zero or negative where the noise floor was removed) on a log scale.

    The speckle average at lo_sigma is taken on the brightness itself, where
    averaging is meaningful, so a pixel whose own mean is not positive still
    counts towards its neighbourhood; only then the log; then the >hi_sigma
    trend is removed, the mask eroded and tapered as in bandpass. Where the
    local average is not positive -- no signal over the whole neighbourhood
    -- there is no log value and the grid is masked."""
    mask = np.isfinite(grid)
    local = _masked_smooth(grid, mask, lo_sigma_px)
    lmask = mask & np.isfinite(local)
    lmask[lmask] = local[lmask] > 0
    log_local = np.full(grid.shape, np.nan)
    log_local[lmask] = np.log(local[lmask])
    trend = _masked_smooth(log_local, lmask, hi_sigma_px)
    ok = lmask & np.isfinite(trend)
    band = np.zeros(grid.shape)
    band[ok] = log_local[ok] - trend[ok]
    # The 2*hi_sigma margin keeps the trend's bias at the mask's outer edge
    # out of the band. Holes inside the data narrower than the speckle
    # averaging (lo_sigma) -- isolated pixels with no positive local mean --
    # are bridged by the masked smoothing and need no such margin, so they
    # are closed for the margin only (they stay 0 in the band).
    edge_mask = scipy.ndimage.binary_closing(lmask, iterations=int(np.ceil(lo_sigma_px)))
    edge_mask &= scipy.ndimage.binary_fill_holes(mask)
    w_eroded = scipy.ndimage.binary_erosion(edge_mask, iterations=int(2 * hi_sigma_px))
    return band * scipy.ndimage.gaussian_filter(w_eroded.astype(float), hi_sigma_px / 4)


XcorrResult = collections.namedtuple("XcorrResult", "dy dx peak significance at_edge")


def xcorr_offset(a, b, search_px):
    """Offset (dy, dx) of b vs a by FFT cross-correlation, sub-pixel peak.

    Sign convention (REPORT §5; enforced by
    test/test_registration_conventions.py): returns the pixel shift that
    REGISTERS b onto a. If b is a's content rolled by +k
    (np.roll(a, +k, axis=1) moves content +k = east), the reported dx is -k;
    likewise dy for axis=0 = north. Apply (dx, dy) to b to align it onto a.

    Returns XcorrResult(dy, dx, peak, significance, at_edge):
      peak: normalized correlation at the peak.
      at_edge: the peak is on the edge of the +-search_px window, so the
        true peak may lie outside it -- not a measurement. Each axis is
        refined to a fraction of a pixel only where it is inside the window.
      significance: the peak over the strongest correlation outside its main
        lobe -- the region around zero lag where a's own autocorrelation is
        positive (it ends where that first crosses zero), centred on the
        peak. Near 1: no stronger than other correlations in the window.
        NaN when the lobe fills the window or nothing outside it is
        positive.
    """
    def centred(x):
        return np.fft.fftshift(np.fft.irfft2(x, s=a.shape))

    fa = np.fft.rfft2(a)
    cc = centred(fa * np.conj(np.fft.rfft2(b)))
    acf = centred(np.abs(fa) ** 2)
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    sl = (slice(cy - search_px, cy + search_px + 1), slice(cx - search_px, cx + search_px + 1))
    win, acf_win = cc[sl], acf[sl]
    k = np.unravel_index(np.argmax(win), win.shape)
    dy, dx = float(k[0] - search_px), float(k[1] - search_px)
    last = 2 * search_px
    at_edge = k[0] in (0, last) or k[1] in (0, last)

    def refine(lo, c, hi):
        den = lo - 2 * c + hi
        return 0.5 * (lo - hi) / den if den < 0 else 0.0

    if 0 < k[0] < last:
        dy += refine(win[k[0] - 1, k[1]], win[k], win[k[0] + 1, k[1]])
    if 0 < k[1] < last:
        dx += refine(win[k[0], k[1] - 1], win[k], win[k[0], k[1] + 1])
    peak = win.max() / np.sqrt((a ** 2).sum() * (b ** 2).sum())

    labels, _ = scipy.ndimage.label(acf_win > 0)
    lobe = labels == labels[search_px, search_px]
    # Move the lobe to the peak without wrapping it round the window.
    lobe = scipy.ndimage.shift(lobe.astype(float), (k[0] - search_px, k[1] - search_px),
                               order=0, mode="constant", cval=0.0) > 0.5
    outside = ~lobe
    sidelobe = win[outside].max() if outside.any() else np.nan
    significance = win.max() / sidelobe if sidelobe > 0 else np.nan
    return XcorrResult(dy, dx, float(peak), float(significance), bool(at_edge))


# A cross-correlation "lock", as reports label it: a peak not on the edge of
# the search window and at least this many times the strongest correlation
# outside its main lobe (xcorr_offset's significance). A labelling
# convention, not a calibrated test: 1.5 is the level this module
# documented as a genuine lock when significance excluded a fixed 0.5 deg
# radius around the peak, and it has not been recalibrated for the
# main-lobe definition. stack_maps does not use it, but two reports select
# data by it (validate_absolute_registration takes the median of its headline offset
# over the locked scales; ata_stockert_crosscheck picks its control scale
# among them), so their headline numbers inherit this uncalibrated choice.
# Reports print every significance alongside the label.
LOCK_SIGNIFICANCE = 1.5


def is_lock(x):
    """Whether an XcorrResult is labelled a lock (see LOCK_SIGNIFICANCE). A
    NaN significance (undefined) is not a lock."""
    return (not x.at_edge) and bool(np.isfinite(x.significance)) \
        and x.significance >= LOCK_SIGNIFICANCE


def report(label, a, b, search_px, step):
    x = xcorr_offset(a, b, search_px)
    dlat, dlon = x.dy * step, x.dx * step
    print(f"  {label:48s} dlon {dlon:+.3f} deg dlat {dlat:+.3f} deg "
          f"({dlon*KM_PER_DEG:+.1f}, {dlat*KM_PER_DEG:+.1f}) km   "
          f"corr {x.peak:.3f}  significance {x.significance:.2f}"
          + ("  AT THE SEARCH EDGE (not a measurement)" if x.at_edge else ""))
    return {"pair": label, "dlon_deg": dlon, "dlat_deg": dlat,
            "dlon_km": dlon * KM_PER_DEG, "dlat_km": dlat * KM_PER_DEG,
            "peak_corr": x.peak, "significance": x.significance,
            "at_edge": x.at_edge}


def main():
    # Gating and radiometry are stack_maps.py's own (imported here, not at
    # module level: stack_maps imports this module).
    import doppler_equator_alignment as dea
    import stack_maps as sm

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stations", nargs="+", required=True)
    parser.add_argument("--chan", required=True)
    parser.add_argument("--step-deg", type=float, default=0.075)
    parser.add_argument("--extent-deg", type=float, default=55.0)
    parser.add_argument("--lo-sigma-deg", type=float, default=0.3)
    parser.add_argument("--hi-sigma-deg", type=float, default=2.5)
    parser.add_argument("--search-deg", type=float, default=1.5)
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    label = "_".join(args.stations) + f"_{args.chan}"   # outputs name their stations

    good, decisions = sm.select_looks(args.run_dir, args.stations, [args.chan])
    print(f"{len(good)} looks pass the stack gates, {len(decisions) - len(good)} excluded")
    if not good:
        raise SystemExit("no look passes the stack gates")
    step = args.step_deg
    lon_axis = np.arange(-args.extent_deg, args.extent_deg + step / 2, step)
    lat_axis = np.arange(-args.extent_deg, args.extent_deg + step / 2, step)
    lo_px, hi_px = args.lo_sigma_deg / step, args.hi_sigma_deg / step
    search_px = int(args.search_deg / step)

    npix = len(np.load(good[0]["map_npy"], mmap_mode="r"))
    cache = {}
    by_session = {}
    for r in good:
        cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), npix, cache,
                                rx_name=r["_rx"])
        try:
            I, r["_gain"] = sm.look_brightness(r, cosi)
        except sm.LookRejected as exc:
            print(f"  - {r['rx_file']}: {exc}")
            decisions = [(d, str(exc) if d is r else why) for d, why in decisions]
            continue
        # Session labels as the stacker forms them (from the file name's date).
        by_session.setdefault(sm.session_of(r["rx_file"]), []).append(I)
    looks_csv = os.path.join(args.out_dir, f"registration_analysis_looks_{label}.csv")
    sm.write_decisions(looks_csv, decisions)
    print(f"Wrote {looks_csv}")

    def band(looks):
        acc = sm.Accumulator(npix)
        for I in looks:
            acc.add(I)
        return sm.band_of(acc.mean(args.min_count), lon_axis, lat_axis, lo_px, hi_px)

    results = []
    print("\nIntra-session stability (early-half stack vs late-half stack):")
    session_bands = {}
    for sess, looks in sorted(by_session.items()):
        h = len(looks) // 2
        session_bands[sess] = band(looks)
        if h < args.min_count:
            print(f"  {sess}: {len(looks)} looks, too few for two half-stacks of "
                  f"{args.min_count} (--min-count); no intra-session measurement")
            continue
        results.append(report(f"{sess}  ({h} vs {len(looks)-h} looks)",
                              band(looks[:h]), band(looks[h:]), search_px, step))

    print("\nCross-session stability (full session stacks):")
    names = sorted(session_bands)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            results.append(report(f"{names[i]} vs {names[j]}",
                                  session_bands[names[i]], session_bands[names[j]],
                                  search_px, step))

    if not results:
        raise SystemExit("no offset could be measured (see the messages above); "
                         "nothing written")
    out_csv = os.path.join(args.out_dir, f"registration_offsets_{label}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {out_csv}")

    # Visual: the band-passed session stacks side by side
    fig, axes = pl.subplots(1, len(names), figsize=(7 * len(names), 7))
    for ax, name in zip(np.atleast_1d(axes), names):
        b = session_bands[name]
        sd = np.nanstd(b[b != 0])
        ax.imshow(b, origin="lower", vmin=-3 * sd, vmax=3 * sd, cmap="gray",
                  extent=[lon_axis[0], lon_axis[-1], lat_axis[0], lat_axis[-1]])
        ax.set_title(name)
        ax.set_xlabel("selenographic lon (deg)")
        ax.set_ylabel("lat (deg)")
    pl.tight_layout()
    out_png = os.path.join(args.out_dir, f"session_stacks_bandpassed_{label}.png")
    pl.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
