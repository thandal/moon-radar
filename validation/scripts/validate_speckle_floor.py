"""Measure how the band-passed stack variance falls with look count.

REPORT section 5 splits the band-passed stack variance into speckle, which
falls as 1/N with the number of looks, and a constant floor (whatever
repeats from look to look). A constant floor alone cannot distinguish
*structure* from N-independent *systematics* (stable artifacts). This
script measures both:

1. Variance vs N: for each rung N of a look-count ladder, draw --draws
   random subsets of N looks (within one channel; optionally one session to
   keep registration out of the picture), stack them, band-pass at
   production's scales (0.3-2.5 deg) with the linear band-pass
   registration_analysis.bandpass (see main for why not the stacker's
   log_bandpass), and take the variance.
   Fit var(N) = A/N + C, each rung weighted by the inverse variance of its
   mean over the draws. Reports the floor fraction C / (A/N_full + C), the
   fit's residuals, the same fit without the smallest-N rung (a result that
   moves much without it is fragile), and the full stack's variance against
   the fit's prediction for it.

2. Split-half correlation: two disjoint half-stacks, band-passed,
   correlated. True structure AND stable systematics both survive this;
   speckle does not. The pair (floor fraction, split-half r) is reported so
   the floor's composition can be argued: a high floor with high split-half
   correlation on *rotating-geometry* looks (the stripe/ring patterns rotate
   between looks) is structure; to probe artifacts, re-run with --session
   restricted to a short session (geometry nearly frozen -> artifacts
   survive the split like structure does).

The looks and their radiometry are production's: stack_maps.select_looks
(the gates) and stack_maps.look_brightness (brightness per unit area, noise
floor removed). Every stack is the plain mean of its looks' brightness, in
linear units, over one fixed set of pixels: those every selected look
covers (see main for both choices). Measurements only: no pass/fail
criterion.

Needs run directories made with the 2026-10-21 pipeline
(registration_stability.py): their per-look maps and runs CSVs.
stack_maps.select_looks refuses older runs CSVs, which lack columns its
gates read (rim_status, timing_status, tone_at_band_edge, dem_file,
spice_kernels, ...), so results/STOCKERT_2025 must be regenerated before
the example below runs. The JSON records the looks left out, with the
reason, and the captures the batch did not map.

Run from the repo root:
  .conda/bin/python validation/scripts/validate_speckle_floor.py \
      --run-dir results/STOCKERT_2025 --stations stockert \
      [--chan chan1] [--session all]
Outputs: validation/results/speckle_floor_<stations>_<chan>[_<session>].json,
log to stdout.
"""

from __future__ import annotations

import argparse

import healpy as hp
import numpy as np

import validation_common as vc

import doppler_equator_alignment as dea
import freq_offset_hunt as foh
import registration_analysis as ra
import stack_maps as sm


def fit_floor(ns, y, se):
    """Weighted least squares of y = A/N + C, weights 1/se^2. Returns
    (A, C, residuals, chi2, degrees of freedom)."""
    ns, y, se = (np.asarray(v, dtype=float) for v in (ns, y, se))
    M = np.column_stack([1.0 / ns, np.ones_like(ns)])
    (A, C), *_ = np.linalg.lstsq(M / se[:, None], y / se, rcond=None)
    resid = y - (A / ns + C)
    return float(A), float(C), resid, float(np.sum((resid / se) ** 2)), len(ns) - 2


def region_corr(a, b, region):
    aa = a[region] - a[region].mean()
    bb = b[region] - b[region].mean()
    return float(np.mean(aa * bb) / (aa.std() * bb.std()))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", nargs="+", required=True,
                   help="run directories with the per-look maps and runs CSVs")
    p.add_argument("--stations", nargs="+", required=True, choices=sorted(dea.STATION_RX),
                   help="receivers whose looks are stacked (as stack_maps.py)")
    p.add_argument("--chan", default="chan1")
    p.add_argument("--session", default="all",
                   help='"all" or one session key (e.g. 2025-09-10_11) to '
                        "freeze geometry for the artifact probe")
    p.add_argument("--draws", type=int, default=12,
                   help="random subsets per ladder rung (at least 2: their spread "
                        "weights the rung in the fit)")
    p.add_argument("--min-count", type=int, default=3,
                   help="the smallest stack, where the ladder starts (stack_maps.py's "
                        "minimum looks per pixel)")
    p.add_argument("--min-snr", type=float, default=foh.MIN_TONE_SNR)
    p.add_argument("--min-tx-s", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.draws < 2:
        p.error("--draws must be at least 2")

    rows, looks, left_out = vc.production_looks(args.run_dir, args.stations, [args.chan],
                                                args.min_snr, args.min_tx_s)
    if args.session != "all":
        inside = [sm.session_of(r["rx_file"]) == args.session for r in rows]
        left_out["looks_left_out"] += [
            {"rx_file": r["rx_file"], "station": r["_station"], "chan": r["_chan"],
             "reason": f"outside --session {args.session}"}
            for r, ok in zip(rows, inside) if not ok]
        rows = [r for r, ok in zip(rows, inside) if ok]
        looks = [I for I, ok in zip(looks, inside) if ok]
    n_full = len(rows)
    print(f"[*] {n_full} looks ({args.stations} {args.chan}, session={args.session})")
    if n_full == 0:
        raise SystemExit("no looks to stack")

    # One fixed set of pixels for every stack: those every look covers.
    # There a stack of N looks averages exactly N values, so its speckle
    # variance is one look's over N -- the A/N + C model holds pixel by
    # pixel, at every rung, on the same pixels. (A pixel some look lacks
    # would average fewer than N looks, and the set of covered pixels would
    # grow with N, mixing coverage into the variance.) --min-count does not
    # thin these pixels: each has all N >= --min-count looks.
    npix = len(looks[0])
    common = np.ones(npix, dtype=bool)
    anywhere = np.zeros(npix, dtype=bool)
    for I in looks:
        common &= np.isfinite(I)
        anywhere |= np.isfinite(I)
    n_common = int(common.sum())
    if n_common == 0:
        raise SystemExit("no pixel is covered by every look; restrict --session")
    print(f"[*] {n_common} pixels covered by all {n_full} looks "
          f"({100 * n_common / anywhere.sum():.0f}% of those any look covers)")
    vals = np.stack([I[common] for I in looks])
    del looks

    step = 0.075
    lon_axis = np.arange(-55, 55 + step / 2, step)
    lat_axis = np.arange(-55, 55 + step / 2, step)
    lo_px, hi_px = 0.3 / step, 2.5 / step

    # Stacks are compared in linear brightness -- the mean production forms
    # (stack_maps.Accumulator.mean), not its log. The A/N + C model holds
    # exactly for a mean of N independent looks in linear units (speckle
    # variance over N, plus the variance of what repeats), only roughly for
    # its log. And with the noise floor removed a pixel's mean can be zero
    # or negative where the echo is weak, which has no log: dropping those
    # pixels would drop more of them at small N, where the mean is noisier,
    # and bias the ladder. In linear units every pixel counts; the variance
    # then weights bright areas more than a log map's would.
    # The band-pass is the linear one (ra.bandpass), not the stacker's
    # log_bandpass (which logs the speckle-averaged brightness): a linear
    # filter of a mean of N looks is the mean of the N filtered looks, so
    # A/N + C still holds after it; a log taken after the average would
    # not keep that.
    def band(idx):
        """Band-passed grid of the mean of the looks idx (indices or a slice)."""
        m = np.full(npix, hp.UNSEEN, dtype=np.float64)
        m[common] = vals[idx].mean(axis=0, dtype=np.float64)
        return ra.bandpass(ra.grid_map(m, lon_axis, lat_axis), lo_px, hi_px)

    # The grid cells every stack has a value at: the mask is the same for
    # all of them, so grid it once.
    mask_map = np.full(npix, hp.UNSEEN)
    mask_map[common] = 0.0
    n_gridded = int(np.isfinite(ra.grid_map(mask_map, lon_axis, lat_axis)).sum())
    if n_gridded == 0:
        raise SystemExit("the pixels every look covers fill no cell of the "
                         "+-55 deg analysis grid")
    # The band-pass erodes and tapers that mask, at its outer edge and around
    # every hole in it (e.g. a pixel one look's degeneracy mask removes), and
    # is zero outside. Variances and correlations are over the cells it
    # leaves a value at -- set by the mask alone, so the same cells for every
    # stack.
    full_band = band(slice(None))  # all looks (a view, no copy)
    region = full_band != 0
    print(f"[*] {n_gridded} grid cells covered by every look; the band-pass leaves "
          f"a value at {int(region.sum())}")
    if not region.any():
        raise SystemExit("the band-pass leaves no grid cell: the pixels every look "
                         "covers have too many holes or too little extent; restrict "
                         "--session")

    # Rungs double from --min-count up to half the looks: two random subsets
    # of N <= n_full / 2 looks share on average at most half their looks, so
    # the spread of a rung's draws (its weight in the fit) reflects how the
    # stack varies with the looks drawn. Larger subsets overlap more and
    # their spread shrinks toward zero whatever the stack's real scatter;
    # the full stack (one subset, no spread) is instead compared with the
    # fit's prediction. Draws overlap even below n_full / 2, so the
    # standard errors, and the chi-square, are approximate.
    ladder = []
    n = args.min_count
    while n <= n_full // 2:
        ladder.append(n)
        n *= 2
    if len(ladder) < 3:
        raise SystemExit(f"{n_full} looks give {len(ladder)} rungs from "
                         f"{args.min_count} to half the looks; the fit with and "
                         "without the smallest rung needs at least 3")

    rng = np.random.default_rng(args.seed)
    var_rows = []
    for n in ladder:
        v = np.array([np.var(band(rng.choice(n_full, size=n, replace=False))[region])
                      for _ in range(args.draws)])
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        if not se > 0:
            raise SystemExit(f"N={n}: the {len(v)} draws have identical variance; "
                             "the rung cannot be weighted")
        var_rows.append({"n": n, "var_mean": float(v.mean()), "var_se": se,
                         "var_draws": v.tolist()})
        print(f"  N={n:4d}: band-passed var {v.mean():.4e} +- {se:.1e} "
              f"({len(v)} draws)")

    var_full = float(np.var(full_band[region]))

    def report_fit(sub, label):
        ns = [r["n"] for r in sub]
        A, C, resid, chi2, dof = fit_floor(ns, [r["var_mean"] for r in sub],
                                           [r["var_se"] for r in sub])
        floor_frac = C / (A / n_full + C)
        pred_full = A / n_full + C
        print(f"[*] fit {label}: var(N) = {A:.4e}/N + {C:.4e}; floor fraction at "
              f"N={n_full}: {100 * floor_frac:.1f}%; chi2 {chi2:.1f} for {dof} "
              f"degrees of freedom")
        for r, e in zip(sub, resid):
            print(f"      N={r['n']:4d}: residual {e:+.3e} ({e / r['var_se']:+.1f} "
                  "standard errors)")
        print(f"      full stack N={n_full}: measured {var_full:.4e}, fit predicts "
              f"{pred_full:.4e} ({var_full / pred_full - 1:+.1%})")
        return {"rungs": ns, "fit_A_over_N": A, "fit_floor": C,
                "floor_fraction_at_full": floor_frac,
                "speckle_fraction_at_full": 1 - floor_frac,
                "residuals": resid.tolist(),
                "residuals_over_se": (resid / np.array([r["var_se"] for r in sub])).tolist(),
                "chi2": chi2, "dof": dof,
                "predicted_var_full": pred_full}

    fit_all = report_fit(var_rows, "over all rungs")
    fit_drop = report_fit(var_rows[1:], f"without N={ladder[0]}")
    print("[*] floor fraction = structure + stable systematics share; speckle "
          f"share at N={n_full}: {100 * fit_all['speckle_fraction_at_full']:.1f}% "
          f"({100 * fit_drop['speckle_fraction_at_full']:.1f}% without the "
          "smallest rung)")

    # Split-half correlation at full depth.
    perm = rng.permutation(n_full)
    h = n_full // 2
    r_half = region_corr(band(perm[:h]), band(perm[h:2 * h]), region)
    print(f"[*] split-half band-passed correlation: {r_half:+.3f} "
          "(structure and stable systematics survive; speckle does not)")
    print("[*] measurements only: no pass/fail criterion")

    tag = "" if args.session == "all" else f"_{args.session}"
    out = vc.write_json(f"speckle_floor_{'_'.join(args.stations)}_{args.chan}{tag}.json", {
        "run_dir": args.run_dir, "stations": args.stations, "chan": args.chan,
        "session": args.session, "criterion": "none: measurements only",
        "units": "linear brightness (stack_maps.look_brightness), band-passed",
        "n_full": n_full, **left_out, "n_common_pixels": n_common,
        "n_grid_cells_covered": n_gridded,
        "n_region_grid_cells": int(region.sum()),
        "draws_per_rung": args.draws, "ladder": var_rows,
        "var_full_stack": var_full,
        "fit_all_rungs": fit_all,
        "fit_without_smallest_rung": fit_drop,
        "split_half_corr": r_half,
    })
    print(f"wrote {vc.report_path(out)}")


if __name__ == "__main__":
    main()
