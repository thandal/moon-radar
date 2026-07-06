"""Substantiate the "structure-limited" stack claim (REPORT §5/§6).

The REPORT states the band-passed stack variance follows speckle ∝ 1/N plus
a constant floor, with residual speckle ~13% of the variance at the full
stack — but no committed computation backs that number, and a constant floor
alone cannot distinguish *structure* from N-independent *systematics* (the
iso-delay ring artifact the REPORT itself names as the leading stack
artifact). This script measures both:

1. Variance-vs-N: draw K random subsets of N looks (within one channel;
   optionally one session to keep registration out of the picture), stack,
   band-pass exactly like production (grid_map + bandpass at 0.3-2.5 deg),
   and fit  var(N) = A/N + C  over the ladder. Reports the floor fraction
   C / (A/N_full + C) — the REPORT's "~13%" if the claim holds.

2. Split-half correlation: two disjoint half-stacks, band-passed, masked
   correlation. True structure AND stable systematics both survive this;
   speckle does not. The pair (floor fraction, split-half r) is reported so
   the floor's composition can be argued: a high floor with high split-half
   correlation on *rotating-geometry* looks (the stripe/ring patterns rotate
   between looks) is structure; to probe artifacts, re-run with
   --session restricted to a short session (geometry nearly frozen →
   artifacts survive the split like structure does).

Needs the per-look map products (results/<run>/... .npy) — data-machine only.

Run from the repo root:
  .conda/bin/python validation/scripts/validate_speckle_floor.py \
      [--run-dir results/LOLA_DEM_REGISTRATION] [--chan chan1] [--session all]
Outputs: validation/results/speckle_floor_<chan>.json, log to stdout.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from validation_common import finite_stats, write_json, report_path

import stack_maps as sm
import registration_analysis as ra


def load_gated_rows(run_dir, chan, min_snr=15.0):
    path = os.path.join(run_dir, f"registration_runs_{chan}.csv")
    if not os.path.exists(path):
        sys.exit(f"[!] {path} not found — this validation needs the "
                 "per-look map products (data machine).")
    rows = []
    for r in csv.DictReader(open(path)):
        gate_snr = float(r.get("pair_snr") or r["tone_snr"])
        shift = float(r.get("applied_shift_samples") or r["shift_refined"])
        if gate_snr < min_snr or abs(shift) >= 39.5:
            continue
        r["map_npy"] = sm.resolve_npy(r["map_npy"], run_dir)
        r["mult_npy"] = sm.resolve_npy(r["mult_npy"], run_dir)
        rows.append(r)
    return rows


def band_grid(intensity_sum, count, lon_axis, lat_axis, lo_px, hi_px, min_count):
    """Production-equivalent band-passed grid of a linear-intensity stack."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count >= min_count, intensity_sum / count, np.nan)
        logm = np.where(np.isfinite(mean) & (mean > 0), np.log10(mean), np.nan)
    import healpy as hp
    m = np.where(np.isfinite(logm), logm, hp.UNSEEN).astype(np.float32)
    return ra.bandpass(ra.grid_map(m, lon_axis, lat_axis), lo_px, hi_px)


def masked_corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return np.nan
    aa, bb = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    return float((aa * bb).mean() / (aa.std() * bb.std()))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", default="results/LOLA_DEM_REGISTRATION")
    p.add_argument("--chan", default="chan1")
    p.add_argument("--session", default="all",
                   help='"all" or one session key (e.g. 2025-09-10_11) to '
                        "freeze geometry for the artifact probe")
    p.add_argument("--n-subsets", type=int, default=12,
                   help="random subsets per ladder point")
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = load_gated_rows(args.run_dir, args.chan)
    if args.session != "all":
        rows = [r for r in rows if sm.session_of(r["rx_file"]) == args.session]
    n_full = len(rows)
    if n_full < 8:
        sys.exit(f"[!] only {n_full} gated looks — not enough for the fit")
    print(f"[*] {n_full} gated looks ({args.chan}, session={args.session})")

    # Per-look masked gain-normalized linear intensity (production identical).
    looks = [sm.look_linear_intensity(r["map_npy"], r["mult_npy"]) for r in rows]
    looks = [np.where(np.isfinite(I), I, np.nan).astype(np.float32) for I in looks]

    step = 0.075
    lon_axis = np.arange(-55, 55 + step / 2, step)
    lat_axis = np.arange(-55, 55 + step / 2, step)
    lo_px, hi_px = 0.3 / step, 2.5 / step

    def stack_grid(idx):
        s = np.nansum([looks[i] for i in idx], axis=0)
        c = np.sum([np.isfinite(looks[i]) for i in idx], axis=0)
        return band_grid(s, c, lon_axis, lat_axis, lo_px, hi_px,
                         min(args.min_count, max(1, len(idx) // 2)))

    rng = np.random.default_rng(args.seed)
    ladder = [n for n in (2, 4, 8, 16, 32, 64, 128) if n <= n_full]
    if ladder[-1] != n_full:
        ladder.append(n_full)

    med_var, var_rows = [], []
    for n in ladder:
        vs = []
        for _ in range(args.n_subsets if n < n_full else 1):
            idx = rng.choice(n_full, size=n, replace=False)
            vs.append(float(np.nanvar(stack_grid(idx))))
        med_var.append(float(np.median(vs)))
        var_rows.append({"n": n, "var_median": med_var[-1],
                         "var_all": vs})
        print(f"  N={n:4d}: band-passed var median {med_var[-1]:.4e} "
              f"({len(vs)} subsets)")

    # Fit var = A/N + C (least squares in [1/N, 1] basis).
    inv_n = np.array([1.0 / n for n in ladder])
    M = np.column_stack([inv_n, np.ones_like(inv_n)])
    (A, C), *_ = np.linalg.lstsq(M, np.array(med_var), rcond=None)
    floor_frac = C / (A / n_full + C)
    print(f"[*] fit: var(N) = {A:.4e}/N + {C:.4e}")
    print(f"[*] floor fraction at N={n_full}: {100 * floor_frac:.1f}% "
          "(REPORT §5 claims ~13% residual speckle => floor ~87%; "
          "floor fraction here = structure+systematics share)")
    print(f"[*] speckle share at N={n_full}: {100 * (1 - floor_frac):.1f}%")

    # Split-half correlation at full depth.
    perm = rng.permutation(n_full)
    r_half = masked_corr(stack_grid(perm[: n_full // 2]),
                         stack_grid(perm[n_full // 2: 2 * (n_full // 2)]))
    print(f"[*] split-half band-passed correlation: {r_half:+.3f} "
          "(structure and stable systematics survive; speckle does not)")

    tag = "" if args.session == "all" else f"_{args.session}"
    out = write_json(f"speckle_floor_{args.chan}{tag}.json", {
        "run_dir": args.run_dir, "chan": args.chan, "session": args.session,
        "n_full": n_full, "ladder": var_rows,
        "fit_A_over_N": float(A), "fit_floor": float(C),
        "floor_fraction_at_full": float(floor_frac),
        "speckle_fraction_at_full": float(1 - floor_frac),
        "split_half_corr": r_half,
    })
    print(f"wrote {report_path(out)}")


if __name__ == "__main__":
    main()
