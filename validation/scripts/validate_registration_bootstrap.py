"""Bootstrap registration robustness from saved per-look map products.

Two modes:
  --mode half     (default) intra-session half-stack vs half-stack offsets --
                  the estimator noise floor under look selection/speckle.
  --mode closure  cross-session closure error bar (REPORT section 5):
                  resample each session's looks with replacement, rebuild
                  the three session stacks exactly as registration_analysis
                  does (grid-space, degeneracy-masked, band-passed), re-solve
                  the three pairwise offsets, and record the loop-closure
                  vector d(A,B)+d(B,C)-d(A,C). The distribution across
                  bootstraps puts an error bar on the 0.009-0.025 deg
                  closures REPORT section 5 quotes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import healpy as hp
import numpy as np

import validation_common as vc

import registration_analysis as ra
import stack_maps as sm


SESSION_OF = {
    "2025_06_21": "2025-06-21",
    "2025_09_10": "2025-09-10_11",
    "2025_09_11": "2025-09-10_11",
    "2025_09_16": "2025-09-16",
}


def session_of(rx_file: str) -> str:
    return SESSION_OF[re.search(r"(\d{4}_\d{2}_\d{2})", rx_file).group(1)]


def load_rows(run_dir, channel: str, min_snr: float) -> list[dict]:
    path = run_dir / f"registration_runs_{channel}.csv"
    rows = []
    for row in csv.DictReader(path.open()):
        snr = float(row.get("pair_snr") or row["tone_snr"])
        shift = float(row.get("applied_shift_samples") or row["shift_refined"])
        if snr >= min_snr and abs(shift) < 39.5:
            rows.append(row)
    rows.sort(key=lambda r: r["rx_start_utc"])
    return rows


def stack_rows(rows: list[dict]) -> np.ndarray:
    npix = len(np.load(rows[0]["map_npy"]))
    acc = sm.Accumulator(npix)
    for row in rows:
        acc.add(sm.look_linear_intensity(row["map_npy"], row["mult_npy"]))
    return acc.stacked_log(min_count=2)


def band_for_map(m, lon_axis, lat_axis, lo_px, hi_px):
    return ra.bandpass(ra.grid_map(m, lon_axis, lat_axis), lo_px, hi_px)


def closure_mode(rows, by_session, args, lon_axis, lat_axis, lo_px, hi_px,
                 search_px, exclude_px, rng):
    """Cross-session closure error bar (see module docstring)."""
    import os

    # Grid every look once (as registration_analysis.main does), float32.
    grids = {}
    for sess, sess_rows in sorted(by_session.items()):
        gs = []
        for row in sess_rows:
            mult = (np.load(row["mult_npy"])
                    if row.get("mult_npy") and os.path.exists(row["mult_npy"])
                    else None)
            gs.append(ra.grid_map(np.load(row["map_npy"]), lon_axis, lat_axis,
                                  multiplicity=mult).astype(np.float32))
        grids[sess] = gs
        print(f"[*] {sess}: {len(gs)} looks gridded")
    names = sorted(grids)
    if len(names) != 3:
        raise SystemExit(f"closure mode expects 3 sessions, got {names}")

    def solve(bands):
        a, b, c = (bands[n] for n in names)
        ab = ra.xcorr_offset(a, b, search_px, exclude_px)
        bc = ra.xcorr_offset(b, c, search_px, exclude_px)
        ac = ra.xcorr_offset(a, c, search_px, exclude_px)
        cy = ab[0] + bc[0] - ac[0]
        cx = ab[1] + bc[1] - ac[1]
        return ab, bc, ac, cy, cx

    step = args.step_deg
    rows_out = []
    for b in range(-1, args.n_boot):
        if b < 0:  # baseline: all looks, no resampling
            sampled = grids
        else:
            sampled = {s: [gs[i] for i in rng.integers(0, len(gs), len(gs))]
                       for s, gs in grids.items()}
        bands = {s: ra.bandpass(ra.nanstack(g), lo_px, hi_px)
                 for s, g in sampled.items()}
        ab, bc, ac, cy, cx = solve(bands)
        rec = {"bootstrap": b, "baseline": b < 0,
               "closure_deg": float(np.hypot(cy, cx) * step),
               "closure_dlon_deg": cx * step, "closure_dlat_deg": cy * step}
        for pair, sol in (("ab", ab), ("bc", bc), ("ac", ac)):
            rec[f"{pair}_dlon_deg"] = sol[1] * step
            rec[f"{pair}_dlat_deg"] = sol[0] * step
            rec[f"{pair}_signif"] = sol[3]
        rows_out.append(rec)
        tag = "baseline" if b < 0 else f"boot {b:2d}"
        print(f"  {tag}: closure {rec['closure_deg']:.4f} deg  "
              f"pairs ab({rec['ab_dlon_deg']:+.3f},{rec['ab_dlat_deg']:+.3f}) "
              f"bc({rec['bc_dlon_deg']:+.3f},{rec['bc_dlat_deg']:+.3f}) "
              f"ac({rec['ac_dlon_deg']:+.3f},{rec['ac_dlat_deg']:+.3f})")

    boot = [r for r in rows_out if not r["baseline"]]
    base = rows_out[0]
    summary = {
        "pairs": names,
        "baseline_closure_deg": base["closure_deg"],
        "closure_deg": vc.finite_stats([r["closure_deg"] for r in boot]),
        "pair_offset_std_deg": {
            p: {"dlon": float(np.std([r[f"{p}_dlon_deg"] for r in boot])),
                "dlat": float(np.std([r[f"{p}_dlat_deg"] for r in boot]))}
            for p in ("ab", "bc", "ac")},
    }
    csv_path = vc.RESULTS_DIR / f"registration_closure_bootstrap_{args.channel}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    payload = {
        "purpose": "bootstrap error bar on the cross-session closure "
                   "(resample looks within each session, re-run the solve)",
        "run_dir": str(args.run_dir), "channel": args.channel,
        "n_boot": args.n_boot, "summary": summary,
    }
    json_path = vc.write_json(f"registration_closure_bootstrap_{args.channel}.json",
                              payload)
    print(f"wrote {vc.report_path(csv_path)}")
    print(f"wrote {vc.report_path(json_path)}")
    s = summary["closure_deg"]
    print(f"closure: baseline {base['closure_deg']:.4f} deg; bootstrap "
          f"median {s['median']:.4f}, p95 {s['p95_abs']:.4f} deg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(vc.REPO_ROOT / "results/LOLA_DEM_REGISTRATION"))
    parser.add_argument("--channel", default="chan1")
    parser.add_argument("--n-boot", type=int, default=20)
    parser.add_argument("--min-snr", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--step-deg", type=float, default=0.075)
    parser.add_argument("--mode", default="half", choices=("half", "closure"),
                        help="half: intra-session half-stack offsets; "
                             "closure: cross-session closure error bar")
    args = parser.parse_args()

    vc.ensure_dirs()
    rng = np.random.default_rng(args.seed)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = vc.REPO_ROOT / run_dir
    rows = load_rows(run_dir, args.channel, args.min_snr)
    by_session = {}
    for row in rows:
        by_session.setdefault(session_of(row["rx_file"]), []).append(row)

    lon_axis = np.arange(-55, 55 + args.step_deg / 2, args.step_deg)
    lat_axis = np.arange(-55, 55 + args.step_deg / 2, args.step_deg)
    lo_px, hi_px = 0.3 / args.step_deg, 2.5 / args.step_deg
    search_px, exclude_px = int(1.5 / args.step_deg), int(0.5 / args.step_deg)

    if args.mode == "closure":
        args.run_dir = str(run_dir)
        closure_mode(rows, by_session, args, lon_axis, lat_axis, lo_px, hi_px,
                     search_px, exclude_px, rng)
        return

    rows_out = []
    for sess, sess_rows in sorted(by_session.items()):
        if len(sess_rows) < 6:
            continue
        for b in range(args.n_boot):
            idx = rng.permutation(len(sess_rows))
            half = len(idx) // 2
            a_rows = [sess_rows[i] for i in idx[:half]]
            b_rows = [sess_rows[i] for i in idx[half:]]
            a = band_for_map(stack_rows(a_rows), lon_axis, lat_axis, lo_px, hi_px)
            c = band_for_map(stack_rows(b_rows), lon_axis, lat_axis, lo_px, hi_px)
            dy, dx, peak, sig = ra.xcorr_offset(a, c, search_px, exclude_px)
            rows_out.append({
                "session": sess,
                "bootstrap": b,
                "n_a": len(a_rows),
                "n_b": len(b_rows),
                "dlon_deg": dx * args.step_deg,
                "dlat_deg": dy * args.step_deg,
                "offset_deg": float(np.hypot(dx, dy) * args.step_deg),
                "peak_corr": peak,
                "significance": sig,
            })

    csv_path = vc.RESULTS_DIR / f"registration_bootstrap_{args.channel}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {}
    for sess in sorted({r["session"] for r in rows_out}):
        subset = [r for r in rows_out if r["session"] == sess]
        summary[sess] = {
            "n_trials": len(subset),
            "offset_deg": vc.finite_stats([r["offset_deg"] for r in subset]),
            "peak_corr": vc.finite_stats([r["peak_corr"] for r in subset]),
            "significance": vc.finite_stats([r["significance"] for r in subset]),
        }
    payload = {
        "purpose": "bootstrap half-stack registration stability under look selection/speckle",
        "run_dir": str(run_dir),
        "channel": args.channel,
        "n_input_rows_gated": len(rows),
        "summary_by_session": summary,
    }
    json_path = vc.write_json(f"registration_bootstrap_{args.channel}.json", payload)
    print(f"wrote {vc.report_path(csv_path)}")
    print(f"wrote {vc.report_path(json_path)}")
    for sess, stats in summary.items():
        print(f"{sess}: median offset={stats['offset_deg']['median']:.4f} deg, "
              f"p95={stats['offset_deg']['p95_abs']:.4f} deg, "
              f"median corr={stats['peak_corr']['median']:.3f}")


if __name__ == "__main__":
    main()
