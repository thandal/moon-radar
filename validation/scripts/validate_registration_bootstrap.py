"""Bootstrap registration robustness from saved per-look map products.

Two modes:
  --mode half     (default) intra-session half-stack vs half-stack offsets --
                  the estimator noise floor under look selection/speckle.
  --mode closure  cross-session closure error bar (REPORT section 5):
                  resample each session's looks with replacement, rebuild
                  the three session stacks as stack_maps.py does, re-solve
                  the three pairwise offsets, and record the loop-closure
                  vector d(A,B)+d(B,C)-d(A,C). The distribution across
                  bootstraps puts an error bar on the closures REPORT
                  section 5 quotes.

Looks, gates and radiometry are production's: stack_maps.select_looks
(the gates, with each look's files resolved in its run directory; a missing
map, degeneracy mask or cell-area file is an error) and
stack_maps.look_brightness (brightness per unit area, noise floor removed,
degeneracy-masked). Each stack is stack_maps.Accumulator's mean of its
looks' brightness (Accumulator.mean, --min-count looks per pixel),
band-passed by stack_maps.band_of as the stacker's session solve does
(gridded; speckle averaged in linear brightness, then logged, then the
trend removed). A session with fewer than --min-count looks, or a stack
with no band-passed content (no pixel with --min-count looks and a
positive local mean), stops the run. A cross-correlation peak on the edge of
the search window is no measurement: its offset is recorded as NaN and
counted. Measurements only: no pass/fail criterion.

Needs run directories made with the 2026-10-21 pipeline
(registration_stability.py): their per-look maps and runs CSVs.
stack_maps.select_looks refuses older runs CSVs, which lack columns its
gates read (rim_status, timing_status, tone_at_band_edge, dem_file,
spice_kernels, ...), so results/STOCKERT_2025 must be regenerated before
the example below runs. The JSON records the looks left out, with the
reason, and the captures the batch did not map.

Run from the repo root:
  .conda/bin/python validation/scripts/validate_registration_bootstrap.py \
      --run-dir results/STOCKERT_2025 --stations stockert [--channel chan1] \
      [--mode half|closure]
Outputs: validation/results/registration_[closure_]bootstrap_<stations>_<channel>
.json/.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import validation_common as vc

import doppler_equator_alignment as dea
import freq_offset_hunt as foh
import registration_analysis as ra
import stack_maps as sm


class Stacker:
    """Session stacks from the looks' brightness, as stack_maps.py builds
    them, band-passed for the cross-correlation."""

    def __init__(self, bright, min_count, step_deg):
        self.bright = bright
        self.min_count = min_count
        self.lon_axis = np.arange(-55, 55 + step_deg / 2, step_deg)
        self.lat_axis = np.arange(-55, 55 + step_deg / 2, step_deg)
        self.lo_px, self.hi_px = 0.3 / step_deg, 2.5 / step_deg

    def band(self, idx):
        """Band-passed stack of the looks idx (repeats count again). Stops
        if it has no content: an empty band cross-correlates to a peak on
        the search edge, which would be counted as a failed measurement
        rather than a stack that does not exist."""
        acc = sm.Accumulator(len(self.bright[idx[0]]))
        for i in idx:
            acc.add(self.bright[i])
        band = sm.band_of(acc.mean(self.min_count), self.lon_axis,
                          self.lat_axis, self.lo_px, self.hi_px)
        if not band.any():
            raise SystemExit(f"a stack of {len(idx)} looks has no band-passed content "
                             f"(no pixel with --min-count {self.min_count} looks and a "
                             "positive local mean)")
        return band


def offset_deg(x, step):
    """(dlon, dlat) in degrees of an XcorrResult: NaN when the peak is on
    the edge of the search window (no measurement)."""
    if x.at_edge:
        return np.nan, np.nan
    return x.dx * step, x.dy * step


def closure_mode(by_session, stacker, args, search_px, rng, label, left_out):
    """Cross-session closure error bar (see module docstring)."""
    names = sorted(by_session)
    if len(names) != 3:
        raise SystemExit(f"closure mode expects 3 sessions, got {names}")
    # A session stack needs --min-count looks before any pixel has a value
    # (a resample draws as many looks as the session has, repeats included,
    # so no pixel can count more).
    few = {s: len(idx) for s, idx in by_session.items() if len(idx) < args.min_count}
    if few:
        raise SystemExit(f"too few looks for a session stack of --min-count "
                         f"{args.min_count}: {few}")
    step = args.step_deg

    def solve(idx_by_sess):
        bands = {s: stacker.band(idx) for s, idx in idx_by_sess.items()}
        a, b, c = (bands[n] for n in names)
        return {"ab": ra.xcorr_offset(a, b, search_px),
                "bc": ra.xcorr_offset(b, c, search_px),
                "ac": ra.xcorr_offset(a, c, search_px)}

    rows_out = []
    for b in range(-1, args.n_boot):
        if b < 0:  # baseline: all looks, no resampling
            sampled = by_session
        else:
            sampled = {s: [idx[i] for i in rng.integers(0, len(idx), len(idx))]
                       for s, idx in by_session.items()}
        sol = solve(sampled)
        rec = {"bootstrap": b, "baseline": b < 0}
        for pair, x in sol.items():
            dlon, dlat = offset_deg(x, step)
            rec[f"{pair}_dlon_deg"] = dlon
            rec[f"{pair}_dlat_deg"] = dlat
            rec[f"{pair}_signif"] = x.significance
            rec[f"{pair}_at_edge"] = x.at_edge
        # A pair at the search edge leaves the closure undefined (NaN).
        cx = rec["ab_dlon_deg"] + rec["bc_dlon_deg"] - rec["ac_dlon_deg"]
        cy = rec["ab_dlat_deg"] + rec["bc_dlat_deg"] - rec["ac_dlat_deg"]
        rec.update(closure_deg=float(np.hypot(cx, cy)), closure_dlon_deg=cx,
                   closure_dlat_deg=cy)
        rows_out.append(rec)
        tag = "baseline" if b < 0 else f"boot {b:2d}"
        edges = [p for p, x in sol.items() if x.at_edge]
        print(f"  {tag}: closure {rec['closure_deg']:.4f} deg  "
              f"pairs ab({rec['ab_dlon_deg']:+.3f},{rec['ab_dlat_deg']:+.3f}) "
              f"bc({rec['bc_dlon_deg']:+.3f},{rec['bc_dlat_deg']:+.3f}) "
              f"ac({rec['ac_dlon_deg']:+.3f},{rec['ac_dlat_deg']:+.3f})"
              + (f"  AT THE SEARCH EDGE: {edges}" if edges else ""))

    boot = [r for r in rows_out if not r["baseline"]]
    base = rows_out[0]
    n_undefined = sum(not np.isfinite(r["closure_deg"]) for r in boot)
    summary = {
        "pairs": names,
        "baseline_closure_deg": base["closure_deg"],
        "n_boot": len(boot),
        "n_closure_undefined_at_edge": n_undefined,
        "n_at_edge_by_pair": {p: int(sum(r[f"{p}_at_edge"] for r in boot))
                              for p in ("ab", "bc", "ac")},
        "closure_deg": vc.finite_stats([r["closure_deg"] for r in boot]),
        # Over the bootstraps where the pair was measured (not at the edge).
        "pair_offset_std_deg": {
            p: {"dlon": float(np.nanstd([r[f"{p}_dlon_deg"] for r in boot])),
                "dlat": float(np.nanstd([r[f"{p}_dlat_deg"] for r in boot]))}
            for p in ("ab", "bc", "ac")},
    }
    csv_path = vc.RESULTS_DIR / f"registration_closure_bootstrap_{label}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    payload = {
        "purpose": "bootstrap error bar on the cross-session closure "
                   "(resample looks within each session, re-run the solve)",
        "criterion": "none: measurements only",
        "run_dir": [str(d) for d in args.run_dir], "stations": args.stations,
        "channel": args.channel, "min_count": args.min_count,
        "n_looks_by_session": {s: len(idx) for s, idx in by_session.items()},
        **left_out,
        "summary": summary,
    }
    json_path = vc.write_json(f"registration_closure_bootstrap_{label}.json", payload)
    print(f"wrote {vc.report_path(csv_path)}")
    print(f"wrote {vc.report_path(json_path)}")
    s = summary["closure_deg"]
    if n_undefined:
        print(f"{n_undefined} of {len(boot)} bootstraps had a pair at the search edge: "
              "no closure; the statistics below are over the others")
    if s["n"] == 0:
        print(f"closure: baseline {base['closure_deg']:.4f} deg; no bootstrap closure")
    else:
        print(f"closure: baseline {base['closure_deg']:.4f} deg; bootstrap "
              f"median {s['median']:.4f}, p95 {s['p95_abs']:.4f} deg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", nargs="+", required=True,
                        help="run directories with the per-look maps and runs CSVs")
    parser.add_argument("--stations", nargs="+", required=True,
                        choices=sorted(dea.STATION_RX),
                        help="receivers whose looks are stacked (as stack_maps.py)")
    parser.add_argument("--channel", default="chan1")
    parser.add_argument("--n-boot", type=int, default=20)
    parser.add_argument("--min-snr", type=float, default=foh.MIN_TONE_SNR)
    parser.add_argument("--min-tx-s", type=float, default=None)
    parser.add_argument("--min-count", type=int, default=3,
                        help="looks per pixel for a stack value (stack_maps.py's default)")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--step-deg", type=float, default=0.075)
    parser.add_argument("--mode", default="half", choices=("half", "closure"),
                        help="half: intra-session half-stack offsets; "
                             "closure: cross-session closure error bar")
    args = parser.parse_args()

    vc.ensure_dirs()
    rng = np.random.default_rng(args.seed)
    args.run_dir = [d if Path(d).is_absolute() else vc.REPO_ROOT / d for d in args.run_dir]
    rows, bright, left_out = vc.production_looks(
        [str(d) for d in args.run_dir], args.stations, [args.channel],
        args.min_snr, args.min_tx_s)
    if not rows:
        raise SystemExit("no looks to stack")
    by_session = {}
    for i, row in enumerate(rows):
        by_session.setdefault(sm.session_of(row["rx_file"]), []).append(i)
    stacker = Stacker(bright, args.min_count, args.step_deg)
    search_px = int(1.5 / args.step_deg)
    label = f"{'_'.join(args.stations)}_{args.channel}"

    if args.mode == "closure":
        closure_mode(by_session, stacker, args, search_px, rng, label, left_out)
        return

    # A half-stack needs --min-count looks before any pixel has a value.
    need = 2 * args.min_count
    rows_out, skipped = [], {}
    for sess, idx in sorted(by_session.items()):
        if len(idx) < need:
            skipped[sess] = (f"{len(idx)} looks: two half-stacks of at least "
                             f"--min-count {args.min_count} looks need {need}")
            print(f"[*] {sess} skipped: {skipped[sess]}")
            continue
        for b in range(args.n_boot):
            perm = rng.permutation(len(idx))
            half = len(perm) // 2
            a_idx = [idx[i] for i in perm[:half]]
            b_idx = [idx[i] for i in perm[half:]]
            x = ra.xcorr_offset(stacker.band(a_idx), stacker.band(b_idx), search_px)
            dlon, dlat = offset_deg(x, args.step_deg)
            rows_out.append({
                "session": sess,
                "bootstrap": b,
                "n_a": len(a_idx),
                "n_b": len(b_idx),
                "dlon_deg": dlon,
                "dlat_deg": dlat,
                "offset_deg": float(np.hypot(dlon, dlat)),
                "peak_corr": x.peak,
                "significance": x.significance,
                "at_edge": x.at_edge,
            })
    if not rows_out:
        raise SystemExit(f"no session has {need} looks: {skipped}")

    csv_path = vc.RESULTS_DIR / f"registration_bootstrap_{label}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {}
    for sess in sorted({r["session"] for r in rows_out}):
        subset = [r for r in rows_out if r["session"] == sess]
        summary[sess] = {
            "n_looks": len(by_session[sess]),
            "n_trials": len(subset),
            "n_at_edge": sum(r["at_edge"] for r in subset),
            "offset_deg": vc.finite_stats([r["offset_deg"] for r in subset]),
            "peak_corr": vc.finite_stats([r["peak_corr"] for r in subset]),
            "significance": vc.finite_stats([r["significance"] for r in subset]),
        }
    payload = {
        "purpose": "bootstrap half-stack registration stability under look selection/speckle",
        "criterion": "none: measurements only",
        "run_dir": [str(d) for d in args.run_dir],
        "stations": args.stations,
        "channel": args.channel,
        "min_count": args.min_count,
        "n_looks_used": len(rows),
        **left_out,
        "sessions_skipped": skipped,
        "summary_by_session": summary,
    }
    json_path = vc.write_json(f"registration_bootstrap_{label}.json", payload)
    print(f"wrote {vc.report_path(csv_path)}")
    print(f"wrote {vc.report_path(json_path)}")
    for sess, stats in summary.items():
        o = stats["offset_deg"]
        edge = (f"; {stats['n_at_edge']} of {stats['n_trials']} at the search edge "
                "(no measurement)" if stats["n_at_edge"] else "")
        if o["n"] == 0:
            print(f"{sess}: no measured offset{edge}")
            continue
        print(f"{sess}: median offset={o['median']:.4f} deg, "
              f"p95={o['p95_abs']:.4f} deg, "
              f"median corr={stats['peak_corr']['median']:.3f}{edge}")


if __name__ == "__main__":
    main()
