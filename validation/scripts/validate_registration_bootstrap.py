"""Bootstrap registration robustness from saved per-look map products."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(vc.REPO_ROOT / "results/LOLA_DEM_REGISTRATION"))
    parser.add_argument("--channel", default="chan1")
    parser.add_argument("--n-boot", type=int, default=20)
    parser.add_argument("--min-snr", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--step-deg", type=float, default=0.075)
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
