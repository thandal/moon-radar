"""
Registration stability batch: corrected lunar maps across all observing
sessions (2025-06-21, 2025-09-10/11, 2025-09-16), saved as healpix arrays
for quantitative cross-look registration analysis.

Each file gets the per-file chain corrections (measured delay shift + line
centroid) before projection. Outputs land in results/REGISTRATION/ with a
combined CSV; analyze with registration_analysis.py.

Usage (from notebooks/):
    ../.conda/bin/python registration_stability.py --per-date 8
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("MPLBACKEND", "Agg")

from astropy import units as au

import doppler_equator_alignment as dea
import freq_offset_hunt as foh

DATES = ["2025-06-21", "2025-09-10", "2025-09-16"]


def process_one(task):
    path, data_root, rx_name, nside, out_dir, corr_lookup = task
    base = os.path.basename(path)
    (rx_samples, tx_samples, sample_rate, frequency,
     rx_start, _tx_file_time, _txf) = dea.load_observation(path, data_root)
    if len(rx_samples) / sample_rate < 20 * au.s:
        return None
    fs = sample_rate.to_value(au.Hz)

    tx_emit_start = rx_start + 1.0 * au.s
    tx_comp = foh.compensated_tx(rx_samples, tx_samples, sample_rate, frequency,
                                 rx_start, tx_emit_start,
                                 tx_name="DWINGELOO", rx_name=rx_name)
    res = foh.measure_offset(rx_samples, tx_comp, fs)
    del tx_comp

    # The specular tone and the horseshoe rim are co-pol features: for
    # cross-pol (chan0) files both the corrections and the rim-calibrated
    # Doppler residual come from the paired co-pol capture (same SDR/clocks,
    # same geometry, same applied compensation -> same residual).
    if corr_lookup is not None and base in corr_lookup:
        applied_shift, applied_df, pair_snr, rim_delta_hz = corr_lookup[base]
    else:
        applied_shift, applied_df, pair_snr, rim_delta_hz = (
            res["shift_refined"], res["f_centroid"], res["snr"], None)
    tx_extra_s = applied_shift / fs

    row = dea.process_file(path, data_root, out_dir, nside, rx_name=rx_name,
                           tx_extra_offset_s=tx_extra_s,
                           freq_offset_hz=applied_df, save_pngs=False,
                           rim_delta_hz=rim_delta_hz)
    row.update({
        "shift_refined": res["shift_refined"],
        "df_centroid_hz": res["f_centroid"],
        "line_width_hz": res["line_width"],
        "tone_snr": res["snr"],
        "applied_shift_samples": applied_shift,
        "applied_df_hz": applied_df,
        "pair_snr": pair_snr,
    })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__),
                                                            "data.camras.nl/lunar-radar"))
    parser.add_argument("--per-date", type=int, default=8)
    parser.add_argument("--chan", default="chan1")
    parser.add_argument("--nside", type=int, default=400)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--corrections-from", default=None,
                        help="runs CSV of the paired channel; corrections are "
                             "looked up by capture (for cross-pol files whose "
                             "own specular tone is too weak to measure)")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__),
                                                          "results/REGISTRATION"))
    args = parser.parse_args()

    corr_lookup = None
    if args.corrections_from:
        import csv as _csv
        corr_lookup = {}
        for r in _csv.DictReader(open(args.corrections_from)):
            paired = r["rx_file"].replace("chan1", args.chan).replace("chan0", args.chan)
            rim = r.get("rim_delta_hz", "")
            corr_lookup[paired] = (float(r["shift_refined"]),
                                   float(r["df_centroid_hz"]),
                                   float(r["tone_snr"]),
                                   float(rim) if rim not in ("", None) else 0.0)
        print(f"corrections from {args.corrections_from}: {len(corr_lookup)} captures")

    tasks = []
    for date in DATES:
        limit = args.per_date if args.per_date > 0 else None
        files = foh.candidate_files(args.data_root, date, "stockert",
                                    limit, chan=args.chan)
        print(f"{date}: {len(files)} files selected")
        tasks += [(p, args.data_root, "STOCKERT", args.nside, args.out_dir, corr_lookup)
                  for p in files]

    rows = []
    failed = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            base = os.path.basename(task[0])
            try:
                row = fut.result()
            except Exception as exc:
                print(f"  {base}: ERROR {exc} (queued for serial retry)")
                failed.append(task)
                continue
            if row is None:
                print(f"  {base}: skipped (short)")
                continue
            rows.append(row)
            print(f"  {base}: shift {row['shift_refined']:+.1f} samp, "
                  f"df {row['df_centroid_hz']:+.3f} Hz, snr {row['tone_snr']:.0f}")

    # Serial retry: transient GPU OOM from concurrent workers, not bad data.
    for task in failed:
        base = os.path.basename(task[0])
        try:
            row = process_one(task)
        except Exception as exc:
            print(f"  {base}: RETRY FAILED {exc}")
            continue
        if row is not None:
            rows.append(row)
            print(f"  {base}: retry OK, shift {row['shift_refined']:+.1f} samp")

    rows.sort(key=lambda r: r["rx_start_utc"])
    if rows:
        out_csv = os.path.join(args.out_dir, f"registration_runs_{args.chan}.csv")
        keys = list(rows[0].keys())
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")
        print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
