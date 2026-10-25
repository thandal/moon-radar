"""
Per-channel batch: calibrate, image and map every selected capture of one
station and channel, and write the runs CSV that stack_maps.py reads.

Each capture gets its measured chain corrections (timing shift and tone
centroid, freq_offset_hunt.measure_offset) before imaging; the opposite-sense
channel then calibrates Doppler from its own rims and refines delay and
Doppler from its echo edges (process_file), and the same-sense channel takes
all of these from its opposite-sense twin (--corrections-from).

Outputs, in --out-dir:
  registration_runs_<station>_<chan>.csv   one row per mapped capture
  registration_skipped_<station>_<chan>.csv  every candidate not mapped, and why
plus each capture's _map.npy, _mapcount.npy, _cellarea.npy (and with
--save-pngs its DD image, _dd.npz, for reproject_maps.py). The batch exits
non-zero if any capture failed with an error.

Usage (from the repo root):
    # 2025 Stockert sessions, opposite-sense channel then same-sense
    .conda/bin/python registration_stability.py --station stockert --chan chan1 \\
        --dates 2025-06-21 2025-09-10 2025-09-16 --per-date 0 --out-dir results/STOCKERT_2025
    .conda/bin/python registration_stability.py --station stockert --chan chan0 \\
        --dates 2025-06-21 2025-09-10 2025-09-16 --per-date 0 --out-dir results/STOCKERT_2025 \\
        --corrections-from results/STOCKERT_2025/registration_runs_stockert_chan1.csv

    # 2026-10-11 session: Stockert and WSRT recorded two linear feeds, so
    # first synthesize opposite-/same-sense captures with make_circular.py (into
    # data/eve2026_circ, chan1 = opposite-sense, chan0 = same-sense), then run as
    # usual. WSRT's timing offset is ~+45 samples, past the default +/-40.
    .conda/bin/python registration_stability.py --data-root data/eve2026_circ \\
        --dates 2026-10-11 --per-date 0 --station wsrt --chan chan1 \\
        --max-shift 80 --save-pngs --out-dir results/EVE2026_CIRC
    .conda/bin/python registration_stability.py --data-root data/eve2026_circ \\
        --dates 2026-10-11 --per-date 0 --station wsrt --chan chan0 \\
        --max-shift 80 --save-pngs --out-dir results/EVE2026_CIRC \\
        --corrections-from results/EVE2026_CIRC/registration_runs_wsrt_chan1.csv
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("MPLBACKEND", "Agg")

import cupy
from astropy import units as au

import doppler_equator_alignment as dea
import freq_offset_hunt as foh


# Timing search half-width (samples), the batch's default: the 2025-09-11
# session has captures with chain offsets up to ~+31 samples. WSRT
# (2026-10-11) sits at ~+45 and needs --max-shift 80.
DEFAULT_MAX_SHIFT = 40


class SkipCapture(Exception):
    """A capture the batch does not map, for a recorded reason (not an error)."""


def look_summary(row):
    """The batch's one line per mapped capture: timing, frequency, rim
    correction and edge-fit result, so a run can be checked as it goes
    (the runs CSV is only written at the end). Blank or non-numeric values
    print as '-'."""
    def num(key, scale, spec):
        try:
            x = float(row.get(key, "")) * scale
        except (TypeError, ValueError):
            return "-"
        return format(x, spec) if math.isfinite(x) else "-"
    return (f"shift {num('applied_shift_samples', 1, '+.1f')} samp, "
            f"df {num('applied_df_hz', 1, '+.3f')} Hz, "
            f"rim {row.get('rim_status', '-')} {num('rim_delta_hz', 1e3, '+.1f')} mHz, "
            f"edge {row.get('edge_status', '-')}: delay {num('edge_delay_us', 1, '+.1f')} "
            f"+- {num('edge_delay_err_us', 1, '.1f')} us, "
            f"Doppler {num('edge_doppler_hz', 1e3, '+.1f')} "
            f"+- {num('edge_doppler_err_hz', 1e3, '.1f')} mHz")


def capture_stem(rx_file):
    """A capture's name without its channel (rx_file is <stem>.chanN.sigmf-meta)."""
    return rx_file.split(".chan")[0]


def read_corrections(path):
    """Opposite-sense corrections keyed by capture stem, from an opposite-sense runs CSV."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[capture_stem(r["rx_file"])] = {
                "shift": float(r["applied_shift_samples"]),
                "tx_extra_offset_s": float(r["tx_extra_offset_s"]),
                "df_hz": float(r["applied_df_hz"]),
                "snr": float(r["tone_snr"]),
                "rim_status": r["rim_status"],
                "rim_delta_hz": r["rim_delta_hz"],
                "edge_status": r["edge_status"],
                "edge_delay_us": r["edge_delay_us"],
                "edge_doppler_hz": r["edge_doppler_hz"],
                "max_shift": int(r["max_shift"]),
            }
    return out


def process_one(task):
    (path, data_root, rx_name, nside, out_dir, corrections,
     max_shift, save_pngs, supersample) = task
    base = os.path.basename(path)
    if corrections is not None:
        # Same-sense: the specular tone and the horseshoe rim are opposite-sense
        # features, so timing, frequency, the rim-calibrated Doppler
        # residual and the edge fit's offsets all come from the opposite-sense twin (same SDR/clocks, same
        # geometry, same compensation -> same residual). Same-sense looks are
        # also gated on the twin's tone (pair_snr) and rim; the same-sense
        # capture itself is not tested (REPORT 3.2).
        twin = corrections.get(capture_stem(base))
        if twin is None:
            raise SkipCapture("no opposite-sense twin in the --corrections-from CSV")
        if twin["rim_status"] == "failed":
            raise SkipCapture("opposite-sense twin's rim calibration failed: no Doppler correction")
        if twin["edge_status"] != "fitted":
            raise SkipCapture(f"opposite-sense twin's edge fit {twin['edge_status']}: "
                              "no delay/Doppler correction")
        row = dea.process_file(path, data_root, out_dir, nside,
                               tx_name=dea.TX_STATION, rx_name=rx_name,
                               tx_extra_offset_s=twin["tx_extra_offset_s"],
                               freq_offset_hz=twin["df_hz"],
                               rim_delta_hz=float(twin["rim_delta_hz"]),
                               edge_offsets=(float(twin["edge_delay_us"]) * 1e-6,
                                             float(twin["edge_doppler_hz"])),
                               save_pngs=save_pngs, supersample=supersample)
        row.update({
            "applied_shift_samples": twin["shift"],
            "applied_df_hz": twin["df_hz"],
            "pair_snr": twin["snr"],
            "pair_rim_status": twin["rim_status"],
            # The search that measured the applied shift (the opposite-sense run's).
            "max_shift": twin["max_shift"],
        })
        return row

    (rx_samples, tx_samples, sample_rate, frequency,
     rx_start, _tx_file_time, _txf) = dea.load_observation(path, data_root)
    fs = sample_rate.to_value(au.Hz)
    capture_s = len(rx_samples) / fs
    if capture_s < dea.MIN_CAPTURE_S:
        raise SkipCapture(f"capture {capture_s:.1f} s < {dea.MIN_CAPTURE_S:g} s")
    tx_comp = foh.compensated_tx(rx_samples, tx_samples, sample_rate, frequency,
                                 rx_start, dea.nominal_tx_emit_start(rx_start),
                                 tx_name=dea.TX_STATION, rx_name=rx_name)
    res = foh.measure_offset(rx_samples, tx_comp, fs, dea.chip_samples_of(_txf, fs),
                             max_shift=max_shift)
    # process_file reloads the samples; drop these copies first (~0.5 GB on
    # the 125 s captures).
    del tx_comp, rx_samples, tx_samples

    row = dea.process_file(path, data_root, out_dir, nside,
                           tx_name=dea.TX_STATION, rx_name=rx_name,
                           tx_extra_offset_s=res["shift_refined"] / fs,
                           freq_offset_hz=res["f_centroid"],
                           save_pngs=save_pngs, supersample=supersample)
    row.update({
        "shift_refined": res["shift_refined"],
        "timing_status": res["timing_status"],
        "shift_strongest": res["shift_strongest"],
        "snr_strongest": res["snr_strongest"],
        "df_centroid_hz": res["f_centroid"],
        "line_width_hz": res["line_width"],
        "tone_snr": res["snr"],
        "tone_at_band_edge": res["tone_at_band_edge"],
        "f_wide_hz": res["f_wide"],
        "snr_wide": res["snr_wide"],
        "applied_shift_samples": res["shift_refined"],
        "applied_df_hz": res["f_centroid"],
        "pair_snr": "",
        "pair_rim_status": "",
        # Search limit, so stack_maps.py can reject shifts at the rail.
        "max_shift": max_shift,
    })
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__),
                                                            "data.camras.nl/lunar-radar"))
    parser.add_argument("--station", required=True, choices=sorted(dea.STATION_RX),
                        help="RX file prefix; also names the runs CSV")
    parser.add_argument("--chan", required=True, help="chan1 (opposite sense) or chan0 (same sense)")
    parser.add_argument("--dates", nargs="+", required=True,
                        help="observing-day folders under --data-root")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-date", type=int, default=0,
                        help="keep only this many captures per date folder, "
                             "evenly spaced (0 = all)")
    parser.add_argument("--min-tx-s", type=float, default=30.0,
                        help="shortest TX waveform to map, seconds (default 30)")
    parser.add_argument("--nside", type=int, default=400)
    parser.add_argument("--supersample", type=int, default=dea.SUPERSAMPLE,
                        help="points averaged per map pixel along each axis "
                             f"(default {dea.SUPERSAMPLE}; 1 = the image cell "
                             "under the pixel centre only)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-shift", type=int, default=DEFAULT_MAX_SHIFT,
                        help=f"timing search half-width in samples (default {DEFAULT_MAX_SHIFT})")
    parser.add_argument("--save-pngs", action="store_true",
                        help="also save each look's DD image (_dd.npz, for "
                             "reproject_maps.py) and PNGs of it and its map")
    parser.add_argument("--corrections-from", default=None,
                        help="opposite-sense runs CSV: same-sense captures take its "
                             "corrections (matched by capture name)")
    args = parser.parse_args()

    rx_name = dea.STATION_RX[args.station]
    tag = f"{args.station}_{args.chan}"
    corrections = read_corrections(args.corrections_from) if args.corrections_from else None

    tasks, skipped = [], []
    for date in args.dates:
        files, skips = foh.candidate_files(args.data_root, date, args.station, args.chan,
                                           args.min_tx_s, args.per_date or None)
        skipped += skips
        print(f"{date}: {len(files)} captures selected, {len(skips)} skipped "
              f"({args.station} -> {rx_name})", flush=True)
        tasks += [(p, args.data_root, rx_name, args.nside, args.out_dir, corrections,
                   args.max_shift, args.save_pngs, args.supersample) for p in files]
    if corrections is not None:
        matched = sum(capture_stem(os.path.basename(t[0])) in corrections for t in tasks)
        print(f"corrections from {args.corrections_from}: {len(corrections)} opposite-sense "
              f"captures, {matched} of {len(tasks)} selected captures matched")

    rows, errors, retry = [], [], []

    def record(task, outcome):
        base = os.path.basename(task[0])
        if isinstance(outcome, SkipCapture):
            skipped.append((task[0], str(outcome)))
            print(f"  {base}: skipped ({outcome})", flush=True)
        elif isinstance(outcome, Exception):
            errors.append((task[0], f"error: {type(outcome).__name__}: {outcome}"))
            print(f"  {base}: ERROR {type(outcome).__name__}: {outcome}", flush=True)
        else:
            rows.append(outcome)
            print(f"  {base}: {look_summary(outcome)}", flush=True)

    ctx = mp.get_context("spawn")
    # A fresh worker per capture: a worker reused across captures keeps
    # memory between them (measured 2026-10-22 on 60 s captures: 0.5 GB
    # held between looks and growing, 1.5 GB at a look's peak), and the
    # 2025 batch crossed its 3 GB cap after 23 looks. Starting a worker costs
    # a few seconds against ~70 s per capture.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                             max_tasks_per_child=1) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                outcome = fut.result()
            except cupy.cuda.memory.OutOfMemoryError as exc:
                # Concurrent workers can exhaust the GPU; retry these alone.
                print(f"  {os.path.basename(task[0])}: GPU out of memory, retrying "
                      f"after the pool ({exc})", flush=True)
                retry.append(task)
                continue
            except Exception as exc:
                outcome = exc
            record(task, outcome)
    for task in retry:
        try:
            outcome = process_one(task)
        except Exception as exc:
            outcome = exc
        record(task, outcome)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"registration_runs_{tag}.csv")
    skip_csv = os.path.join(args.out_dir, f"registration_skipped_{tag}.csv")
    not_mapped = sorted(skipped + errors)
    if not_mapped:
        foh.write_rows(skip_csv, [{"rx_file": os.path.basename(p), "reason": r}
                                  for p, r in not_mapped])
        print(f"{len(not_mapped)} captures not mapped, listed in {skip_csv}")
    elif os.path.exists(skip_csv):
        os.remove(skip_csv)
    if rows:
        rows.sort(key=lambda r: r["rx_start_utc"])
        foh.write_rows(out_csv, rows)
        print(f"Wrote {len(rows)} rows to {out_csv}")
    elif os.path.exists(out_csv):
        # A rerun replaces the station/channel CSV; one that mapped nothing
        # must not leave the previous run's CSV looking current.
        os.remove(out_csv)
        print(f"No capture mapped; removed the previous {out_csv}")
    if errors or not rows:
        sys.exit(f"{len(errors)} captures failed with errors; {len(rows)} mapped")


if __name__ == "__main__":
    main()
