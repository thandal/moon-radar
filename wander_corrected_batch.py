"""
A/B batch: per-file chain-offset estimation + corrected DD images.

For each observation:
  1. Measure the residual specular tone (freq_offset_hunt.measure_offset):
     delay shift (samples), line peak, noise-floor-subtracted centroid, and
     RMS line width. Captures whose measurement cannot be used -- a tone
     weaker than --min-snr (default freq_offset_hunt.MIN_TONE_SNR), a
     timing shift at the +-max_shift search limit, a timing peak that is
     not resolved (timing_status other than "resolved"), or a tone at the
     edge of the frequency search band: the timing gates stack_maps.
     select_looks applies to opposite-sense looks -- are not processed, and are
     listed with the reason.
  2. Process the file twice through the standard pipeline, differing only
     in the chain corrections:
       - no chain correction (timing offset 0, tone offset 0)
                                             -> <out-root>/BATCH_BASELINE/
       - chain-corrected (measured timing offset + tone centroid)
                                             -> <out-root>/BATCH_CORRECTED/
     Both passes still run the rim self-calibration and the edge fit
     (process_file with rim_delta_hz=None), which measure and remove a
     Doppler offset and (within one chip) a delay offset from the image
     itself; their results for each pass are recorded, so the A/B shows how
     much of the chain offset the image's own calibration absorbs anyway.
  3. Save a per-file line-shape panel so the "wander" being corrected can be
     inspected (a clean displaced tone = real chain offset; a broad lumpy
     line = fading structure, and the correction is centroid-centering).

Outputs, in <out-root>/FREQ_OFFSET/:
  wander_batch_<date>_<station>_<chan>.csv          one row per processed capture
  wander_batch_<date>_<station>_<chan>_skipped.csv  every candidate not processed, and why
  line_shapes_<date>_<station>_<chan>.png
A rerun that processes nothing removes the previous CSV and figure rather
than leave them looking current. The batch exits non-zero if any capture
failed with an error.

Files are processed in parallel worker processes (spawn context: required
for CUDA; each worker furnishes its own SPICE kernels on import). Three
workers overlap the CPU-bound SPICE/resample work with the GPU correlation
bursts; GPU memory is the limit (~3 GB peak per worker on a 10 GB card).

Usage (from the repo root):
    .conda/bin/python wander_corrected_batch.py --date 2025-09-16 --station stockert --chan chan1 --limit 8
"""

import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import cupy
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl
from astropy import units as au

import doppler_equator_alignment as dea
import freq_offset_hunt as foh
from registration_stability import SkipCapture


def process_one(task):
    """Worker: measure offsets and run the uncorrected and corrected
    pipeline passes. Raises SkipCapture (with the reason) for a capture
    whose measurement cannot be applied."""
    (path, data_root, rx_name, max_shift, min_snr, nside,
     baseline_dir, corrected_dir) = task
    base = os.path.basename(path)
    (rx_samples, tx_samples, sample_rate, frequency,
     rx_start, _tx_file_time, tx_file) = dea.load_observation(path, data_root)
    fs = sample_rate.to_value(au.Hz)
    capture_s = len(rx_samples) / fs
    if capture_s < dea.MIN_CAPTURE_S:
        raise SkipCapture(f"capture {capture_s:.1f} s < {dea.MIN_CAPTURE_S:g} s")

    # --- 1. Measure chain offsets ---
    tx_comp = foh.compensated_tx(rx_samples, tx_samples, sample_rate, frequency,
                                 rx_start, dea.nominal_tx_emit_start(rx_start),
                                 tx_name=dea.TX_STATION, rx_name=rx_name)
    res = foh.measure_offset(rx_samples, tx_comp, fs, dea.chip_samples_of(tx_file, fs),
                             max_shift=max_shift)
    # process_file reloads the samples; drop these copies first.
    del tx_comp, rx_samples, tx_samples
    problems = foh.measurement_problems(res, max_shift, min_snr)
    if problems:
        raise SkipCapture("; ".join(problems) + ": not a usable timing/frequency measurement")
    tx_extra_s = res["shift_refined"] / fs
    df = res["f_centroid"]
    seg_spread = max(res["segments"]) - min(res["segments"])

    # --- 2. Process without and with the chain corrections ---
    row_b = dea.process_file(path, data_root, baseline_dir, nside,
                             tx_name=dea.TX_STATION, rx_name=rx_name,
                             tx_extra_offset_s=0.0, freq_offset_hz=0.0)
    row_c = dea.process_file(path, data_root, corrected_dir, nside,
                             tx_name=dea.TX_STATION, rx_name=rx_name,
                             tx_extra_offset_s=tx_extra_s, freq_offset_hz=df)
    tx_extra_us = tx_extra_s * 1e6
    row = {
        "rx_file": base,
        "rx_start_utc": rx_start.utc.value,
        "capture_s": capture_s,
        "tx_overlap_s": row_c["tx_overlap_s"],
        "max_shift": max_shift,
        "shift_samples": res["shift"],
        "shift_refined": res["shift_refined"],
        "shift_strongest": res["shift_strongest"],
        "timing_status": res["timing_status"],
        "tx_extra_offset_us": tx_extra_us,
        "df_peak_hz": res["f_peak"],
        "df_centroid_hz": df,
        "line_width_hz": res["line_width"],
        "segment_spread_hz": seg_spread,
        "tone_snr": res["snr"],
        "score_baseline": row_b["alignment_score"],
        "score_corrected": row_c["alignment_score"],
        # Rim self-calibration ran in both passes; what it found in each.
        "rim_status_baseline": row_b["rim_status"],
        "rim_delta_hz_baseline": row_b["rim_delta_hz"],
        "rim_status_corrected": row_c["rim_status"],
        "rim_delta_hz_corrected": row_c["rim_delta_hz"],
        "edge_status_baseline": row_b["edge_status"],
        "edge_delay_us_baseline": row_b["edge_delay_us"],
        "edge_doppler_hz_baseline": row_b["edge_doppler_hz"],
        "edge_status_corrected": row_c["edge_status"],
        "edge_delay_us_corrected": row_c["edge_delay_us"],
        "edge_doppler_hz_corrected": row_c["edge_doppler_hz"],
        # Terrain under the SRP (extract_srp_elevation): the timing offset
        # (the tone's, refined by the edge fit) is measured relative to that
        # terrain, so it is (hardware delay) - 2h/c. srp_topo_delay_us is
        # +2h/c; adding it isolates the SDR/hardware part of the per-look
        # timing offset. Blank when the corrected pass's edge fit did not fit.
        "srp_elevation_km": row_c["srp_elevation_km"],
        "srp_topo_delay_us": row_c["srp_topo_delay_us"],
        "hardware_delay_us": (tx_extra_us + float(row_c["edge_delay_us"])
                              + row_c["srp_topo_delay_us"]
                              if row_c["edge_status"] == "fitted" else ""),
    }
    spectrum = {k: res[k] for k in ("spectrum_f", "spectrum_mag",
                                    "f_peak", "f_centroid")}
    return row, (base, spectrum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__),
                                                            "data.camras.nl/lunar-radar"))
    parser.add_argument("--date", required=True, help="observing-day folder under --data-root")
    parser.add_argument("--station", required=True, choices=sorted(dea.STATION_RX),
                        help="RX file prefix")
    parser.add_argument("--chan", required=True,
                        help="chan1 only (opposite sense: the timing and frequency measurement "
                             "needs the specular tone, an opposite-sense feature)")
    parser.add_argument("--limit", type=int, default=8,
                        help="keep this many captures, evenly spaced (0 = all)")
    parser.add_argument("--min-tx-s", type=float, default=30.0,
                        help="shortest TX waveform to process, seconds (default 30)")
    # +-40 matches the batch standard (REPORT 8.3).
    parser.add_argument("--max-shift", type=int, default=40)
    parser.add_argument("--min-snr", type=float, default=foh.MIN_TONE_SNR,
                        help="weakest tone whose corrections are applied "
                             f"(default {foh.MIN_TONE_SNR:g}, the stack's gate)")
    parser.add_argument("--nside", type=int, default=400)
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel worker processes (GPU memory bound)")
    parser.add_argument("--out-root", default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()
    if args.chan != "chan1":
        parser.error(f"--chan {args.chan}: only chan1 (opposite sense) can be measured here. "
                     "The timing and frequency measurement needs the specular tone, "
                     "which is an opposite-sense feature; a same-sense capture takes its "
                     "opposite-sense twin's corrections (registration_stability.py "
                     "--corrections-from)")

    rx_name = dea.STATION_RX[args.station]
    tag = f"{args.date}_{args.station}_{args.chan}"
    baseline_dir = os.path.join(args.out_root, "BATCH_BASELINE")
    corrected_dir = os.path.join(args.out_root, "BATCH_CORRECTED")
    diag_dir = os.path.join(args.out_root, "FREQ_OFFSET")
    os.makedirs(diag_dir, exist_ok=True)

    files, skipped = foh.candidate_files(args.data_root, args.date, args.station,
                                         args.chan, args.min_tx_s, args.limit or None)
    print(f"{len(files)} candidate files ({args.station} -> {rx_name}), "
          f"{len(skipped)} not selected, {args.workers} workers")
    tasks = [(path, args.data_root, rx_name, args.max_shift, args.min_snr, args.nside,
              baseline_dir, corrected_dir) for path in files]

    rows, spectra, errors, retry = [], [], [], []

    def record(path, outcome):
        base = os.path.basename(path)
        if isinstance(outcome, SkipCapture):
            skipped.append((path, str(outcome)))
            print(f"  {base}: skipped ({outcome})")
        elif isinstance(outcome, Exception):
            errors.append((path, f"error: {type(outcome).__name__}: {outcome}"))
            print(f"  {base}: ERROR {type(outcome).__name__}: {outcome}")
        else:
            row, spectrum = outcome
            rows.append(row)
            spectra.append(spectrum)
            print(f"  {base}: shift {row['shift_refined']:+.1f} samp "
                  f"({row['tx_extra_offset_us']:.0f} us), "
                  f"peak {row['df_peak_hz']:+.3f} Hz, "
                  f"centroid {row['df_centroid_hz']:+.3f} Hz, "
                  f"width {row['line_width_hz']:.3f} Hz, "
                  f"seg spread {row['segment_spread_hz']:.3f} Hz, "
                  f"rim {row['rim_status_baseline']} / {row['rim_status_corrected']}")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                outcome = fut.result()
            except cupy.cuda.memory.OutOfMemoryError as exc:
                # Concurrent workers can exhaust the GPU; retry these alone.
                print(f"  {os.path.basename(task[0])}: GPU out of memory, retrying "
                      f"after the pool ({exc})")
                retry.append(task)
                continue
            except Exception as exc:
                outcome = exc
            record(task[0], outcome)
    for task in retry:
        try:
            outcome = process_one(task)
        except Exception as exc:
            outcome = exc
        record(task[0], outcome)

    rows.sort(key=lambda r: r["rx_start_utc"])
    out_csv = os.path.join(diag_dir, f"wander_batch_{tag}.csv")
    skip_csv = os.path.join(diag_dir, f"wander_batch_{tag}_skipped.csv")
    not_processed = sorted(skipped + errors)
    if not_processed:
        foh.write_rows(skip_csv, [{"rx_file": os.path.basename(p), "reason": r}
                                  for p, r in not_processed])
        print(f"{len(not_processed)} captures not processed, listed in {skip_csv}")
    elif os.path.exists(skip_csv):
        os.remove(skip_csv)

    # --- 3. Diagnostics ---
    shapes_png = os.path.join(diag_dir, f"line_shapes_{tag}.png")
    if spectra:
        spectra.sort(key=lambda s: s[0])
        ncols = 3
        nrows_fig = int(np.ceil(len(spectra) / ncols))
        fig, axes = pl.subplots(nrows_fig, ncols, figsize=(5 * ncols, 3 * nrows_fig),
                                squeeze=False)
        for ax, (base, res) in zip(axes.flat, spectra):
            ax.plot(res["spectrum_f"], res["spectrum_mag"], lw=0.6)
            ax.axvline(res["f_peak"], color="tab:red", lw=0.8, label="peak")
            ax.axvline(res["f_centroid"], color="tab:green", lw=0.8, label="centroid")
            ax.axvline(0, color="gray", lw=0.5)
            # The spectrum spans exactly measure_offset's tone search band.
            ax.set_xlim(res["spectrum_f"][0], res["spectrum_f"][-1])
            ax.set_title(base.split(".chan")[0], fontsize=8)
            ax.legend(fontsize=6)
        for ax in axes.flat[len(spectra):]:
            ax.set_visible(False)
        fig.suptitle("Specular line shapes (rx * conj(tx_comp) spectrum, best delay shift)")
        pl.tight_layout()
        pl.savefig(shapes_png, dpi=130)
        print(f"Saved {shapes_png}")
    elif os.path.exists(shapes_png):
        # A rerun must not leave the previous run's figure looking current.
        os.remove(shapes_png)
        print(f"No capture processed; removed the previous {shapes_png}")

    if rows:
        foh.write_rows(out_csv, rows)
        print(f"Wrote {len(rows)} rows to {out_csv}")
        widths = np.array([r["line_width_hz"] for r in rows])
        cents = np.array([r["df_centroid_hz"] for r in rows])
        sb = np.array([r["score_baseline"] for r in rows])
        sc = np.array([r["score_corrected"] for r in rows])
        print(f"Statistics over the {len(rows)} processed captures "
              f"({len(skipped)} skipped, {len(errors)} failed are not in them):")
        print(f"  centroids: mean {cents.mean():+.3f} Hz, std {cents.std():.3f} Hz")
        print(f"  line widths: mean {widths.mean():.3f} Hz (vs centroid scatter "
              f"{cents.std():.3f})")
        print(f"  alignment score, no chain correction -> corrected: "
              f"{sb.mean():.3f} -> {sc.mean():.3f} "
              f"(higher after correction on {int((sc > sb).sum())} of {len(rows)})")
    elif os.path.exists(out_csv):
        # A rerun must not leave the previous run's CSV looking current.
        os.remove(out_csv)
        print(f"No capture processed; removed the previous {out_csv}")
    if errors or not rows:
        sys.exit(f"{len(errors)} captures failed with errors; {len(rows)} processed")


if __name__ == "__main__":
    main()
