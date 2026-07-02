"""
A/B batch: per-file chain-offset estimation + corrected DD images.

For each observation:
  1. Measure the residual specular tone (freq_offset_hunt product method):
     delay shift (samples), line peak, noise-floor-subtracted centroid, and
     RMS line width.
  2. Process the file twice through the standard pipeline:
       - baseline  (no corrections)        -> <out-root>/BATCH_BASELINE/
       - corrected (tx delay + centroid)   -> <out-root>/BATCH_CORRECTED/
  3. Save a per-file line-shape panel so the "wander" being corrected can be
     inspected (a clean displaced tone = real chain offset; a broad lumpy
     line = fading structure, and the correction is centroid-centering).

Files are processed in parallel worker processes (spawn context: required
for CUDA; each worker furnishes its own SPICE kernels on import). Three
workers overlap the CPU-bound SPICE/resample work with the GPU correlation
bursts; GPU memory is the limit (~3 GB peak per worker on a 10 GB card).

Usage (from the repo root):
    .conda/bin/python wander_corrected_batch.py --date 2025-09-16 --chan chan1 --limit 8
"""

import argparse
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl
from astropy import units as au

import doppler_equator_alignment as dea
import freq_offset_hunt as foh


def process_one(task):
    """Worker: measure offsets and run baseline + corrected pipeline passes."""
    (path, data_root, rx_name, max_shift, nside,
     baseline_dir, corrected_dir) = task
    base = os.path.basename(path)
    (rx_samples, tx_samples, sample_rate, frequency,
     rx_start, _tx_file_time, _txf) = dea.load_observation(path, data_root)
    if len(rx_samples) / sample_rate < 20 * au.s:
        return None
    fs = sample_rate.to_value(au.Hz)

    # --- 1. Measure chain offsets ---
    tx_emit_start = rx_start + 1.0 * au.s
    tx_comp = foh.compensated_tx(rx_samples, tx_samples, sample_rate, frequency,
                                 rx_start, tx_emit_start,
                                 tx_name="DWINGELOO", rx_name=rx_name)
    res = foh.measure_offset(rx_samples, tx_comp, fs, max_shift=max_shift)
    del tx_comp
    tx_extra_s = res["shift_refined"] / fs
    df = res["f_centroid"]
    seg_spread = max(res["segments"]) - min(res["segments"])

    # --- 2. Process baseline and corrected ---
    row_b = dea.process_file(path, data_root, baseline_dir, nside, rx_name=rx_name)
    row_c = dea.process_file(path, data_root, corrected_dir, nside, rx_name=rx_name,
                             tx_extra_offset_s=tx_extra_s, freq_offset_hz=df)
    row = {
        "rx_file": base,
        "rx_start_utc": rx_start.utc.value,
        "shift_samples": res["shift"],
        "shift_refined": res["shift_refined"],
        "tx_extra_offset_us": tx_extra_s * 1e6,
        "df_peak_hz": res["f_peak"],
        "df_centroid_hz": df,
        "line_width_hz": res["line_width"],
        "segment_spread_hz": seg_spread,
        "tone_snr": res["snr"],
        "score_baseline": row_b["alignment_score"],
        "score_corrected": row_c["alignment_score"],
        # Terrain under the SRP (REPORT 8.4): subtracting srp_topo_delay_us
        # from tx_extra_offset_us isolates the SDR/hardware part of the
        # per-look timing offset.
        "srp_elevation_km": row_c["srp_elevation_km"],
        "srp_topo_delay_us": row_c["srp_topo_delay_us"],
    }
    spectrum = {k: res[k] for k in ("spectrum_f", "spectrum_mag",
                                    "f_peak", "f_centroid")}
    return row, (base, spectrum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__),
                                                            "data.camras.nl/lunar-radar"))
    parser.add_argument("--date", default="2025-09-16")
    parser.add_argument("--station", default="stockert")
    parser.add_argument("--chan", default="chan1")
    parser.add_argument("--limit", type=int, default=8)
    # ±40 matches the batch standard (REPORT §8.3); ±20 re-rails the three
    # recovered 2025-09-11 captures.
    parser.add_argument("--max-shift", type=int, default=40)
    parser.add_argument("--nside", type=int, default=400)
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel worker processes (GPU memory bound)")
    parser.add_argument("--out-root", default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()

    rx_name = "STOCKERT" if args.station == "stockert" else "DWINGELOO"
    baseline_dir = os.path.join(args.out_root, "BATCH_BASELINE")
    corrected_dir = os.path.join(args.out_root, "BATCH_CORRECTED")
    diag_dir = os.path.join(args.out_root, "FREQ_OFFSET")
    os.makedirs(diag_dir, exist_ok=True)

    files = foh.candidate_files(args.data_root, args.date, args.station,
                                args.limit, chan=args.chan)
    print(f"{len(files)} candidate files, {args.workers} workers")
    tasks = [(path, args.data_root, rx_name, args.max_shift, args.nside,
              baseline_dir, corrected_dir) for path in files]

    rows = []
    spectra = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(process_one, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            base = os.path.basename(futures[fut])
            try:
                result = fut.result()
            except Exception as exc:
                print(f"  {base}: ERROR {exc}")
                continue
            if result is None:
                print(f"  {base}: skipped (short)")
                continue
            row, spectrum = result
            rows.append(row)
            spectra.append(spectrum)
            print(f"  {base}: shift {row['shift_samples']} samp "
                  f"({row['tx_extra_offset_us']:.0f} us), "
                  f"peak {row['df_peak_hz']:+.3f} Hz, "
                  f"centroid {row['df_centroid_hz']:+.3f} Hz, "
                  f"width {row['line_width_hz']:.3f} Hz, "
                  f"seg spread {row['segment_spread_hz']:.3f} Hz")

    rows.sort(key=lambda r: r["rx_start_utc"])

    # --- 3. Diagnostics ---
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
            ax.set_xlim(-1.5, 1.5)
            ax.set_title(base.split("_1299")[0], fontsize=8)
            ax.legend(fontsize=6)
        for ax in axes.flat[len(spectra):]:
            ax.set_visible(False)
        fig.suptitle("Specular line shapes (rx * conj(tx_comp) spectrum, best delay shift)")
        pl.tight_layout()
        shapes_png = os.path.join(diag_dir, f"line_shapes_{args.date}_{args.station}.png")
        pl.savefig(shapes_png, dpi=130)
        print(f"Saved {shapes_png}")

    if rows:
        out_csv = os.path.join(diag_dir, f"wander_batch_{args.date}_{args.station}.csv")
        keys = list(rows[0].keys())
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row[k]) for k in keys) + "\n")
        print(f"Wrote {len(rows)} rows to {out_csv}")
        widths = np.array([r["line_width_hz"] for r in rows])
        cents = np.array([r["df_centroid_hz"] for r in rows])
        sb = np.array([r["score_baseline"] for r in rows])
        sc = np.array([r["score_corrected"] for r in rows])
        print(f"centroids: mean {cents.mean():+.3f} Hz, std {cents.std():.3f} Hz")
        print(f"line widths: mean {widths.mean():.3f} Hz (vs centroid scatter {cents.std():.3f})")
        print(f"alignment score baseline -> corrected: {sb.mean():.3f} -> {sc.mean():.3f}")


if __name__ == "__main__":
    main()
