"""Re-project saved delay-Doppler images (<rx_file>_dd.npz) onto a HEALPix
grid of another nside, without redoing the correlation. Uses the pipeline's
own lunar_projection, so the only change from registration_stability.py's
maps is the grid.

    reproject.py OUT_DIR NSIDE RUNS_CSV [RUNS_CSV ...] [--check N]

Writes <rx_file>_map.npy / _mapcount.npy into OUT_DIR and a copy of each runs
CSV with map_npy / mult_npy pointing at them (so stack_maps.py --run-dir
OUT_DIR works). --check N: only re-project the first N looks and compare
with the saved maps (run it at the CSV's own nside: it should reproduce them).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import healpy as hp
from astropy import time as at

sys.path.insert(0, "/home/than/code/moon-radar")
import doppler_equator as de
import doppler_equator_alignment as dea

DATA = "/home/than/code/moon-radar/data/eve2026_circ/2026-10-11"
RX_NAME = {"stockert": "STOCKERT", "wsrt": "WSRT"}

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("nside", type=int)
ap.add_argument("csvs", nargs="+")
ap.add_argument("--check", type=int, default=0)
ap.add_argument("--supersample", type=int, default=1)
ap.add_argument("--long-only", action="store_true")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
de.load_lola_dem()

for path in args.csvs:
    rows = list(csv.DictReader(open(path)))
    if args.check:
        rows = rows[:args.check]
    if args.long_only:
        rows = [r for r in rows if float(r["tx_s"]) > 100]
    for r in rows:
        t0 = time.time()
        base = r["rx_file"]
        rx_name = RX_NAME[base.split("_")[0]]
        meta = json.load(open(os.path.join(DATA, base)))
        rx_time_s = de.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
        fs = float(r["sample_rate_hz"])
        f_hz = float(r["frequency_hz"])
        rx_duration_s = float(r["capture_s"])
        tx_extra = float(r["tx_extra_offset_s"])

        dd = np.load(r["dd_npz"])
        log_A = dd["log_A"].astype(np.float32)
        # Undo plot axes (process_file): doppler_hz = -(dlt_cal - dlt_srp) f,
        # delay_s = lt_min_image + tx_extra + delay_values_s. The projection
        # only uses lt_min_image + delay_values_s, so carry it all in the
        # delay axis (integer lags / fs, as compute_dd_image builds it).
        _, dlt_srp = de.moonSRP_DLT_BCK(rx_time_s, "DWINGELOO", rx_name,
                                        dt=rx_duration_s / 2)
        dlt_shifts_cal = dlt_srp - dd["doppler_hz"] / f_hz
        delay_values_s = (dd["delay_s"][0] - tx_extra) + np.arange(log_A.shape[1]) / fs
        dlt_rate_srp = de.srp_dlt_rate_bck(rx_time_s, rx_duration_s, "DWINGELOO", rx_name)

        val, valid_frac, mult = dea.lunar_projection(
            log_A, dlt_shifts_cal, delay_values_s, 0.0, rx_time_s, args.nside,
            rx_duration_s=rx_duration_s, dlt_rate_srp=dlt_rate_srp,
            tx_name="DWINGELOO", rx_name=rx_name, use_dem=True,
            supersample=args.supersample)
        del log_A, dd

        if args.check:
            old = np.load(r["map_npy"])
            ok_old, ok_new = old != hp.UNSEEN, val != hp.UNSEEN
            both = ok_old & ok_new
            d = np.abs(old[both] - val[both])
            mold = np.load(r["mult_npy"])
            print(f"   values bit-identical: {np.array_equal(old, val)}; multiplicity identical: {np.array_equal(mold, mult)}")
            print(f"{base[:40]}: valid {valid_frac:.4f} vs CSV {float(r['valid_lunar_fraction']):.4f}; "
                  f"valid-mask mismatches {(ok_old != ok_new).sum()}; "
                  f"|diff| median {np.median(d):.4f}, 99.9% {np.quantile(d, 0.999):.4f}, "
                  f"max {d.max():.3f}; pixels off by > 0.05: {(d > 0.05).mean():.2e}")
            continue

        r["map_npy"] = os.path.join(args.out_dir, f"{base}_map.npy")
        r["mult_npy"] = os.path.join(args.out_dir, f"{base}_mapcount.npy")
        np.save(r["map_npy"], val)
        np.save(r["mult_npy"], mult)
        del val, mult
        print(f"{base[:52]}: valid {valid_frac:.4f}, {time.time() - t0:.0f} s", flush=True)

    if not args.check:
        out_csv = os.path.join(args.out_dir, os.path.basename(path))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv}")
