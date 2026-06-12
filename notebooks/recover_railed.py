"""
Recover the 2025-09-11 captures whose chain delay railed the original
+/-20-sample shift search.

Re-measured with a +/-40-sample window, the three captures resolve cleanly
at +91 / +80 / +125 us (tone SNR 68-94) -- the largest chain offsets in the
dataset (typical is +35-45 us). This script re-processes them (chan1 with
rim self-calibration, chan0 twins with corrections inherited from chan1,
same conventions as registration_stability.py) and patches their rows in
results/REGISTRATION/registration_runs_{chan1,chan0}.csv in place.

Afterwards restack with stack_maps.py (its rail gate now matches +/-40).

Usage (from notebooks/):
    ../.conda/bin/python recover_railed.py
"""

import csv
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from astropy import units as au

import doppler_equator_alignment as dea
import freq_offset_hunt as foh

STAMPS = ["06_39_21", "07_52_07", "08_05_44"]
DATA_ROOT = os.path.join(os.path.dirname(__file__), "data.camras.nl/lunar-radar")
RUN_DIR = os.path.join(os.path.dirname(__file__), "results/REGISTRATION")
NSIDE = 400
MAX_SHIFT = 40


def reprocess(path, corr=None):
    """Mirror registration_stability.process_one for one file (serial GPU)."""
    (rx, tx, fs_q, freq, rx_start, _t, _f) = dea.load_observation(path, DATA_ROOT)
    fs = fs_q.to_value(au.Hz)
    tx_comp = foh.compensated_tx(rx, tx, fs_q, freq, rx_start,
                                 rx_start + 1.0 * au.s,
                                 tx_name="DWINGELOO", rx_name="STOCKERT")
    res = foh.measure_offset(rx, tx_comp, fs, max_shift=MAX_SHIFT)
    del tx_comp

    if corr is not None:  # cross-pol: corrections from the co-pol twin
        applied_shift, applied_df, pair_snr, rim_delta_hz = corr
    else:
        applied_shift, applied_df, pair_snr, rim_delta_hz = (
            res["shift_refined"], res["f_centroid"], res["snr"], None)

    row = dea.process_file(path, DATA_ROOT, RUN_DIR, NSIDE,
                           tx_extra_offset_s=applied_shift / fs,
                           freq_offset_hz=applied_df, save_pngs=True,
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


def patch_csv(chan, new_rows):
    path = os.path.join(RUN_DIR, f"registration_runs_{chan}.csv")
    rows = list(csv.DictReader(open(path)))
    keys = list(rows[0].keys())
    by_file = {r["rx_file"]: r for r in new_rows}
    n = 0
    for i, r in enumerate(rows):
        if r["rx_file"] in by_file:
            rows[i] = {k: str(by_file[r["rx_file"]][k]) for k in keys}
            n += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(r[k] for k in keys) + "\n")
    print(f"patched {n} rows in {path}")


def main():
    chan1_rows, chan0_rows = [], []
    for st in STAMPS:
        base = f"stockert_radar_2025_09_11_{st}_1299.500MHz_0.25Msps_ci16_le"
        p1 = os.path.join(DATA_ROOT, "2025-09-10", f"{base}.chan1.sigmf-meta")
        r1 = reprocess(p1)
        print(f"{st} chan1: shift {r1['shift_refined']:+.2f} samp, "
              f"df {r1['df_centroid_hz']:+.3f} Hz, snr {r1['tone_snr']:.0f}, "
              f"rim delta {r1['rim_delta_hz']:+.4f} Hz (n={r1['rim_n']})")
        chan1_rows.append(r1)

        p0 = os.path.join(DATA_ROOT, "2025-09-10", f"{base}.chan0.sigmf-meta")
        corr = (r1["shift_refined"], r1["df_centroid_hz"], r1["tone_snr"],
                r1["rim_delta_hz"])
        r0 = reprocess(p0, corr=corr)
        print(f"{st} chan0: own shift {r0['shift_refined']:+.2f} samp "
              f"(applied {r0['applied_shift_samples']:+.2f} from chan1 twin)")
        chan0_rows.append(r0)

    patch_csv("chan1", chan1_rows)
    patch_csv("chan0", chan0_rows)


if __name__ == "__main__":
    main()
