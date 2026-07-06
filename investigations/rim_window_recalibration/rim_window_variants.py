"""Rim-window variant study on cached DD images (both A/B blocks).

Variants (per-look row offsets for measure_rim_offset):
  legacy   : (10,50)/(30,90) rows                      [wins on 5.5 mHz pitch]
  adaptive : production scaling to 80 mHz capture      [wins on 1.8 mHz pitch]
  physfreq : strips fixed in PHYSICAL frequency units, floored at the legacy
             row geometry: inner (55..275 mHz), outer (165..495 mHz) --
             equals legacy at 5.5 mHz pitch, scales up at fine pitch.
  physfreq6: physfreq with up to 6 convergence passes (capture via crawl).

Judged by smoothness of total chain offset (applied_df + delta) per block.
"""
import csv
import os
import sys

import numpy as np
from astropy import units as au

REPO = "/home/than/code/moon-radar"
sys.path.insert(0, REPO)
os.chdir(REPO)

import doppler_equator_alignment as dea

SCRATCH = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRATCH, "dd_cache")
os.makedirs(CACHE, exist_ok=True)
FROZEN_CSV = os.path.join(REPO, "results/LOLA_DEM_REGISTRATION/registration_runs_chan1.csv")
DATA_ROOT = os.path.join(REPO, "data.camras.nl/lunar-radar")

BLOCKS = [
    ("2025-09-16", "2025-09-16", ["13_55_52", "13_57_38", "13_58_31",
                                  "13_59_25", "14_00_20", "14_06_11"]),
    ("2025-09-10", "2025-09-11", ["08_05_44", "08_06_26", "08_07_08",
                                  "08_07_50", "08_08_32", "08_09_14",
                                  "08_09_56", "08_10_38", "08_11_20",
                                  "08_12_02"]),
]

frozen = {r["rx_file"]: r for r in csv.DictReader(open(FROZEN_CSV))}


def get_look(sess_dir, date, stamp):
    base = (f"stockert_radar_{date.replace('-', '_')}_{stamp}"
            "_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta")
    fr = frozen[base]
    npz = os.path.join(CACHE, base + ".npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return fr, {k: d[k] for k in d.files}
    path = os.path.join(DATA_ROOT, sess_dir, base)
    applied_shift = float(fr["applied_shift_samples"])
    applied_df = float(fr["applied_df_hz"])
    (rx, tx, sample_rate, frequency, rx_start, _t, _f) = \
        dea.load_observation(path, DATA_ROOT)
    fs = sample_rate.to_value(au.Hz)
    f_hz = frequency.to_value(au.Hz)
    tx_emit_start = rx_start + 1.0 * au.s
    rx_duration_s = len(rx) / fs
    log_A, dlt_shifts, delay_values_s, lt_min_image, dlt_rate_srp = \
        dea.compute_dd_image(rx, tx, sample_rate, frequency, rx_start,
                             tx_emit_start, applied_shift / fs, 0.0,
                             "DWINGELOO", "STOCKERT", freq_offset_hz=applied_df)
    del rx, tx
    rx_time_s = dea.et_from_astropy(rx_start)
    lt_min_eq, delay_up, dlt_up, delay_down, dlt_down = \
        dea.compute_doppler_equator_velocity(
            rx_time_s, n_points=500, rx_duration_s=rx_duration_s,
            dlt_rate_srp=dlt_rate_srp, tx_name="DWINGELOO", rx_name="STOCKERT")
    data = dict(log_A=log_A.astype(np.float32), dlt_shifts=dlt_shifts,
                delay_values_s=delay_values_s,
                lt_min_image=np.float64(lt_min_image),
                lt_min_eq=np.float64(lt_min_eq),
                delay_up=delay_up, dlt_up=dlt_up,
                delay_down=delay_down, dlt_down=dlt_down,
                f_hz=np.float64(f_hz))
    np.savez(npz, **data)
    return fr, data


def measure(d, inner, outer, cap, n_pass):
    log_A, dlt_shifts = d["log_A"], d["dlt_shifts"]
    ddlt_bin = dlt_shifts[1] - dlt_shifts[0]
    args = (d["delay_values_s"], float(d["lt_min_image"]), float(d["lt_min_eq"]),
            d["delay_up"], d["dlt_up"], d["delay_down"], d["dlt_down"])
    delta_dlt, rim, resid = 0.0, None, None
    for _ in range(n_pass):
        r = dea.measure_rim_offset(log_A, dlt_shifts - delta_dlt, *args,
                                   inner_off=inner, outer_off=outer,
                                   delta_capture_dlt=cap)
        if r is None:
            break
        rim = rim or r
        delta_dlt += r["delta_dlt"]
        resid = r["delta_dlt"]
        if abs(r["delta_dlt"]) < 0.5 * ddlt_bin:
            break
    if rim is None:
        return None
    rim_f = dea.measure_rim_offset(log_A, dlt_shifts - delta_dlt, *args,
                                   inner_off=inner, outer_off=outer,
                                   delta_capture_dlt=cap)
    conv = rim_f or rim
    return dict(delta_dlt=delta_dlt, spread_conv=conv["spread_dlt"],
                n=conv["n_up"] + conv["n_down"],
                resid=(rim_f or rim)["delta_dlt"] if rim_f else resid)


def physfreq_offsets(pitch_hz):
    """Strips in physical frequency units, floored at legacy rows."""
    def rows(mhz, legacy_rows):
        return max(legacy_rows, int(np.ceil(mhz * 1e-3 / pitch_hz)))
    inner = (rows(55, 10), rows(275, 50))
    outer = (rows(165, 30), rows(495, 90))
    return inner, outer


VARIANTS = ["legacy", "adaptive", "physfreq", "physfreq6"]
results = []
for sess_dir, date, stamps in BLOCKS:
    for stamp in stamps:
        fr, d = get_look(sess_dir, date, stamp)
        f_hz = float(d["f_hz"])
        pitch = (d["dlt_shifts"][1] - d["dlt_shifts"][0]) * f_hz
        applied_df = float(fr["applied_df_hz"])
        row = {"date": date, "stamp": stamp, "pitch_mhz": pitch * 1e3,
               "applied_df_hz": applied_df}
        for v in VARIANTS:
            if v == "legacy":
                inner, outer, cap, n_pass = (10, 50), (30, 90), None, 3
            elif v == "adaptive":
                inner, outer, cap, n_pass = (10, 50), (30, 90), 0.080 / f_hz, 3
            else:
                inner, outer = physfreq_offsets(pitch)
                cap = None
                n_pass = 6 if v == "physfreq6" else 3
            m = measure(d, inner, outer, cap, n_pass)
            if m is None:
                row[f"{v}_delta"] = row[f"{v}_total"] = row[f"{v}_spread"] = None
                row[f"{v}_n"] = 0
            else:
                delta_hz = -m["delta_dlt"] * f_hz
                row[f"{v}_delta"] = delta_hz * 1e3
                row[f"{v}_total"] = (applied_df + delta_hz) * 1e3
                row[f"{v}_spread"] = -m["spread_conv"] * f_hz * 1e3
                row[f"{v}_n"] = m["n"]
        results.append(row)
        print(f"{date} {stamp} pitch {pitch*1e3:5.2f} mHz  " +
              "  ".join(f"{v}: d={row[f'{v}_delta']:+7.1f} t={row[f'{v}_total']:+7.1f}"
                        if row[f"{v}_delta"] is not None else f"{v}: ---"
                        for v in VARIANTS), flush=True)

out = os.path.join(SCRATCH, "rim_window_variants.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print("\nwrote", out)

print(f"\n{'variant':10s} {'block':10s} {'n':>2s} {'fails':>5s} "
      f"{'tot range':>9s} {'l2l RMS':>8s} {'med |spread|':>12s}")
for v in VARIANTS:
    for date in ("2025-09-16", "2025-09-11"):
        rs = [r for r in results if r["date"] == date]
        # Exclude the known tone-centroid outlier from smoothness stats.
        tots = [r[f"{v}_total"] for r in rs
                if r[f"{v}_total"] is not None and r["stamp"] != "13_57_38"]
        fails = sum(1 for r in rs if r[f"{v}_total"] is None)
        spr = [abs(r[f"{v}_spread"]) for r in rs if r[f"{v}_spread"] is not None]
        if len(tots) > 2:
            dd = np.diff(tots)
            print(f"{v:10s} {date:10s} {len(tots):2d} {fails:5d} "
                  f"{max(tots)-min(tots):9.1f} {np.sqrt(np.mean(dd**2)):8.1f} "
                  f"{np.median(spr):12.1f}")

print("\n13_57_38 (tone outlier, true total ~ -20 mHz):")
r = next(r for r in results if r["stamp"] == "13_57_38")
for v in VARIANTS:
    print(f"  {v:10s} delta {r[f'{v}_delta'] if r[f'{v}_delta'] is not None else '---'}"
          f"  total {r[f'{v}_total'] if r[f'{v}_total'] is not None else '---'}")
