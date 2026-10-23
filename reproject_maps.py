"""
Rebuild a run's per-look lunar maps from the saved delay-Doppler images,
without redoing the correlation.

registration_stability.py --save-pngs saves each look's DD image as
<rx_file>_dd.npz. When only the projection or the edge fit changes
(lunar_projection, fit_edge_offsets in doppler_equator_alignment.py), this
fits each co-pol look's edges again (its cross-pol twin takes the result,
as in process_file), re-projects the images and replaces each look's
_map.npy, _mapcount.npy and _cellarea.npy and its runs-CSV row
(valid_lunar_fraction, supersample, edge_*). The saved images are float16
(log amplitude to ~0.004); that is the only difference from a full rerun.
Co-pol channels (chan1) are done first, so --chans must include chan1
whenever it includes chan0.

Everything is read from and written to --run-dir (recorded paths are used
only for their file names), each capture's metadata is found through its
recorded rx_meta_path under --data-root, and nothing is replaced until every
look has been rebuilt: the run is checked first, the new maps are written to
a temporary folder, and only then moved into place with the CSVs.

Usage (from the repo root):
    .conda/bin/python reproject_maps.py --run-dir results/NEW_REGISTRATION \\
        --data-root data.camras.nl/lunar-radar --stations stockert --chans chan1 chan0
"""

import argparse
import csv
import json
import os
import shutil

import numpy as np
import healpy as hp
from astropy import time as at

import doppler_equator_alignment as dea

NEEDED = ("dd_npz", "rx_meta_path", "capture_s", "tx_overlap_s", "srp_topo_delay_us",
          "use_dem", "tx_extra_offset_s", "tx_file", "spice_kernels", "dem_file")


def reproject_row(r, run_dir, data_root, rx_name, supersample, twin_edge=None):
    """New (map, multiplicity, cell area, valid fraction, edge fit) for one
    runs-CSV row. twin_edge: None for a co-pol row (its edges are fitted
    here, as process_file does); the co-pol twin's edge fit for a cross-pol
    row."""
    with open(os.path.join(data_root, r["rx_meta_path"]), encoding="utf-8") as f:
        meta = json.load(f)
    # As load_observation: the capture's own start time.
    rx_time_s = dea.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
    fs = float(r["sample_rate_hz"])
    f_hz = float(r["frequency_hz"])
    rx_duration_s = float(r["capture_s"])
    dd = np.load(os.path.join(run_dir, os.path.basename(r["dd_npz"])))
    log_A = dd["log_A"].astype(np.float32)
    # Undo the saved axes (process_file): doppler_hz = -(dlt_cal - dlt_srp) f
    # and delay_s = lt_min_image + tx_extra_offset_s + delay_values_s. Carry
    # lt_min_image in the delay axis (integer lags / fs, as compute_dd_image
    # builds it), so the reference is just the terrain's offset from it.
    _, dlt_srp = dea.moonSRP_DLT_BCK(rx_time_s, dea.TX_STATION, rx_name, dt=rx_duration_s / 2)
    dlt_shifts_cal = dlt_srp - dd["doppler_hz"] / f_hz
    delay_values_s = ((dd["delay_s"][0] - float(r["tx_extra_offset_s"]))
                      + np.arange(log_A.shape[1]) / fs)
    use_dem = r["use_dem"] == "True"
    lt_ref = -float(r["srp_topo_delay_us"]) * 1e-6 if use_dem else 0.0
    # Same rate compute_dd_image compensated (same calls, same stencils).
    dlt_rate_srp = dea.srp_dlt_rate_bck(rx_time_s, rx_duration_s, dea.TX_STATION, rx_name)
    nside = hp.npix2nside(len(np.load(os.path.join(run_dir, os.path.basename(r["map_npy"])),
                                      mmap_mode="r")))
    chip_samples = dea.chip_samples_of(r["tx_file"], fs)
    # The saved axes are before the edge fit (process_file); as there: fit
    # (co-pol, when the rims were measured), or take the twin's.
    if twin_edge is None:
        edge = {"status": "not_run", "delay_s": np.nan, "doppler_hz": np.nan,
                "delay_err_s": np.nan, "doppler_err_hz": np.nan, "n_columns": 0}
        if r["rim_status"] in ("converged", "not_converged"):
            edge = dea.fit_edge_offsets(
                log_A, dlt_shifts_cal, delay_values_s, lt_ref, rx_time_s,
                rx_duration_s=rx_duration_s, dlt_rate_srp=dlt_rate_srp,
                chip_samples=chip_samples, tx_overlap_s=float(r["tx_overlap_s"]),
                f_hz=f_hz, noise_log_amp=dea.noise_floor_log_amp(log_A),
                tx_name=dea.TX_STATION, rx_name=rx_name, use_dem=use_dem)
        applied = ((edge["delay_s"], edge["doppler_hz"]) if edge["status"] == "fitted"
                   else (0.0, 0.0))
    else:
        # A twin whose fit failed gives nothing to apply (registration_stability
        # would skip the capture; stack_maps.py rejects it on the twin's status).
        fitted = twin_edge["status"] == "fitted"
        edge = {"status": "inherited",
                "delay_s": twin_edge["delay_s"] if fitted else np.nan,
                "doppler_hz": twin_edge["doppler_hz"] if fitted else np.nan,
                "delay_err_s": np.nan, "doppler_err_hz": np.nan, "n_columns": 0}
        applied = (edge["delay_s"], edge["doppler_hz"]) if fitted else (0.0, 0.0)
    val, valid_fraction, mult, area = dea.lunar_projection(
        log_A, dlt_shifts_cal + applied[1] / f_hz, delay_values_s, lt_ref - applied[0],
        rx_time_s, nside, tx_name=dea.TX_STATION, rx_name=rx_name,
        chip_samples=chip_samples,
        doppler_res_dlt=1.0 / (float(r["tx_overlap_s"]) * f_hz),
        rx_duration_s=rx_duration_s, dlt_rate_srp=dlt_rate_srp,
        use_dem=use_dem, supersample=supersample)
    return val, mult, area, valid_fraction, edge


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True,
                        help="folder with registration_runs_<station>_<chan>.csv")
    parser.add_argument("--data-root", required=True,
                        help="data root the run was made from (for the capture "
                             "start times in the .sigmf-meta files)")
    parser.add_argument("--stations", nargs="+", required=True,
                        choices=sorted(dea.STATION_RX))
    parser.add_argument("--chans", nargs="+", required=True)
    parser.add_argument("--supersample", type=int, default=dea.SUPERSAMPLE,
                        help="points averaged per pixel along each axis "
                             f"(default {dea.SUPERSAMPLE})")
    args = parser.parse_args()
    dem_file = os.path.basename(dea.load_lola_dem())
    kernels = dea.kernel_fingerprint()

    if "chan0" in args.chans and "chan1" not in args.chans:
        parser.error("cross-pol (chan0) looks take their co-pol twin's edge fit: "
                     "include chan1")
    runs = []
    for station in args.stations:
        # co-pol first: its edge fits are the cross-pol looks'
        for chan in sorted(args.chans, key=lambda c: c != "chan1"):
            path = os.path.join(args.run_dir, f"registration_runs_{station}_{chan}.csv")
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                missing = [k for k in NEEDED if not r.get(k)
                           and not (k == "dem_file" and r.get("use_dem") == "False")]
                if missing:
                    parser.error(f"{path}: {r['rx_file']} has no {missing}; rerun "
                                 "registration_stability.py --save-pngs instead")
                for k in ("dd_npz", "map_npy"):
                    if not os.path.exists(os.path.join(args.run_dir, os.path.basename(r[k]))):
                        parser.error(f"{path}: {os.path.basename(r[k])} is not in {args.run_dir}")
                if not os.path.exists(os.path.join(args.data_root, r["rx_meta_path"])):
                    parser.error(f"{path}: {r['rx_meta_path']} is not under {args.data_root}")
                # The row's terrain delay and calibration belong to the terrain
                # model and kernels it was made with; re-projecting with others
                # would mix them under the row's old labels.
                if r["spice_kernels"] != kernels:
                    parser.error(f"{path}: {r['rx_file']} was made with SPICE kernels "
                                 f"{r['spice_kernels']}, now {kernels}; reprocess instead")
                if r["use_dem"] == "True" and r["dem_file"] != dem_file:
                    parser.error(f"{path}: {r['rx_file']} was made with {r['dem_file']}, "
                                 f"now {dem_file}; reprocess instead")
            runs.append((path, dea.STATION_RX[station], chan, rows))

    tmp = os.path.join(args.run_dir, ".reproject_tmp")
    os.makedirs(tmp, exist_ok=True)
    moves = []
    copol_edges = {}                    # (rx_name, capture stem) -> edge fit
    for path, rx_name, chan, rows in runs:
        for r in rows:
            stem = r["rx_file"].split(".chan")[0]
            twin_edge = None
            if chan != "chan1":
                twin_edge = copol_edges.get((rx_name, stem))
                if twin_edge is None:
                    parser.error(f"{path}: {r['rx_file']} has no co-pol twin in this run")
            val, mult, area, valid_fraction, edge = reproject_row(
                r, args.run_dir, args.data_root, rx_name, args.supersample, twin_edge)
            if chan == "chan1":
                copol_edges[(rx_name, stem)] = edge
            r.update(dea.edge_row(edge))
            for key, arr in (("map_npy", val), ("mult_npy", mult), ("area_npy", area)):
                name = os.path.basename(r[key])
                np.save(os.path.join(tmp, name), arr)
                moves.append((os.path.join(tmp, name), os.path.join(args.run_dir, name)))
            r["valid_lunar_fraction"] = valid_fraction
            r["supersample"] = args.supersample
            print(f"  {r['rx_file']}: valid {valid_fraction:.4f}, edge fit {edge['status']} "
                  f"(delay {edge['delay_s'] * 1e6:+.2f} us, Doppler "
                  f"{edge['doppler_hz'] * 1e3:+.2f} mHz)", flush=True)
    # Every look rebuilt: replace the maps, then the CSVs.
    for src, dst in moves:
        os.replace(src, dst)
    for path, _, _, rows in runs:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"Rewrote {len(rows)} looks in {path}")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
