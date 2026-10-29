"""
Rebuild a run's per-look lunar maps from the saved delay-Doppler images,
without redoing the correlation.

registration_stability.py --save-pngs saves each look's DD image as
<rx_file>_dd.npz. When only the projection or the edge fit changes
(lunar_projection, fit_edge_offsets in doppler_equator_alignment.py), this
fits each opposite-sense look's edges again (its same-sense twin takes the result,
as in process_file), re-projects the images and replaces each look's
_map.npy, _mapcount.npy and _cellarea.npy and its runs-CSV row
(valid_lunar_fraction, supersample, edge_*). The saved images are float16
(log amplitude to ~0.004); that is the only difference from a full rerun.
Opposite-sense channels (chan1) are done first, so --chans must include chan1
whenever it includes chan0.

--terrain re-projects (and re-fits the edges) on another terrain model than
the run was mapped with (doppler_equator.terrain_model): "figure", the Moon's
smooth overall shape (the maps' model since 2026-10-27), "sphere" or "lola";
each row then records the new model, its terrain height and delay at the
specular point, and the terrain grid's name only for "lola". Copy the run
directory first to keep the old maps.

Everything is read from and written to --run-dir (recorded paths are used
only for their file names), each capture's metadata is found through its
recorded rx_meta_path under --data-root, and nothing is replaced until every
look has been rebuilt: the run is checked first, the new maps are written to
a temporary folder, and only then moved into place with the CSVs.

Usage (from the repo root):
    .conda/bin/python reproject_maps.py --run-dir results/NEW_REGISTRATION \\
        --data-root data.camras.nl/lunar-radar --stations stockert --chans chan1 chan0
    .conda/bin/python reproject_maps.py --run-dir results/STOCKERT_2025_FIGURE \\
        --data-root data.camras.nl/lunar-radar --stations stockert --chans chan1 chan0 \\
        --terrain figure
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import healpy as hp
from astropy import time as at

import doppler_equator_alignment as dea

NEEDED = ("dd_npz", "rx_meta_path", "capture_s", "tx_overlap_s", "srp_topo_delay_us",
          "use_dem", "tx_extra_offset_s", "tx_file", "spice_kernels", "dem_file")


def reproject_row(r, run_dir, data_root, rx_name, supersample, twin_edge=None, terrain=None):
    """New (map, multiplicity, cell area, valid fraction, edge fit, terrain
    fields) for one runs-CSV row. twin_edge: None for an opposite-sense row (its
    edges are fitted here, as process_file does); the opposite-sense twin's edge
    fit for a same-sense row. terrain: the terrain model to map on (None: the
    row's own)."""
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
    use_dem = dea.terrain_model(r["use_dem"] if terrain is None else terrain)
    # As process_file: lag 0 sits at the terrain's light time at the SRP.
    srp_elevation_km = srp_topo_delay_s = np.nan
    lt_ref = 0.0
    if use_dem != "sphere":
        srp_elevation_km, srp_topo_delay_s = dea.extract_srp_elevation(
            rx_time_s, dea.TX_STATION, rx_name, use_dem=use_dem)
        lt_ref = -srp_topo_delay_s
    # Same rate compute_dd_image compensated (same calls, same stencils).
    dlt_rate_srp = dea.srp_dlt_rate_bck(rx_time_s, rx_duration_s, dea.TX_STATION, rx_name)
    nside = hp.npix2nside(len(np.load(os.path.join(run_dir, os.path.basename(r["map_npy"])),
                                      mmap_mode="r")))
    chip_samples = dea.chip_samples_of(r["tx_file"], fs)
    # The saved axes are before the edge fit (process_file); as there: fit
    # (opposite-sense looks, when the rims were measured), or take the twin's.
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
    fields = {"use_dem": use_dem, "srp_elevation_km": srp_elevation_km,
              "srp_topo_delay_us": srp_topo_delay_s * 1e6}
    return val, mult, area, valid_fraction, edge, fields


def reproject_task(task):
    """One look for a worker: re-project it, write its maps into tmp, return
    (valid fraction, edge fit, terrain fields, map file names)."""
    r, run_dir, data_root, rx_name, supersample, twin_edge, terrain, tmp = task
    if dea.terrain_model(r["use_dem"] if terrain is None else terrain) == "lola":
        dea.load_lola_dem()
    val, mult, area, valid_fraction, edge, fields = reproject_row(
        r, run_dir, data_root, rx_name, supersample, twin_edge, terrain)
    names = []
    for key, arr in (("map_npy", val), ("mult_npy", mult), ("area_npy", area)):
        name = os.path.basename(r[key])
        np.save(os.path.join(tmp, name), arr)
        names.append(name)
    return valid_fraction, edge, fields, names


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
    parser.add_argument("--workers", type=int, default=1,
                        help="looks re-projected at once, each in its own process "
                             "(about 1.2 GB each); results are the same")
    parser.add_argument("--terrain", choices=dea.TERRAIN_MODELS, default=None,
                        help="terrain model to map on (default: each row's own)")
    args = parser.parse_args()
    kernels = dea.kernel_fingerprint()

    if "chan0" in args.chans and "chan1" not in args.chans:
        parser.error("same-sense (chan0) looks take their opposite-sense twin's edge fit: "
                     "include chan1")
    runs = []
    for station in args.stations:
        # the opposite-sense channel first: its edge fits are the same-sense looks'
        for chan in sorted(args.chans, key=lambda c: c != "chan1"):
            path = os.path.join(args.run_dir, f"registration_runs_{station}_{chan}.csv")
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                missing = [k for k in NEEDED if not r.get(k)
                           and not (k == "dem_file" and dea.terrain_model(r.get("use_dem")) != "lola")]
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
                if args.terrain is None and dea.terrain_model(r["use_dem"]) == "lola":
                    dem_file = os.path.basename(dea.load_lola_dem())
                    if r["dem_file"] != dem_file:
                        parser.error(f"{path}: {r['rx_file']} was made with {r['dem_file']}, "
                                     f"now {dem_file}; reprocess instead")
            runs.append((path, dea.STATION_RX[station], chan, rows))
    if args.terrain == "lola":
        dem_file = os.path.basename(dea.load_lola_dem())

    tmp = os.path.join(args.run_dir, ".reproject_tmp")
    os.makedirs(tmp, exist_ok=True)
    moves = []
    opposite_sense_edges = {}           # (rx_name, capture stem) -> edge fit

    def run_phase(items):
        """Re-project the (path, rx_name, chan, row) items, --workers at a
        time; update each row and queue its maps' moves."""
        tasks = []
        for path, rx_name, chan, r in items:
            stem = r["rx_file"].split(".chan")[0]
            twin_edge = None
            if chan != "chan1":
                twin_edge = opposite_sense_edges.get((rx_name, stem))
                if twin_edge is None:
                    parser.error(f"{path}: {r['rx_file']} has no opposite-sense twin in this run")
            tasks.append((r, args.run_dir, args.data_root, rx_name, args.supersample, twin_edge,
                          args.terrain, tmp))
        if args.workers == 1:
            results = map(reproject_task, tasks)
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"))
            results = pool.map(reproject_task, tasks)
        for (path, rx_name, chan, r), (valid_fraction, edge, fields, names) in zip(items, results):
            if chan == "chan1":
                opposite_sense_edges[(rx_name, r["rx_file"].split(".chan")[0])] = edge
            r.update(dea.edge_row(edge))
            r.update(fields)
            if args.terrain is not None:
                r["dem_file"] = dem_file if args.terrain == "lola" else ""
            for name in names:
                moves.append((os.path.join(tmp, name), os.path.join(args.run_dir, name)))
            r["valid_lunar_fraction"] = valid_fraction
            r["supersample"] = args.supersample
            print(f"  {r['rx_file']}: {fields['use_dem']}, valid {valid_fraction:.4f}, edge fit {edge['status']} "
                  f"(delay {edge['delay_s'] * 1e6:+.2f} us, Doppler "
                  f"{edge['doppler_hz'] * 1e3:+.2f} mHz)", flush=True)
        if args.workers != 1:
            pool.shutdown()

    # Opposite-sense looks first: their edge fits are the same-sense looks'.
    run_phase([(path, rx_name, chan, r) for path, rx_name, chan, rows in runs if chan == "chan1" for r in rows])
    run_phase([(path, rx_name, chan, r) for path, rx_name, chan, rows in runs if chan != "chan1" for r in rows])
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
