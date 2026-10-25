"""
What each look's imaging geometry alone does to a known surface: a
synthetic reflectivity map is turned into the delay-Doppler image the look
would record (no noise), projected back exactly as the look's real image
was, and written as a map like the look's own, so that stack_maps'
radiometry and stacking can be run on it unchanged. The difference between
what comes back and the surface that went in is the look's geometry error:
its mirror ghosts, and blur where its delay-Doppler cells are large (near
its specular point and its Doppler equator). Sessions' errors, and how they
are shared between sessions, follow from stacking these maps
(combine_sessions.py).

Forward model, per look (same inputs as reproject_maps.reproject_row: the
saved image's axes and size, the edge-fit corrections, the run's timing):
every point of the projection's fine grid (nside x supersample, the points
lunar_projection reads) whose bistatic incidence is below 90 deg puts power
reflectivity x law(incidence) into the image (every point has the same
area), at its delay at the look's
start and at its Doppler in each of n_sub sub-intervals of the look (so its
drift over the look spreads it, as in edge_model_image), spread bilinearly
between cells. The image is then convolved with the code's delay response
(triangle_power, one chip either side) and the coherent integration's
Doppler response (sinc^2 with sidelobes). Occlusion by terrain is ignored,
as in the projection. The law is the one measured from the real looks
(--law: stack_maps.ScatterLaw's bin centres and values, written by
combine_sessions.py --write-law), linear between bin centres and constant
beyond the outermost ones; it rises steeply towards normal incidence, and
spreading that peak over the large cells near a look's specular point is
part of its geometry error there. Without --law: cos(incidence)^1.5, the
edge model's generic law.

Back-projection: the same cell for each fine point as lunar_projection
(same rounding, same placement), the mean intensity over each pixel's
points, stored as log amplitude. Checked against production with
--check: the real saved image read back this way must reproduce the look's
production map (the saved images are float16, so to ~0.003 rms).

The surface: a random field with equal contrast per octave of feature size
(as validation/scripts/validate_mirror_ghosts.py), down to the maps'
finest features (lmax = 3 nside - 1, about 9 km at nside 400), on the fine
grid; reflectivity 1 + CONTRAST x field, clipped at REFLECTIVITY_FLOOR. It
is written once per seed to OUT_DIR/surface_seed<N>.npy (fine grid) and
surface_seed<N>_map.npy (the mean over each map pixel's points: what a
perfect look would give).

Each simulated look is written to OUT_DIR as <rx_file>_map.npy (simulated
map), with a copy of the look's runs-CSV row in
OUT_DIR/simulated_runs_<station>_<chan>.csv whose map_npy points to it and
whose mult_npy / area_npy point to the look's own production files (they
depend on geometry only); noise_log_amp is set far below the signal (no
noise). Finished looks are skipped, so a run can be resumed.

Usage (from the repo root):
    .conda/bin/python simulate_geometry.py --run-dir results/STOCKERT_2025_EDGEFIT \\
        --data-root data.camras.nl/lunar-radar --station stockert --chan chan1 \\
        --out-dir results/GEOMETRY_SIM/seed1 --seed 1
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import healpy as hp
import scipy.ndimage
import scipy.signal
from astropy import time as at

import doppler_equator as de
import doppler_equator_alignment as dea

CONTRAST = 0.2
REFLECTIVITY_FLOOR = 0.01
SIM_NOISE_LOG_AMP = -50.0       # log amplitude far below any simulated cell
SIDELOBE_NULLS = 20             # Doppler response kept out to this many nulls


def make_surface(nside_map, supersample, seed):
    """(reflectivity on the fine grid, its mean per map pixel), float32."""
    lmax = 3 * nside_map - 1
    ell = np.arange(lmax + 1)
    cl = 1.0 / (ell + 20.0) ** 2                  # equal contrast per octave
    cl[:2] = 0
    np.random.seed(seed)
    field = hp.synfast(cl, nside_map * supersample, lmax=lmax).astype(np.float32)
    field /= field.std()
    refl = np.clip(1.0 + CONTRAST * field, REFLECTIVITY_FLOOR, None).astype(np.float32)
    # Mean over each map pixel's points (nside 400 is not a power of 2, so
    # no ud_grade): parents found as lunar_projection finds them.
    npix = hp.nside2npix(nside_map)
    total = np.zeros(npix)
    for i0 in range(0, len(refl), 2_000_000):
        idx = np.arange(i0, min(i0 + 2_000_000, len(refl)))
        v = hp.pix2vec(nside_map * supersample, idx)
        total += np.bincount(hp.vec2pix(nside_map, *v), weights=refl[idx], minlength=npix)
    return refl, (total / supersample ** 2).astype(np.float32)


def look_geometry(r, run_dir, data_root, rx_name):
    """The projection inputs of a look, as reproject_maps.reproject_row
    rebuilds them, with the look's applied edge-fit corrections."""
    with open(os.path.join(data_root, r["rx_meta_path"]), encoding="utf-8") as f:
        meta = json.load(f)
    rx_time_s = dea.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
    fs = float(r["sample_rate_hz"])
    f_hz = float(r["frequency_hz"])
    T = float(r["capture_s"])
    dd = np.load(os.path.join(run_dir, os.path.basename(r["dd_npz"])))
    log_A = dd["log_A"]
    _, dlt_srp = dea.moonSRP_DLT_BCK(rx_time_s, dea.TX_STATION, rx_name, dt=T / 2)
    dlt_shifts = dlt_srp - dd["doppler_hz"] / f_hz
    delay_values_s = ((dd["delay_s"][0] - float(r["tx_extra_offset_s"]))
                      + np.arange(log_A.shape[1]) / fs)
    use_dem = r["use_dem"] == "True"
    lt_ref = -float(r["srp_topo_delay_us"]) * 1e-6 if use_dem else 0.0
    if r["edge_status"] in ("fitted", "inherited") and r["edge_delay_us"] not in ("", "nan"):
        delay_s = float(r["edge_delay_us"]) * 1e-6
        doppler_hz = float(r["edge_doppler_hz"])
    else:
        delay_s = doppler_hz = 0.0
    return dict(log_A=log_A, rx_time_s=rx_time_s, T=T, f_hz=f_hz,
                dlt_shifts=dlt_shifts + doppler_hz / f_hz,
                delay_values_s=delay_values_s, lt_ref=lt_ref - delay_s,
                dlt_rate_srp=dea.srp_dlt_rate_bck(rx_time_s, T, dea.TX_STATION, rx_name),
                chip_samples=dea.chip_samples_of(r["tx_file"], fs),
                doppler_res_dlt=1.0 / (float(r["tx_overlap_s"]) * f_hz),
                use_dem=use_dem, supersample=int(r["supersample"]))


def drift_subintervals(g, rx_name):
    """Sub-intervals so a point drifts by less than one Doppler resolution
    cell within each (as edge_model_image: points along the Doppler
    equator, both halves of the look)."""
    T, t0 = g["T"], g["rx_time_s"]
    s_hat = de.specular_point_bck(t0 + T / 2, dea.TX_STATION, rx_name)
    s_hat = s_hat / np.linalg.norm(s_hat)
    v_tan, _, _ = de.srp_velocity_analytic(t0 + T / 2, dea.TX_STATION, rx_name)
    v_tan = v_tan - (v_tan @ s_hat) * s_hat
    w_hat = v_tan / np.linalg.norm(v_tan)
    a = np.radians(np.arange(-85.0, 85.1, 5.0))
    p = de.moon_surface_points(np.cos(a)[:, None] * s_hat + np.sin(a)[:, None] * w_hat)
    lt = [lt_field(p, dea.apparent_station_positions(t0 + k * T / 2, dea.TX_STATION, rx_name))
          for k in range(3)]
    d = [(lt[k + 1] - lt[k]) / (T / 2) for k in range(2)]
    drift_dlt = 2 * np.max(np.abs(d[1] - d[0] - g["dlt_rate_srp"] * T / 2))
    return max(1, int(np.ceil(drift_dlt / g["doppler_res_dlt"])))


def lt_field(p, sta):
    R_rx, R_tx, c = sta
    return (np.linalg.norm(p - R_rx, axis=1) + np.linalg.norm(p - R_tx, axis=1)) / c


def law_function(path):
    """Brightness vs cos(incidence) from a --law file (None: cos^1.5)."""
    if path is None:
        return lambda c: c ** 1.5
    z = np.load(path)
    good = np.isfinite(z["law"])
    x, y = z["centers"][good], z["law"][good]
    return lambda c: np.interp(c, x, y)


def simulate_look(g, nside, rx_name, refl_fine=None, log_A_check=None, law=None,
                  chunk=2_000_000):
    """Simulated map (log amplitude, UNSEEN where no point falls in the
    image) of reflectivity refl_fine (fine grid) seen through look geometry
    g. With log_A_check instead, read that image back (no forward model):
    the check against production."""
    s = g["supersample"]
    npix, fine_nside = hp.nside2npix(nside), nside * s
    n_fine = hp.nside2npix(fine_nside)
    n_rows, n_cols = g["log_A"].shape
    dlt0, ddlt = g["dlt_shifts"][0], g["dlt_shifts"][1] - g["dlt_shifts"][0]
    d0 = g["delay_values_s"][0]
    ddelay = g["delay_values_s"][1] - d0
    T, t0 = g["T"], g["rx_time_s"]
    sta_start = dea.apparent_station_positions(t0, dea.TX_STATION, rx_name)
    sta_end = dea.apparent_station_positions(t0 + T, dea.TX_STATION, rx_name)
    forward = log_A_check is None
    if forward:
        n_sub = drift_subintervals(g, rx_name)
        sta_sub = [dea.apparent_station_positions(t0 + k * T / n_sub, dea.TX_STATION, rx_name)
                   for k in range(n_sub + 1)]
        image = np.zeros(n_rows * n_cols)
    cell = np.empty(n_fine, dtype=np.int32)
    parent_all = np.empty(n_fine, dtype=np.int32)
    for i0 in range(0, n_fine, chunk):
        idx = np.arange(i0, min(i0 + chunk, n_fine))
        v = np.array(hp.pix2vec(fine_nside, idx)).T
        parent_all[idx] = hp.vec2pix(nside, v[:, 0], v[:, 1], v[:, 2])
        p = de.moon_surface_points(v, use_dem=g["use_dem"])
        del v
        lt = lt_field(p, sta_start)
        # lunar_projection's placement: delay at the start, window-averaged Doppler.
        dlt = (lt_field(p, sta_end) - lt) / T - g["dlt_rate_srp"] * T / 2
        di = np.rint((dlt - dlt0) / ddlt).astype(np.int64)
        ci = np.rint((lt - g["lt_ref"] - d0) / ddelay).astype(np.int64)
        ok = (di >= 0) & (di < n_rows) & (ci >= 0) & (ci < n_cols)
        cell[idx] = np.where(ok, di * n_cols + ci, -1)
        if not forward:
            continue
        # Forward: bistatic incidence on the sphere normal (as stack_maps.incidence_cos).
        R_rx, R_tx, _ = sta_start
        u = R_rx - p
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        w = R_tx - p
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        b = u + w
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        cosi = np.einsum("ij,ij->i", p / np.linalg.norm(p, axis=1, keepdims=True), b)
        lit = cosi > 0
        p, lt, cosi = p[lit], lt[lit], cosi[lit]
        power = refl_fine[idx[lit]].astype(np.float64) * law(cosi) / n_sub
        col = (lt - g["lt_ref"] - d0) / ddelay
        c0 = np.floor(col).astype(np.int64)
        fc = col - c0
        lts = [lt] + [lt_field(p, st) for st in sta_sub[1:]]
        for k in range(n_sub):
            dk = (lts[k + 1] - lts[k]) / (T / n_sub) - g["dlt_rate_srp"] * (k + 0.5) * T / n_sub
            row = (dk - dlt0) / ddlt
            r0 = np.floor(row).astype(np.int64)
            fr = row - r0
            for dr, wr in ((0, 1 - fr), (1, fr)):
                for dc, wc in ((0, 1 - fc), (1, fc)):
                    rr, cc = r0 + dr, c0 + dc
                    inside = (rr >= 0) & (rr < n_rows) & (cc >= 0) & (cc < n_cols)
                    image += np.bincount(rr[inside] * n_cols + cc[inside],
                                         weights=(power * wr * wc)[inside],
                                         minlength=n_rows * n_cols)
        del p, lt, lts, cosi, power
    if forward:
        img = image.reshape(n_rows, n_cols)
        img = scipy.ndimage.convolve1d(img, dea.triangle_power(g["chip_samples"]),
                                       axis=1, mode="constant")
        null_rows = g["doppler_res_dlt"] / abs(ddlt)
        k = np.arange(-int(SIDELOBE_NULLS * null_rows), int(SIDELOBE_NULLS * null_rows) + 1)
        sinc2 = np.sinc(k / null_rows) ** 2
        img = scipy.signal.fftconvolve(img, (sinc2 / sinc2.sum())[:, None], mode="same", axes=0)
        intensity = np.clip(img, 0.0, None).ravel()
        ref = 0.0
    else:
        ref = float(np.max(log_A_check))
        intensity = np.exp(2.0 * (np.asarray(log_A_check, dtype=np.float64) - ref)).ravel()
    inside = cell >= 0
    sum_i = np.bincount(parent_all[inside], weights=intensity[cell[inside]], minlength=npix)
    n_in = np.bincount(parent_all[inside], minlength=npix)
    out = np.full(npix, hp.UNSEEN, dtype=np.float32)
    good = (n_in > 0) & (sum_i > 0)
    out[good] = 0.5 * np.log(sum_i[good] / n_in[good]) + ref
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--station", required=True, choices=sorted(dea.STATION_RX))
    parser.add_argument("--chan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--looks", nargs="*", default=None,
                        help="only these captures (a time in the file name, e.g. 2025_09_11_06_25_19)")
    parser.add_argument("--law", default=None,
                        help="measured scattering law (combine_sessions.py --write-law); "
                             "default cos(incidence)^1.5")
    parser.add_argument("--check", action="store_true",
                        help="read each look's real saved image back and compare with its "
                             "production map; writes nothing")
    args = parser.parse_args()

    rx_name = dea.STATION_RX[args.station]
    dea.load_lola_dem()
    with open(os.path.join(args.run_dir, f"registration_runs_{args.station}_{args.chan}.csv")) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("dd_npz")]
    if args.looks:
        rows = [r for r in rows if any(f"_{t}_" in r["rx_file"] for t in args.looks)]
    nside = hp.npix2nside(len(np.load(os.path.join(args.run_dir, os.path.basename(rows[0]["map_npy"])),
                                      mmap_mode="r")))
    if args.check:
        for r in rows:
            g = look_geometry(r, args.run_dir, args.data_root, rx_name)
            t = time.time()
            m = simulate_look(g, nside, rx_name, log_A_check=g["log_A"].astype(np.float32))
            prod = np.load(os.path.join(args.run_dir, os.path.basename(r["map_npy"])))
            a, b = m != hp.UNSEEN, prod != hp.UNSEEN
            both = a & b
            d = m[both] - prod[both]
            print(f"{r['rx_file']}: same pixels {np.mean(a == b):.6f}; read-back minus production "
                  f"rms {np.sqrt(np.mean(d ** 2)):.4f}, max {np.abs(d).max():.3f} "
                  f"({time.time() - t:.0f} s)", flush=True)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    s = int(rows[0]["supersample"])
    surf = os.path.join(args.out_dir, f"surface_seed{args.seed}.npy")
    if not os.path.exists(surf):
        refl, refl_map = make_surface(nside, s, args.seed)
        np.save(os.path.join(args.out_dir, f"surface_seed{args.seed}_map.npy"), refl_map)
        np.save(surf, refl)
    refl = np.load(surf)
    law = law_function(args.law)
    out_csv = os.path.join(args.out_dir, f"simulated_runs_{args.station}_{args.chan}.csv")
    done = {}
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            done = {r["rx_file"]: r for r in csv.DictReader(f)}
    for r in rows:
        if r["rx_file"] in done:
            continue
        t = time.time()
        g = look_geometry(r, args.run_dir, args.data_root, rx_name)
        m = simulate_look(g, nside, rx_name, refl_fine=refl, law=law)
        path = os.path.join(args.out_dir, r["rx_file"] + "_map.npy")
        np.save(path, m)
        row = dict(r)
        row["map_npy"] = os.path.abspath(path)
        for k in ("mult_npy", "area_npy"):
            row[k] = os.path.abspath(os.path.join(args.run_dir, os.path.basename(r[k])))
        row["noise_log_amp"] = SIM_NOISE_LOG_AMP
        done[r["rx_file"]] = row
        with open(out_csv + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(done.values())
        os.replace(out_csv + ".tmp", out_csv)
        print(f"{r['rx_file']}: simulated ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
