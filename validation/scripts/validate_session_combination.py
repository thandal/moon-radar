"""Does combine_sessions.py's stack (sessions weighted by geometry and
noise, each session's looks averaged band by band within the cell-area
cuts) leave less error than stack_maps' equal weight per look?

1. Synthetic, on a surface independent of the one the weights were built
   from (--sim-dir: simulate_geometry.py with another seed): per
   feature-size band, the geometry error each stack keeps (sum over
   sessions of weight x the session's simulated error, the sessions'
   simulated looks averaged with the same level-1 masks as the stack, and
   the surface's whole detail in the band where no session has it --
   weights summing to less than one -- scaled to the real surface by the
   combination's KAPPA) plus the noise
   it keeps (sum of weight^2 x the session's noise variance, measured on
   the real looks), as mean squares over the covered disk and within
   PATCH_KM of the patch centre, relative to the surface's own detail
   power there. Equal weights: each session in proportion to its looks at
   the pixel, with stack_maps' mask (its per-look mean).
2. Real data: repeatable detail between two independent half-stacks (each
   session's interleaved half-session means, combined with the same
   weights -- written by combine_sessions.py; for equal weights the
   per-look mean of the halves), per DiskTiles band, over the disk and
   over the tiles centred within PATCH_KM of the patch centre.

Patch centre: --patch-lon/--patch-lat, by default the 2025 sessions' mean
specular point (the blurry spot) or, with no 2025 looks, the mean
specular point of all looks. Combinations made with
combine_sessions.py --level1 limit or cuts (the per-band cell-area cuts,
the main product from 2026-10-26 to 2026-10-29) are reported under
session_combination_<label>_single_limit.json or _per_band_cuts.json.

Usage (from the repo root):
  .conda/bin/python validation/scripts/validate_session_combination.py \\
      --combination results/ALL_SEASONS_FIGURE/combination_stockert_wsrt_chan1.npz \\
      --run-dir results/STOCKERT_2025_FIGURE results/EVE2026_CIRC_FIGURE --stations stockert wsrt \\
      --chans chan1 --equal-stack-dir results/ALL_SEASONS_FIGURE --sim-dir results/GEOMETRY_SIM_FIGURE/seed2
Output: validation/results/session_combination_<label>[_single_limit|_per_band_cuts].json
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import healpy as hp
import numpy as np

import validation_common as vc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import combine_sessions as cs  # noqa: E402
import doppler_equator as de  # noqa: E402
import doppler_equator_alignment as dea  # noqa: E402
import stack_maps as sm  # noqa: E402
import validate_mirror_ghosts as g  # noqa: E402

PATCH_KM = 250.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--combination", required=True)
    p.add_argument("--run-dir", nargs="+", required=True)
    p.add_argument("--stations", nargs="+", required=True)
    p.add_argument("--chans", nargs="+", required=True, help="as combine_sessions.py --chans")
    p.add_argument("--equal-stack-dir", required=True)
    p.add_argument("--sim-dir", required=True, help="simulated looks on an independent surface")
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--patch-lon", type=float, default=None)
    p.add_argument("--patch-lat", type=float, default=None)
    args = p.parse_args()
    label = "_".join(args.stations) + "_" + ("dual" if len(args.chans) > 1 else args.chans[0])

    z = np.load(args.combination)
    names = list(z["sessions"])
    if os.path.abspath(args.sim_dir) == str(z["sim_dir"]):
        raise SystemExit("--sim-dir is the surface the weights were built from; use another seed")
    kappa, n_b = z["kappa"], len(z["bands"])
    W, NV = z["weights"], z["noise_variance"]          # read once (npz reloads per access)
    levels = [tuple(lv) for lv in z["levels"]]
    lob = list(z["level_of_band"])
    footprint_cuts = bool(z["footprint_cuts"])
    mode = str(z["level1"]) if "level1" in z.files else ("cuts" if footprint_cuts else "limit")
    looks = z["session_looks"].astype(np.float64)
    ref_mean, ref_h = z["session_mean"], (z["half0"], z["half1"])
    half_combined = z["half_combined"]
    present = np.isfinite(ref_mean)
    npix = present.shape[1]
    nside = hp.npix2nside(npix)
    n_s = len(names)
    if not np.all(np.isfinite(kappa)):
        raise SystemExit("the combination has a single session (no KAPPA): nothing was weighted; "
                         "only its level-1 cuts differ from the equal-weight stack")

    # The looks and their simulated twins on the independent surface.
    good, _ = sm.select_looks(args.run_dir, args.stations, args.chans)
    with open(os.path.join(args.equal_stack_dir, f"stacked_looks_{label}.csv")) as f:
        used = {u["rx_file"] for u in csv.DictReader(f) if u["used"] == "True"}
    rows = [r for r in good if r["rx_file"] in used]
    srows, truth = cs.simulated_twins(rows, args.sim_dir, args.stations)
    cache = {}
    laws = cs.fit_laws(srows, cache, npix)
    _, _, reference = cs.level_plan()
    all_levels = levels if reference in levels else levels + [reference]
    smooth = cs.Smoother(nside)

    # Patch centre.
    if args.patch_lon is not None:
        lon_c, lat_c = args.patch_lon, args.patch_lat
        c = hp.ang2vec(lon_c, lat_c, lonlat=True)
        what = "given"
    else:
        sp = []
        pick = [r for r in rows if r["rx_start_utc"].startswith("2025")] or rows
        for r in pick:
            t = dea.csp.str2et(r["rx_start_utc"]) + float(r["capture_s"]) / 2
            v = de.specular_point_bck(t, dea.TX_STATION, r["_rx"])
            sp.append(v / np.linalg.norm(v))
        c = np.mean(sp, axis=0)
        c /= np.linalg.norm(c)
        lon_c, lat_c = np.degrees(np.arctan2(c[1], c[0])), np.degrees(np.arcsin(c[2]))
        what = ("2025 mean specular point" if pick[0]["rx_start_utc"].startswith("2025")
                else "mean specular point of the looks")
    v_pix = np.array(hp.pix2vec(nside, np.arange(npix))).T
    in_patch = np.arccos(np.clip(v_pix @ c, -1, 1)) * cs.R_KM < PATCH_KM
    print(f"patch: within {PATCH_KM:g} km of lon {lon_c:+.2f} lat {lat_c:+.2f} ({what})")
    print("level 1: " + {"cuts": "cell-area cuts per band", "limit": "single multiplicity limit",
                         "none": "no mask"}[mode])

    # 1. Synthetic: error kept by each weighting. Simulated session errors
    # with the stack's level-1 masks (weighted) and stack_maps' mask (equal).
    E, E_eq = {}, {}
    for s, out in cs.session_levels(srows, laws, None, cache, npix, all_levels, args.min_count,
                                    sim=True):
        E[s] = []
        for b in range(n_b):
            S = out[levels[lob[b]]]["M"]
            E[s].append(cs.band(S - truth, np.isfinite(S), ("s", s, lob[b]), smooth, b))
        S = out[reference]["M"]
        E_eq[s] = cs.bands(S - truth, np.isfinite(S), ("q", s), smooth)
        smooth.cache.clear()
    E = [E[s] for s in names]
    E_eq = [E_eq[s] for s in names]
    # Noise of the equal-weight stack's sessions: from their own half-session
    # means with stack_maps' mask.
    NV_eq = np.zeros((n_b, n_s, npix), dtype=np.float32)
    for i in range(n_s):
        hm = np.isfinite(ref_h[0][i]) & np.isfinite(ref_h[1][i])
        h0 = cs.bands(ref_h[0][i], hm, ("h", i), smooth)
        h1 = cs.bands(ref_h[1][i], hm, ("h", i), smooth)
        for b in range(n_b):
            NV_eq[b, i] = cs.local_noise(h0[b], h1[b], smooth, ("n", i, b), b)
        smooth.cache.clear()
    Tb = cs.bands(truth, np.ones(npix, bool), ("t",), smooth)
    covered = np.isfinite(half_combined[0]) | present.any(axis=0)
    w_eq = np.where(present, looks, 0.0)
    w_eq /= np.maximum(w_eq.sum(axis=0), 1e-12)
    result = {"patch_center_lon_lat_deg": [lon_c, lat_c], "patch_km": PATCH_KM,
              "level1": mode, "levels": [list(lv) for lv in levels],
              "level_of_band": lob, "bands": list(z["bands"]), "kappa": kappa.tolist(),
              "synthetic": {}, "real": {}}
    print("\nSynthetic (independent surface): error kept / surface detail power, per band")
    print(f"  {'':34s}" + "".join(f"{b:>11s}" for b in z["bands"]))
    rows_out = {}
    for region, sel in (("disk", covered), ("patch", covered & in_patch)):
        for name in ("equal", "geomweighted"):
            geo, noi, rel = [], [], []
            for b in range(n_b):
                if name == "equal":
                    w, err, nv = w_eq, [E_eq[i][b] for i in range(n_s)], NV_eq[b]
                else:
                    w, err, nv = W[b].astype(np.float64), [E[i][b] for i in range(n_s)], NV[b]
                # Error of sum_s w_s B_s against the true band: the weighted
                # session errors, minus the detail missing where the weights sum
                # to less than one (no session has the band there).
                eg = (sum(w[i] * np.nan_to_num(err[i]) for i in range(n_s))
                      - (1.0 - sum(w[i] for i in range(n_s))) * Tb[b])
                en = sum(w[i] ** 2 * np.nan_to_num(nv[i]) for i in range(n_s))
                P = kappa[b] * np.mean(Tb[b][sel] ** 2)
                geo.append(kappa[b] * np.mean(eg[sel] ** 2) / P)
                noi.append(np.mean(en[sel]) / P)
                rel.append(geo[-1] + noi[-1])
            rows_out[(region, name)] = (geo, noi, rel)
            for what_, vals in (("geometry", geo), ("noise", noi), ("total", rel)):
                print(f"  {region + ', ' + name + ', ' + what_:34s}" + "".join(f"{x:11.3f}" for x in vals))
            result["synthetic"][f"{region}/{name}"] = {"geometry": geo, "noise": noi, "total": rel}
        ratio = np.array(rows_out[(region, "geomweighted")][2]) / np.array(rows_out[(region, "equal")][2])
        print(f"  {region + ', total: weighted / equal':34s}" + "".join(f"{x:11.3f}" for x in ratio))
        result["synthetic"][f"{region}/total_ratio"] = ratio.tolist()

    # 2. Real: repeatable detail between independent half-stacks.
    halves = {}
    for k in (0, 1):
        H = ref_h[k].astype(np.float64)
        hm = np.isfinite(H)
        eq_num = sum(np.where(hm[i], H[i] * looks[i], 0.0) for i in range(n_s))
        eq_den = sum(np.where(hm[i], looks[i], 0.0) for i in range(n_s))
        halves[("equal", k)] = np.where(eq_den > 0, eq_num / np.maximum(eq_den, 1e-12), np.nan)
        halves[("geomweighted", k)] = half_combined[k].astype(np.float64)
    tiles = g.DiskTiles()
    xs = np.arange(-g.EXT_KM, g.EXT_KM + g.PX_KM / 2, g.PX_KM)
    cx = cs.R_KM * np.cos(np.radians(lat_c)) * np.sin(np.radians(lon_c))
    cy = cs.R_KM * np.sin(np.radians(lat_c))
    near = [(i, j) for i, j in tiles.origins
            if np.hypot(xs[j + tiles.nt // 2] - cx, xs[i + tiles.nt // 2] - cy) < PATCH_KM]

    def repeatable(a, b, origins):
        A, B = tiles.grid(a), tiles.grid(b)
        x = np.zeros(len(g.BANDS_KM)); pa = np.zeros_like(x); pb = np.zeros_like(x); n = 0
        for i, j in origins:
            ta, tb = A[i:i + tiles.nt, j:j + tiles.nt], B[i:i + tiles.nt, j:j + tiles.nt]
            ok = np.isfinite(ta) & np.isfinite(tb)
            if ok.mean() < 0.95:
                continue
            n += 1
            Fa = np.fft.fft2(np.where(ok, ta - np.nanmean(ta), 0.0) * tiles.taper)
            Fb = np.fft.fft2(np.where(ok, tb - np.nanmean(tb), 0.0) * tiles.taper)
            for k, s in enumerate(tiles.sel):
                x[k] += np.real(Fa[s] * np.conj(Fb[s])).sum()
                pa[k] += (np.abs(Fa[s]) ** 2).sum(); pb[k] += (np.abs(Fb[s]) ** 2).sum()
        return x / np.sqrt(pa * pb), n

    print("\nReal: repeatable detail between independent half-stacks, per DiskTiles band")
    print(f"  {'':34s}" + "".join(f"{f'{a:g}-{b:g} km':>11s}" for a, b in g.BANDS_KM))
    for region, origins in (("disk", tiles.origins), ("patch", near)):
        for name in ("equal", "geomweighted"):
            cc, n = repeatable(halves[(name, 0)], halves[(name, 1)], origins)
            print(f"  {region + ', ' + name + f' ({n} tiles)':34s}" + "".join(f"{x:11.3f}" for x in cc))
            result["real"][f"{region}/{name}"] = {"repeatable": cc.tolist(), "tiles": n}
    suffix = {"none": "", "limit": "_single_limit", "cuts": "_per_band_cuts"}[mode]
    out = vc.write_json(f"session_combination_{label}{suffix}.json", result)
    print(f"\nWrote {vc.report_path(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
