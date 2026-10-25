"""Does weighting sessions by geometry and noise (combine_sessions.py) leave
less error than stack_maps' equal weight per look?

1. Synthetic, on a surface independent of the one the weights were built
   from (--sim-dir: simulate_geometry.py with another seed): per
   feature-size band, the geometry error each stack keeps (sum over
   sessions of weight x the session's simulated error, scaled to the real
   surface by the combination's KAPPA) plus the noise it keeps (sum of
   weight^2 x the session's noise variance, measured on the real looks),
   as mean squares over the covered disk and within PATCH_KM of the 2025
   sessions' mean specular point (the blurry spot), and relative to the
   surface's own detail power there. Equal weights: each session in
   proportion to its looks at the pixel (stack_maps' per-look mean).
2. Real data: repeatable detail between two independent half-stacks (each
   session's interleaved half-session means, combined with the same weights;
   for equal weights the per-look mean of the halves), per DiskTiles band,
   over the disk and over the tiles centred within PATCH_KM of that point.

Usage (from the repo root):
  .conda/bin/python validation/scripts/validate_session_combination.py \\
      --combination results/ALL_SEASONS_EDGEFIT/combination_stockert_wsrt_chan1.npz \\
      --run-dir results/STOCKERT_2025_EDGEFIT results/EVE2026_CIRC --stations stockert wsrt \\
      --chans chan1 --equal-stack-dir results/ALL_SEASONS_EDGEFIT --sim-dir results/GEOMETRY_SIM/seed2
Output: validation/results/session_combination_<label>.json
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
    args = p.parse_args()
    label = "_".join(args.stations) + "_" + ("dual" if len(args.chans) > 1 else args.chans[0])

    z = np.load(args.combination)
    names = list(z["sessions"])
    if os.path.abspath(args.sim_dir) == str(z["sim_dir"]):
        raise SystemExit("--sim-dir is the surface the weights were built from; use another seed")
    kappa, n_b = z["kappa"], len(z["bands"])
    W, NV = z["weights"], z["noise_variance"]          # read once (npz reloads per access)
    looks = z["session_looks"].astype(np.float64)
    present = np.isfinite(z["session_mean"])
    npix = present.shape[1]
    nside = hp.npix2nside(npix)

    # The looks and their simulated twins on the independent surface.
    good, _ = sm.select_looks(args.run_dir, args.stations, args.chans)
    with open(os.path.join(args.equal_stack_dir, f"stacked_looks_{label}.csv")) as f:
        used = {u["rx_file"] for u in csv.DictReader(f) if u["used"] == "True"}
    rows = [r for r in good if r["rx_file"] in used]
    srows, truth = cs.simulated_twins(rows, args.sim_dir, args.stations)
    cache = {}
    laws = cs.fit_laws(srows, cache, npix, args.chans)
    sim = cs.session_sums(srows, laws, None, cache, npix, sim=True)
    smooth = cs.Smoother(nside)

    # The blurry spot: around the 2025 sessions' mean specular point.
    sp = []
    for r in rows:
        if r["rx_start_utc"].startswith("2025"):
            t = dea.csp.str2et(r["rx_start_utc"]) + float(r["capture_s"]) / 2
            v = de.specular_point_bck(t, dea.TX_STATION, r["_rx"])
            sp.append(v / np.linalg.norm(v))
    c = np.mean(sp, axis=0)
    c /= np.linalg.norm(c)
    lon_c, lat_c = np.degrees(np.arctan2(c[1], c[0])), np.degrees(np.arcsin(c[2]))
    v_pix = np.array(hp.pix2vec(nside, np.arange(npix))).T
    in_patch = np.arccos(np.clip(v_pix @ c, -1, 1)) * cs.R_KM < PATCH_KM
    print(f"blurry spot: within {PATCH_KM:g} km of lon {lon_c:+.2f} lat {lat_c:+.2f} "
          f"(2025 mean specular point)")

    # 1. Synthetic: error kept by each weighting.
    E = []
    for i, s in enumerate(names):
        S = cs.mean_of(sim[s]["sum"], sim[s]["n"], args.min_count)
        m = np.isfinite(S)
        E.append(cs.bands(S - truth, m, ("s", s), smooth))
    Tb = cs.bands(truth, np.ones(npix, bool), ("t",), smooth)
    covered = present.any(axis=0)
    w_eq = np.where(present, looks, 0.0)
    w_eq /= np.maximum(w_eq.sum(axis=0), 1e-12)
    result = {"patch_center_lon_lat_deg": [lon_c, lat_c], "patch_km": PATCH_KM,
              "bands": list(z["bands"]), "kappa": kappa.tolist(), "synthetic": {}, "real": {}}
    print("\nSynthetic (independent surface): error kept / surface detail power, per band")
    print(f"  {'':34s}" + "".join(f"{b:>11s}" for b in z["bands"]))
    rows_out = {}
    for region, sel in (("disk", covered), ("blurry spot", covered & in_patch)):
        for name in ("equal", "geomweighted"):
            geo, noi, rel = [], [], []
            for b in range(n_b):
                w = w_eq if name == "equal" else W[b].astype(np.float64)
                eg = sum(w[i] * np.nan_to_num(E[i][b]) for i in range(len(names)))
                en = sum(w[i] ** 2 * np.nan_to_num(NV[b][i]) for i in range(len(names)))
                P = kappa[b] * np.mean(Tb[b][sel] ** 2)
                geo.append(kappa[b] * np.mean(eg[sel] ** 2) / P)
                noi.append(np.mean(en[sel]) / P)
                rel.append(geo[-1] + noi[-1])
            rows_out[(region, name)] = (geo, noi, rel)
            for what, vals in (("geometry", geo), ("noise", noi), ("total", rel)):
                print(f"  {region + ', ' + name + ', ' + what:34s}" + "".join(f"{x:11.3f}" for x in vals))
            result["synthetic"][f"{region}/{name}"] = {"geometry": geo, "noise": noi, "total": rel}
        ratio = np.array(rows_out[(region, "geomweighted")][2]) / np.array(rows_out[(region, "equal")][2])
        print(f"  {region + ', total: weighted / equal':34s}" + "".join(f"{x:11.3f}" for x in ratio))
        result["synthetic"][f"{region}/total_ratio"] = ratio.tolist()

    # 2. Real: repeatable detail between independent half-stacks.
    halves = {}
    for k in (0, 1):
        H = z[f"half{k}"].astype(np.float64)
        hm = np.isfinite(H)
        eq_num = sum(np.where(hm[i], H[i] * looks[i], 0.0) for i in range(len(names)))
        eq_den = sum(np.where(hm[i], looks[i], 0.0) for i in range(len(names)))
        halves[("equal", k)] = np.where(eq_den > 0, eq_num / np.maximum(eq_den, 1e-12), np.nan)
        hb = [cs.bands(H[i], hm[i], ("h", k, i), smooth) for i in range(len(names))]
        tot = np.zeros(npix)
        for b in range(n_b):
            wk = np.where(hm, W[b].astype(np.float64), 0.0)
            wk /= np.maximum(wk.sum(axis=0), 1e-12)
            tot += sum(wk[i] * np.nan_to_num(hb[i][b]) for i in range(len(names)))
        tot[~hm.any(axis=0)] = np.nan
        halves[("geomweighted", k)] = tot
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
    for region, origins in (("disk", tiles.origins), ("blurry spot", near)):
        for name in ("equal", "geomweighted"):
            cc, n = repeatable(halves[(name, 0)], halves[(name, 1)], origins)
            print(f"  {region + ', ' + name + f' ({n} tiles)':34s}" + "".join(f"{x:11.3f}" for x in cc))
            result["real"][f"{region}/{name}"] = {"repeatable": cc.tolist(), "tiles": n}
    out = vc.write_json(f"session_combination_{label}.json", result)
    print(f"\nWrote {vc.report_path(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
