"""How much of a place's surface detail a look's map pixel carries, against
the ground area inside the image's resolution cell there (its footprint):
the measurement behind the degeneracy mask (2026-10-25).

A known surface (simulate_geometry.py's random field and the measured
scattering law, the same files as results/GEOMETRY_SIM) goes through each
look's real geometry with simulate_geometry.simulate_look. By default the
image is focused: every point at its mid-look Doppler, as the drift-
correcting imaging (investigations/focused_dd_prototype.py) makes it
(simulate_look with one sub-interval); --unfocused spreads each point's
drift over the look, as production's image does now. No noise. The map
becomes brightness per unit area as stack_maps makes it (intensity / cell
area), and the surface's true brightness at each pixel is the surface times
the law at the pixel's incidence; both are divided by their mean at 20-50
deg incidence. Both are split into combine_sessions' feature-size bands,
over the same pixels. For each band and each footprint bin:

  retained: sum(look x true) / sum(true^2), the share of the true detail at
            the place that comes back at that place (1 when the pixel is
            fully localized; blur lowers it; mirror ghosts and smear from
            elsewhere add power uncorrelated with the place and leave it
            unbiased);
  corr:     their correlation (blur and added power both lower it);
  share:    the fraction of the look's seen pixels in the bin;
  level:    mean look / mean true brightness over the bin's pixels, before
            the bands (1 when the cell area divides out the image's response
            correctly).

Footprint: the look's saved cell area (area_npy) times
dea.resolution_cells for its image grid, in map pixels (km^2 / pixel area);
also as a multiple of the look's median over its seen pixels. Looks are
grouped by session, capture length and chip rate.

Results, 2026-10-25 (results/FOOTPRINT_TEST/report.txt; 24 focused
Stockert opposite-sense looks: all 9 of 2026-10-11 and 16 of the 2025
sessions, 3 per session and look type; seed 1 only):
  - The footprint does not depend on the image grid: on 09:41:25 it agrees
    to 4 % (median 0.964) between production's rows (7.72 mHz) and the
    prototype's (3.05 mHz); the overlap count it replaces differs by
    0.36-0.88, and its cut of 3 masks 1.4 % of that look in production
    and nothing in the prototype.
  - No cliff: retained detail falls steadily with the footprint, faster
    for smaller features. Off the line of highest overlap it stays near 0.5
    even for small footprints, because each cell also holds the mirror
    point.
  - The fall lines up across look types against the footprint relative to
    the look's own median better than against the footprint in pixels.
    Relative to each type's value at 0.5-1x median, detail falls to half at
    about 2x median for features below 10 km, 4x for 10-20 km, 8x for
    20-40 km and 16-32x for 40-75 km. Above 75 km nothing is lost.
  - Share of each look above 2x / 4x / 8x / 16x its median: 25 / 12-13 /
    2-7 / under 0.6 %. The old cut (overlap count 3) sat at 9-13x the
    median in production (6x on two 2025-09-11 looks) and masked 1.1-2.1 %.
  - Unfocused (2026 long looks): the mean brightness of the largest
    footprints (16-32 px) comes back at 0.56 of true (focused 0.92); count>3
    pixels beyond 400 km from the specular point read 0.91-0.95 of the
    focused brightness, but 1.00 within 400 km.
  - In the focused images, count>3 pixels 100-300 km from the 2026
    specular point read 3-9 % darker than the rest of their ring. The
    cause is not found; it is not the main-lobe-only cell area (with the
    full sinc^2 to 20 nulls, 09:41:25's 0.928 at 150-300 km became 0.932).
    The real 2026 stack is darker there still: 0.71-0.91 at 25-125 km
    (session moon-radar-c3, 2026-alone stack without the mask over the
    2025 stack), against at most 3-6 % from the simulation. Projecting
    09:41:25's simulated image as if its echo arrived 5 us (2 us) earlier
    than it does (an edge-fit delay too small by that much) gives
    0.84-0.93 (0.96-1.01) at 25-100 km and 0.70-0.73 (0.85-0.86) at
    100-150 km; the opposite offset brightens the line slightly
    (results/FOOTPRINT_TEST/delay_check.py). But WSRT's direct signal
    from Dwingeloo (moon-radar-c3, 2026-10-24) puts the 40 us-chip echoes
    about 3.4 us earlier than the edge fit does: the sign that does not
    darken. So either a delay error is not the cause, or one of the two
    measurements is biased; the direct-path difference changes sign on the
    20 us-chip look (-0.72 samples), which points at something that
    depends on the chip, such as a real delay response unlike the triangle
    the edge fit assumes. Unresolved.

Usage (from the repo root; one run directory per call, results appended to
OUT_DIR/results.json; finished looks are skipped):
    .conda/bin/python investigations/footprint_localization.py \\
        --run-dir results/EVE2026_CIRC --data-root data/eve2026_circ \\
        --out-dir results/FOOTPRINT_TEST [--looks 2026_10_11_09_41_25 ...] [--unfocused]
    .conda/bin/python investigations/footprint_localization.py --report results/FOOTPRINT_TEST
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import doppler_equator_alignment as dea
import simulate_geometry as sg
import stack_maps as sm
import combine_sessions as cs

SIM_DIR = "results/GEOMETRY_SIM/seed1"
SURFACE = os.path.join(SIM_DIR, "surface_seed1.npy")
LAW = "results/GEOMETRY_SIM/law_stockert_wsrt_chan1.npz"
STATION, CHAN = "stockert", "chan1"
PIX_EDGES = 2.0 ** np.arange(-2, 8)          # footprint, map pixels
REL_EDGES = 2.0 ** np.arange(-2, 6)          # footprint / the look's median


def group_of(r):
    chips = r["tx_file"].split("-fzc")[1].split("-")[0]
    return f"{sm.session_of(r['rx_file'])} {float(r['tx_overlap_s']):.0f} s {int(chips) // 1000} kchip/s"


def bin_sums(x, edges, look, true, valid):
    """Per bin of x (below the first edge, between edges, above the last):
    n, sum(look x true), sum(true^2), sum(look^2)."""
    k = np.searchsorted(edges, x[valid])
    nb = len(edges) + 1
    a, b = look[valid], true[valid]
    return [np.bincount(k, minlength=nb).tolist(),
            np.bincount(k, weights=a * b, minlength=nb).tolist(),
            np.bincount(k, weights=b * b, minlength=nb).tolist(),
            np.bincount(k, weights=a * a, minlength=nb).tolist()]


def level_sums(x, edges, look, true, valid):
    """Per bin of x, as bin_sums: n, sum(look), sum(true)."""
    k = np.searchsorted(edges, x[valid])
    nb = len(edges) + 1
    return [np.bincount(k, minlength=nb).tolist(),
            np.bincount(k, weights=look[valid], minlength=nb).tolist(),
            np.bincount(k, weights=true[valid], minlength=nb).tolist()]


def measure_look(r, run_dir, data_root, refl, truth_map, law, focused, sim_path):
    rx_name = dea.STATION_RX[STATION]
    g = sg.look_geometry(r, run_dir, data_root, rx_name)
    npix = len(truth_map)
    nside = hp.npix2nside(npix)
    if focused:
        keep = sg.drift_subintervals
        sg.drift_subintervals = lambda g_, rx_: 1
    try:
        m = sg.simulate_look(g, nside, rx_name, refl_fine=refl, law=law)
    finally:
        if focused:
            sg.drift_subintervals = keep
    np.save(sim_path, m)
    area = np.load(sm.resolve_npy(r["area_npy"], run_dir))
    ddlt = abs(g["dlt_shifts"][1] - g["dlt_shifts"][0])
    cells = dea.resolution_cells(g["chip_samples"], g["doppler_res_dlt"] / ddlt)
    pix_km2 = 4 * np.pi * dea.moon_radii()[0] ** 2 / npix
    cosi = sm.incidence_cos(g["rx_time_s"], npix, {}, rx_name=rx_name)
    valid = (m != hp.UNSEEN) & np.isfinite(area) & (area > 0) & (cosi > 0)
    foot = np.where(valid, area * cells / pix_km2, np.nan)
    look = np.where(valid, np.exp(2.0 * (m.astype(np.float64) - np.max(m[valid]))) / area, np.nan)
    true = np.where(valid, truth_map * law(np.clip(cosi, 0.0, 1.0)), np.nan)
    lo, hi = np.cos(np.radians(sm.GAIN_INCIDENCE_DEG[::-1]))
    gain = valid & (cosi >= lo) & (cosi <= hi)
    look /= np.mean(look[gain])
    true /= np.mean(true[gain])
    smoother = cs.Smoother(nside)
    bl = cs.bands(look, valid, "look", smoother)
    bt = cs.bands(true, valid, "look", smoother)
    med = float(np.median(foot[valid]))
    out = dict(group=group_of(r), median_footprint_px=med, n_seen=int(valid.sum()),
               resolution_cells=cells, rows_per_resolution=g["doppler_res_dlt"] / ddlt,
               footprint_max_px=float(np.nanmax(foot)), bands={},
               level={sc: level_sums(x, e, look, true, valid)
                      for sc, x, e in (("pix", foot, PIX_EDGES), ("rel", foot / med, REL_EDGES))})
    for name, a, b in zip(cs.BAND_NAMES, bl, bt):
        ok = valid & np.isfinite(a) & np.isfinite(b)
        out["bands"][name] = dict(pix=bin_sums(foot, PIX_EDGES, a, b, ok),
                                  rel=bin_sums(foot / med, REL_EDGES, a, b, ok))
    return out


def report(out_dir):
    res = json.load(open(os.path.join(out_dir, "results.json")))
    for mode in ("focused", "unfocused"):
        looks = {k: v for k, v in res.items() if v["mode"] == mode}
        if not looks:
            continue
        groups = sorted({v["group"] for v in looks.values()})
        print(f"\n=== {mode} images; {len(looks)} looks ===")
        for gname in groups:
            gl = [v for v in looks.values() if v["group"] == gname]
            print(f"\n{gname}: {len(gl)} looks; median footprint "
                  + ", ".join(f"{v['median_footprint_px']:.2f}" for v in gl)
                  + " px; largest " + ", ".join(f"{v['footprint_max_px']:.0f}" for v in gl)
                  + f" px; rows per 1/T {gl[0]['rows_per_resolution']:.2f}")
            for scale, edges in (("pix", PIX_EDGES), ("rel", REL_EDGES)):
                unit = "px" if scale == "pix" else "x median"
                labels = ([f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]
                          + [f">{edges[-1]:g}"])
                print(f"  footprint ({unit}): " + " ".join(f"{s:>9}" for s in labels))
                n = np.sum([v["bands"][cs.BAND_NAMES[0]][scale][0] for v in gl], axis=0)
                tot = np.sum([v["n_seen"] for v in gl])
                print("  share of seen      " + " ".join(f"{x / tot:9.4f}" for x in n))
                if all("level" in v for v in gl):
                    cnt, sa, sb = np.sum([v["level"][scale] for v in gl], axis=0)
                    print("  level (look/true)  " + " ".join(
                        f"{a / b:9.3f}" if k >= 200 else f"{'':9}" for a, b, k in zip(sa, sb, cnt)))
                for band in cs.BAND_NAMES:
                    s = np.sum([v["bands"][band][scale] for v in gl], axis=0)
                    cnt, ab, bb, aa = s
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ret = ab / bb
                        corr = ab / np.sqrt(aa * bb)
                    cells = [f"{x:5.2f}/{c:4.2f}" if k >= 200 else f"{'':9}"
                             for x, c, k in zip(ret, corr, cnt)]
                    print(f"  {band:>9} ret/corr " + " ".join(f"{c:>9}" for c in cells))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir")
    p.add_argument("--data-root")
    p.add_argument("--out-dir", required=False)
    p.add_argument("--looks", nargs="*", default=None)
    p.add_argument("--unfocused", action="store_true")
    p.add_argument("--report", default=None)
    a = p.parse_args()
    if a.report:
        report(a.report)
        return
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    mode = "unfocused" if a.unfocused else "focused"
    rows = list(csv.DictReader(open(os.path.join(a.run_dir, f"registration_runs_{STATION}_{CHAN}.csv"))))
    rows = [r for r in rows if r["dd_npz"] and r["rim_status"] == "converged"
            and r["edge_status"] == "fitted"]
    if a.looks:
        rows = [r for r in rows if any(f"_{t}_" in r["rx_file"] for t in a.looks)]
    dea.load_lola_dem()
    refl = np.load(SURFACE, mmap_mode="r")
    truth_map = np.load(SURFACE.replace(".npy", "_map.npy")).astype(np.float64)
    law = sg.law_function(LAW)
    for r in rows:
        key = f"{r['rx_file']} {mode}"
        if key in res and "level" in res[key]:
            continue
        t = time.time()
        os.makedirs(os.path.join(a.out_dir, "sims"), exist_ok=True)
        out = measure_look(r, a.run_dir, a.data_root, refl, truth_map, law, mode == "focused",
                           os.path.join(a.out_dir, "sims", f"{r['rx_file']}_{mode}.npy"))
        out["mode"] = mode
        res[key] = out
        with open(path + ".tmp", "w") as f:
            json.dump(res, f)
        os.replace(path + ".tmp", path)
        print(f"{r['rx_file'][:52]} {mode}: median footprint {out['median_footprint_px']:.2f} px, "
              f"largest {out['footprint_max_px']:.0f} px ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
