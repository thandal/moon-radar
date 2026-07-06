"""Absolute selenolocation tie: cross-correlate the stacked radar map
against an external LOLA-derived reference (REPORT §5 open item).

The registration chain is internally consistent (closed-loop residuals
0.009-0.025 deg) but nothing ties the stack to the true lunar frame: a
common offset of ALL sessions is unobservable by the session solve, and the
DEM feature check is an A/B against the pipeline's own maps. This script
provides the missing absolute check: the reference is LOLA topographic
slope (|grad h|), which shares structure with L-band reflectivity (crater
rims, maria boundaries, rough ejecta) while being derived from an entirely
independent measurement (laser altimetry in the LOLA/LRO frame).

Method: grid the stacked log map and the healpix-sampled LOLA slope proxy
onto the production lon/lat grid, band-pass both at the production scales
(0.3-2.5 deg), and measure xcorr_offset within a +-search window, sweeping
an extra low-pass ladder for robustness (the correlation is expected to be
modest — reflectivity is not slope — so the offset stability across scales
matters more than any single peak). Reports offset (deg and km), peak
correlation, and significance (peak vs strongest sidelobe; >~1.5 = lock,
per registration_analysis).

Interpretation: a stable offset <~ 0.075 deg (1 grid px) with signif > 1.5
certifies absolute placement at the pixel level; a stable larger offset
reveals a real absolute misregistration (chain-level, e.g. timing/frame);
signif ~ 1 means this reference cannot certify — try the scatnorm dual map
or a shaded-relief reference instead.

Needs the stacked map products and the LOLA DEM — data-machine only.

Run from the repo root:
  .conda/bin/python validation/scripts/validate_absolute_registration.py \
      [--map results/LOLA_DEM_REGISTRATION/stacked_map_dual_scatnorm.npy]
Outputs: validation/results/absolute_registration.json.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from validation_common import write_json, report_path

import healpy as hp

import doppler_equator as de
import registration_analysis as ra

MOON_KM_PER_DEG = 2 * np.pi * 1737.4 / 360.0


def lola_slope_proxy(nside, ref="slope"):
    """LOLA-derived reference sampled at healpix pixel centers.

    ref="slope": |grad h| (roughness proxy); "shade_e"/"shade_n": signed
    directional derivative (shaded relief), which preserves the rim/wall
    light-dark asymmetry that |grad| folds together."""
    de.load_lola_dem()
    vecs = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
    h = de.get_lola_elevation(vecs)
    # Gradient via neighbor differences on the healpix sphere.
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    dtheta = np.radians(0.08)  # ~0.08 deg (~2.4 km) baseline
    # Clamp at the poles (pixels there are far outside the +-55 deg analysis
    # grid; the clamped slope value is never used).
    h_n = de.get_lola_elevation(np.array(hp.ang2vec(
        np.clip(theta - dtheta, 0.0, np.pi), phi)))
    h_e = de.get_lola_elevation(np.array(hp.ang2vec(
        theta, phi + dtheta / np.maximum(np.sin(theta), 1e-6))))
    base_km = dtheta * 1737.4
    if ref == "shade_e":
        return (h_e - h) / base_km
    if ref == "shade_n":
        return (h_n - h) / base_km
    return np.hypot((h_n - h) / base_km, (h_e - h) / base_km)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--map",
                   default="results/LOLA_DEM_REGISTRATION/stacked_map_dual_scatnorm.npy")
    p.add_argument("--search-deg", type=float, default=3.0)
    p.add_argument("--ref", default="slope", choices=("slope", "shade_e", "shade_n"),
                   help="LOLA reference: |grad h| or signed E/N shaded relief")
    args = p.parse_args()

    if not os.path.exists(args.map):
        sys.exit(f"[!] {args.map} not found — needs the stacked map products "
                 "(data machine).")
    from spice_setup import furnsh_kernels
    furnsh_kernels()

    m = np.load(args.map)
    nside = hp.npix2nside(len(m))
    valid = np.isfinite(m) & (m != hp.UNSEEN)
    print(f"[*] stack: {args.map} (nside {nside}, {valid.sum()} valid px)")

    ref = lola_slope_proxy(nside, args.ref).astype(np.float32)
    # |slope| is log-normal-ish -> log-compress; signed shade stays linear.
    ref_v = (np.log10(np.maximum(ref, 1e-6)) if args.ref == "slope" else ref)
    ref_m = np.where(valid, ref_v, hp.UNSEEN).astype(np.float32)
    stack_m = np.where(valid, m, hp.UNSEEN).astype(np.float32)

    step = 0.075
    lon_axis = np.arange(-55, 55 + step / 2, step)
    lat_axis = np.arange(-55, 55 + step / 2, step)
    search_px, exclude_px = int(args.search_deg / step), int(0.5 / step)

    rows = []
    for lo_deg in (0.3, 0.45, 0.7, 1.0):
        lo_px, hi_px = lo_deg / step, 2.5 / step
        a = ra.bandpass(ra.grid_map(ref_m, lon_axis, lat_axis), lo_px, hi_px)
        b = ra.bandpass(ra.grid_map(stack_m, lon_axis, lat_axis), lo_px, hi_px)
        dy, dx, peak, signif = ra.xcorr_offset(a, b, search_px, exclude_px)
        rows.append({"lo_deg": lo_deg,
                     "dlon_deg": dx * step, "dlat_deg": dy * step,
                     "dlon_km": dx * step * MOON_KM_PER_DEG,
                     "dlat_km": dy * step * MOON_KM_PER_DEG,
                     "peak": float(peak), "signif": float(signif)})
        print(f"  band {lo_deg:.2f}-2.5 deg: offset "
              f"({dx * step:+.3f}, {dy * step:+.3f}) deg = "
              f"({dx * step * MOON_KM_PER_DEG:+.1f}, "
              f"{dy * step * MOON_KM_PER_DEG:+.1f}) km, "
              f"peak {peak:+.3f}, signif {signif:.2f}")

    good = [r for r in rows if r["signif"] >= 1.5]
    if good:
        med_lon = float(np.median([r["dlon_deg"] for r in good]))
        med_lat = float(np.median([r["dlat_deg"] for r in good]))
        print(f"[*] locked scales: {len(good)}/{len(rows)}; median absolute "
              f"offset ({med_lon:+.3f}, {med_lat:+.3f}) deg "
              f"({med_lon * MOON_KM_PER_DEG:+.1f}, "
              f"{med_lat * MOON_KM_PER_DEG:+.1f}) km vs LOLA frame")
    else:
        print("[*] no scale reaches signif >= 1.5 — this reference cannot "
              "certify absolute placement; see docstring for alternatives.")

    name = ("absolute_registration.json" if args.ref == "slope"
            else f"absolute_registration_{args.ref}.json")
    out = write_json(name,
                     {"map": args.map, "ref": args.ref, "scales": rows,
                      "locked_scales": len(good)})
    print(f"wrote {report_path(out)}")


if __name__ == "__main__":
    main()
