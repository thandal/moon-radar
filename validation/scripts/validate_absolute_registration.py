"""Absolute selenolocation tie: cross-correlate the stacked radar map
against an external LOLA-derived reference (REPORT §5 open item).

The registration chain is internally consistent (small closed-loop
residuals) but nothing ties the stack to the true lunar frame: a common
offset of ALL sessions is unobservable by the session solve, and the DEM
feature check is an A/B against the pipeline's own maps. This script
provides the missing absolute check: the reference is LOLA topographic
slope (|grad h|), which shares structure with L-band reflectivity (crater
rims, maria boundaries, rough ejecta) while being derived from an entirely
independent measurement (laser altimetry in the LOLA/LRO frame).

Method: grid the stack's mean brightness (stack_maps' stacked_linear_*.npy,
NaN where unstacked; values can be <= 0 where the noise floor was removed)
and the healpix-sampled LOLA reference onto the production lon/lat grid,
both masked to the pixels where both have a value. Band-pass them at the
production scales (0.3-2.5 deg): the stack with ra.log_bandpass, as the
stacker band-passes it for registration (stack_maps.band_of: speckle
averaged in linear brightness, then logged, then the trend removed), and
the reference with ra.bandpass (it is not noise-subtracted brightness and
has no holes). log_bandpass also masks the stack where its local average is
not positive, so both band-passed maps are then cut to the cells where both
have content, and cover the same cells (the numbers cut are recorded).
Then measure xcorr_offset within a +-search window, sweeping an extra
low-pass ladder for robustness (the correlation is expected to be modest --
reflectivity is not slope -- so the offset's stability across scales
matters more than any single peak). The stacked log map
(stacked_map_*.npy) is not used: it has no value where the mean is not
positive, and a band-pass erodes its mask around each such hole.

Sign: registration_analysis.xcorr_offset(a, b) returns the correction that
moves b onto a (test/test_registration_conventions.py: if b is a's content
moved +k pixels east, it returns dx = -k). Here a = LOLA and b = the stack,
so the stack's offset from the LOLA frame -- where its features sit
relative to LOLA's, +east/+north -- is MINUS that correction, and that is
what this script reports. A peak on the edge of the search window is no
measurement and has no offset.

Each scale reports its offset (deg and km; east km at the mean
cos(latitude) of the gridded stack), peak correlation, significance (peak
over the strongest correlation outside its main lobe) and whether it is a
lock (validation_common.LOCK_SIGNIFICANCE, a labelling convention).

Interpretation: a stable offset <~ 0.075 deg (1 grid px) on locked scales
places the stack absolutely at the pixel level; a stable larger offset
reveals a real absolute misregistration (chain-level, e.g. timing/frame);
no lock means this reference does not place this stack -- the shaded-relief
references (--ref shade_e / shade_n) are the alternatives offered here.

Needs the stacked map products and the LOLA DEM -- data-machine only.

Run from the repo root (--map names the stack; there is no default):
  .conda/bin/python validation/scripts/validate_absolute_registration.py \
      --map results/STOCKERT_2025/stacked_linear_stockert_dual_scatnorm.npy
(stacked_linear_*.npy is new with the 2026-10-21 stack_maps; this file
exists only once the 2025 looks have been reprocessed and restacked.)
Outputs: validation/results/absolute_registration_<map label>[_<ref>].json,
the map label being the stack's file name without stacked_linear_ and .npy
(it names the stations and channels).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import validation_common as vc

import healpy as hp

import doppler_equator as de
import registration_analysis as ra


def lola_reference(nside, pix, ref="slope"):
    """LOLA-derived reference at healpix pixels pix: ref="slope" gives
    |grad h| (roughness proxy); "shade_e"/"shade_n" the signed east/north
    derivative (shaded relief), which keeps the rim/wall light-dark
    asymmetry that |grad| folds together.

    The derivatives are differences over a 0.08 deg baseline to the north
    and east. Within that baseline of a pole there is no north neighbour
    and east is undefined, so those pixels get NaN; returns (values, their
    count)."""
    de.load_lola_dem()
    theta, phi = hp.pix2ang(nside, pix)
    dtheta = np.radians(0.08)  # ~2.4 km
    ok = (theta > dtheta) & (theta < np.pi - dtheta)
    th, ph = theta[ok], phi[ok]
    h = de.get_lola_elevation(np.array(hp.ang2vec(th, ph)))
    h_n = de.get_lola_elevation(np.array(hp.ang2vec(th - dtheta, ph)))
    h_e = de.get_lola_elevation(np.array(hp.ang2vec(th, ph + dtheta / np.sin(th))))
    base_km = dtheta * 1737.4
    if ref == "shade_e":
        v = (h_e - h) / base_km
    elif ref == "shade_n":
        v = (h_n - h) / base_km
    else:
        v = np.hypot((h_n - h) / base_km, (h_e - h) / base_km)
    out = np.full(len(pix), np.nan)
    out[ok] = v
    return out, int((~ok).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--map", required=True,
                   help="a stack's mean brightness, stack_maps.py's "
                        "stacked_linear_<label>[_scatnorm].npy")
    p.add_argument("--search-deg", type=float, default=3.0)
    p.add_argument("--ref", default="slope", choices=("slope", "shade_e", "shade_n"),
                   help="LOLA reference: |grad h| or signed E/N shaded relief")
    args = p.parse_args()

    name = os.path.basename(args.map)
    if not (name.startswith("stacked_linear_") and name.endswith(".npy")):
        sys.exit(f"[!] {args.map} is not a stack's mean brightness "
                 "(stacked_linear_<label>.npy); the log map stacked_map_*.npy has "
                 "holes where the mean is not positive")
    if not os.path.exists(args.map):
        sys.exit(f"[!] {args.map} not found — needs the stacked map products "
                 "(data machine).")
    from spice_setup import furnsh_kernels
    furnsh_kernels()

    m = np.load(args.map)
    nside = hp.npix2nside(len(m))
    valid = np.isfinite(m)
    pix = np.nonzero(valid)[0]
    print(f"[*] stack: {args.map} (nside {nside}, {len(pix)} stacked px, "
          f"{int(np.sum(m[valid] <= 0))} of them with a mean brightness <= 0)")

    ref, n_pole = lola_reference(nside, pix, args.ref)
    n_zero_slope = 0
    if args.ref == "slope":
        # |slope| is roughly log-normal, so compare its log. A zero slope
        # has no log: such pixels get no reference value (counted).
        n_zero_slope = int(np.sum(ref == 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = np.where(ref > 0, np.log10(ref), np.nan)
    has_ref = np.isfinite(ref)
    print(f"[*] reference {args.ref}: no value at {n_pole} px within the baseline "
          f"of a pole and {n_zero_slope} px of zero slope; both maps are masked "
          "to the pixels where both have a value")
    ref_m = np.full(len(m), np.nan, dtype=np.float32)
    ref_m[pix[has_ref]] = ref[has_ref]
    stack_m = np.full(len(m), np.nan, dtype=np.float32)
    stack_m[pix[has_ref]] = m[pix[has_ref]]

    step = 0.075
    lon_axis = np.arange(-55, 55 + step / 2, step)
    lat_axis = np.arange(-55, 55 + step / 2, step)
    search_px = int(args.search_deg / step)
    g_ref = ra.grid_map(ref_m, lon_axis, lat_axis)
    g_stack = ra.grid_map(stack_m, lon_axis, lat_axis)
    covered = np.isfinite(g_stack)
    if not covered.any():
        sys.exit("[!] the stack has no value on the +-55 deg analysis grid")
    # East km per degree of longitude at the mean cos(latitude) of the
    # grid cells the stack covers (the region the correlation sees).
    lat_g = np.broadcast_to(lat_axis[:, None], covered.shape)
    cos_lat = float(np.mean(np.cos(np.radians(lat_g[covered]))))

    rows = []
    for lo_deg in (0.3, 0.45, 0.7, 1.0):
        lo_px, hi_px = lo_deg / step, 2.5 / step
        a = ra.bandpass(g_ref, lo_px, hi_px)
        b = ra.log_bandpass(g_stack, lo_px, hi_px)
        # Both band-passes return 0 where they have no value; keep the cells
        # where both have one.
        both = (a != 0) & (b != 0)
        if not both.any():
            sys.exit(f"[!] band {lo_deg:.2f}-2.5 deg: the band-passed stack and "
                     "reference share no grid cell")
        n_cut_ref, n_cut_stack = int(np.sum(a != 0) - both.sum()), int(np.sum(b != 0) - both.sum())
        a, b = np.where(both, a, 0.0), np.where(both, b, 0.0)
        x = ra.xcorr_offset(a, b, search_px)
        row = {"lo_deg": lo_deg, "peak": x.peak, "signif": x.significance,
               "at_edge": x.at_edge, "lock": vc.is_lock(x), "n_cells": int(both.sum()),
               "n_cells_cut_from_ref": n_cut_ref, "n_cells_cut_from_stack": n_cut_stack}
        print(f"  band {lo_deg:.2f}-2.5 deg: {both.sum()} grid cells with both maps "
              f"({n_cut_ref} cut from the reference, {n_cut_stack} from the stack)")
        if x.at_edge:
            row.update(offset_dlon_deg=None, offset_dlat_deg=None,
                       offset_east_km=None, offset_north_km=None)
            print(f"  band {lo_deg:.2f}-2.5 deg: peak at the +-{args.search_deg:g} deg "
                  f"search edge -- no measurement (peak {x.peak:+.3f}, "
                  f"signif {x.significance:.2f})")
        else:
            # The stack's offset from LOLA: minus the correction that would
            # move the stack (b) onto LOLA (a).
            dlon, dlat = -x.dx * step, -x.dy * step
            row.update(offset_dlon_deg=dlon, offset_dlat_deg=dlat,
                       offset_east_km=dlon * ra.KM_PER_DEG * cos_lat,
                       offset_north_km=dlat * ra.KM_PER_DEG)
            print(f"  band {lo_deg:.2f}-2.5 deg: stack offset from LOLA "
                  f"({dlon:+.3f} lon, {dlat:+.3f} lat) deg = "
                  f"({row['offset_east_km']:+.1f} east, {row['offset_north_km']:+.1f} "
                  f"north) km, peak {x.peak:+.3f}, signif {x.significance:.2f}"
                  + ("  LOCK" if row["lock"] else ""))
        rows.append(row)

    locked = [r for r in rows if r["lock"]]
    if locked:
        med_lon = float(np.median([r["offset_dlon_deg"] for r in locked]))
        med_lat = float(np.median([r["offset_dlat_deg"] for r in locked]))
        print(f"[*] locked scales: {len(locked)}/{len(rows)}; median stack offset "
              f"from LOLA ({med_lon:+.3f} lon, {med_lat:+.3f} lat) deg = "
              f"({med_lon * ra.KM_PER_DEG * cos_lat:+.1f} east, "
              f"{med_lat * ra.KM_PER_DEG:+.1f} north) km")
    else:
        print(f"[*] no scale locks (significance >= {vc.LOCK_SIGNIFICANCE:g}, not at "
              "the search edge): this reference does not place this stack; the "
              "other references are --ref shade_e / shade_n.")

    label = name[len("stacked_linear_"):-len(".npy")]
    out = vc.write_json(f"absolute_registration_{label}"
                        + ("" if args.ref == "slope" else f"_{args.ref}") + ".json", {
        "map": args.map, "ref": args.ref,
        "stack_band_pass": "ra.log_bandpass of the mean brightness (stack_maps.band_of)",
        "ref_band_pass": "ra.bandpass",
        "offset_meaning": "the stack's offset from the LOLA frame (+east, +north): "
                          "minus xcorr_offset(LOLA, stack)",
        "cos_lat_for_east_km": cos_lat,
        "lock_significance": vc.LOCK_SIGNIFICANCE,
        "n_ref_no_value_pole": n_pole, "n_ref_zero_slope": n_zero_slope,
        "scales": rows, "locked_scales": len(locked)})
    print(f"wrote {vc.report_path(out)}")


if __name__ == "__main__":
    main()
