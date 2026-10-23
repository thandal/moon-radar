"""How much of the mirror ghosts a stack cancels, by feature size.

Each delay-Doppler cell holds the echo of two surface points, one each side
of the look's Doppler equator, and the projection gives the cell to both:
every look's map carries, at each point, a copy (ghost) of the surface at
its mirror point. The mirror is the reflection across the plane through
the Moon's centre that contains the line of sight to the specular point and
is perpendicular to the Doppler gradient: it keeps a point's delay (its
distance along the line of sight) and its Doppler (its distance along the
gradient). As the Doppler equator turns between looks, the ghosts land in
different places and average down in the stack; a single look's ghosts do
not cancel at all.

The surface here is synthetic (a random field with equal contrast per
octave of feature size, the same at every point), so the ghost term can be
told apart from the real one exactly. It holds features down to the finest
the maps resolve (about two pixels: 9 km at nside 400) and is made on a grid
SURFACE_OVERSAMPLE times finer than the maps, so that reading it at a
mirror point, which falls between pixels, does not smooth it: the real and
the ghost terms are both read from that grid the same way. Each look's map
is the surface at P plus the surface at its mirror point, on the pixels that look contributes
to the stack (the production selection and masks: stack_maps.select_looks
and look_brightness, through validation_common.production_looks); the stack
is the plain mean over looks where at least --min-count cover a pixel, as
stack_maps stacks. Session registration shifts (fractions of a degree) are
left out: they move whole sessions, not the ghosts within them.

Ghost leakage per feature-size band, measured on 300 km tiles of the disk
seen from longitude 0 (2 km grid): the ghost term's detail power, less the
part that copies the real surface (close to the Doppler equator a mirror
point is next to its source, which blurs rather than ghosts), over the real
surface's detail power. A single look is about 1; the stack's value is the
fraction of one look's ghost power it keeps. Reported for the stack and for
up to 10 single looks spread through the set (median and range).

Also reported: how far each look's Doppler equator is turned from the
first look's, the geometry that decides the cancellation.

Usage (from the repo root; looks made with the 2026-10-21 pipeline):
  .conda/bin/python validation/scripts/validate_mirror_ghosts.py \\
      --run-dir results/EVE2026_CIRC --stations stockert wsrt --chan chan1
Output: validation/results/mirror_ghosts_<run folders>_<stations>_<chan>[_txmin<N>s].json
(the run folders' names joined by +, e.g. STOCKERT_2025+EVE2026_CIRC).
"""

from __future__ import annotations

import argparse
import os

import healpy as hp
import numpy as np

import validation_common as vc

import doppler_equator as de
import doppler_equator_alignment as dea
import freq_offset_hunt as foh

R_KM = 1737.4
PX_KM = 2.0                 # disk grid spacing
EXT_KM = 1300.0             # half-width of the disk grid
TILE_KM, STEP_KM = 300.0, 150.0
BANDS_KM = [(150, 75), (75, 40), (40, 20), (20, 12), (12, 10)]
SURFACE_OVERSAMPLE = 4      # synthetic surface grid: this many times the maps' nside
# Points used to fit the Doppler gradient: the near side within this much
# two-way light time of the nearest point (the delay window, ~11.6 ms).
FIT_DELAY_S = 0.0115


def mirror_normal(rx_time, capture_s, rx_name):
    """Unit normal of a look's mirror plane, at mid-look, in MOON_ME."""
    tm = rx_time + capture_s / 2
    s = de.specular_point_bck(tm, dea.TX_STATION, rx_name)
    s = s / np.linalg.norm(s)
    v = np.array(hp.pix2vec(64, np.arange(hp.nside2npix(64)))).T
    p = de.moon_surface_points(v)
    lt = []
    for t in (tm - 1.0, tm + 1.0):
        R_rx, R_tx, c = de.apparent_station_positions(t, dea.TX_STATION, rx_name)
        lt.append((np.linalg.norm(p - R_rx, axis=1) + np.linalg.norm(p - R_tx, axis=1)) / c)
    dlt = (lt[1] - lt[0]) / 2.0
    near = (lt[0] - lt[0].min()) < FIT_DELAY_S
    A = np.column_stack([np.ones(near.sum()), v[near]])
    g = np.linalg.lstsq(A, dlt[near], rcond=None)[0][1:]
    w = g - (g @ s) * s                      # Doppler gradient across the line of sight
    w /= np.linalg.norm(w)
    return np.cross(s, w)


class DiskTiles:
    """The near side on a 2 km orthographic grid, cut into overlapping
    tiles, with per-band detail power and cross power."""

    def __init__(self):
        xs = np.arange(-EXT_KM, EXT_KM + PX_KM / 2, PX_KM)
        X, Y = np.meshgrid(xs, xs)                       # east, north (km)
        self.inside = X ** 2 + Y ** 2 < (0.98 * R_KM) ** 2
        Z = np.sqrt(np.clip(R_KM ** 2 - X ** 2 - Y ** 2, 0, None))
        self.theta, self.phi = hp.vec2ang(np.column_stack([Z.ravel(), X.ravel(), Y.ravel()]))
        self.shape = X.shape
        self.nt = int(TILE_KM / PX_KM)
        step = int(STEP_KM / PX_KM)
        self.origins = [(i, j) for i in range(0, len(xs) - self.nt + 1, step)
                        for j in range(0, len(xs) - self.nt + 1, step)]
        self.taper = np.outer(np.hanning(self.nt), np.hanning(self.nt))
        f = np.fft.fftfreq(self.nt, d=PX_KM)
        kr = np.hypot(*np.meshgrid(f, f))
        with np.errstate(divide="ignore"):
            wl = np.where(kr > 0, 1 / kr, np.inf)
        self.sel = [(wl <= hi) & (wl > lo) for hi, lo in BANDS_KM]

    def grid(self, hmap):
        """A healpix map (NaN where empty) on the disk grid; NaN where any of
        the interpolation's pixels is empty."""
        mask = np.isfinite(hmap)
        v = hp.get_interp_val(np.where(mask, hmap, 0.0), self.theta, self.phi)
        w = hp.get_interp_val(mask.astype(float), self.theta, self.phi)
        out = np.where(w > 0.99, v / np.maximum(w, 1e-9), np.nan).reshape(self.shape)
        return np.where(self.inside, out, np.nan)

    def leakage(self, real, ghost):
        """Per band: (ghost power not copying the real surface) / real power,
        summed over tiles with at least 95 % coverage; and the tile count."""
        a, b = self.grid(ghost), self.grid(real)
        pa = np.zeros(len(BANDS_KM)); pb = np.zeros_like(pa); x = np.zeros_like(pa)
        n = 0
        for i, j in self.origins:
            ta, tb = a[i:i + self.nt, j:j + self.nt], b[i:i + self.nt, j:j + self.nt]
            ok = np.isfinite(ta) & np.isfinite(tb)
            if ok.mean() < 0.95:
                continue
            n += 1
            A = np.fft.fft2(np.where(ok, ta - np.nanmean(ta), 0.0) * self.taper)
            B = np.fft.fft2(np.where(ok, tb - np.nanmean(tb), 0.0) * self.taper)
            for k, s in enumerate(self.sel):
                pa[k] += (np.abs(A[s]) ** 2).sum()
                pb[k] += (np.abs(B[s]) ** 2).sum()
                x[k] += np.real(A[s] * np.conj(B[s])).sum()
        if n == 0:
            return np.full(len(BANDS_KM), np.nan), 0
        return (pa - x ** 2 / pb) / pb, n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", nargs="+", required=True,
                   help="run directories made by registration_stability.py")
    p.add_argument("--stations", nargs="+", required=True, choices=sorted(dea.STATION_RX))
    p.add_argument("--chan", required=True)
    p.add_argument("--min-count", type=int, default=3,
                   help="looks needed at a pixel, as stack_maps (default 3)")
    p.add_argument("--min-snr", type=float, default=foh.MIN_TONE_SNR)
    p.add_argument("--min-tx-s", type=float, default=None)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    rows, looks, left_out = vc.production_looks(args.run_dir, args.stations, [args.chan],
                                                args.min_snr, args.min_tx_s)
    if len(rows) < args.min_count:
        raise SystemExit(f"{len(rows)} looks, fewer than --min-count {args.min_count}")
    npix = len(looks[0])
    nside = hp.npix2nside(npix)
    v_pix = np.array(hp.pix2vec(nside, np.arange(npix))).T

    lmax = 3 * nside - 1                          # the maps' resolution
    finest_km = 2 * np.pi * R_KM / lmax
    if BANDS_KM[-1][1] < finest_km:
        raise SystemExit(f"nside {nside} resolves features down to {finest_km:.1f} km, "
                         f"coarser than the finest band ({BANDS_KM[-1][1]:g} km)")
    ell = np.arange(lmax + 1)
    cl = 1.0 / (ell + 20.0) ** 2                  # equal contrast per octave
    cl[:2] = 0
    np.random.seed(args.seed)
    surface = hp.synfast(cl, nside * SURFACE_OVERSAMPLE, lmax=lmax).astype(np.float32)
    surface /= surface.std()

    def read(vectors):
        return hp.get_interp_val(surface, *hp.vec2ang(vectors))

    order = np.argsort([r["rx_start_utc"] for r in rows], kind="stable")
    rows = [rows[i] for i in order]
    looks = [looks[i] for i in order]
    names = [f"{r['_station']} {r['rx_start_utc'][:19]}" for r in rows]

    real_all = read(v_pix).astype(np.float32)
    ghost_sum = np.zeros(npix)
    count = np.zeros(npix, dtype=np.int32)
    normals = []
    tiles = DiskTiles()
    single_at = set(np.unique(np.linspace(0, len(rows) - 1, min(10, len(rows))).astype(int)))
    single = []
    for k, (r, I) in enumerate(zip(rows, looks)):
        u = mirror_normal(dea.csp.str2et(r["rx_start_utc"]), float(r["capture_s"]), r["_rx"])
        normals.append(u)
        on = np.nonzero(np.isfinite(I))[0]
        ghost = read(v_pix[on] - 2 * (v_pix[on] @ u)[:, None] * u)
        ghost_sum[on] += ghost
        count[on] += 1
        if k in single_at:
            g1 = np.full(npix, np.nan); g1[on] = ghost
            r1 = np.full(npix, np.nan); r1[on] = real_all[on]
            single.append(tiles.leakage(r1, g1)[0])
        print(f"  {names[k]}: {len(on)} pixels", flush=True)
    normals = np.array(normals)

    ok = count >= args.min_count
    ghost = np.full(npix, np.nan); ghost[ok] = ghost_sum[ok] / count[ok]
    real = np.full(npix, np.nan); real[ok] = real_all[ok]
    stack, n_tiles = tiles.leakage(real, ghost)
    single = np.array(single)

    turn = np.degrees(np.arccos(np.clip(np.abs(normals @ normals[0]), 0, 1)))
    print(f"\n{len(rows)} looks; Doppler-equator turn from {names[0]} (deg):")
    for n, t in zip(names, turn):
        print(f"  {n}  {t:6.2f}")
    bands = [f"{a:g}-{b:g} km" for a, b in BANDS_KM]
    print(f"\nGhost leakage (ghost detail power / real detail power; one look ~1), "
          f"{n_tiles} tiles of {TILE_KM:g} km:")
    print(f"  {'':24s}" + "".join(f"{b:>11s}" for b in bands))
    print(f"  {'stack':24s}" + "".join(f"{x:11.3f}" for x in stack))
    print(f"  {'single looks, median':24s}" + "".join(f"{x:11.3f}" for x in np.median(single, axis=0)))
    print(f"  {'single looks, min':24s}" + "".join(f"{x:11.3f}" for x in single.min(axis=0)))
    print(f"  {'single looks, max':24s}" + "".join(f"{x:11.3f}" for x in single.max(axis=0)))

    tag = f"_txmin{args.min_tx_s:g}s" if args.min_tx_s is not None else ""
    runs = "+".join(os.path.basename(os.path.normpath(d)) for d in args.run_dir)
    out = vc.write_json(f"mirror_ghosts_{runs}_{'_'.join(args.stations)}_{args.chan}{tag}.json", {
        "run_dirs": args.run_dir, "stations": args.stations, "chan": args.chan,
        "min_count": args.min_count, "min_tx_s": args.min_tx_s, "seed": args.seed,
        "looks": names, "turn_from_first_deg": turn.tolist(),
        "bands_km": BANDS_KM, "tiles": n_tiles,
        "stack_leakage": stack.tolist(),
        "single_look_leakage": single.tolist(),
        **left_out,
    })
    print(f"\nWrote {vc.report_path(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
