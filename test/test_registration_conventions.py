"""Enforce the registration sign conventions (REPORT §5).

Pins the convention each primitive implements so a future sign-flip in any one
of them fails HERE instead of silently corrupting the stack. The production
closed-loop ± check in stack_maps.main cannot catch an even-parity double flip
(it would just pick sign=-1 and still close the loop) -- these unit-level
checks can, because each function is validated in isolation against synthetic
ground truth.

Canonical convention:
  +lon = east, +lat = north.
  grid_map(m, lon, lat)          : feature at (lon,lat) -> grid (lon,lat).
  shift_intensity(I, dlon, dlat) : moves content +dlon east, +dlat north.
  xcorr_offset(a, b) -> (dy,dx)  : pixel correction registering b onto a;
                                   b = np.roll(a, +k) -> returns -k.
  solve_offsets + apply compose so the closed-loop global sign is +1.

Pure synthetic data; no SDR/GPU. Run: .conda/bin/python test/test_registration_conventions.py
"""
import os
import sys

import numpy as np
import healpy as hp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import registration_analysis as ra  # noqa: E402
import stack_maps as sm  # noqa: E402

NSIDE = 256
STEP = 0.075
LON = np.arange(-20, 20 + STEP / 2, STEP)
LAT = np.arange(-20, 20 + STEP / 2, STEP)
LO_PX, HI_PX = 0.3 / STEP, 2.5 / STEP
SEARCH_PX, EXCLUDE_PX = int(1.5 / STEP), int(0.5 / STEP)

_VECS = np.array(hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE)))).T


def _gauss(lon0, lat0, sigma=1.3):
    """Unit-peak gaussian blob centered at (lon0, lat0) on the healpix sphere."""
    v0 = hp.ang2vec(np.radians(90.0 - lat0), np.radians(lon0))
    ang = np.degrees(np.arccos(np.clip(_VECS @ v0, -1, 1)))
    return np.exp(-0.5 * (ang / sigma) ** 2)


def _map(positions, sigma=1.3):
    m = np.full(hp.nside2npix(NSIDE), 1e-3)
    for lon0, lat0 in positions:
        m = np.maximum(m, _gauss(lon0, lat0, sigma))
    return m


def _centroid_lonlat(m):
    """Sub-pixel (lon,lat) of a blob: intensity-weighted mean of its bright half."""
    m = np.where(np.isfinite(m), m, 0.0)
    idx = np.where(m >= 0.5 * m.max())[0]
    v = (_VECS[idx] * m[idx][:, None]).sum(0)
    v /= np.linalg.norm(v)
    lon, lat = hp.vec2ang(v[None, :], lonlat=True)
    return float((lon[0] + 180) % 360 - 180), float(lat[0])


def _band(m):
    return ra.bandpass(ra.grid_map(m, LON, LAT), LO_PX, HI_PX)


# --- A. xcorr_offset returns the negative of the content roll, both axes ---

def test_xcorr_sign_lon():
    a = np.outer(np.sin(np.linspace(0, 6, 200)), np.cos(np.linspace(0, 9, 200)))
    a = a + 0.3 * np.roll(a, 17, axis=0)
    _, dx, _, _ = ra.xcorr_offset(a, np.roll(a, +5, axis=1), 12, 4)
    assert abs(dx - (-5)) < 0.5, f"lon roll +5 -> dx={dx:+.2f}, expected -5"


def test_xcorr_sign_lat():
    a = np.outer(np.sin(np.linspace(0, 6, 200)), np.cos(np.linspace(0, 9, 200)))
    a = a + 0.3 * np.roll(a, 17, axis=1)
    dy, _, _, _ = ra.xcorr_offset(a, np.roll(a, +4, axis=0), 12, 4)
    assert abs(dy - (-4)) < 0.5, f"lat roll +4 -> dy={dy:+.2f}, expected -4"


# --- B. grid_map is identity in selenographic coordinates ---

def test_grid_map_identity():
    for lon0, lat0 in [(10.0, 5.0), (-8.0, -12.0)]:
        g = ra.grid_map(_gauss(lon0, lat0), LON, LAT)
        iy, ix = np.unravel_index(np.nanargmax(g), g.shape)
        assert np.hypot(LON[ix] - lon0, LAT[iy] - lat0) < 0.3, \
            f"blob ({lon0},{lat0}) -> grid ({LON[ix]:.2f},{LAT[iy]:.2f})"


# --- C. shift_intensity moves content +dlon east, +dlat north ---

def test_shift_intensity_east():
    moved = sm.shift_intensity(_gauss(0.0, 0.0), +3.0, 0.0)
    lon, lat = _centroid_lonlat(moved)
    assert abs(lon - 3.0) < 0.4 and abs(lat - 0.0) < 0.4, \
        f"+3 east -> centroid ({lon:+.2f},{lat:+.2f})"


def test_shift_intensity_north():
    moved = sm.shift_intensity(_gauss(0.0, 0.0), 0.0, +2.0)
    lon, lat = _centroid_lonlat(moved)
    assert abs(lon - 0.0) < 0.4 and abs(lat - 2.0) < 0.4, \
        f"+2 north -> centroid ({lon:+.2f},{lat:+.2f})"


# --- D. end-to-end: solve recovers -displacement, composition sign is +1 ---

def _scene():
    pos = [(5, 3), (-4, 6), (8, -5), (-7, -8), (2, -2)]
    dA, dB = (+0.45, -0.30), (-0.30, +0.55)   # (east, north) injected displacements
    maps = {
        "ref": _map(pos),
        "A": _map([(lo + dA[0], la + dA[1]) for lo, la in pos]),
        "B": _map([(lo + dB[0], la + dB[1]) for lo, la in pos]),
    }
    return maps, dA, dB


def test_solve_recovers_negative_displacement():
    maps, dA, dB = _scene()
    names = ["ref", "A", "B"]
    bands = {n: _band(maps[n]) for n in names}
    offs = sm.solve_offsets(
        sm.measure_pairwise(bands, names, SEARCH_PX, EXCLUDE_PX, STEP), names, "ref")
    # solved offset registers the session ONTO ref, i.e. the negative of its displacement
    assert np.hypot(offs["A"][0] + dA[0], offs["A"][1] + dA[1]) < 0.12, offs["A"]
    assert np.hypot(offs["B"][0] + dB[0], offs["B"][1] + dB[1]) < 0.12, offs["B"]


def test_closed_loop_sign_is_plus_one():
    maps, _, _ = _scene()
    names = ["ref", "A", "B"]
    bands = {n: _band(maps[n]) for n in names}
    offs = sm.solve_offsets(
        sm.measure_pairwise(bands, names, SEARCH_PX, EXCLUDE_PX, STEP), names, "ref")

    def resid(sign):
        sh = {n: _band(sm.shift_intensity(maps[n], sign * offs[n][0], sign * offs[n][1]))
              for n in names}
        m = sm.measure_pairwise(sh, names, SEARCH_PX, EXCLUDE_PX, STEP)
        return np.sqrt(sum(v[0] ** 2 + v[1] ** 2 for v in m.values()))

    rp, rm = resid(+1.0), resid(-1.0)
    assert rp < 0.5 * rm, f"closed-loop sign not +1: resid(+1)={rp:.3f} resid(-1)={rm:.3f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
