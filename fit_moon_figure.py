"""
Fit the Moon's smooth overall shape (doppler_equator.MOON_FIGURE_KM) to the
LOLA grid.

The shape is the height above the 1737.4 km reference sphere as a
polynomial of degree <= 2 in the components (x, y, z) of the unit vector in
MOON_ME: basis (1, x, y, z, xy, yz, zx, x^2 - y^2, 3z^2 - 1), i.e. the mean
radius, the offset of the shape's centre and the triaxial figure. It is
fitted by least squares to the LOLA grid sampled every 0.5 deg, each sample
weighted by the area it stands for. Nine numbers: LOLA's measurement
reduced to the Moon's overall shape, with no crater or terrain detail (the
basis varies over thousands of km), so maps made on it take their detail
from the radar alone (REPORT section 2).

Prints the nine numbers, the shape's departure from a sphere at its own
mean radius over the near side and within 60 deg of the disk centre, the
rms of the terrain it leaves out there, and the equivalent ellipsoid (mean
radius, centre offset, principal radii and axes).

Usage (from the repo root):
    .conda/bin/python fit_moon_figure.py [--dem lola_dem/ldem_64.img]
"""

import argparse
import os

import numpy as np

import doppler_equator_alignment  # noqa: F401  (loads the SPICE kernels)
import doppler_equator as de

BASIS_NAMES = ("mean", "x", "y", "z", "xy", "yz", "zx", "x2-y2", "3z2-1")


def figure_basis(u):
    """The degree <= 2 basis at unit vectors u (..., 3) -> (..., 9)."""
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    return np.stack([np.ones_like(x), x, y, z, x * y, y * z, z * x, x * x - y * y,
                     3 * z * z - 1], axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dem", default=None, help="LOLA grid (default: the finest in lola_dem/)")
    p.add_argument("--step-deg", type=float, default=0.5)
    args = p.parse_args()
    dem = de.load_lola_dem(args.dem)
    s = args.step_deg
    lat = np.radians(np.arange(-90 + s / 2, 90, s))
    lon = np.radians(np.arange(-180 + s / 2, 180, s))
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    u = np.stack([np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)], -1)
    h = de.get_lola_elevation(u)
    B = figure_basis(u)
    w = np.sqrt(np.cos(LAT)).ravel()
    coef, *_ = np.linalg.lstsq(B.reshape(-1, 9) * w[:, None], h.ravel() * w, rcond=None)
    fit = B @ coef
    print(f"fitted to {os.path.basename(dem)}, every {s:g} deg, area-weighted")
    print("MOON_FIGURE_KM = (" + ", ".join(f"{c:+.4f}" for c in coef) + ")")
    print("  basis: " + ", ".join(BASIS_NAMES))
    x = u[..., 0]
    for title, sel in (("near side", x > 0),
                       ("within 60 deg of the disk centre", np.degrees(np.arccos(np.clip(x, -1, 1))) <= 60)):
        f = fit[sel] - coef[0]
        rms = np.sqrt(np.average((h - fit)[sel] ** 2, weights=np.cos(LAT)[sel]))
        print(f"{title}: shape minus the sphere at its mean radius {f.min():+.2f} .. {f.max():+.2f} km; "
              f"terrain left out, rms {rms:.2f} km")
    R0 = 1737.4 + coef[0]
    Q = np.array([[coef[7] - coef[8], coef[4] / 2, coef[6] / 2],
                  [coef[4] / 2, -coef[7] - coef[8], coef[5] / 2],
                  [coef[6] / 2, coef[5] / 2, 2 * coef[8]]])
    ev, evec = np.linalg.eigh(Q)
    print(f"equivalent ellipsoid: mean radius {R0:.3f} km, centre offset (x, y, z) "
          f"({coef[1]:+.3f}, {coef[2]:+.3f}, {coef[3]:+.3f}) km, principal radii "
          + ", ".join(f"{R0 + e:.3f}" for e in ev) + " km along "
          + "; ".join("(" + ", ".join(f"{c:+.2f}" for c in evec[:, i]) + ")" for i in range(3)))


if __name__ == "__main__":
    main()
