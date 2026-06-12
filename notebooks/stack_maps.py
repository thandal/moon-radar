"""
Deep stacks of registered lunar maps: per-channel or dual-channel, with and
without empirical scattering-law normalization, session offsets applied.

Pipeline:
  1. Load runs CSVs for the requested channels; gate looks on correction
     quality (pair_snr for cross-pol looks corrected via their co-pol twin,
     own tone_snr otherwise) and non-railed delay shifts.
  2. Pass 1 (streaming): per-session accumulators for the offset solve, and
     per-channel (cos incidence angle, intensity) samples for the empirical
     scattering law. The law is the median gain-normalized intensity in
     cos(theta) bins -- fitted from the data, no model assumed -- and
     dividing it out flattens the specular falloff so the stack approaches
     a uniform-illumination reflectivity map.
  3. Solve session shifts (LSQ, ref 2025-09-16) and verify closed-loop.
  4. Pass 2 (streaming): accumulate raw and scattering-normalized stacks
     simultaneously, with session shifts applied.

Usage (from notebooks/):
    ../.conda/bin/python stack_maps.py --chans chan1            # co-pol
    ../.conda/bin/python stack_maps.py --chans chan0            # cross-pol
    ../.conda/bin/python stack_maps.py --chans chan1 chan0      # dual
"""

import argparse
import csv
import os
import re

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

import doppler_equator_alignment as dea
import registration_analysis as ra

SESSION_OF = {"2025_06_21": "2025-06-21", "2025_09_10": "2025-09-10_11",
              "2025_09_11": "2025-09-10_11", "2025_09_16": "2025-09-16"}
REF_SESSION = "2025-09-16"


def session_of(rx_file):
    return SESSION_OF[re.search(r"(\d{4}_\d{2}_\d{2})", rx_file).group(1)]


def look_linear_intensity(map_npy, mult_npy, max_mult_factor=3.0):
    """Per-look masked, gain-normalized linear intensity (NaN where invalid)."""
    m = np.load(map_npy)
    mult = np.load(mult_npy)
    mask = (m != hp.UNSEEN) & np.isfinite(m)
    med_mult = np.median(mult[mult > 0])
    mask &= mult <= max_mult_factor * med_mult
    med = np.median(m[mask])
    out = np.full(m.shape, np.nan, dtype=np.float64)
    out[mask] = np.exp(2.0 * (m[mask] - med))
    return out


_PIX_CACHE = {}

def pixel_vectors(npix):
    if npix not in _PIX_CACHE:
        nside = hp.npix2nside(npix)
        v = np.array(hp.pix2vec(nside, np.arange(npix))).T
        _PIX_CACHE[npix] = dea.moon_surface_points(v)
    return _PIX_CACHE[npix]


def incidence_cos(rx_time, npix, station_cache):
    """cos(incidence angle) per healpix pixel for a look (bisector geometry)."""
    if rx_time not in station_cache:
        station_cache[rx_time] = dea.apparent_station_positions(rx_time)
    R_rx, R_tx, _ = station_cache[rx_time]
    p = pixel_vectors(npix)
    u = R_rx - p
    v = R_tx - p
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    b = u + v
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    n = p / np.linalg.norm(p, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", n, b)


def shift_intensity(I, dlon_deg, dlat_deg):
    """Shift a NaN-masked healpix intensity map by (+dlon, +dlat) degrees."""
    if abs(dlon_deg) < 1e-6 and abs(dlat_deg) < 1e-6:
        return I
    nside = hp.npix2nside(len(I))
    theta, phi = hp.pix2ang(nside, np.arange(len(I)))
    th_src = np.clip(theta + np.radians(dlat_deg), 1e-6, np.pi - 1e-6)
    ph_src = phi - np.radians(dlon_deg)
    mask = np.isfinite(I)
    v = hp.get_interp_val(np.where(mask, I, 0.0), th_src, ph_src)
    w = hp.get_interp_val(mask.astype(float), th_src, ph_src)
    return np.where(w > 0.99, v / np.maximum(w, 1e-9), np.nan)


class Accumulator:
    def __init__(self, npix):
        self.sum = np.zeros(npix)
        self.count = np.zeros(npix, dtype=np.int32)

    def add(self, I):
        ok = np.isfinite(I)
        self.sum[ok] += I[ok]
        self.count[ok] += 1

    def stacked_log(self, min_count=3):
        out = np.full(len(self.sum), hp.UNSEEN, dtype=np.float32)
        ok = self.count >= min_count
        out[ok] = 0.5 * np.log(self.sum[ok] / self.count[ok])
        return out


class ScatterLaw:
    """Empirical median intensity vs cos(incidence), per channel."""

    def __init__(self, nbins=40):
        self.samples = []
        self.nbins = nbins
        self.centers = None
        self.law = None

    def collect(self, I, cosi, decimate=64):
        ok = np.isfinite(I[::decimate]) & (cosi[::decimate] > 0.02)
        self.samples.append(np.column_stack([cosi[::decimate][ok], I[::decimate][ok]]))

    def fit(self):
        s = np.vstack(self.samples)
        edges = np.linspace(0.02, 1.0, self.nbins + 1)
        self.centers = (edges[:-1] + edges[1:]) / 2
        idx = np.clip(np.digitize(s[:, 0], edges) - 1, 0, self.nbins - 1)
        med = np.array([np.median(s[idx == b, 1]) if (idx == b).any() else np.nan
                        for b in range(self.nbins)])
        ok = np.isfinite(med)
        self.law = np.interp(self.centers, self.centers[ok], med[ok])
        self.samples = []  # free memory

    def apply(self, I, cosi):
        law = np.interp(np.clip(cosi, 0.02, 1.0), self.centers, self.law)
        return I / np.maximum(law, 1e-12)


def band_of(stacked, lon_axis, lat_axis, lo_px, hi_px):
    return ra.bandpass(ra.grid_map(stacked, lon_axis, lat_axis), lo_px, hi_px)


def measure_pairwise(bands, names, search_px, exclude_px, step):
    meas = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dy, dx, peak, sig = ra.xcorr_offset(bands[names[i]], bands[names[j]],
                                                search_px, exclude_px)
            meas[(names[i], names[j])] = (dx * step, dy * step, peak, sig)
    return meas


def solve_offsets(meas, names, ref):
    free = [n for n in names if n != ref]
    A, b_lon, b_lat = [], [], []
    for (ni, nj), (dlon, dlat, _, _) in meas.items():
        row = np.zeros(len(free))
        if nj in free:
            row[free.index(nj)] += 1
        if ni in free:
            row[free.index(ni)] -= 1
        A.append(row)
        b_lon.append(dlon)
        b_lat.append(dlat)
    A = np.array(A)
    o_lon = np.linalg.lstsq(A, np.array(b_lon), rcond=None)[0]
    o_lat = np.linalg.lstsq(A, np.array(b_lat), rcond=None)[0]
    out = {ref: (0.0, 0.0)}
    for k, n in enumerate(free):
        out[n] = (o_lon[k], o_lat[k])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=os.path.join(os.path.dirname(__file__),
                                                          "results/REGISTRATION"))
    parser.add_argument("--chans", nargs="+", default=["chan1"])
    parser.add_argument("--min-snr", type=float, default=15.0)
    parser.add_argument("--min-count", type=int, default=3)
    args = parser.parse_args()
    label = "dual" if len(args.chans) > 1 else args.chans[0]

    rows = []
    for chan in args.chans:
        path = os.path.join(args.run_dir, f"registration_runs_{chan}.csv")
        for r in csv.DictReader(open(path)):
            r["_chan"] = chan
            rows.append(r)
    rows.sort(key=lambda r: r["rx_start_utc"])

    # --- Gate ---
    good, excluded = [], []
    for r in rows:
        gate_snr = float(r.get("pair_snr") or r["tone_snr"])
        shift = float(r.get("applied_shift_samples") or r["shift_refined"])
        if gate_snr < args.min_snr:
            excluded.append((r["rx_file"], f"snr {gate_snr:.0f}"))
        elif abs(shift) >= 39.5:
            # measurement rail is +/-40 samples (registration_stability.py)
            excluded.append((r["rx_file"], f"railed shift {shift:+.1f}"))
        else:
            good.append(r)
    print(f"[{label}] {len(good)} looks pass gating, {len(excluded)} excluded")
    for name, why in excluded:
        print(f"  - {name}: {why}")

    step = 0.075
    lon_axis = np.arange(-55, 55 + step / 2, step)
    lat_axis = np.arange(-55, 55 + step / 2, step)
    lo_px, hi_px = 0.3 / step, 2.5 / step
    search_px, exclude_px = int(1.5 / step), int(0.5 / step)

    npix = len(np.load(good[0]["map_npy"]))
    session_names = sorted({session_of(r["rx_file"]) for r in good})
    station_cache = {}
    for r in good:
        r["_et"] = dea.csp.str2et(r["rx_start_utc"])

    # --- Pass 1: unshifted session stacks + scattering-law samples ---
    acc1 = {s: Accumulator(npix) for s in session_names}
    laws = {c: ScatterLaw() for c in args.chans}
    n_looks = {s: 0 for s in session_names}
    for r in good:
        I = look_linear_intensity(r["map_npy"], r["mult_npy"])
        acc1[session_of(r["rx_file"])].add(I)
        n_looks[session_of(r["rx_file"])] += 1
        laws[r["_chan"]].collect(I, incidence_cos(r["_et"], npix, station_cache))
    print({s: n_looks[s] for s in session_names})
    for c in args.chans:
        laws[c].fit()

    # Scattering-law diagnostic plot
    pl.figure(figsize=(7, 5))
    for c in args.chans:
        pl.plot(np.degrees(np.arccos(laws[c].centers)),
                10 * np.log10(laws[c].law), label=c)
    pl.xlabel("incidence angle (deg)")
    pl.ylabel("relative backscatter (dB)")
    pl.title("Empirical scattering law (1299.5 MHz)")
    pl.grid(alpha=0.3)
    pl.legend()
    law_png = os.path.join(args.run_dir, f"scattering_law_{label}.png")
    pl.savefig(law_png, dpi=130)
    print(f"Saved {law_png}")

    # --- Session offsets (band-passed, insensitive to the smooth law) ---
    bands = {s: band_of(acc1[s].stacked_log(args.min_count), lon_axis, lat_axis,
                        lo_px, hi_px) for s in session_names}
    meas0 = measure_pairwise(bands, session_names, search_px, exclude_px, step)
    offsets = solve_offsets(meas0, session_names, REF_SESSION)
    print("Solved session shifts (deg, ref = " + REF_SESSION + "):")
    for s, o in offsets.items():
        print(f"  {s}: dlon {o[0]:+.3f} dlat {o[1]:+.3f}")

    def residual_norm(sign):
        shifted = {}
        for s in session_names:
            stk = acc1[s].stacked_log(args.min_count)
            v = np.where(stk == hp.UNSEEN, np.nan, stk.astype(np.float64))
            v = shift_intensity(v, sign * offsets[s][0], sign * offsets[s][1])
            shifted[s] = band_of(np.where(np.isfinite(v), v, hp.UNSEEN).astype(np.float32),
                                 lon_axis, lat_axis, lo_px, hi_px)
        m = measure_pairwise(shifted, session_names, search_px, exclude_px, step)
        return np.sqrt(sum(v[0] ** 2 + v[1] ** 2 for v in m.values()))

    base_norm = np.sqrt(sum(v[0] ** 2 + v[1] ** 2 for v in meas0.values()))
    res_plus, res_minus = residual_norm(+1.0), residual_norm(-1.0)
    sign = +1.0 if res_plus <= res_minus else -1.0
    res = min(res_plus, res_minus)
    print(f"Closed-loop: residual norm {base_norm:.3f} -> {res:.3f} deg (sign {sign:+.0f})")
    if res > base_norm:
        print("WARNING: offsets do not close; not applying session shifts.")
        sign = 0.0

    # --- Pass 2: raw + scattering-normalized stacks, shifts applied ---
    grand_raw = Accumulator(npix)
    grand_norm = Accumulator(npix)
    for r in good:
        sess = session_of(r["rx_file"])
        I = look_linear_intensity(r["map_npy"], r["mult_npy"])
        In = laws[r["_chan"]].apply(I, incidence_cos(r["_et"], npix, station_cache))
        sh = (sign * offsets[sess][0], sign * offsets[sess][1])
        grand_raw.add(shift_intensity(I, *sh))
        grand_norm.add(shift_intensity(In, *sh))

    n_total = sum(n_looks.values())
    for suffix, accum in [("", grand_raw), ("_scatnorm", grand_norm)]:
        g = accum.stacked_log(args.min_count)
        band = band_of(g, lon_axis, lat_axis, lo_px, hi_px)
        print(f"[{label}{suffix}] noise floor: {band[band != 0].std():.4f} "
              f"({n_total} looks)")
        np.save(os.path.join(args.run_dir, f"stacked_map_{label}{suffix}.npy"), g)
        valid = g != hp.UNSEEN
        vmin, vmax = np.percentile(g[valid], [5, 99.5])
        dea.save_lunar_image(
            g, os.path.join(args.run_dir, f"stacked_map_{label}{suffix}.png"),
            f"{label}{suffix} stack ({n_total} looks)", vmin=vmin, vmax=vmax)
    np.save(os.path.join(args.run_dir, f"stacked_counts_{label}.npy"), grand_raw.count)
    print(f"valid fraction {(grand_raw.count >= args.min_count).mean():.3f}, "
          f"median looks/pixel {np.median(grand_raw.count[grand_raw.count > 0]):.0f}")


if __name__ == "__main__":
    main()
