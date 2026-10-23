"""
Deep stacks of registered lunar maps: per-channel or dual-channel, with and
without empirical scattering-law normalization, session offsets applied.

Pipeline:
  1. select_looks: read the runs CSVs and gate each look (every decision is
     written to stacked_looks_<label>.csv with its reason, together with the
     captures the batch did not map).
  2. Pass 1 (streaming): each look becomes brightness per unit area
     (look_brightness); per-session accumulators for the offset solve, and
     per-channel sums for the empirical scattering law (ScatterLaw: the mean
     brightness in cos(incidence) bins -- fitted from the data, no model
     assumed -- whose division flattens the specular falloff so the stack
     approaches a uniform-illumination reflectivity map).
  3. Solve session shifts (least squares, --ref-session pinned) from
     band-passed session stacks; stop if a peak is at the search edge or the
     shifts close better with the opposite sign; record the measurements and
     how well the shifts close (stacked_registration_<label>.json).
     --no-registration stacks without shifts, named ..._unregistered.
  4. Pass 2 (streaming): accumulate raw and scattering-normalized stacks,
     session shifts applied.
All outputs are written to a temporary folder and moved into --out-dir only
when the run completes, so a run that stops leaves the previous outputs of
the same name untouched.

Radiometry: each look's pixel intensity minus its image's noise floor,
divided by the ground area one delay-Doppler cell covers there (which grows
as 1/distance from the Doppler equator), normalized by the look's mean at
20-50 deg incidence; looks are averaged with equal weights. (Weights from
how well each look agreed with the rest of its session were tried and
dropped on 2026-10-21: mirror ghosts count as disagreement, so they marked
down the looks whose ghosts fall elsewhere -- the ones that cancel the
others' ghosts; REPORT section 6.) Means, not medians, throughout: the stack
averages brightness, and a single look's brightness is skewed by speckle,
so its median sits well below its mean and, with the noise removed, goes
negative where the echo is weak. A mean is used only where it is clearly
above zero (SIGNIFICANCE_SE standard errors): a look's gain, and each
incidence bin of the scattering law.

Outputs (named after stations and channels):
  stacked_map_<label>[_scatnorm].npy/.png  log amplitude, 0.5 ln(mean); a
      pixel whose mean brightness is not positive (no signal left after the
      noise floor is removed) has no log value and is UNSEEN
  stacked_linear_<label>[_scatnorm].npy   the mean brightness itself (NaN
      below --min-count looks), including values <= 0
  stacked_counts_<label>.npy               looks per pixel
  stacked_looks_<label>.csv                every look read, used or not, why
  stacked_registration_<label>.json        pairwise offsets, solved shifts,
                                           closure (with several sessions)
  scattering_law_<label>.png

Usage (from the repo root):
    .conda/bin/python stack_maps.py --run-dir results/STOCKERT_2025 \\
        --stations stockert --chans chan1           # co-pol
    .conda/bin/python stack_maps.py --run-dir results/STOCKERT_2025 \\
        --stations stockert --chans chan1 chan0     # co- + cross-pol

    # Several receivers in one stack (2026-10-11 Stockert + WSRT):
    .conda/bin/python stack_maps.py --run-dir results/EVE2026_CIRC \\
        --stations stockert wsrt --chans chan1 chan0

    # Only looks whose transmission lasted at least 100 s (the 2026-10-11
    # 120 s looks); outputs are named ..._txmin100s_<chan>:
    .conda/bin/python stack_maps.py --run-dir results/EVE2026_CIRC \\
        --stations stockert wsrt --chans chan1 chan0 --min-tx-s 100

    # Seasons processed into separate run directories, stacked together
    # (a station or channel may be missing from some of them):
    .conda/bin/python stack_maps.py --run-dir results/STOCKERT_2025 \\
        results/EVE2026_CIRC --out-dir results/ALL_SEASONS \\
        --stations stockert wsrt --chans chan1 chan0
"""

import argparse
import csv
import json
import os
import re
import shutil
import tempfile

import numpy as np
import healpy as hp
import scipy.stats
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

import doppler_equator_alignment as dea
import freq_offset_hunt as foh
import registration_analysis as ra

# Session labels (used in output file names): 2025-09-10 and 09-11 form one
# registration session.
SESSION_OF = {"2025_06_21": "2025-06-21", "2025_09_10": "2025-09-10_11",
              "2025_09_11": "2025-09-10_11", "2025_09_16": "2025-09-16",
              "2026_10_11": "2026-10-11"}

# Incidence band (deg) for the per-look gain: bright enough that every look
# has signal there, and away from the specular peak.
GAIN_INCIDENCE_DEG = (20.0, 50.0)

# A mean is used only if it is clearly above zero: SIGNIFICANCE_SE standard
# errors (the conventional three; no tuning), at the same one-sided level
# under Student's t when the standard error comes from few pieces
# (significance_factor). Standard errors come from independent pieces: the
# looks themselves for the scattering law, and for one look's gain, blocks
# of healpix pixels at GAIN_BLOCK_NSIDE (~110 km across, many times the
# image's resolution cells, so their means are close to independent).
SIGNIFICANCE_SE = 3.0
GAIN_BLOCK_NSIDE = 16

# Rim-spread gate: a converged rim spread far off its session's median marks
# intra-look frequency wander (bursty ~25 mHz drift within the window smears
# the rims: sign-flipped spread, low rim_n) -- no single delta calibrates
# such a look, so its map carries an uncorrectable Doppler warp/blur. The
# threshold is 3 robust standard deviations of the session's spreads, at
# least SPREAD_FLOOR_MHZ; on the 2025 data it flags the three
# wander-affected 06-21 looks and nothing else
# (investigations/rim_window_recalibration_2026-07-03.md). The wander is
# chain-common, so both polarization channels gate on the co-pol spread.
SPREAD_FLOOR_MHZ = 20.0

# Columns every runs-CSV row must have, non-blank (registration_stability.py
# since 2026-10-21; edge_status since 2026-10-22); older run directories must
# be reprocessed (or, with saved DD images, re-projected: reproject_maps.py). Co-pol rows
# also carry what the gates read for both channels.
REQUIRED_COLUMNS = ("rim_status", "edge_status", "noise_log_amp", "area_npy", "max_shift",
                    "tx_s", "use_dem", "dem_file", "spice_kernels", "supersample",
                    "applied_shift_samples", "applied_df_hz", "map_npy", "mult_npy")
COPOL_COLUMNS = ("rim_status", "edge_status", "tone_at_band_edge", "tone_snr",
                 "timing_status", "applied_shift_samples", "applied_df_hz")


def session_of(rx_file):
    """Registration session of a capture, from the date in its file name."""
    m = re.search(r"(\d{4}_\d{2}_\d{2})", rx_file)
    if m is None or m.group(1) not in SESSION_OF:
        raise ValueError(f"{rx_file}: no known observing date in the name; add the "
                         "session to stack_maps.SESSION_OF")
    return SESSION_OF[m.group(1)]


def significance_factor(n):
    """How many standard errors a mean estimated from n independent pieces
    must clear: the one-sided level of SIGNIFICANCE_SE normal standard
    errors (0.13 %), taken from Student's t with n - 1 degrees of freedom,
    since the standard error itself is estimated from those n pieces: about
    236 standard errors with 2 looks, 6.6 with 5, 3.3 with 30, tending to 3."""
    p = scipy.stats.norm.sf(SIGNIFICANCE_SE)
    return scipy.stats.t.isf(p, np.asarray(n) - 1)


def resolve_npy(path, run_dir):
    """A look's file in its run directory: run_dir/<basename of the recorded
    path>. The run directory is the authority -- a recorded path may point
    into another directory (a copied run) or a layout that has since moved.
    Raises if the file is not there."""
    out = os.path.join(run_dir, os.path.basename(path))
    if not os.path.exists(out):
        raise FileNotFoundError(f"{out} (recorded as {path})")
    return out


def degeneracy_mask(m, mult):
    """Pixels with a value whose DD cells are not degenerate
    (multiplicity <= dea.MAX_MULTIPLICITY)."""
    return (m != hp.UNSEEN) & np.isfinite(m) & (mult <= dea.MAX_MULTIPLICITY)


class LookRejected(Exception):
    """A look that cannot enter the stack, for the stated reason."""


_BLOCK_CACHE = {}


def gain_blocks(npix):
    """Index of each map pixel's GAIN_BLOCK_NSIDE block."""
    if npix not in _BLOCK_CACHE:
        theta, phi = hp.pix2ang(hp.npix2nside(npix), np.arange(npix))
        _BLOCK_CACHE[npix] = hp.ang2pix(GAIN_BLOCK_NSIDE, theta, phi)
    return _BLOCK_CACHE[npix]


def look_brightness(r, cosi):
    """Per-look brightness per unit area with the noise floor removed,
    normalized by its mean at GAIN_INCIDENCE_DEG (NaN where masked).

    r is a runs-CSV row with map_npy, mult_npy, area_npy and noise_log_amp;
    cosi the look's cos(incidence) per pixel. Returns (brightness, gain):
    gain is the look's mean brightness in the band, in units of its own
    median map pixel. Raises LookRejected if the band is empty or the mean
    is not clearly above zero (SIGNIFICANCE_SE standard errors, from the
    scatter of ~110 km block means): no signal above the noise floor where
    every good look has plenty, so normalizing would only amplify noise.
    """
    m = np.load(r["map_npy"])
    mult = np.load(r["mult_npy"])
    area = np.load(r["area_npy"])
    mask = degeneracy_mask(m, mult) & np.isfinite(area) & (area > 0)
    ref = np.median(m[mask])        # numerical scale only
    noise = np.exp(2.0 * (float(r["noise_log_amp"]) - ref))
    bright = np.full(m.shape, np.nan)
    bright[mask] = (np.exp(2.0 * (m[mask] - ref)) - noise) / area[mask]
    lo, hi = np.cos(np.radians(GAIN_INCIDENCE_DEG[::-1]))
    band = np.nonzero(mask & (cosi >= lo) & (cosi <= hi))[0]
    deg = f"{GAIN_INCIDENCE_DEG[0]:g}-{GAIN_INCIDENCE_DEG[1]:g} deg incidence"
    if len(band) == 0:
        raise LookRejected(f"no valid pixels at {deg}")
    gain = float(np.mean(bright[band]))
    blocks = gain_blocks(len(m))[band]
    n = np.bincount(blocks)
    has = n > 0
    block_means = np.bincount(blocks, weights=bright[band])[has] / n[has]
    if len(block_means) < 2:
        raise LookRejected(f"band at {deg} spans fewer than two blocks")
    se = float(np.std(block_means, ddof=1) / np.sqrt(len(block_means)))
    if not gain > significance_factor(len(block_means)) * se:
        raise LookRejected(f"mean brightness at {deg} is {gain:.3g} +- {se:.2g} "
                           f"over {len(block_means)} blocks, not clearly above zero: "
                           "no clear signal above the image's noise floor")
    return bright / gain, gain


_PIX_CACHE = {}

def pixel_vectors(npix):
    if npix not in _PIX_CACHE:
        nside = hp.npix2nside(npix)
        v = np.array(hp.pix2vec(nside, np.arange(npix))).T
        _PIX_CACHE[npix] = dea.moon_surface_points(v)
    return _PIX_CACHE[npix]


def incidence_cos(rx_time, npix, station_cache, *, rx_name):
    """cos(incidence angle) per healpix pixel for a look (bisector geometry)."""
    key = (rx_time, rx_name)
    if key not in station_cache:
        station_cache[key] = dea.apparent_station_positions(
            rx_time, tx_name=dea.TX_STATION, rx_name=rx_name)
    R_rx, R_tx, _ = station_cache[key]
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
    """Shift a NaN-masked healpix intensity map, moving map CONTENT +dlon east
    and +dlat north: a feature at (lon, lat) moves to (lon+dlon, lat+dlat).
    Sign convention REPORT §5; enforced by test/test_registration_conventions.py."""
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

    def mean(self, min_count=3):
        """Mean brightness per pixel; NaN with fewer than min_count looks."""
        out = np.full(len(self.sum), np.nan)
        ok = self.count >= min_count
        out[ok] = self.sum[ok] / self.count[ok]
        return out

    def stacked_log(self, min_count=3):
        """0.5 ln(mean brightness): log amplitude, as the per-look maps.
        UNSEEN with fewer than min_count looks, and where the mean is not
        positive -- no signal above the noise floor, which has no log value."""
        mean = self.mean(min_count)
        out = np.full(len(mean), hp.UNSEEN, dtype=np.float32)
        pos = np.isfinite(mean)
        pos[pos] = mean[pos] > 0
        out[pos] = 0.5 * np.log(mean[pos])
        return out


class ScatterLaw:
    """Empirical mean brightness vs cos(incidence), per channel: sums over
    every valid pixel of every look in NBINS equal bins of cos(incidence)
    from COS_MIN to 1.

    A bin has a value only if its mean is clearly above zero
    (significance_factor standard errors), the standard error taken from the
    scatter of the looks' own means in that bin (looks are independent;
    pixels within a look are not); bins seen by fewer than two looks have
    none. Pixels in a
    bin without a value have no normalized value (NaN)."""
    NBINS = 40
    COS_MIN = 0.02

    def __init__(self):
        self.edges = np.linspace(self.COS_MIN, 1.0, self.NBINS + 1)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.sum = np.zeros(self.NBINS)
        self.count = np.zeros(self.NBINS, dtype=np.int64)
        self.look_means = []     # one array of per-bin means (NaN: no pixels) per look
        self.law = None

    def _bins(self, cosi):
        b = np.digitize(cosi, self.edges) - 1
        return np.where((b >= 0) & (b < self.NBINS), b, -1)

    def collect(self, I, cosi):
        b = self._bins(cosi)
        ok = np.isfinite(I) & (b >= 0)
        s = np.bincount(b[ok], weights=I[ok], minlength=self.NBINS)
        n = np.bincount(b[ok], minlength=self.NBINS)
        self.sum += s
        self.count += n
        look_mean = np.full(self.NBINS, np.nan)
        has = n > 0
        look_mean[has] = s[has] / n[has]
        self.look_means.append(look_mean)

    def fit(self):
        """Fit the law; returns (bins with no looks, bins with one look,
        bins whose mean is not clearly above zero)."""
        mean = np.full(self.NBINS, np.nan)
        se = np.full(self.NBINS, np.nan)
        seen = self.count > 0
        mean[seen] = self.sum[seen] / self.count[seen]
        lm = np.array(self.look_means)          # looks x bins
        n_looks = np.isfinite(lm).sum(axis=0)
        multi = n_looks >= 2
        se[multi] = np.nanstd(lm[:, multi], axis=0, ddof=1) / np.sqrt(n_looks[multi])
        good = seen & multi
        good[good] = mean[good] > significance_factor(n_looks[good]) * se[good]
        self.law = np.where(good, mean, np.nan)
        return (int(np.sum(~seen)), int(np.sum(seen & ~multi)),
                int(np.sum(seen & multi & ~good)))

    def value(self, cosi):
        """The law at each cos(incidence): linear between neighbouring bin
        centres that both have a value; otherwise the pixel's own bin's value
        (e.g. beyond the outermost centre of the end bins); NaN in bins
        without a value and outside COS_MIN..1."""
        c, L = self.centers, self.law
        good = np.isfinite(L)
        b = self._bins(cosi)
        own = b >= 0
        own[own] = good[b[own]]
        val = np.full(np.shape(cosi), np.nan)
        val[own] = L[b[own]]
        j = np.clip(np.searchsorted(c, cosi) - 1, 0, self.NBINS - 2)
        inner = (cosi >= c[0]) & (cosi <= c[-1])
        both = own & inner & good[j] & good[j + 1]
        t = (cosi[both] - c[j[both]]) / (c[j[both] + 1] - c[j[both]])
        val[both] = L[j[both]] + t * (L[j[both] + 1] - L[j[both]])
        return val


def band_of(mean_map, lon_axis, lat_axis, lo_px, hi_px):
    """Registration product of a stack's mean brightness map (NaN where
    unstacked): gridded, then ra.log_bandpass (speckle-averaged in linear
    brightness, then logged, then the large-scale trend removed)."""
    return ra.log_bandpass(ra.grid_map(mean_map, lon_axis, lat_axis), lo_px, hi_px)


def measure_pairwise(bands, names, search_px, step):
    meas = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x = ra.xcorr_offset(bands[names[i]], bands[names[j]], search_px)
            meas[(names[i], names[j])] = (x.dx * step, x.dy * step, x.peak,
                                          x.significance, x.at_edge)
    return meas


def solve_offsets(meas, names, ref):
    """Least-squares per-session registration offsets from pairwise measurements.

    Convention (REPORT §5): each solved (dlon, dlat) is the east/north
    correction, in degrees, that registers that session onto `ref` (pinned to
    (0, 0)); apply it via shift_intensity(I, +dlon, +dlat). Under this
    convention the closed-loop global sign is +1 (register_sessions' sign
    test); a measured -1 signals a convention regression. Enforced by
    test/test_registration_conventions.py.
    """
    if ref not in names:
        raise ValueError(f"reference session {ref} is not among {names}")
    free = [n for n in names if n != ref]
    A, b_lon, b_lat = [], [], []
    for (ni, nj), (dlon, dlat, *_rest) in meas.items():
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


def closure_noise_probability(ratio, n_sessions):
    """Probability that pure noise would close at least this well.

    Shifting a map shifts its cross-correlations, so re-measuring after the
    solved shifts returns the least-squares residual. If the pairwise
    offsets were pure noise (independent, equal variance on both axes), the
    squared residual / squared raw offsets would follow Beta(pairs - (n-1),
    n-1) (two axes pooled), pairs = n(n-1)/2. A small probability means the
    offsets are consistent across pairs; a large one means they are no more
    consistent than noise (they may still be real but small). Needs n >= 3:
    two sessions have no redundant pair."""
    pairs = n_sessions * (n_sessions - 1) // 2
    return float(scipy.stats.beta.cdf(ratio ** 2, pairs - (n_sessions - 1), n_sessions - 1))


def _float_eq(a, b):
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)), abs(float(b)))


def select_looks(run_dirs, stations, chans, min_snr=foh.MIN_TONE_SNR, min_tx_s=None):
    """Read the runs CSVs and gate every look. Returns (good, decisions):
    good is the list of usable rows (with _chan, _rx, _station and resolved
    file paths); decisions is one (row, reason) per look read, reason ""
    when used, followed by one per capture the batch did not map (from
    registration_skipped_<station>_<chan>.csv), reason "not mapped: ...".

    Gates, in order: transmission length (min_tx_s), tone strength
    (min_snr: the co-pol twin's for cross-pol looks), a timing shift at the
    search limit, a co-pol timing peak that is not resolved
    (freq_offset_hunt.leading_peak), a tone at the edge of the frequency
    search band, a rim calibration that did not converge (co-pol, applied to
    both channels), and the session rim-spread test (SPREAD_FLOOR_MHZ).

    Refused outright (ValueError): looks mapped with different terrain
    models, SPICE kernels or supersampling, or without terrain; duplicate
    looks; blank required columns; a run directory that contributes no runs
    CSV, or has a station's looks but no co-pol CSV; a cross-pol row whose
    corrections differ from its co-pol twin's (the cross-pol run is stale)
    or that was not made with --corrections-from.
    """
    rows, copol, not_mapped = [], {}, []
    for run_dir in run_dirs:
        contributed = []
        for station in stations:
            rx_name = dea.STATION_RX[station]
            have = {c: os.path.join(run_dir, f"registration_runs_{station}_{c}.csv")
                    for c in set(chans) | {"chan1"}}
            if not any(os.path.exists(have[c]) for c in chans):
                continue
            if not os.path.exists(have["chan1"]):
                raise ValueError(f"{run_dir} has {station} looks but no co-pol "
                                 f"{have['chan1']}, which the gates need")
            with open(have["chan1"], encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    blank = [k for k in COPOL_COLUMNS if not r.get(k)]
                    if blank:
                        raise ValueError(f"{have['chan1']}: {r['rx_file']} has no {blank}; "
                                         "rerun registration_stability.py for it")
                    copol[(station, r["rx_file"].split(".chan")[0])] = r
            for chan in chans:
                if not os.path.exists(have[chan]):
                    continue
                contributed.append(os.path.basename(have[chan]))
                with open(have[chan], encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        blank = [k for k in REQUIRED_COLUMNS if not r.get(k)]
                        if blank or "pair_snr" not in r or "rim_delta_hz" not in r:
                            raise ValueError(f"{have[chan]}: {r['rx_file']} has no "
                                             f"{blank or ['pair_snr/rim_delta_hz']}; "
                                             "rerun registration_stability.py for it")
                        r["_chan"], r["_rx"], r["_station"] = chan, rx_name, station
                        r["_run_dir"] = run_dir
                        for k in ("map_npy", "mult_npy", "area_npy"):
                            r[k] = resolve_npy(r[k], run_dir)
                        rows.append(r)
                skip_csv = os.path.join(run_dir, f"registration_skipped_{station}_{chan}.csv")
                if os.path.exists(skip_csv):
                    with open(skip_csv, encoding="utf-8") as f:
                        for s in csv.DictReader(f):
                            not_mapped.append(({"rx_file": s["rx_file"], "_station": station,
                                                "_chan": chan}, f"not mapped: {s['reason']}"))
        if not contributed:
            raise ValueError(f"{run_dir} has no registration_runs_<station>_<chan>.csv "
                             f"for {stations} x {chans}")
        print(f"{run_dir}: {', '.join(contributed)}")
    seen = set()
    for r in rows:
        if r["rx_file"] in seen:
            raise ValueError(f"{r['rx_file']} appears in more than one run directory")
        seen.add(r["rx_file"])
    for key in ("use_dem", "dem_file", "spice_kernels", "supersample"):
        values = {r[key] for r in rows}
        if len(values) > 1:
            raise ValueError(f"looks were mapped with different {key}: {sorted(values)}")
    if rows[0]["use_dem"] != "True":
        raise ValueError("looks were mapped without the terrain model")
    for r in rows:
        if r["_chan"] == "chan1":
            continue
        twin = copol.get((r["_station"], r["rx_file"].split(".chan")[0]))
        if twin is None:
            continue    # gated below
        if r["rim_status"] != "inherited":
            raise ValueError(f"{r['rx_file']}: cross-pol row made without --corrections-from "
                             f"(rim_status {r['rim_status']})")
        if r["edge_status"] != "inherited":
            raise ValueError(f"{r['rx_file']}: cross-pol row made without its co-pol "
                             f"twin's edge fit (edge_status {r['edge_status']})")
        # The twin's edge offsets apply only when it was fitted (a twin that
        # was not is rejected below, with its cross-pol look).
        edge_same = (twin["edge_status"] != "fitted"
                     or (_float_eq(r["edge_delay_us"], twin["edge_delay_us"])
                         and _float_eq(r["edge_doppler_hz"], twin["edge_doppler_hz"])))
        if not (_float_eq(r["applied_shift_samples"], twin["applied_shift_samples"])
                and _float_eq(r["applied_df_hz"], twin["applied_df_hz"])
                and twin["rim_delta_hz"] != ""
                and _float_eq(r["rim_delta_hz"], twin["rim_delta_hz"])
                and edge_same):
            raise ValueError(f"{r['rx_file']}: corrections differ from its co-pol twin's; "
                             "the cross-pol run is stale -- rerun registration_stability.py "
                             "for it with --corrections-from")
    rows.sort(key=lambda r: r["rx_start_utc"])

    by_sess = {}
    for (station, stem), r in copol.items():
        if r["rim_status"] == "converged":
            by_sess.setdefault((session_of(r["rx_file"]), station), []).append(
                (stem, float(r["rim_spread_hz"]) * 1e3))
    spread_anom = {}
    for (sess, station), v in by_sess.items():
        sp = np.array([x[1] for x in v])
        med = np.median(sp)
        thr = max(3 * 1.4826 * np.median(np.abs(sp - med)), SPREAD_FLOOR_MHZ)
        for stem, s in v:
            if abs(s - med) > thr:
                spread_anom[(station, stem)] = (
                    f"rim spread {s:+.1f} mHz vs session median {med:+.1f} "
                    f"(threshold {thr:.1f}): intra-look wander")

    good, decisions = [], []
    for r in rows:
        stem = r["rx_file"].split(".chan")[0]
        twin = copol.get((r["_station"], stem))
        snr = float(r["pair_snr"] or r["tone_snr"])
        shift = float(r["applied_shift_samples"])
        if min_tx_s is not None and float(r["tx_s"]) < min_tx_s:
            why = f"transmission {float(r['tx_s']):.0f} s < {min_tx_s:g} s"
        elif snr < min_snr:
            why = f"tone snr {snr:.0f} < {min_snr:g}"
        elif abs(shift) >= int(r["max_shift"]) - 0.5:
            why = f"timing shift {shift:+.1f} at the search limit +-{r['max_shift']}"
        elif twin is None:
            why = "no co-pol row for this capture"
        elif twin["timing_status"] != "resolved":
            why = f"timing peak {twin['timing_status']}"
        elif twin["tone_at_band_edge"] == "True":
            why = "specular tone at the edge of the +-2 Hz search band"
        elif twin["rim_status"] != "converged":
            why = f"co-pol rim calibration {twin['rim_status']}"
        elif twin["edge_status"] != "fitted":
            why = f"co-pol edge fit {twin['edge_status']}"
        elif (r["_station"], stem) in spread_anom:
            why = spread_anom[(r["_station"], stem)]
        else:
            why = ""
        decisions.append((r, why))
        if not why:
            good.append(r)
    return good, decisions + not_mapped


def write_decisions(path, decisions):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["rx_file", "station", "chan", "session", "used", "reason", "gain"])
        for r, why in decisions:
            # A capture the batch did not map is in no session.
            session = "" if why.startswith("not mapped") else session_of(r["rx_file"])
            w.writerow([r["rx_file"], r["_station"], r["_chan"], session,
                        not why, why, r.get("_gain", "")])


def register_sessions(acc1, session_names, ref, min_count, grid):
    """Solve and check the session shifts; returns (offsets, record).

    Stops (SystemExit) if a cross-correlation peak is on the search edge
    (not a measurement) or if the shifts close clearly better with the
    opposite sign (a convention regression). How well the shifts close is
    recorded, not used to stop: with three sessions the test has little
    power, and offsets no more consistent than noise may be real but small."""
    lon_axis, lat_axis, lo_px, hi_px, search_px, step = grid
    means = {s: acc1[s].mean(min_count) for s in session_names}
    bands = {s: band_of(means[s], lon_axis, lat_axis, lo_px, hi_px) for s in session_names}
    meas0 = measure_pairwise(bands, session_names, search_px, step)
    print("Pairwise cross-correlation (deg / peak / significance):")
    for (ni, nj), (dlon, dlat, peak, sig, edge) in meas0.items():
        print(f"  {ni} / {nj}: dlon {dlon:+.3f} dlat {dlat:+.3f}  "
              f"peak {peak:.3f} significance {sig:.2f}"
              + ("  AT THE SEARCH EDGE" if edge else ""))
    at_edge = [p for p, m in meas0.items() if m[4]]
    if at_edge:
        raise SystemExit(f"cross-correlation peak at the +-{search_px * step:g} deg "
                         f"search edge for {at_edge}: no registration measured. "
                         "Rerun with --no-registration to stack without shifts.")
    offsets = solve_offsets(meas0, session_names, ref)
    print(f"Solved session shifts (deg, ref = {ref}):")
    for s, o in offsets.items():
        print(f"  {s}: dlon {o[0]:+.3f} dlat {o[1]:+.3f}")

    def remeasure(sign):
        shifted = {s: band_of(shift_intensity(means[s], sign * offsets[s][0],
                                              sign * offsets[s][1]),
                              lon_axis, lat_axis, lo_px, hi_px) for s in session_names}
        return measure_pairwise(shifted, session_names, search_px, step)

    def norm(m):
        return float(np.sqrt(sum(v[0] ** 2 + v[1] ** 2 for v in m.values())))

    m_plus, m_minus = remeasure(+1.0), remeasure(-1.0)
    base, res_plus, res_minus = norm(meas0), norm(m_plus), norm(m_minus)
    # With the right sign the re-measured offsets are the residual r; with
    # the wrong sign r + 2 A o, so res_minus^2 - res_plus^2 = 4 |A o|^2,
    # A o the offsets' pairwise differences. sign_test is that difference in
    # units of 4 |A o|^2: +1 for the right convention, -1 for a flipped one,
    # near 0 when the shifts are too small to tell against re-measurement
    # noise (then the sign hardly matters). 0.5 is the midpoint between the
    # two hypotheses.
    ao2 = sum((offsets[nj][0] - offsets[ni][0]) ** 2 + (offsets[nj][1] - offsets[ni][1]) ** 2
              for ni, nj in meas0)
    sign_test = (res_minus ** 2 - res_plus ** 2) / (4 * ao2) if ao2 > 0 else float("nan")
    record = {"pairs": {f"{ni}/{nj}": dict(zip(("dlon_deg", "dlat_deg", "peak",
                                                   "significance", "at_edge"), v))
                        for (ni, nj), v in meas0.items()},
              "offsets_deg": {s: list(o) for s, o in offsets.items()},
              "reference": ref, "raw_norm_deg": base, "closed_norm_deg": res_plus,
              "opposite_sign_norm_deg": res_minus, "sign_test": sign_test,
              "remeasured_at_edge": [f"{ni}/{nj}" for (ni, nj), v in m_plus.items() if v[4]]}
    print(f"Closed loop: residual {base:.3f} -> {res_plus:.3f} deg; opposite sign "
          f"{res_minus:.3f} deg (sign test {sign_test:+.2f}: +1 right, -1 flipped)")
    if sign_test < -0.5:
        raise SystemExit("session shifts close clearly better with the opposite sign: "
                         "convention regression; run test/test_registration_conventions.py")
    if not sign_test > 0.5:
        print("  the shifts are too small against re-measurement noise to check the "
              "sign convention")
    if len(session_names) >= 3:
        ratio = res_plus / base
        p = closure_noise_probability(ratio, len(session_names))
        record.update({"closure_ratio": ratio, "noise_closes_this_well_probability": p})
        print(f"  closure ratio {ratio:.2f}: pure noise would close this well with "
              f"probability {p:.2f}")
    else:
        print("  two sessions: no redundant pair, so closure cannot check the shift; "
              "it rests on the one cross-correlation above")
    return offsets, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", nargs="+", required=True,
                        help="one or more run directories; looks from all of "
                             "them are stacked together (e.g. two seasons "
                             "processed separately)")
    parser.add_argument("--out-dir", default=None,
                        help="where stacks are written; default the run "
                             "directory (required with several)")
    parser.add_argument("--stations", nargs="+", required=True,
                        choices=sorted(dea.STATION_RX),
                        help="receivers to stack, as registration_stability.py "
                             "--station; names the outputs")
    parser.add_argument("--chans", nargs="+", required=True)
    parser.add_argument("--min-snr", type=float, default=foh.MIN_TONE_SNR)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--min-tx-s", type=float, default=None,
                        help="keep only looks whose transmission lasted at least "
                             "this many seconds (runs-CSV column tx_s); the "
                             "outputs are then named ..._txmin<N>s_<chan>")
    parser.add_argument("--ref-session", default="2025-09-16",
                        help="session the others are registered onto")
    parser.add_argument("--no-registration", action="store_true",
                        help="stack without session shifts; outputs are named "
                             "..._unregistered")
    args = parser.parse_args()
    if args.out_dir is None:
        if len(args.run_dir) > 1:
            parser.error("--out-dir is required with several --run-dir")
        args.out_dir = args.run_dir[0]
    os.makedirs(args.out_dir, exist_ok=True)
    label = "dual" if len(args.chans) > 1 else args.chans[0]
    selection = f"_txmin{args.min_tx_s:g}s" if args.min_tx_s is not None else ""
    out_label = ("_".join(args.stations) + selection + f"_{label}"
                 + ("_unregistered" if args.no_registration else ""))
    tmp = tempfile.mkdtemp(prefix=f".stack_{out_label}_", dir=args.out_dir)
    try:
        run(args, label, out_label, tmp)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    for name in sorted(os.listdir(tmp)):
        os.replace(os.path.join(tmp, name), os.path.join(args.out_dir, name))
    os.rmdir(tmp)
    print(f"Wrote the {out_label} outputs to {args.out_dir}")


def run(args, label, out_label, out):
    """The stacking run, writing every output into the folder out."""
    good, decisions = select_looks(args.run_dir, args.stations, args.chans,
                                   args.min_snr, args.min_tx_s)
    n_read = sum(1 for _, why in decisions if not why.startswith("not mapped"))
    print(f"[{label}] {len(good)} looks pass gating, {n_read - len(good)} excluded, "
          f"{len(decisions) - n_read} captures not mapped by the batch")
    for r, why in decisions:
        if why:
            print(f"  - {r['rx_file']}: {why}")
    if not good:
        raise SystemExit("no look passes the gates")

    step = 0.075
    lon_axis = np.arange(-55, 55 + step / 2, step)
    lat_axis = np.arange(-55, 55 + step / 2, step)
    lo_px, hi_px = 0.3 / step, 2.5 / step
    search_px = int(1.5 / step)
    grid = (lon_axis, lat_axis, lo_px, hi_px, search_px, step)

    npix = len(np.load(good[0]["map_npy"], mmap_mode="r"))
    station_cache = {}
    for r in good:
        r["_et"] = dea.csp.str2et(r["rx_start_utc"])
    print("radiometry: brightness per unit area, noise floor removed")

    def look(r):
        """(brightness, cos incidence) for a look."""
        cosi = incidence_cos(r["_et"], npix, station_cache, rx_name=r["_rx"])
        I, r["_gain"] = look_brightness(r, cosi)
        return I, cosi

    # --- Pass 1: unshifted session stacks + scattering-law sums ---
    decision_of = {id(r): i for i, (r, _) in enumerate(decisions)}
    used = []
    acc1 = {}
    laws = {c: ScatterLaw() for c in args.chans}
    for r in good:
        try:
            I, cosi = look(r)
        except LookRejected as exc:
            decisions[decision_of[id(r)]] = (r, str(exc))
            print(f"  - {r['rx_file']}: {exc}")
            continue
        used.append(r)
        s = session_of(r["rx_file"])
        if s not in acc1:
            acc1[s] = Accumulator(npix)
        acc1[s].add(I)
        laws[r["_chan"]].collect(I, cosi)
    write_decisions(os.path.join(out, f"stacked_looks_{out_label}.csv"), decisions)
    if not used:
        raise SystemExit("no look left to stack")
    session_names = sorted(acc1)
    n_maps = len(used)
    n_captures = len({r["rx_file"].split(".chan")[0] for r in used})
    print({s: sum(session_of(r["rx_file"]) == s for r in used) for s in session_names})
    for c in args.chans:
        none, one, weak = laws[c].fit()
        if none + one + weak:
            print(f"  scattering law {c}: of {ScatterLaw.NBINS} incidence bins, {none} "
                  f"have no pixels, {one} only one look, {weak} a mean not clearly "
                  "above zero; their pixels have no normalized value")

    # Scattering-law diagnostic plot (only bins with a value)
    pl.figure(figsize=(7, 5))
    for c in args.chans:
        ok = np.isfinite(laws[c].law)
        pl.plot(np.degrees(np.arccos(laws[c].centers[ok])),
                10 * np.log10(laws[c].law[ok]), "o-", ms=3, label=c)
    pl.xlabel("incidence angle (deg)")
    pl.ylabel("relative backscatter (dB)")
    pl.title(f"Empirical scattering law ({float(used[0]['frequency_hz']) / 1e6:.2f} MHz); "
             "bins without a clear mean not drawn", fontsize=9)
    pl.grid(alpha=0.3)
    pl.legend()
    pl.savefig(os.path.join(out, f"scattering_law_{out_label}.png"), dpi=130)
    pl.close()

    # --- Session offsets (band-passed, insensitive to the smooth law) ---
    if args.no_registration or len(session_names) < 2:
        offsets = {s: (0.0, 0.0) for s in session_names}
        why = ("--no-registration" if args.no_registration
               else f"single session ({session_names[0]})")
        print("No session shifts: " + why)
        record = {"offsets_deg": {s: [0.0, 0.0] for s in session_names}, "note": why}
    else:
        if args.ref_session not in session_names:
            raise SystemExit(f"--ref-session {args.ref_session} is not among the "
                             f"stacked sessions {session_names}")
        offsets, record = register_sessions(acc1, session_names, args.ref_session,
                                            args.min_count, grid)
    with open(os.path.join(out, f"stacked_registration_{out_label}.json"), "w") as f:
        json.dump(record, f, indent=1)

    # --- Pass 2: raw + scattering-normalized stacks, shifts applied ---
    grand_raw = Accumulator(npix)
    grand_norm = Accumulator(npix)
    for r in used:
        sess = session_of(r["rx_file"])
        I, cosi = look(r)
        law = laws[r["_chan"]].value(cosi)
        sh = offsets[sess]
        grand_raw.add(shift_intensity(I, *sh))
        grand_norm.add(shift_intensity(I / law, *sh))

    title_n = (f"{n_maps} maps, {n_captures} captures" if len(args.chans) > 1
               else f"{n_maps} looks")
    for suffix, accum in [("", grand_raw), ("_scatnorm", grand_norm)]:
        g = accum.stacked_log(args.min_count)
        mean = accum.mean(args.min_count)
        band = band_of(mean, lon_axis, lat_axis, lo_px, hi_px)
        enough = np.isfinite(mean)
        not_pos = enough.copy()
        not_pos[enough] = ~(mean[enough] > 0)
        print(f"[{label}{suffix}] noise floor: {band[band != 0].std():.4f} ({title_n}); "
              f"{not_pos.sum() / enough.sum():.1%} of pixels with >= {args.min_count} "
              "looks have no positive mean brightness")
        np.save(os.path.join(out, f"stacked_map_{out_label}{suffix}.npy"), g)
        np.save(os.path.join(out, f"stacked_linear_{out_label}{suffix}.npy"),
                mean.astype(np.float32))
        valid = g != hp.UNSEEN
        vmin, vmax = np.percentile(g[valid], [5, 99.5])
        dea.save_lunar_image(
            g, os.path.join(out, f"stacked_map_{out_label}{suffix}.png"),
            f"{out_label}{suffix} stack ({title_n})", vmin=vmin, vmax=vmax)
    np.save(os.path.join(out, f"stacked_counts_{out_label}.npy"), grand_raw.count)
    covered = grand_raw.count >= args.min_count
    print(f"valid fraction {covered.mean():.3f}, "
          f"median looks/pixel {np.median(grand_raw.count[covered]):.0f}")


if __name__ == "__main__":
    main()
