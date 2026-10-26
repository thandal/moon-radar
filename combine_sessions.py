"""
Stack looks in two levels, so that many looks sharing one geometry cannot
outvote fewer looks that see a place better. Since 2026-10-26 this is the
main stacked product; stack_maps.py's equal-weight stack is the step it is
built from (its looks, scattering laws and session shifts) and a check.

Each look's map holds the surface plus two kinds of error. Speckle and
receiver noise change from look to look, so averaging any looks reduces
them. Geometry error -- mirror ghosts, and blur where the look's
delay-Doppler cells are large (near its specular point and its Doppler
equator) -- is fixed by the geometry, so looks with the same geometry
repeat the same error and averaging them does not reduce it.

Feature-size bands: each map is split into the variation finer than 10 km,
10-20 km, 20-40 km, 40-75 km, 75-150 km and broader (differences of
Gaussian smoothings at SCALES_KM, full width at half maximum, each
smoothing normalized by the smoothed coverage so a map's edge does not
pull it towards zero). Both levels work band by band.

Level 1, within a session (one night's looks, as stack_maps.session_of;
its specular point hardly moves): for each band, the plain mean of the
looks that place features of that size at the pixel. A look's pixel
counts in band b only if the look's cell area there is at most
CUT_X_MEDIAN[b] times its median cell area over the pixels it sees: 2x,
4x, 8x and 16x for features below 10, 10-20, 20-40 and 40-75 km, no cut
above 75 km. Measured 2026-10-25 by passing a known surface through 24
looks' geometry (investigations/footprint_localization.py): the detail a
look keeps at a place fades steadily as its cell grows, faster for small
features, and falls to half its typical value at about these multiples,
whatever the look's length or chip rate; broad features are kept even in
the largest cells. Half is the line for an equal-weight mean of equally
noisy looks: a look keeping less than half the others' detail lowers the
mean's signal to noise. So a large cell no longer removes the pixel (the
single multiplicity limit used before left holes where every look of a
session had large cells, as at 2026's specular point): the session still
gives the broad features there, and fine detail only where it places it.
--level1 limit (stack_maps' single multiplicity limit) or none (every
seen pixel of every look) instead, for comparison. On the independent
synthetic surface (2026-10-26, all seasons, opposite-sense) both of those
keep less error below 10 km than the cuts (0.84 against 0.87 of equal
weights' over the disk, 0.90 against 1.00 at the 2025 specular point): the
cuts compare a look with its own typical cell, not with the other looks at
the place, so they also drop 2026's finest detail along an edge near the
2025 specular point. The cuts were kept because the full maps look cleaner
(chosen by eye, 2026-10-26). Brightness is scattering-law-normalized as stack_maps
normalizes it (one law per channel and season), with the session shifts
of the equal-weight stack applied. Each session's noise is measured from
its own looks: half the difference of two interleaved half-session means
(alternate captures in time; both channels of a capture in the same
half), which cancels the shared geometry error and keeps the noise of the
full mean.

Level 2, across sessions: at each place and in each band, the sessions are
combined with the weights that minimize the expected error, given the
covariance of their errors: noise (independent between sessions) plus
geometry error, which can be shared (the three 2025 sessions' specular
points are within a few degrees of each other). The geometry error is not
guessed: simulate_geometry.py passes a synthetic surface through every
look's own geometry, the simulated looks are stacked the same way (same
cuts), and each session's simulated mean minus the synthetic surface is
its geometry error. The products of these errors, averaged over a
neighbourhood (Gaussian, full width LOCAL_KM per band), give the error
covariance between sessions; it is scaled to the real surface by the ratio
of real to simulated cross-session detail power in each band (KAPPA,
printed). The weights are zero or positive and sum to one at every place
(see combine: unconstrained, the minimum subtracts a session to cancel
shared blur); a session with no value in a band at a place gets none
there. 1 / (sum of the inverse covariance) is the combination's expected
error variance, written per band. With a single session there is nothing
to weigh: the product is its level-1 bands added up.

Channels as stack_maps.py --chans: one, or the opposite- and same-sense
channels together (dual), each with its own scattering law. The simulated
twins are found by capture, so one simulation serves both channels (the
geometry is the same; simulate_geometry.py is run on the opposite-sense
rows).

Outputs in --out-dir, named as stack_maps names them with _geomweighted:
stacked_linear_<label>_geomweighted[_scatnorm].npy (mean brightness),
stacked_map_<label>_geomweighted[_scatnorm].npy/.png (log, as stack_maps),
and combination_<label>.npz: session names, per-band weights, the
combination's expected error variance and each session's noise variance,
KAPPA, the level-1 cuts, the two half-stacks combined with the weights,
and, as stack_maps' mask gives them, each session's mean, half-session
means and look counts (for validate_session_combination.py).

Usage (from the repo root):
    .conda/bin/python combine_sessions.py \\
        --run-dir results/STOCKERT_2025_EDGEFIT results/EVE2026_CIRC \\
        --stations stockert wsrt --chans chan1 \\
        --equal-stack-dir results/ALL_SEASONS_EDGEFIT \\
        --sim-dir results/GEOMETRY_SIM/seed1 --out-dir results/ALL_SEASONS_EDGEFIT
"""

import argparse
import csv
import itertools
import json
import os

import numpy as np
import healpy as hp

import doppler_equator_alignment as dea
import freq_offset_hunt as foh
import stack_maps as sm

R_KM = 1737.4
SCALES_KM = (10.0, 20.0, 40.0, 75.0, 150.0)
BAND_NAMES = ("<10 km", "10-20 km", "20-40 km", "40-75 km", "75-150 km", ">150 km")
LOCAL_KM = (150.0, 150.0, 160.0, 300.0, 600.0, 900.0)
CUT_X_MEDIAN = (2.0, 4.0, 8.0, 16.0, np.inf, np.inf)
RIDGE = 1e-6        # relative to the mean diagonal: keeps every solve well posed


LEVEL1_MODES = ("cuts", "limit", "none")


def level_plan(mode="cuts"):
    """The level-1 masks, as (multiplicity limit, cell-area cut as a multiple
    of the look's median) pairs, and which one each band uses; plus
    stack_maps' own mask (the reference: equal-weight comparisons, look
    counts). mode: "cuts" (CUT_X_MEDIAN per band), "limit" (stack_maps'
    single multiplicity limit for every band) or "none" (every seen pixel
    of every look in every band)."""
    reference = (float(dea.MAX_MULTIPLICITY), np.inf)
    if mode == "limit":
        return [reference], [0] * len(BAND_NAMES), reference
    if mode == "none":
        return [(np.inf, np.inf)], [0] * len(BAND_NAMES), reference
    if mode != "cuts":
        raise ValueError(f"level-1 mode {mode!r}: one of {LEVEL1_MODES}")
    levels = sorted({(np.inf, c) for c in CUT_X_MEDIAN})
    return levels, [levels.index((np.inf, c)) for c in CUT_X_MEDIAN], reference


class Smoother:
    """Gaussian smoothing of healpix maps, NaN-aware (normalized by the
    smoothed coverage), with the coverage smoothings cached."""

    def __init__(self, nside):
        self.nside = nside
        self.cache = {}

    def lmax(self, fwhm_km):
        sigma = fwhm_km / 2.3548 / R_KM
        return int(min(3 * self.nside - 1, np.ceil(4.5 / sigma)))

    def plain(self, m, fwhm_km):
        return hp.smoothing(np.asarray(m, dtype=np.float64), fwhm=fwhm_km / R_KM,
                            lmax=self.lmax(fwhm_km), iter=1).astype(np.float32)

    def coverage(self, mask, key, fwhm_km):
        k = (key, fwhm_km)
        if k not in self.cache:
            self.cache[k] = self.plain(mask.astype(np.float32), fwhm_km)
        return self.cache[k]

    def normalized(self, m, mask, key, fwhm_km):
        den = self.coverage(mask, key, fwhm_km)
        num = self.plain(np.where(mask, m, 0.0), fwhm_km)
        return np.where(mask & (den > 0.05), num / np.maximum(den, 1e-6), np.nan).astype(np.float32)


def bands(m, mask, key, sm_):
    """The feature-size bands of map m (NaN outside mask); they sum to m."""
    levels = [np.where(mask, m, np.nan).astype(np.float32)]
    for f in SCALES_KM:
        levels.append(sm_.normalized(m, mask, key, f))
    out = [levels[k] - levels[k + 1] for k in range(len(SCALES_KM))]
    out.append(levels[-1])
    return out


def band(m, mask, key, sm_, b):
    """Band b alone of map m (as bands(m, ...)[b])."""
    fine = (np.where(mask, m, np.nan).astype(np.float32) if b == 0
            else sm_.normalized(m, mask, key, SCALES_KM[b - 1]))
    if b == len(SCALES_KM):
        return fine
    return fine - sm_.normalized(m, mask, key, SCALES_KM[b])


def half_of_capture(rows):
    """Which half (0 or 1) of its session each capture goes to: alternate
    captures in time within each session."""
    half, count = {}, {}
    for r in sorted(rows, key=lambda r: (r["rx_start_utc"], r["rx_file"])):
        c = capture_of(r["rx_file"])
        if c not in half:
            s = sm.session_of(r["rx_file"])
            half[c] = count.get(s, 0) % 2
            count[s] = count.get(s, 0) + 1
    return half


def session_levels(rows, laws, offsets, cache, npix, levels, min_count, sim=False):
    """Session by session (yields (session, out)): for each level-1 mask
    in levels, the session's mean normalized brightness M, mean raw
    brightness Mraw, interleaved half-session means H0 and H1 (float32, NaN
    with fewer than min_count looks) and look counts n."""
    half = half_of_capture(rows)
    by_session = {}
    for r in sorted(rows, key=lambda r: (r["rx_start_utc"], r["rx_file"])):
        by_session.setdefault(sm.session_of(r["rx_file"]), []).append(r)
    for s in sorted(by_session):
        acc = {}
        for lv in levels:
            acc[lv] = {k: np.zeros(npix) for k in ("sum", "raw", "sum0", "sum1")}
            acc[lv].update({k: np.zeros(npix, dtype=np.int32) for k in ("n", "nr", "n0", "n1")})
        for r in by_session[s]:
            cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), npix, cache, rx_name=r["_rx"])
            I, _ = sm.look_brightness(r, cosi, max_multiplicity=np.inf)
            seen = np.isfinite(I)
            area = np.load(r["area_npy"]).astype(np.float64)
            ratio = np.where(seen, area / np.median(area[seen]), np.nan)
            mult = np.where(seen, np.load(r["mult_npy"]).astype(np.float64), np.nan)
            N = I / laws[sm.law_key(r)].value(cosi)
            if not sim:
                sh = offsets[s]
                N, I, ratio, mult = (sm.shift_intensity(x, *sh) for x in (N, I, ratio, mult))
            h = half[capture_of(r["rx_file"])]
            for lv in levels:
                with np.errstate(invalid="ignore"):
                    keep = (mult <= lv[0]) & (ratio <= lv[1])
                ok = keep & np.isfinite(N)
                okr = keep & np.isfinite(I)
                a = acc[lv]
                a["sum"][ok] += N[ok]
                a["n"][ok] += 1
                a["raw"][okr] += I[okr]
                a["nr"][okr] += 1
                a[f"sum{h}"][ok] += N[ok]
                a[f"n{h}"][ok] += 1
        out = {}
        for lv, a in acc.items():
            out[lv] = {"M": mean_of(a["sum"], a["n"], min_count),
                       "Mraw": mean_of(a["raw"], a["nr"], min_count),
                       "H0": mean_of(a["sum0"], a["n0"], min_count),
                       "H1": mean_of(a["sum1"], a["n1"], min_count),
                       "n": a["n"]}
        del acc
        yield s, out


def capture_of(rx_file):
    """The capture a look's file belongs to (both channels share it)."""
    return rx_file.split(".chan")[0]


def simulated_twins(rows, sim_dir, stations):
    """Each look's simulated twin (simulate_geometry.py's row for the same
    capture, any channel: the geometry is the same) with the look's channel
    and receiver, and the synthetic surface's mean per map pixel."""
    sim_rows = {}
    for f in sorted(os.listdir(sim_dir)):
        if f.startswith("simulated_runs_") and f.endswith(".csv") and \
                any(f.startswith(f"simulated_runs_{st}_") for st in stations):
            with open(os.path.join(sim_dir, f)) as fh:
                sim_rows.update({capture_of(r["rx_file"]): r for r in csv.DictReader(fh)})
    missing = [r["rx_file"] for r in rows if capture_of(r["rx_file"]) not in sim_rows]
    if missing:
        raise SystemExit(f"{len(missing)} looks have no simulated map in {sim_dir}, e.g. {missing[0]}")
    srows = []
    for r in rows:
        q = dict(sim_rows[capture_of(r["rx_file"])])
        q.update({k: r[k] for k in ("rx_file", "_chan", "_rx", "_station")})
        srows.append(q)
    seed = [f for f in os.listdir(sim_dir) if f.startswith("surface_seed") and f.endswith("_map.npy")]
    if len(seed) != 1:
        raise SystemExit(f"expected one synthetic surface in {sim_dir}, found {seed}")
    return srows, np.load(os.path.join(sim_dir, seed[0]))


def mean_of(total, n, min_count):
    return np.where(n >= min_count, total / np.maximum(n, 1), np.nan).astype(np.float32)


def fit_laws(rows, cache, npix, by_season=True):
    """Scattering laws of the looks, as stack_maps fits them: one per
    channel and season (stack_maps.law_key), or with by_season=False one per
    channel (keyed (channel, None))."""
    laws = {}
    for r in rows:
        cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), npix, cache, rx_name=r["_rx"])
        I, _ = sm.look_brightness(r, cosi)
        key = sm.law_key(r) if by_season else (r["_chan"], None)
        laws.setdefault(key, sm.ScatterLaw()).collect(I, cosi)
    for law in laws.values():
        law.fit()
    return laws


def local_noise(Bh0, Bh1, sm_, key, b):
    """Noise variance of a session's band b, averaged over LOCAL_KM[b]: the
    squared half-difference of its two half-session bands."""
    d = (Bh0.astype(np.float32) - Bh1.astype(np.float32)) / 2
    m = np.isfinite(d)
    n2 = np.where(m, d, 0.0) ** 2
    return sm_.plain(n2, LOCAL_KM[b]) / np.maximum(sm_.coverage(m, key, LOCAL_KM[b]), 1e-3)


def combine(B, G, N, present, chunk=250_000):
    """Per pixel: the weights, each zero or positive and together one, that
    minimize the expected error of sum_s w_s B_s for error covariance
    G (geometry, sessions x sessions per pixel) + diag(N) (noise), among
    the sessions present there. Unconstrained, the minimum can subtract a
    session to cancel an error it shares with others (sharpening by
    differencing two blurred maps), which leans entirely on the simulation
    matching the real looks; the constraint keeps a certainty-weighted
    average. Solved exactly: for every subset of the sessions present, the
    best weights using only that subset (inverse covariance, summing to
    one); of the subsets whose weights are all non-negative, the one with
    the least expected error. Returns (combined, weights, variance)."""
    n_s, npix = len(B), len(B[0])
    P = np.array(present)
    subsets = [list(c) for k in range(1, n_s + 1) for c in itertools.combinations(range(n_s), k)]
    W = np.zeros((n_s, npix), dtype=np.float32)
    VAR = np.full(npix, np.nan, dtype=np.float32)
    for a in range(0, npix, chunk):
        sl = slice(a, min(a + chunk, npix))
        n = sl.stop - sl.start
        K = np.empty((n, n_s, n_s))
        for i in range(n_s):
            for j in range(n_s):
                K[:, i, j] = G[i][j][sl]
            K[:, i, i] += N[i][sl]
        K = np.nan_to_num(K, nan=0.0)
        diag = np.einsum("pii->p", K) / n_s
        K[:, np.arange(n_s), np.arange(n_s)] += RIDGE * np.maximum(diag, 1e-30)[:, None]
        best_var = np.full(n, np.inf)
        best_w = np.zeros((n, n_s))
        for S in subsets:
            ok = np.nonzero(P[S][:, sl].all(axis=0))[0]
            if len(ok) == 0:
                continue
            KS = K[ok][:, S][:, :, S]
            u = np.linalg.solve(KS, np.ones((len(ok), len(S), 1)))[..., 0]
            tot = u.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                w = u / tot[:, None]
                var = 1.0 / tot
            feas = (tot > 0) & np.all(w >= -1e-9, axis=1)
            idx = ok[feas]
            better = var[feas] < best_var[idx]
            idx = idx[better]
            best_var[idx] = var[feas][better]
            best_w[idx] = 0.0
            best_w[np.ix_(idx, S)] = np.clip(w[feas][better], 0.0, None)
        W[:, sl] = best_w.T
        VAR[sl] = np.where(np.isfinite(best_var), best_var, np.nan)
    comb = np.zeros(npix)
    for i in range(n_s):
        comb += W[i] * np.where(present[i], np.nan_to_num(B[i]), 0.0)
    comb[~P.any(axis=0)] = np.nan
    return comb.astype(np.float32), W, VAR


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dir", nargs="+", required=True)
    p.add_argument("--stations", nargs="+", required=True, choices=sorted(dea.STATION_RX))
    p.add_argument("--chans", nargs="+", required=True,
                   help="as stack_maps.py --chans (two channels: the dual stack)")
    p.add_argument("--equal-stack-dir", required=True,
                   help="the equal-weight stack of the same looks (stack_maps.py): its "
                        "looks list and session shifts are used")
    p.add_argument("--sim-dir", default=None,
                   help="simulate_geometry.py output for every look of the stack")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-snr", type=float, default=foh.MIN_TONE_SNR)
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--level1", choices=LEVEL1_MODES, default="cuts",
                   help="level-1 masks: per-band cell-area cuts (default), stack_maps' single "
                        "multiplicity limit, or none (for comparison)")
    p.add_argument("--no-footprint-cuts", action="store_true", help="same as --level1 limit")
    p.add_argument("--write-law", default=None,
                   help="only write the scattering law of the stack's looks, one law over all "
                        "seasons (bin centres and values), to this .npz, for "
                        "simulate_geometry.py --law")
    args = p.parse_args()
    label = "_".join(args.stations) + "_" + ("dual" if len(args.chans) > 1 else args.chans[0])
    os.makedirs(args.out_dir, exist_ok=True)

    good, _ = sm.select_looks(args.run_dir, args.stations, args.chans, args.min_snr)
    with open(os.path.join(args.equal_stack_dir, f"stacked_looks_{label}.csv")) as f:
        used = {u["rx_file"] for u in csv.DictReader(f) if u["used"] == "True"}
    rows = [r for r in good if r["rx_file"] in used]
    if len(rows) != len(used):
        raise SystemExit(f"{len(used)} looks in the equal-weight stack, {len(rows)} pass the gates now")
    with open(os.path.join(args.equal_stack_dir, f"stacked_registration_{label}.json")) as f:
        offsets = {s: tuple(o) for s, o in json.load(f)["offsets_deg"].items()}
    if args.write_law:
        if len(args.chans) != 1:
            raise SystemExit("--write-law takes one channel")
        c = args.chans[0]
        npix = len(np.load(rows[0]["map_npy"], mmap_mode="r"))
        law = fit_laws(rows, {}, npix, by_season=False)[(c, None)]
        np.savez(args.write_law, centers=law.centers, edges=law.edges, law=law.law)
        print(f"wrote the {c} scattering law of {len(rows)} looks to {args.write_law}")
        return
    if args.sim_dir is None:
        raise SystemExit("--sim-dir is required")
    # The simulated twin of every look: same row, simulated map, no noise.
    srows, truth = simulated_twins(rows, args.sim_dir, args.stations)

    npix = len(truth)
    nside = hp.npix2nside(npix)
    mode = "limit" if args.no_footprint_cuts else args.level1
    levels, lob, reference = level_plan(mode)
    all_levels = levels if reference in levels else levels + [reference]
    n_b = len(BAND_NAMES)
    cache = {}
    print(f"{len(rows)} looks; fitting scattering laws (real and simulated)", flush=True)
    laws = fit_laws(rows, cache, npix)
    laws_sim = fit_laws(srows, cache, npix)
    print("level 1: " + {"limit": f"single multiplicity limit {dea.MAX_MULTIPLICITY:g}",
                         "none": "no mask (every seen pixel of every look)",
                         "cuts": "cell-area cuts per band, x the look's median: "
                         + ", ".join(f"{n} {c:g}" for n, c in zip(BAND_NAMES, CUT_X_MEDIAN))}[mode],
          flush=True)

    sm_ = Smoother(nside)
    names, Bm, Braw, Bh0, Bh1, Bs, Be = [], {}, {}, {}, {}, {}, {}
    ref_mean, ref_h0, ref_h1, ref_looks = {}, {}, {}, {}
    for s, out in session_levels(rows, laws, offsets, cache, npix, all_levels, args.min_count):
        names.append(s)
        ref = out[reference]
        ref_mean[s], ref_h0[s], ref_h1[s], ref_looks[s] = ref["M"], ref["H0"], ref["H1"], ref["n"]
        Bm[s], Braw[s], Bh0[s], Bh1[s] = [], [], [], []
        for b in range(n_b):
            lv = out[levels[lob[b]]]
            k = (s, lob[b])
            Bm[s].append(band(lv["M"], np.isfinite(lv["M"]), ("m",) + k, sm_, b))
            Braw[s].append(band(lv["Mraw"], np.isfinite(lv["Mraw"]), ("r",) + k, sm_, b))
            hm = np.isfinite(lv["H0"]) & np.isfinite(lv["H1"])
            Bh0[s].append(band(lv["H0"], hm, ("h",) + k, sm_, b).astype(np.float16))
            Bh1[s].append(band(lv["H1"], hm, ("h",) + k, sm_, b).astype(np.float16))
        print(f"  {s}: {int(np.median(ref['n'][ref['n'] > 0]))} looks", flush=True)
        del out
        sm_.cache.clear()
    for s, out in session_levels(srows, laws_sim, offsets, cache, npix, levels, args.min_count,
                                 sim=True):
        Bs[s], Be[s] = [], []
        for b in range(n_b):
            S = out[levels[lob[b]]]["M"]
            m = np.isfinite(S)
            Bs[s].append(band(S, m, ("s", s, lob[b]), sm_, b).astype(np.float16))
            Be[s].append(band(S - truth, m, ("s", s, lob[b]), sm_, b))
        S = out[levels[lob[-1]]]["M"]
        print(f"  {s}: simulated session mean / surface over its pixels "
              f"{np.nanmean(S) / np.mean(truth[np.isfinite(S)]):.3f}", flush=True)
        del out
        sm_.cache.clear()
    n_s = len(names)

    kappa = np.full(n_b, np.nan)
    W = np.zeros((n_b, n_s, npix), dtype=np.float16)
    VAR = np.zeros((n_b, npix), dtype=np.float32)
    NV = np.zeros((n_b, n_s, npix), dtype=np.float32)
    comb = np.zeros(npix)
    comb_raw = np.zeros(npix)
    halves = np.zeros((2, npix))
    # Scale of the real surface's detail against the synthetic one's, per
    # band: real vs simulated cross-session detail power (the sessions'
    # noise is independent, so it drops out of the cross power).
    for b in range(n_b):
        num = den = 0.0
        for i, si in enumerate(names):
            for sj in names[i + 1:]:
                both = (np.isfinite(Bm[si][b]) & np.isfinite(Bm[sj][b])
                        & np.isfinite(Bs[si][b]) & np.isfinite(Bs[sj][b]))
                num += float(np.sum((Bm[si][b] * Bm[sj][b])[both]))
                den += float(np.sum((Bs[si][b].astype(np.float32) * Bs[sj][b])[both]))
        if den > 0:
            kappa[b] = num / den
    del Bs
    Bt = bands(truth, np.ones(npix, dtype=bool), ("t",), sm_)
    sm_.cache.clear()
    for b in range(n_b):
        present = [np.isfinite(Bm[s][b]) for s in names]
        Nv = [local_noise(Bh0[s][b], Bh1[s][b], sm_, ("n", s, b), b) for s in names]
        NV[b] = np.array(Nv)
        if n_s == 1:
            w = present[0][None].astype(np.float32)
            var = Nv[0]
            cb = Bm[names[0]][b]
            print(f"  band {BAND_NAMES[b]:>9s}: one session, nothing to weigh", flush=True)
        else:
            L = LOCAL_KM[b]
            sm_mask = {s: np.isfinite(Be[s][b]) for s in names}
            E = {s: np.where(sm_mask[s], np.nan_to_num(Be[s][b]), 0.0) for s in names}
            D = {s: np.maximum(sm_.coverage(sm_mask[s], ("e", s, b), L), 1e-3) for s in names}
            G = [[None] * n_s for _ in names]
            for i, si in enumerate(names):
                for j, sj in enumerate(names):
                    if j < i:
                        G[i][j] = G[j][i]
                        continue
                    G[i][j] = kappa[b] * sm_.plain(E[si] * E[sj], L) / np.sqrt(D[si] * D[sj])
            # Over the whole disk, in units of the surface's own detail (rms):
            # each session's geometry error and noise, and how strongly the
            # sessions' geometry errors are shared (correlation).
            print(f"  band {BAND_NAMES[b]:>9s}: KAPPA {kappa[b]:.3g}; error / surface detail (rms):",
                  flush=True)
            for i, si in enumerate(names):
                P_surf = float(np.mean(Bt[b][sm_mask[si]] ** 2))
                geo = np.sqrt(np.mean(E[si][sm_mask[si]] ** 2) / P_surf)
                noi = np.sqrt(np.nanmean(Nv[i][present[i]]) / (kappa[b] * P_surf))
                corr = []
                for sj in names:
                    both = sm_mask[si] & sm_mask[sj]
                    corr.append(np.sum(E[si][both] * E[sj][both])
                                / np.sqrt(np.sum(E[si][both] ** 2) * np.sum(E[sj][both] ** 2)))
                print(f"    {si:14s} geometry {geo:6.3f}  noise {noi:6.3f}  geometry-error correlation "
                      + " ".join(f"{c:+.2f}" for c in corr), flush=True)
            cb, w, var = combine([Bm[s][b] for s in names], G, Nv, present)
            del G, E, D
        W[b] = w
        VAR[b] = var
        comb += np.nan_to_num(cb)
        for i, s in enumerate(names):
            comb_raw += w[i] * np.where(present[i], np.nan_to_num(Braw[s][b]), 0.0)
        # The two half-stacks, each combined with the same weights (renormalized
        # over the sessions whose half has a value): validation's real test.
        for k, Bh in enumerate((Bh0, Bh1)):
            hk = [Bh[s][b].astype(np.float32) for s in names]
            wk = np.where([np.isfinite(x) for x in hk], w.astype(np.float64), 0.0)
            wk /= np.maximum(wk.sum(axis=0), 1e-12)
            halves[k] += sum(wk[i] * np.nan_to_num(hk[i]) for i in range(n_s))
        if n_s > 1:
            share = {s: float(np.mean(w[i][present[i]])) for i, s in enumerate(names)}
            print("    mean weight where present: "
                  + ", ".join(f"{s} {v:.2f}" for s, v in share.items()), flush=True)
        sm_.cache.clear()
    # A place is covered where some session has its broadest band.
    covered = np.any([np.isfinite(Bm[s][-1]) for s in names], axis=0)
    comb[~covered] = np.nan
    comb_raw[~covered] = np.nan
    halves[:, ~covered] = np.nan

    for suffix, mean in (("", comb_raw), ("_scatnorm", comb)):
        tag = f"{label}_geomweighted{suffix}"
        np.save(os.path.join(args.out_dir, f"stacked_linear_{tag}.npy"), mean.astype(np.float32))
        g = np.full(npix, hp.UNSEEN, dtype=np.float32)
        pos = np.isfinite(mean)
        pos[pos] = mean[pos] > 0
        g[pos] = 0.5 * np.log(mean[pos])
        np.save(os.path.join(args.out_dir, f"stacked_map_{tag}.npy"), g)
        valid = g != hp.UNSEEN
        vmin, vmax = np.percentile(g[valid], [5, 99.5])
        dea.save_lunar_image(g, os.path.join(args.out_dir, f"stacked_map_{tag}.png"),
                             f"{tag} stack ({len(rows)} looks, sessions weighted by "
                             "geometry and noise)", vmin=vmin, vmax=vmax)
        print(f"[{tag}] {np.mean(~pos[covered]):.1%} of covered pixels have no positive mean", flush=True)
    session_looks = np.array([ref_looks[s] for s in names])
    np.savez_compressed(os.path.join(args.out_dir, f"combination_{label}.npz"),
        sessions=np.array(names), bands=np.array(BAND_NAMES), scales_km=np.array(SCALES_KM),
        local_km=np.array(LOCAL_KM), kappa=kappa, weights=W, variance=VAR, noise_variance=NV,
        levels=np.array(levels, dtype=np.float64), level_of_band=np.array(lob),
        footprint_cuts=mode == "cuts", level1=mode, half_combined=halves.astype(np.float32),
        session_mean=np.array([ref_mean[s] for s in names]),
        half0=np.array([ref_h0[s] for s in names]), half1=np.array([ref_h1[s] for s in names]),
        session_looks=session_looks, equal_weight_looks=session_looks.sum(axis=0),
        offsets=np.array([offsets[s] for s in names]), sim_dir=os.path.abspath(args.sim_dir))
    print(f"Wrote the {label}_geomweighted outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
