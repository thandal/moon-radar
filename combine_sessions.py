"""
Stack looks in two levels, so that many looks sharing one geometry cannot
outvote fewer looks that see a place better.

Each look's map holds the surface plus two kinds of error. Speckle and
receiver noise change from look to look, so averaging any looks reduces
them. Geometry error -- mirror ghosts, and blur where the look's
delay-Doppler cells are large (near its specular point and its Doppler
equator) -- is fixed by the geometry, so looks with the same geometry
repeat the same error and averaging them does not reduce it.

Level 1, within a session (one night's looks, as stack_maps.session_of;
its specular point hardly moves): the plain mean of its looks, as
stack_maps stacks them (scattering-law-normalized brightness, the session
shifts of the equal-weight stack applied). Its noise is measured from its
own looks: half the difference of two interleaved half-session means
(alternate captures in time; both channels of a capture in the same half),
which cancels the shared geometry error and keeps the noise of the full
mean.

Level 2, across sessions: at each place and for each range of feature
sizes separately (a look with large cells still measures large features),
the sessions are combined with the weights that minimize the expected
error, given the covariance of their errors: noise (independent between
sessions) plus geometry error, which can be shared (the three 2025
sessions' specular points are within a few degrees of each other). The
geometry error is not guessed: simulate_geometry.py passes a synthetic
surface through every look's own geometry, the simulated looks are
stacked the same way, and each session's simulated mean minus the
synthetic surface is its geometry error. The products of these errors,
averaged over a neighbourhood, give the error covariance between sessions;
it is scaled to the real surface by the ratio of real to simulated
cross-session detail power in each band (KAPPA, printed). The weights are
zero or positive and sum to one at every place (see combine: unconstrained,
the minimum subtracts a session to cancel shared blur); a session that does
not cover a place gets none.
1 / (sum of the inverse covariance) is the combination's expected error
variance, written as an uncertainty map per band.

Feature-size bands: differences of Gaussian smoothings at SCALES_KM (full
width at half maximum), each smoothing normalized by the smoothed coverage
(so a map's edge does not pull it towards zero); the bands add back up to
the map exactly. Neighbourhoods for the error statistics: Gaussian, full
width LOCAL_KM per band.

Channels as stack_maps.py --chans: one, or the opposite- and same-sense channels together
(dual), each with its own scattering law. The simulated twins are found
by capture, so one simulation serves both channels (the geometry is the
same; simulate_geometry.py is run on the opposite-sense rows).

Outputs in --out-dir, named as stack_maps names them with _geomweighted:
stacked_linear_<label>_geomweighted[_scatnorm].npy (mean brightness),
stacked_map_<label>_geomweighted[_scatnorm].npy/.png (log, as stack_maps),
and combination_<label>.npz: session names, per-band weights, the
combination's expected error variance and each session's noise variance,
KAPPA, the session means and half-session means (for
validate_session_combination.py), and each session's look counts.

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
RIDGE = 1e-6        # relative to the mean diagonal: keeps every solve well posed


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


def session_sums(rows, laws, offsets, cache, npix, sim=False):
    """Per session: sums and counts of the looks' normalized brightness
    (all looks, and the two interleaved halves: alternate captures in time,
    both channels of a capture in the same half), and of raw brightness."""
    out = {}
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["rx_start_utc"], rows[i]["rx_file"]))
    half_of = {}
    for i in order:
        c = capture_of(rows[i]["rx_file"])
        s = sm.session_of(rows[i]["rx_file"])
        if c not in half_of:
            half_of[c] = sum(sm.session_of(k) == s for k in half_of) % 2
    for i in order:
        r = rows[i]
        s = sm.session_of(r["rx_file"])
        cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), npix, cache, rx_name=r["_rx"])
        I, _ = sm.look_brightness(r, cosi)
        N = I / laws[r["_chan"]].value(cosi)
        if not sim:
            N = sm.shift_intensity(N, *offsets[s])
            I = sm.shift_intensity(I, *offsets[s])
        if s not in out:
            out[s] = {k: np.zeros(npix) for k in ("sum", "raw", "sum0", "sum1")}
            out[s].update({k: np.zeros(npix, dtype=np.int32) for k in ("n", "n0", "n1")})
        h = half_of[capture_of(r["rx_file"])]
        ok = np.isfinite(N)
        out[s]["sum"][ok] += N[ok]
        out[s]["n"][ok] += 1
        okr = np.isfinite(I)
        out[s]["raw"][okr] += I[okr]
        out[s][f"sum{h}"][ok] += N[ok]
        out[s][f"n{h}"][ok] += 1
    return out


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


def fit_laws(rows, cache, npix, chans):
    laws = {c: sm.ScatterLaw() for c in chans}
    for r in rows:
        cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), npix, cache, rx_name=r["_rx"])
        I, _ = sm.look_brightness(r, cosi)
        laws[r["_chan"]].collect(I, cosi)
    for law in laws.values():
        law.fit()
    return laws


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
    p.add_argument("--write-law", default=None,
                   help="only write the scattering law of the stack's looks (bin centres and "
                        "values, as stack_maps fits it) to this .npz, for simulate_geometry.py --law")
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
        law = fit_laws(rows, {}, len(np.load(rows[0]["map_npy"], mmap_mode="r")), [c])[c]
        np.savez(args.write_law, centers=law.centers, edges=law.edges, law=law.law)
        print(f"wrote the {c} scattering law of {len(rows)} looks to {args.write_law}")
        return
    if args.sim_dir is None:
        raise SystemExit("--sim-dir is required")
    # The simulated twin of every look: same row, simulated map, no noise.
    srows, truth = simulated_twins(rows, args.sim_dir, args.stations)

    npix = len(truth)
    nside = hp.npix2nside(npix)
    cache = {}
    print(f"{len(rows)} looks; fitting scattering laws (real and simulated)", flush=True)
    laws = fit_laws(rows, cache, npix, args.chans)
    laws_sim = fit_laws(srows, cache, npix, args.chans)
    real = session_sums(rows, laws, offsets, cache, npix)
    sim = session_sums(srows, laws_sim, offsets, cache, npix, sim=True)
    names = sorted(real)
    print("sessions: " + ", ".join(f"{s} ({int(np.median(real[s]['n'][real[s]['n'] > 0]))} looks)"
                                   for s in names), flush=True)

    sm_ = Smoother(nside)
    M = {s: mean_of(real[s]["sum"], real[s]["n"], args.min_count) for s in names}
    Mraw = {s: mean_of(real[s]["raw"], real[s]["n"], args.min_count) for s in names}
    H0 = {s: mean_of(real[s]["sum0"], real[s]["n0"], args.min_count) for s in names}
    H1 = {s: mean_of(real[s]["sum1"], real[s]["n1"], args.min_count) for s in names}
    S = {s: mean_of(sim[s]["sum"], sim[s]["n"], args.min_count) for s in names}
    mask = {s: np.isfinite(M[s]) for s in names}
    nmask = {s: np.isfinite(H0[s]) & np.isfinite(H1[s]) for s in names}
    smask = {s: np.isfinite(S[s]) for s in names}
    for s in names:
        print(f"  {s}: simulated session mean / surface over its pixels "
              f"{np.nanmean(S[s]) / np.mean(truth[smask[s]]):.3f}", flush=True)

    print("splitting into feature-size bands", flush=True)
    Bm = {s: bands(M[s], mask[s], ("m", s), sm_) for s in names}
    Braw = {s: bands(Mraw[s], mask[s], ("m", s), sm_) for s in names}
    Bn = {s: bands((H0[s] - H1[s]) / 2, nmask[s], ("n", s), sm_) for s in names}
    Bs = {s: bands(S[s], smask[s], ("s", s), sm_) for s in names}
    # Geometry error per band (bands are linear for a fixed coverage).
    Be = {s: bands(S[s] - truth, smask[s], ("s", s), sm_) for s in names}
    session_looks = np.array([real[s]["n"] for s in names])
    del real, sim
    sm_.cache.clear()

    n_b = len(BAND_NAMES)
    kappa = np.zeros(n_b)
    W = np.zeros((n_b, len(names), npix), dtype=np.float16)
    VAR = np.zeros((n_b, npix), dtype=np.float32)
    NV = np.zeros((n_b, len(names), npix), dtype=np.float32)
    comb = np.zeros(npix)
    comb_raw = np.zeros(npix)
    present = [mask[s] for s in names]
    # Scale of the real surface's detail against the synthetic one's, per
    # band: real vs simulated cross-session detail power (the sessions'
    # noise is independent, so it drops out of the cross power).
    for b in range(n_b):
        num = den = 0.0
        for i, si in enumerate(names):
            for sj in names[i + 1:]:
                both = mask[si] & mask[sj] & smask[si] & smask[sj]
                num += float(np.nansum((Bm[si][b] * Bm[sj][b])[both]))
                den += float(np.nansum((Bs[si][b] * Bs[sj][b])[both]))
        kappa[b] = num / den
    del Bs
    Bt = bands(truth, np.ones(npix, dtype=bool), ("t",), sm_)
    sm_.cache.clear()
    for b in range(n_b):
        L = LOCAL_KM[b]
        E = {s: np.where(smask[s], np.nan_to_num(Be[s][b]), 0.0) for s in names}
        D = {s: np.maximum(sm_.coverage(smask[s], ("s", s), L), 1e-3) for s in names}
        G = [[None] * len(names) for _ in names]
        for i, si in enumerate(names):
            for j, sj in enumerate(names):
                if j < i:
                    G[i][j] = G[j][i]
                    continue
                G[i][j] = kappa[b] * sm_.plain(E[si] * E[sj], L) / np.sqrt(D[si] * D[sj])
        Nv = []
        for s in names:
            n2 = np.where(nmask[s], np.nan_to_num(Bn[s][b]) ** 2, 0.0)
            Nv.append(sm_.plain(n2, L) / np.maximum(sm_.coverage(nmask[s], ("n", s), L), 1e-3))
        NV[b] = np.array(Nv)
        # Over the whole disk, in units of the surface's own detail (rms):
        # each session's geometry error and noise, and how strongly the
        # sessions' geometry errors are shared (correlation).
        print(f"  band {BAND_NAMES[b]:>9s}: KAPPA {kappa[b]:.3g}; error / surface detail (rms):", flush=True)
        for i, si in enumerate(names):
            P_surf = float(np.mean(Bt[b][smask[si]] ** 2))
            geo = np.sqrt(np.mean(E[si][smask[si]] ** 2) / P_surf)
            noi = np.sqrt(np.mean(Nv[i][nmask[si]]) / (kappa[b] * P_surf))
            corr = []
            for sj in names:
                both = smask[si] & smask[sj]
                corr.append(np.sum(E[si][both] * E[sj][both])
                            / np.sqrt(np.sum(E[si][both] ** 2) * np.sum(E[sj][both] ** 2)))
            print(f"    {si:14s} geometry {geo:6.3f}  noise {noi:6.3f}  geometry-error correlation "
                  + " ".join(f"{c:+.2f}" for c in corr), flush=True)
        cb, w, var = combine([Bm[s][b] for s in names], G, Nv, present)
        W[b] = w
        VAR[b] = var
        comb += np.nan_to_num(cb)
        rb = np.zeros(npix)
        for i, s in enumerate(names):
            rb += w[i] * np.where(present[i], np.nan_to_num(Braw[s][b]), 0.0)
        comb_raw += rb
        share = {s: float(np.mean(w[i][present[i]])) for i, s in enumerate(names)}
        print("    mean weight where present: "
              + ", ".join(f"{s} {v:.2f}" for s, v in share.items()), flush=True)
        sm_.cache.clear()
    covered = np.any(present, axis=0)
    comb[~covered] = np.nan
    comb_raw[~covered] = np.nan

    n_looks = session_looks.sum(axis=0)
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
    np.savez_compressed(os.path.join(args.out_dir, f"combination_{label}.npz"),
        sessions=np.array(names), bands=np.array(BAND_NAMES), scales_km=np.array(SCALES_KM),
        local_km=np.array(LOCAL_KM), kappa=kappa, weights=W, variance=VAR, noise_variance=NV,
        session_mean=np.array([M[s] for s in names]),
        half0=np.array([H0[s] for s in names]), half1=np.array([H1[s] for s in names]),
        session_looks=session_looks,
        equal_weight_looks=n_looks, offsets=np.array([offsets[s] for s in names]),
        sim_dir=os.path.abspath(args.sim_dir))
    print(f"Wrote the {label}_geomweighted outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
