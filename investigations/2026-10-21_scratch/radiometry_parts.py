"""Which part of the new radiometry helps or hurts: area, noise removal,
weights, each alone and together; measured on log brightness (as before)
and on linear contrast (brightness / its 40 km local mean - 1)."""
import csv, os, sys
import numpy as np, healpy as hp, scipy.ndimage
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
import tiles as T
import compare800b as C
import stack_maps as sm
import doppler_equator_alignment as dea

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
C.BANDS = [(40, 20), (20, 12), (12, 9)]
C.SEL = [(C.WL <= hi) & (C.WL > lo) for hi, lo in C.BANDS]
npix = hp.nside2npix(400); cache = {}
LO, HI = np.cos(np.radians(sm.GAIN_INCIDENCE_DEG[::-1]))


def variant(r, cosi, area_on, noise_on):
    m = np.load(r["map_npy"]); mult = np.load(r["mult_npy"]); A = np.load(r["area_npy"])
    mask = (m != hp.UNSEEN) & np.isfinite(A) & (A > 0) & (mult <= 3 * np.median(mult[mult > 0]))
    ref = np.median(m[mask]); N = np.exp(2 * (float(r["noise_log_amp"]) - ref))
    b = np.full(npix, np.nan); n = np.full(npix, np.nan)
    b[mask] = np.exp(2 * (m[mask] - ref)) - (N if noise_on else 0.0)
    n[mask] = N
    if area_on:
        b[mask] /= A[mask]; n[mask] /= A[mask]
    band = mask & (cosi >= LO) & (cosi <= HI)
    g = np.median(b[band])
    return b / g, n / g


def disk_contrast(hmap_log):
    """linear brightness / its >40 km local mean - 1, on the disk grid"""
    lin = np.where(hmap_log != hp.UNSEEN, np.exp(2.0 * hmap_log.astype(float)), np.nan)
    ok = np.isfinite(lin)
    v = hp.get_interp_val(np.where(ok, lin, 0.0), T.theta, T.phi).reshape(T.X.shape)
    w = hp.get_interp_val(ok.astype(float), T.theta, T.phi).reshape(T.X.shape)
    g = np.where((w > 0.99) & T.inside, v / np.maximum(w, 1e-9), np.nan)
    okg = np.isfinite(g); s = 20.0 / T.PX
    trend = scipy.ndimage.gaussian_filter(np.where(okg, g, 0), s) / np.maximum(
        scipy.ndimage.gaussian_filter(okg.astype(float), s), 1e-9)
    return np.where(okg & (trend > 0), g / trend - 1, np.nan)


for chan in ("chan1", "chan0"):
    rows = []
    for st in ("stockert", "wsrt"):
        for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_{chan}.csv")):
            if float(r.get("pair_snr") or r["tone_snr"]) >= 15:
                r["_rx"] = st.upper(); r["_et"] = dea.csp.str2et(r["rx_start_utc"]); rows.append(r)
    cos = {id(r): sm.incidence_cos(r["_et"], npix, cache, rx_name=r["_rx"]) for r in rows}
    times = sorted({r["rx_start_utc"][11:19] for r in rows})
    half = {h: [r for r in rows if times.index(r["rx_start_utc"][11:19]) % 2 == h] for h in (0, 1)}
    results = []
    for area_on, noise_on, weights_on in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)]:
        vals = {id(r): variant(r, cos[id(r)], area_on, noise_on) for r in rows}
        tot = sm.Accumulator(npix)
        for r in rows:
            tot.add(vals[id(r)][0])
        signal = sm.smooth_map(tot.mean(min_count=3))
        w = {}
        for r in rows:
            I, n = vals[id(r)]
            if weights_on:
                others = np.full(npix, np.nan); ok = np.isfinite(I) & (tot.count >= 3)
                others[ok] = (tot.sum[ok] - I[ok]) / (tot.count[ok] - 1)
                K, _ = sm.independent_samples(I, others, cos[id(r)], n)
                w[id(r)] = K / (signal + (n if noise_on else 0 * n)) ** 2
            else:
                w[id(r)] = None
        stacks = []
        for h in (0, 1):
            acc = sm.Accumulator(npix)
            for r in half[h]:
                acc.add(vals[id(r)][0], w[id(r)])
            stacks.append(acc.stacked_log(3))
        results.append(((area_on, noise_on, weights_on), stacks))
    gl = [(k, T.disk_grid(a), T.disk_grid(b)) for k, (a, b) in results]
    gc = [(k, disk_contrast(a), disk_contrast(b)) for k, (a, b) in results]
    common = C.good_tiles(*[x for _, a, b in gl for x in (a, b)], *[x for _, a, b in gc for x in (a, b)])
    print(f"\n===== {chan}: {len(half[0])} + {len(half[1])} looks, {len(common)} tiles; "
          "correlation for features 40-20 / 20-12 / 12-9 km =====")
    print(f"{'area  noise  weights':>22s} {'on log brightness':>28s} {'on linear contrast':>28s}")
    for (k, a, b), (_, ca, cb) in zip(gl, gc):
        c1 = C.band_corr(a, b, common); c2 = C.band_corr(ca, cb, common)
        print(f"{'  '.join('yes' if x else ' no' for x in k):>22s} {' '.join(f'{x:8.3f}' for x in c1):>28s} {' '.join(f'{x:8.3f}' for x in c2):>28s}")
