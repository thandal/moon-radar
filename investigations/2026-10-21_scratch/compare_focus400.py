"""nside 400: production vs block method without drift correction vs focused,
the 8 long looks per channel (both stations). Repeatable detail between
independent halves (09:35 + 09:46 vs 09:41 + 10:16) per feature size, overall
and by direction, on log brightness and on linear contrast. Each version is
stacked with stack_maps' radiometry (per unit area, noise removed, equal
weights); the block and focused looks use the run directories written by
make_focused_run.py (production's noise floor carried over)."""
import csv, glob, json, sys
import numpy as np, healpy as hp, scipy.ndimage
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
import tiles as T
import stack_maps as sm
import doppler_equator_alignment as dea

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
FOCUS = "/var/tmp/focus400"
BANDS = [(40, 20), (20, 12), (12, 9)]
nt, stp = int(T.TILE / T.PX), int(T.STEP / T.PX)
TAPER = np.outer(np.hanning(nt), np.hanning(nt))
f = np.fft.fftfreq(nt, d=T.PX)
KX, KY = np.meshgrid(f, f)
with np.errstate(divide="ignore"):
    WL = np.where(np.hypot(KX, KY) > 0, 1 / np.hypot(KX, KY), np.inf)
ang = np.degrees(np.arctan2(KY, KX))
d_along = np.abs((ang - T.EQ_DEG + 90) % 180 - 90)    # wavevector along the equator = Doppler direction
SECTORS = {"all directions": np.ones_like(WL, bool), "along Doppler": d_along < 30, "along delay": d_along > 60}
ORIGINS = [(i0, j0) for i0 in range(0, len(T.xs) - nt + 1, stp) for j0 in range(0, len(T.xs) - nt + 1, stp)]
HALF_A = {"09:35:24", "09:46:36"}
npix = hp.nside2npix(400)


def grid_linear(mean):
    """Linear contrast on the disk grid: brightness / its >40 km trend - 1."""
    ok = np.isfinite(mean)
    v = hp.get_interp_val(np.where(ok, mean, 0.0), T.theta, T.phi).reshape(T.X.shape)
    w = hp.get_interp_val(ok.astype(float), T.theta, T.phi).reshape(T.X.shape)
    g = np.where((w > 0.99) & T.inside, v / np.maximum(w, 1e-9), np.nan)
    okg = np.isfinite(g); gv = np.where(okg, g, 0.0); s = 20.0 / T.PX
    trend = scipy.ndimage.gaussian_filter(gv, s) / np.maximum(scipy.ndimage.gaussian_filter(okg.astype(float), s), 1e-9)
    return np.where(okg & (trend > 0), g / trend - 1, np.nan)


def good_tiles(*grids):
    return [o for o in ORIGINS if all(np.isfinite(g[o[0]:o[0] + nt, o[1]:o[1] + nt]).mean() >= 0.95 for g in grids)]


def band_corr(a, b, tiles, sector):
    sel = [(WL <= hi) & (WL > lo) & sector for hi, lo in BANDS]
    cross = np.zeros(len(BANDS)); pa = np.zeros_like(cross); pb = np.zeros_like(cross)
    for i0, j0 in tiles:
        ta, tb = a[i0:i0 + nt, j0:j0 + nt], b[i0:i0 + nt, j0:j0 + nt]
        ok = np.isfinite(ta) & np.isfinite(tb)
        A = np.fft.fft2(np.where(ok, ta - np.nanmean(ta), 0.0) * TAPER)
        B = np.fft.fft2(np.where(ok, tb - np.nanmean(tb), 0.0) * TAPER)
        for k, s in enumerate(sel):
            cross[k] += np.real(A[s] * np.conj(B[s])).sum()
            pa[k] += (np.abs(A[s]) ** 2).sum(); pb[k] += (np.abs(B[s]) ** 2).sum()
    return cross / np.sqrt(pa * pb)


cache = {}
for chan in ("chan1", "chan0"):
    versions = {"production": [], "block, no drift correction": [], "focused": []}
    for st in ("stockert", "wsrt"):
        for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_{chan}.csv")):
            if float(r["tx_s"]) > 100 and float(r.get("pair_snr") or r["tone_snr"]) >= 15:
                for k in ("map_npy", "mult_npy", "area_npy"):
                    r[k] = sm.resolve_npy(r[k], RUN)
                versions["production"].append(r)
    for ver, d in (("block, no drift correction", "/var/tmp/focus400/run_block"),
                   ("focused", "/home/than/code/moon-radar/results/EVE2026_CIRC_FOCUSED")):
        for st in ("stockert", "wsrt"):
            for r in csv.DictReader(open(f"{d}/registration_runs_{st}_{chan}.csv")):
                if float(r["tx_s"]) > 100 and float(r.get("pair_snr") or r["tone_snr"]) >= 15:
                    versions[ver].append(r)
    grids = {}
    for ver, rows in versions.items():
        looks = []
        for r in rows:
            rx = r["rx_file"].split("_")[0].upper()
            cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), npix, cache, rx_name=rx)
            looks.append(dict(t=r["rx_start_utc"][11:19], I=sm.look_brightness(r, cosi)))
        g = {"log": [], "linear": []}
        for inA in (True, False):
            acc = sm.Accumulator(npix)
            for L in looks:
                if (L["t"] in HALF_A) == inA:
                    acc.add(L["I"])
            g["log"].append(T.disk_grid(acc.stacked_log(2)))
            mean = np.where(acc.count >= 2, acc.sum / np.maximum(acc.count, 1), np.nan)
            g["linear"].append(grid_linear(mean))
        grids[ver] = g
        print(f"{chan} {ver}: {len(looks)} looks", flush=True)
        del looks
    for scale in ("log", "linear"):
        tiles = good_tiles(*(h for v in grids.values() for h in v[scale]))
        print(f"\n===== {chan} ({'opposite sense' if chan == 'chan1' else 'same sense'}), {scale} brightness, "
              f"{len(tiles)} tiles: half-stack correlation per feature size =====")
        for sname, sector in SECTORS.items():
            print(f"  {sname:28s}" + "".join(f"{f'{a:g}-{b:g} km':>10s}" for a, b in BANDS))
            cs = {}
            for ver, g in grids.items():
                cs[ver] = band_corr(*g[scale], tiles, sector)
                print(f"  {ver:>28s}" + "".join(f"{x:10.3f}" for x in cs[ver]))
            snr = lambda c: c / (1 - c)
            print(f"  {'focused / production':>28s}" + "".join(f"{x:9.2f}x" for x in snr(cs["focused"]) / snr(cs["production"])))
            a, b = grids["focused"][scale]
            bs = np.roll(b, (37, 37), axis=(0, 1))
            cn = band_corr(a, bs, good_tiles(a, bs), sector)
            print(f"  {'chance (shifted halves)':>28s}" + "".join(f"{x:10.3f}" for x in cn))
