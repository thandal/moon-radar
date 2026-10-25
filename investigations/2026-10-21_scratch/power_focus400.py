"""Total vs repeatable fine-scale power, nside 400 long looks: production,
block (no drift correction), focused [, flipped]. Total = mean auto power of
the two half-stacks; repeatable = their cross power; noise = the rest. Per
feature size and direction, linear contrast and log brightness."""
import csv, sys
import numpy as np, healpy as hp
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
import tiles as T
import stack_maps as sm
import doppler_equator_alignment as dea
src = open("/tmp/eve_check/blur/compare_focus400.py").read()
exec(src[:src.index("cache = {}")])        # constants, grid_linear, good_tiles
RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
DIRS = {"production": RUN, "block, no drift correction": "/var/tmp/focus400/run_block",
        "focused": "/home/than/code/moon-radar/results/EVE2026_CIRC_FOCUSED"}
if len(sys.argv) > 1:
    DIRS["focused, sign flipped"] = sys.argv[1]
BANDS = [(40, 20), (20, 12), (12, 9), (9, 6)]


def powers(a, b, tiles, sector):
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
    return (pa + pb) / 2, cross


cache = {}
for chan in ("chan1", "chan0"):
    grids = {}
    for ver, d in DIRS.items():
        looks = []
        for st in ("stockert", "wsrt"):
            for r in csv.DictReader(open(f"{d}/registration_runs_{st}_{chan}.csv")):
                if float(r["tx_s"]) > 100 and float(r.get("pair_snr") or r["tone_snr"]) >= 15:
                    for k in ("map_npy", "mult_npy", "area_npy"):
                        r[k] = sm.resolve_npy(r[k], RUN)
                    rx = r["rx_file"].split("_")[0].upper()
                    cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), npix, cache, rx_name=rx)
                    looks.append((r["rx_start_utc"][11:19], sm.look_brightness(r, cosi)))
        g = []
        for inA in (True, False):
            acc = sm.Accumulator(npix)
            for t, I in looks:
                if (t in HALF_A) == inA:
                    acc.add(I)
            mean = np.where(acc.count >= 2, acc.sum / np.maximum(acc.count, 1), np.nan)
            g.append(grid_linear(mean))
        grids[ver] = g
        if ver == "production":                  # brightness of each tile, for the bright subset
            acc = sm.Accumulator(npix)
            for t, I in looks:
                acc.add(I)
            mean = np.where(acc.count >= 2, acc.sum / np.maximum(acc.count, 1), np.nan)
            ok = np.isfinite(mean)
            v = hp.get_interp_val(np.where(ok, mean, 0.0), T.theta, T.phi).reshape(T.X.shape)
            w = hp.get_interp_val(ok.astype(float), T.theta, T.phi).reshape(T.X.shape)
            bright_grid = np.where((w > 0.99) & T.inside, v / np.maximum(w, 1e-9), np.nan)
    tiles = good_tiles(*(h for v in grids.values() for h in v))
    tb = np.array([np.nanmean(bright_grid[i0:i0 + nt, j0:j0 + nt]) for i0, j0 in tiles])
    bright_tiles = [t for t, b in zip(tiles, tb) if b >= np.percentile(tb, 67)]
    print(f"\n===== {chan} ({'opposite sense' if chan == 'chan1' else 'same sense'}), linear contrast, {len(tiles)} tiles; "
          "power relative to production =====")
    for sname, sector, tset in [(k, v, tiles) for k, v in SECTORS.items()] + \
            [("brightest third", SECTORS["all directions"], bright_tiles),
             ("brightest, along Dop.", SECTORS["along Doppler"], bright_tiles)]:
        base = powers(*grids["production"], tset, sector)
        print(f"  {sname:20s}" + "".join(f"{f'{a:g}-{b:g} km':>27s}" for a, b in BANDS))
        print(f"  {'':20s}" + "    total  repeat.   noise" * len(BANDS))
        for ver, g in grids.items():
            tot, rep = powers(*g, tset, sector)
            cells = [f"{tot[k] / base[0][k]:9.2f}{rep[k] / base[1][k]:9.2f}{(tot[k] - rep[k]) / (base[0][k] - base[1][k]):8.2f}"
                     for k in range(len(BANDS))]
            print(f"  {ver[:20]:>20s}" + "".join(cells))
