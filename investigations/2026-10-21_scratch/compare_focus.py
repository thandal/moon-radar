"""Focused vs unfocused (block method, no drift correction), nside 800, the
8 long looks per channel: detail that repeats between independent halves
(09:35 + 09:46 vs 09:41 + 10:16, both stations), per feature size, overall
and split by direction (wave crests across / along the Doppler direction).
Stacked with stack_maps' radiometry (per unit area, noise removed, weighted)."""
import json, sys
import numpy as np, healpy as hp
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
import tiles as T
import compare800b as C
import stack_maps as sm
import doppler_equator_alignment as dea

BANDS = [(40, 20), (20, 12), (12, 8), (8, 6), (6, 4.4)]
n, st = int(T.TILE / T.PX), int(T.STEP / T.PX)
f = np.fft.fftfreq(n, d=T.PX)
KX, KY = np.meshgrid(f, f)
with np.errstate(divide="ignore"):
    WL = np.where(np.hypot(KX, KY) > 0, 1 / np.hypot(KX, KY), np.inf)
ang = np.degrees(np.arctan2(KY, KX))
e = T.EQ_DEG
d_along = np.abs((ang - e + 90) % 180 - 90)          # wavevector along the equator (Doppler direction)
SECTORS = {"all": np.ones_like(WL, bool), "Doppler direction": d_along < 30,
           "across it": d_along > 60}

import glob
rows = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + '/*/rows.json'))]
print(len(rows), 'looks')
npix = hp.nside2npix(800); cache = {}
HALF_A = {"09:35:24", "09:46:36"}
for chan in ("chan1", "chan0"):
    res = {}
    for ver in ("alpha0", "focused"):
        looks = []
        for pair in rows:
            r = pair[ver]
            if not r["rx_file"].endswith(f"{chan}.sigmf-meta"):
                continue
            rx = r["rx_file"].split("_")[0].upper()
            cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), npix, cache, rx_name=rx)
            I, nz = sm.look_brightness(r, cosi, max_mult_factor=12.0)   # nside 800: 4x the pixels per cell
            looks.append(dict(t=r["rx_start_utc"][11:19], I=I.astype(np.float32), n=nz.astype(np.float32), cosi=cosi))
        tot = sm.Accumulator(npix)
        for L in looks:
            tot.add(L["I"])
        signal = sm.smooth_map(tot.mean(min_count=3))
        for L in looks:
            others = np.full(npix, np.nan); ok = np.isfinite(L["I"]) & (tot.count >= 3)
            others[ok] = (tot.sum[ok] - L["I"][ok]) / (tot.count[ok] - 1)
            L["K"], _ = sm.independent_samples(L["I"], others, L["cosi"], L["n"])
            L["w"] = L["K"] / (signal + L["n"]) ** 2
        halves = []
        for inA in (True, False):
            acc = sm.Accumulator(npix)
            for L in looks:
                if (L["t"] in HALF_A) == inA:
                    acc.add(L["I"], L["w"])
            halves.append(T.disk_grid(acc.stacked_log(3)))
        res[ver] = (halves, [round(L["K"], 2) for L in looks])
        del looks
    common = C.good_tiles(*(h for v in res.values() for h in v[0]))
    print(f"\n===== {chan} ({'opposite sense' if chan == 'chan1' else 'same sense'}), {len(common)} tiles =====")
    for ver in res:
        print(f"  K per look, {ver}: {res[ver][1]}")
    for sname, sector in SECTORS.items():
        C.BANDS = BANDS
        C.SEL = [(WL <= hi) & (WL > lo) & sector for hi, lo in BANDS]
        print(f"  {sname:18s} " + " ".join(f"{f'{a:g}-{b:g} km':>9s}" for a, b in BANDS))
        cs = {}
        for ver in res:
            a, b = res[ver][0]
            cs[ver] = C.band_corr(a, b, common)
            print(f"  {ver:>18s} " + " ".join(f"{x:9.3f}" for x in cs[ver]))
        snr = lambda c: c / (1 - c)
        print(f"  {'focused/unfocused':>18s} " + " ".join(f"{x:8.2f}x" for x in snr(cs["focused"]) / snr(cs["alpha0"])))
        a, b = res["focused"][0]
        cn = C.band_corr(a, np.roll(b, (37, 37), axis=(0, 1)), C.good_tiles(a, np.roll(b, (37, 37), axis=(0, 1))))
        print(f"  {'chance (shifted)':>18s} " + " ".join(f"{x:9.3f}" for x in cn))
