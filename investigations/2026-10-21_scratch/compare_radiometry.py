"""Equal-weight median-normalized stacking (before) vs brightness per unit
area, noise removed, inverse-variance weights (now): half-stack correlation
per feature size on the same tiles, all looks, halves by alternating time."""
import csv, os, sys
import numpy as np, healpy as hp
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
import tiles as T
import compare800b as C
import stack_maps as sm
import doppler_equator_alignment as dea

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
C.BANDS = [(40, 20), (20, 12), (12, 9)]
C.SEL = [(C.WL <= hi) & (C.WL > lo) for hi, lo in C.BANDS]
npix = hp.nside2npix(400); cache = {}
for chan in ("chan1", "chan0"):
    rows = []
    for st in ("stockert", "wsrt"):
        for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_{chan}.csv")):
            if float(r.get("pair_snr") or r["tone_snr"]) >= 15:
                r["_rx"] = st.upper(); r["_et"] = dea.csp.str2et(r["rx_start_utc"]); rows.append(r)
    looks = {}
    for r in rows:
        cosi = sm.incidence_cos(r["_et"], npix, cache, rx_name=r["_rx"])
        I, n = sm.look_brightness(r, cosi)
        looks[(r["_rx"], r["rx_start_utc"][11:19])] = dict(
            old=sm.look_linear_intensity(r["map_npy"], r["mult_npy"]), new=I, n=n, cosi=cosi)
    tot = sm.Accumulator(npix)
    for L in looks.values():
        tot.add(L["new"])
    signal = sm.smooth_map(tot.mean(min_count=3))
    for k, L in looks.items():
        others = np.full(npix, np.nan); ok = np.isfinite(L["new"]) & (tot.count >= 3)
        others[ok] = (tot.sum[ok] - L["new"][ok]) / (tot.count[ok] - 1)
        L["K"], _ = sm.independent_samples(L["new"], others, L["cosi"], L["n"])
        L["w"] = L["K"] / (signal + L["n"]) ** 2
    times = sorted({t for _, t in looks})
    half = {h: [k for k in looks if times.index(k[1]) % 2 == h] for h in (0, 1)}

    def stack(keys, how):
        acc = sm.Accumulator(npix)
        for k in keys:
            if how == "old":
                acc.add(looks[k]["old"])
            else:
                acc.add(looks[k]["new"], looks[k]["w"] if how == "new" else None)
        return acc.stacked_log(3)

    hows = [("old", "before: equal weights, intensity"), ("area", "per unit area, noise removed, equal weights"),
            ("new", "per unit area, noise removed, weighted")]
    gA = {h: T.disk_grid(stack(half[0], h)) for h, _ in hows}
    gB = {h: T.disk_grid(stack(half[1], h)) for h, _ in hows}
    common = C.good_tiles(*gA.values(), *gB.values())
    print(f"\n===== {chan}: halves of {len(half[0])} + {len(half[1])} looks, same {len(common)} tiles =====")
    print(f"{'feature size, km':>46s} " + " ".join(f"{f'{a:g}-{b:g}':>8s}" for a, b in C.BANDS) + "   signal-to-noise vs before")
    base = None
    for h, label in hows:
        c = C.band_corr(gA[h], gB[h], common); snr = c / (1 - c); base = snr if base is None else base
        print(f"{label:>46s} " + " ".join(f"{x:8.3f}" for x in c) + "   " + " ".join(f"{x:5.2f}x" for x in snr / base))
