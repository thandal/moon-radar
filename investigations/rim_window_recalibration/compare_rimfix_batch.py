"""Compare the RIMFIX batch against the frozen LOLA_DEM_REGISTRATION run.

Reports per-session chain-offset (applied_df + rim_delta) health, δ changes,
recovered/censored looks, spread stats, and the registration offsets/closure.
"""
import csv
import os
import sys

import numpy as np

REPO = "/home/than/code/moon-radar"
os.chdir(REPO)

FROZEN = "results/LOLA_DEM_REGISTRATION"
RIMFIX = "results/LOLA_DEM_REGISTRATION_RIMFIX"


def sess_of(f):
    if "06_21" in f:
        return "06-21"
    if "09_16" in f:
        return "09-16"
    return "09-10" if "_2025_09_10_" in f else "09-11"


def load(run, chan):
    p = os.path.join(run, f"registration_runs_{chan}.csv")
    return {r["rx_file"]: r for r in csv.DictReader(open(p))}


for chan in ("chan1", "chan0"):
    old = load(FROZEN, chan)
    new = load(RIMFIX, chan)
    common = sorted(set(old) & set(new))
    print(f"\n================ {chan}: {len(old)} frozen, {len(new)} rimfix, "
          f"{len(common)} common")
    by_sess = {}
    for f in common:
        o, n = old[f], new[f]
        do, dn = float(o["rim_delta_hz"]), float(n["rim_delta_hz"])
        to = (float(o["applied_df_hz"]) + do) * 1e3
        tn = (float(n["applied_df_hz"]) + dn) * 1e3
        by_sess.setdefault(sess_of(f), []).append(
            (f, do * 1e3, dn * 1e3, to, tn, int(o["rim_n"]), int(n["rim_n"])))
    for s, v in sorted(by_sess.items()):
        to = np.array([x[3] for x in v])
        tn = np.array([x[4] for x in v])
        mo, mn = np.median(to), np.median(tn)
        mado = np.median(np.abs(to - mo))
        madn = np.median(np.abs(tn - mn))
        out_o = int(np.sum(np.abs(to - mo) > 20))
        out_n = int(np.sum(np.abs(tn - mn) > 20))
        fails_o = sum(1 for x in v if x[5] == 0)
        fails_n = sum(1 for x in v if x[6] == 0)
        dd = np.array([x[2] - x[1] for x in v])
        print(f"{s}: n={len(v):3d}  total MAD {mado:5.1f} -> {madn:5.1f} mHz  "
              f">20mHz outliers {out_o:2d} -> {out_n:2d}  rim fails "
              f"{fails_o} -> {fails_n}  |ddelta| median "
              f"{np.median(np.abs(dd)):5.1f} max {np.abs(dd).max():6.1f} mHz")
        worst = sorted(v, key=lambda x: -abs(x[4] - mn))[:3]
        for f, do_, dn_, to_, tn_, ro, rn in worst:
            if abs(tn_ - mn) > 20:
                print(f"    still off-trend: {f[15:34]} total {tn_:+7.1f} "
                      f"(delta {do_:+7.1f} -> {dn_:+7.1f}, rim_n {ro}->{rn})")

reg_new = os.path.join(RIMFIX, "registration_offsets.csv")
if os.path.exists(reg_new):
    print("\n================ registration offsets (chan1 grid solve)")
    for run in (FROZEN, RIMFIX):
        print(f"-- {run}")
        for r in csv.DictReader(open(os.path.join(run, "registration_offsets.csv"))):
            print(f"  {r['pair'][:52]:52s} dlon {float(r['dlon_deg']):+7.3f} "
                  f"dlat {float(r['dlat_deg']):+7.3f} deg  corr "
                  f"{float(r['peak_corr']):.3f} signif {float(r['significance']):.2f}")
