"""Refinements: (a) mirror correlation by distance from the Doppler-equator
plane, single look and stack; (b) noise floor from lags well before the echo."""
import csv, json, os, sys
import numpy as np, healpy as hp
from astropy import time as at
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
from trace_maps_helpers import *

rows = [(st, r) for st in ("stockert", "wsrt")
        for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_chan1.csv"))]
r = [r for st, r in rows if st == "stockert" and r["rx_start_utc"][11:19] == "09:41:25"][0]
meta = json.load(open(os.path.join(DATA, r["rx_file"])))
t0 = de.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
s, w, u, geo = geometry("STOCKERT", t0, float(r["capture_s"]))
mirror_pix = hp.vec2pix(NSIDE, *(v_pix - 2 * (v_pix @ u)[:, None] * u).T)
dist_km = np.abs(v_pix @ u) * 1737.4                 # distance from the plane
dist_grid = hp.get_interp_val(dist_km, T.theta, T.phi).reshape(T.X.shape)

def corr_by_band(m):
    mm = np.where(m[mirror_pix] != hp.UNSEEN, m[mirror_pix], hp.UNSEEN)
    a, b = T.disk_grid(m), T.disk_grid(mm)
    out = []
    for lo, hi in ((0, 100), (100, 300), (300, 600), (600, 1800)):
        ok = np.isfinite(a) & np.isfinite(b) & (dist_grid >= lo) & (dist_grid < hi)
        out.append((lo, hi, np.corrcoef(a[ok], b[ok])[0, 1], ok.sum()))
    return out

m = np.load(r["map_npy"]); mu = np.load(r["mult_npy"])
m = np.where((m != hp.UNSEEN) & (mu <= 3 * np.median(mu[mu > 0])), m, hp.UNSEEN)
L = {(st, rr["rx_start_utc"][11:19]): (rr["map_npy"], rr["mult_npy"]) for st, rr in rows if float(rr["tone_snr"]) >= 15}
acc = None
for k in L:
    I = sm.look_linear_intensity(*L[k]); acc = acc or sm.Accumulator(len(I)); acc.add(I)
stk = acc.stacked_log(3)
print("(a) correlation of 4-40 km detail with the reflection across the Doppler-equator")
print("    plane (09:41 geometry), by distance of the point from that plane")
for name, mp in (("single look Stockert 09:41", m), (f"stack of all {len(L)} opposite-sense looks", stk)):
    print(f"    {name}: " + ", ".join(f"{lo}-{hi} km: {c:.2f}" for lo, hi, c, n in corr_by_band(mp)))

print("\n(b) noise floor vs lag before the echo (Stockert 09:41 and 09:49:57), mean intensity / median map intensity")
for t in ("09:41:25", "09:49:57"):
    rr = [rr for st, rr in rows if st == "stockert" and rr["rx_start_utc"][11:19] == t][0]
    dd = np.load(rr["dd_npz"]); I = np.exp(2.0 * dd["log_A"].astype(np.float64))
    mm = np.load(rr["map_npy"]); ok = mm != hp.UNSEEN
    med = np.exp(2.0 * np.median(mm[ok]))
    lag_us = (dd["delay_s"] - dd["delay_s"][20]) * 1e6
    prof = I.mean(axis=0) / med
    print(f"    {t}: " + ", ".join(f"{lag_us[k]:+.0f} us: {prof[k]:.3f}" for k in (0, 3, 6, 9, 12, 15, 18, 20, 25, 30)))
    # far-from-SRP noise: Doppler rows near the limb at small delay hold no echo
    rows_far = np.r_[0:150, 2850:3000]
    print(f"      rows within 5% of the limbs, lags +100..+400 us (no surface there): {I[rows_far][:, 45:120].mean() / med:.3f}")
