"""Check the focusing prototype's output for one look.
usage: validate_focus.py RUN_CSV TIME OUT_DIR"""
import csv, json, os, sys
import numpy as np, healpy as hp
from astropy import time as at
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
from trace_maps_helpers import *          # geometry(), tiles as T, de, NSIDE, v_pix
import tiles as T

csv_path, when, out = sys.argv[1:4]
r = [r for r in csv.DictReader(open(csv_path)) if r["rx_start_utc"][11:19] == when][0]
rx_name = r["rx_file"].split("_")[0].upper()

# (a) a = 0 image vs today's image, on today's rows
old = np.load(r["dd_npz"]); new = np.load(os.path.join(out, "dd_alpha0.npz"))
Io = np.exp(2.0 * old["log_A"].astype(float)); In = np.exp(2.0 * new["log_A"].astype(float))
fo, fn = old["doppler_hz"], new["doppler_hz"]
order = np.argsort(fn)
In_on_old = np.array([np.interp(fo, fn[order], In[order, k]) for k in range(In.shape[1])]).T
print(f"(a) delay axes agree to {np.abs(old['delay_s'] - new['delay_s']).max() * 1e9:.2f} ns; "
      f"Doppler rows today {abs(fo[1] - fo[0]) * 1e3:.2f} mHz, block method {abs(fn[1] - fn[0]) * 1e3:.2f} mHz")
sel = (np.abs(fo) < 10)[:, None] & (np.arange(Io.shape[1]) > 30)[None, :]
a, b = np.log(Io[sel]), np.log(In_on_old[sel])
print(f"    log-intensity correlation over the echo {np.corrcoef(a, b)[0, 1]:.4f}; "
      f"median ratio block / today {np.median(In_on_old[sel] / Io[sel]):.3f}")
# best Doppler shift between the two (should be ~0)
prof_o = Io[:, 40:400].sum(axis=1); prof_n = In_on_old[:, 40:400].sum(axis=1)
lags = np.arange(-20, 21)
cc = [np.corrcoef(prof_o[30:-30], np.roll(prof_n, k)[30:-30])[0, 1] for k in lags]
print(f"    Doppler profile best shift {lags[int(np.argmax(cc))]} rows (corr {max(cc):.4f})")

# (b) maps: a = 0 vs today's map; (c) focused vs a = 0
m_today = np.load(r["map_npy"])
m0 = np.load(os.path.join(out, "map_alpha0.npy")); m1 = np.load(os.path.join(out, "map_focused.npy"))
mu0 = np.load(os.path.join(out, "mapcount_alpha0.npy")); mu1 = np.load(os.path.join(out, "mapcount_focused.npy"))
mu_today = np.load(r["mult_npy"])
def masked(m, mu):
    return np.where((m != hp.UNSEEN) & (mu <= 3 * np.median(mu[mu > 0])), m, hp.UNSEEN)
g_today, g0, g1 = (T.disk_grid(masked(m, mu)) for m, mu in ((m_today, mu_today), (m0, mu0), (m1, mu1)))
ok = np.isfinite(g_today) & np.isfinite(g0)
print(f"(b) detail 4-40 km, block a = 0 map vs today's map: correlation {np.corrcoef(g_today[ok], g0[ok])[0, 1]:.3f}")
ok = np.isfinite(g1) & np.isfinite(g0)
print(f"(c) focused vs a = 0 map: correlation {np.corrcoef(g1[ok], g0[ok])[0, 1]:.3f}; "
      f"detail rms a = 0 {np.std(g0[ok]):.3f}, focused {np.std(g1[ok]):.3f}")

# mirror correlation, as trace_maps.py
meta = json.load(open(os.path.join(DATA, r["rx_file"])))
t0 = de.et_from_astropy(at.Time(meta["captures"][0]["core:datetime"]))
s, w, u, geo = geometry(rx_name, t0, float(r["capture_s"]))
mp = hp.vec2pix(NSIDE, *(v_pix - 2 * (v_pix @ u)[:, None] * u).T)
dist = hp.get_interp_val(np.abs(v_pix @ u) * 1737.4, T.theta, T.phi).reshape(T.X.shape)
for name, m, mu in (("today", m_today, mu_today), ("a = 0", m0, mu0), ("focused", m1, mu1)):
    mm = masked(m, mu)
    a, b = T.disk_grid(mm), T.disk_grid(np.where(mm[mp] != hp.UNSEEN, mm[mp], hp.UNSEEN))
    out_s = []
    for lo, hi in ((100, 300), (300, 600), (600, 1800)):
        ok = np.isfinite(a) & np.isfinite(b) & (dist >= lo) & (dist < hi)
        out_s.append(f"{lo}-{hi} km {np.corrcoef(a[ok], b[ok])[0, 1]:.2f}")
    print(f"    mirror correlation, {name:8s}: " + ", ".join(out_s))
