import json, glob, sys, numpy as np, healpy as hp
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
import tiles as T
PARENT = hp.ang2pix(400, *hp.pix2ang(800, np.arange(hp.nside2npix(800))))
def to400(m):
    ok = m != hp.UNSEEN
    I = np.exp(2.0 * m[ok].astype(float))
    s = np.bincount(PARENT[ok], weights=I, minlength=hp.nside2npix(400)); c = np.bincount(PARENT[ok], minlength=hp.nside2npix(400))
    return np.where(c > 0, 0.5 * np.log(np.maximum(s, 1e-300) / np.maximum(c, 1)), hp.UNSEEN).astype(np.float32)
for f in sorted(glob.glob("/var/tmp/focus/nside800/*/rows.json")):
    pair = json.load(open(f)); r = pair["alpha0"]
    prod = np.load(f"results/EVE2026_CIRC/{r['rx_file']}_map.npy")
    proto = to400(np.load(r["map_npy"]))
    a, b = T.disk_grid(prod), T.disk_grid(proto); ok = np.isfinite(a) & np.isfinite(b)
    print(f"{r['rx_file'][:45]} {r['rx_file'][-16:-11]}: detail corr prototype(a=0, to nside 400) vs production {np.corrcoef(a[ok], b[ok])[0, 1]:.3f}; "
          f"noise {float(r['noise_log_amp']):.2f}; shift {float(r['applied_shift_samples']):+.1f} df {float(r['applied_df_hz']):+.3f} rim {float(r['rim_delta_hz']):+.4f}")
