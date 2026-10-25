import csv, os, numpy as np, healpy as hp
OLD = "/var/tmp/moon_backup/EVE2026_CIRC_one_point_per_pixel"; NEW = "results/EVE2026_CIRC"
for st in ("stockert", "wsrt"):
    for r in list(csv.DictReader(open(f"{NEW}/registration_runs_{st}_chan1.csv")))[::3]:
        b = os.path.basename(r["mult_npy"])
        mo, mn = np.load(f"{OLD}/{b}"), np.load(f"{NEW}/{b}")
        seen = mo > 0
        q = lambda m: np.quantile(m[seen], [.5, .9, .99])
        print(f"{st:8s} {r['rx_start_utc'][11:19]} ({float(r['tx_s']):3.0f} s): median old/new {np.median(mo[mo>0]):.2f}/{np.median(mn[mn>0]):.2f}; "
              f"masked (> 3 x median) old {(mo[seen] > 3*np.median(mo[mo>0])).mean():.3f} new {(mn[seen] > 3*np.median(mn[mn>0])).mean():.3f}; "
              f"90/99% old {q(mo)[1]:.1f}/{q(mo)[2]:.1f} new {q(mn)[1]:.1f}/{q(mn)[2]:.1f}")
