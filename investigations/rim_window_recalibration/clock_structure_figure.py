"""Final clock structure-function figure: clean looks vs the wander trio."""
import csv
import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as pl
from astropy.time import Time

REPO = "/home/than/code/moon-radar"
os.chdir(REPO)
OUT = "investigations/rim_window_recalibration/clock_excursion_structure_function.png"
BAD = ("10_09_25", "10_14_34", "10_17_40")

rows = list(csv.DictReader(
    open("results/LOLA_DEM_REGISTRATION/registration_runs_chan1.csv")))


def sess(f):
    if "06_21" in f:
        return "06-21"
    if "09_16" in f:
        return "09-16"
    return "09-10/11"


looks = []
for r in rows:
    if int(r["rim_n"]) == 0:
        continue
    looks.append((sess(r["rx_file"]), Time(r["rx_start_utc"], scale="utc").unix,
                  (float(r["applied_df_hz"]) + float(r["rim_delta_hz"])) * 1e3,
                  any(b in r["rx_file"] for b in BAD)))

clean_pairs, trio_pairs = [], []
for s in {x[0] for x in looks}:
    sl = sorted([x for x in looks if x[0] == s], key=lambda x: x[1])
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            tau = sl[j][1] - sl[i][1]
            if tau > 6 * 3600:
                break
            d = abs(sl[j][2] - sl[i][2])
            (trio_pairs if (sl[i][3] or sl[j][3]) else clean_pairs).append((tau, d))
clean = np.array(clean_pairs)
trio = np.array(trio_pairs)

bins = np.geomspace(30, 6 * 3600, 12)
xs, ys = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = clean[(clean[:, 0] >= lo) & (clean[:, 0] < hi)]
    if len(sel) >= 5:
        xs.append(np.sqrt(lo * hi))
        ys.append(np.sqrt(np.mean(sel[:, 1] ** 2)))

fig, ax = pl.subplots(figsize=(7.5, 5))
ax.loglog(clean[:, 0], np.maximum(clean[:, 1], 0.2), ".", ms=3, alpha=0.2,
          color="C0", label="same-session pairs (clean looks)")
ax.loglog(trio[:, 0], np.maximum(trio[:, 1], 0.2), "x", ms=4, alpha=0.6,
          color="C3", label="pairs involving the 06-21 wander trio")
ax.loglog(xs, ys, "o-", color="C1", lw=2, label="clean RMS per lag bin")
tt = np.geomspace(30, 6 * 3600, 50)
ax.loglog(tt, 10.6 * np.sqrt(tt / 33.0), "k--", alpha=0.6,
          label=r"random walk from old intra-look 10.6 mHz@33s")
ax.axhline(5.5e-11 * 1.2995e9 * 1e3, color="gray", ls=":",
           label=r"logged Rb 5.5$\times 10^{-11}$ (71 mHz)")
ax.set_ylim(0.15, 400)
ax.set_xlabel("lag tau (s)")
ax.set_ylabel("|Delta chain offset| (mHz)")
ax.set_title("Chain-offset stability, rim-verified looks (2026-07-04)\n"
             "clean floor ~5-7 mHz (4-5e-12), flat 30 s - 3 h; "
             "trio shows bursty wander")
ax.legend(fontsize=8, loc="upper left")
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
