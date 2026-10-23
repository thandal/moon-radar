"""Final clock structure-function figure: clean looks vs the wander trio.

HISTORICAL (July 2026): reads the July run's CSV, which predates
rim_status (rim_n > 0 meant a measured rim; see the directory-rename note
in ../rim_window_recalibration_2026-07-03.md).
"""
import csv
import os
import re

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as pl
from astropy.time import Time

REPO = "/home/than/code/moon-radar"
os.chdir(REPO)
OUT = "investigations/rim_window_recalibration/clock_excursion_structure_function.png"
BAD = ("10_09_25", "10_14_34", "10_17_40")
F_HZ = 1.2995e9   # carrier of the 2025 sessions (1299.5 MHz, the capture names)

rows = list(csv.DictReader(
    open("results/LOLA_DEM_REGISTRATION/registration_runs_chan1.csv")))


# Session of each capture date in the July runs CSV; a capture from any
# other date is an error, not a member of some default session.
SESSION_OF_DATE = {"2025_06_21": "06-21", "2025_09_10": "09-10/11",
                   "2025_09_11": "09-10/11", "2025_09_16": "09-16"}


def sess(f):
    m = re.search(r"_(\d{4}_\d{2}_\d{2})_", f)
    if m is None or m.group(1) not in SESSION_OF_DATE:
        raise ValueError(f"{f}: capture date not one of {sorted(SESSION_OF_DATE)}")
    return SESSION_OF_DATE[m.group(1)]


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
# A log axis has no place for a zero difference: pairs below FLOOR_MHZ
# (zeros included) are drawn at it with their own marker, and counted.
FLOOR_MHZ = 0.2
for pairs_, marker, ms, alpha, color, label in (
        (clean, ".", 3, 0.2, "C0", "same-session pairs (clean looks)"),
        (trio, "x", 4, 0.6, "C3", "pairs involving the 06-21 wander trio")):
    below = pairs_[:, 1] < FLOOR_MHZ
    ax.loglog(pairs_[~below, 0], pairs_[~below, 1], marker, ms=ms, alpha=alpha,
              color=color, label=label)
    if below.any():
        ax.loglog(pairs_[below, 0], np.full(int(below.sum()), FLOOR_MHZ), "v", ms=4,
                  alpha=0.6, color=color,
                  label=f"  {int(below.sum())} of them below {FLOOR_MHZ:g} mHz (zeros "
                        f"included), drawn at {FLOOR_MHZ:g}")
ax.loglog(xs, ys, "o-", color="C1", lw=2, label="clean RMS per lag bin")
tt = np.geomspace(30, 6 * 3600, 50)
ax.loglog(tt, 10.6 * np.sqrt(tt / 33.0), "k--", alpha=0.6,
          label=r"random walk from old intra-look 10.6 mHz@33s (stated)")
ax.axhline(5.5e-11 * F_HZ * 1e3, color="gray", ls=":",
           label=rf"5.5$\times 10^{{-11}}$ of the carrier ({5.5e-11 * F_HZ * 1e3:.0f} mHz): "
                 "a Stockert Rb figure; attribution provisional (REPORT 4)")
ax.set_ylim(0.15, 400)
ax.set_xlabel("lag tau (s)")
ax.set_ylabel("|Delta chain offset| (mHz)")
ax.set_title("Chain-offset stability, rim-verified looks (2026-07-04)\n"
             f"clean RMS per lag bin {min(ys):.1f}-{max(ys):.1f} mHz "
             f"({min(ys) * 1e-3 / F_HZ:.1e}-{max(ys) * 1e-3 / F_HZ:.1e}) "
             f"over lags {xs[0]:.0f} s - {xs[-1] / 3600:.1f} h")
ax.legend(fontsize=8, loc="upper left")
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
