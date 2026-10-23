"""Are the off-trend rim-verified looks consistent with Rb random-walk FM?

Structure function of the total chain offset (applied_df + rim_delta) over
rim-verified looks (rim_n > 0) vs the random-walk extrapolation of an
earlier intra-look wander estimate, 10.6 mHz rms at 33 s half-windows
(stated: the July 2026 analysis, whose log
../rim_window_recalibration_2026-07-03.md then traced it to measurement
noise plus three burst looks; REPORT section 4 no longer uses it):
sigma(tau) = 10.6 mHz * sqrt(tau / 33 s).

HISTORICAL (July 2026): reads the July run's CSV, which predates
rim_status (rim_n > 0 meant a measured rim; see the directory-rename note
in the log).
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
OUT_PNG = "investigations/rim_window_recalibration/clock_excursion_structure_function.png"

rows = list(csv.DictReader(
    open("results/LOLA_DEM_REGISTRATION/registration_runs_chan1.csv")))


# Session of each capture date in the July runs CSV; a capture from any
# other date is an error, not a member of some default session.
SESSION_OF_DATE = {"2025_06_21": "06-21", "2025_09_10": "09-10/11",
                   "2025_09_11": "09-10/11", "2025_09_16": "09-16"}


def sess_of(f):
    m = re.search(r"_(\d{4}_\d{2}_\d{2})_", f)
    if m is None or m.group(1) not in SESSION_OF_DATE:
        raise ValueError(f"{f}: capture date not one of {sorted(SESSION_OF_DATE)}")
    return SESSION_OF_DATE[m.group(1)]


looks = []
for r in rows:
    if int(r["rim_n"]) == 0:
        continue  # not rim-verified
    t = Time(r["rx_start_utc"], scale="utc").unix
    tot = (float(r["applied_df_hz"]) + float(r["rim_delta_hz"])) * 1e3
    looks.append((sess_of(r["rx_file"]), t, tot, r["rx_file"]))
print(f"{len(looks)} rim-verified looks")

pairs = []
for s in ("06-21", "09-10/11", "09-16"):
    sl = sorted([x for x in looks if x[0] == s], key=lambda x: x[1])
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            tau = sl[j][1] - sl[i][1]
            if tau > 6 * 3600:
                break
            pairs.append((tau, abs(sl[j][2] - sl[i][2])))
pairs = np.array(pairs)
print(f"{len(pairs)} same-session pairs (tau <= 6 h)")

bins = np.geomspace(30, 6 * 3600, 12)
print(f"\n{'tau bin':>16s} {'n':>5s} {'RMS |dTotal|':>13s} {'RW model':>9s} {'ratio':>6s}")
xs, ys = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = pairs[(pairs[:, 0] >= lo) & (pairs[:, 0] < hi)]
    if len(sel) < 5:
        continue
    rms = np.sqrt(np.mean(sel[:, 1] ** 2))
    tau_c = np.sqrt(lo * hi)
    model = 10.6 * np.sqrt(tau_c / 33.0)
    xs.append(tau_c)
    ys.append(rms)
    print(f"{lo:7.0f}-{hi:<8.0f} {len(sel):5d} {rms:10.1f} mHz {model:6.1f} mHz {rms/model:6.2f}")

# The three flagged 06-21 looks against their nearest rim-verified neighbors.
print("\n06-21 morning sequence (rim-verified totals, mHz):")
sl = sorted([x for x in looks if x[0] == "06-21"], key=lambda x: x[1])
t0 = sl[0][1]
for s, t, tot, f in sl:
    hhmm = f.split("2025_06_21_")[1][:8]
    flag = " <-- flagged" if hhmm.replace("_", ":") in ("10:09:25", "10:14:34", "10:17:40") else ""
    print(f"  {hhmm}  {tot:+7.1f}{flag}")

fig, ax = pl.subplots(figsize=(7, 5))
# A log axis has no place for a zero difference: pairs below FLOOR_MHZ
# (zeros included) are drawn at it with their own marker, and counted.
FLOOR_MHZ = 0.3
below = pairs[:, 1] < FLOOR_MHZ
ax.loglog(pairs[~below, 0], pairs[~below, 1], ".", ms=3, alpha=0.25,
          label="same-session pairs")
if below.any():
    ax.loglog(pairs[below, 0], np.full(int(below.sum()), FLOOR_MHZ), "v", ms=4,
              color="C2", alpha=0.6,
              label=f"{int(below.sum())} pairs below {FLOOR_MHZ:g} mHz (zeros "
                    f"included), drawn at {FLOOR_MHZ:g}")
ax.loglog(xs, ys, "o-", color="C1", label="RMS per bin")
tt = np.geomspace(30, 6 * 3600, 50)
ax.loglog(tt, 10.6 * np.sqrt(tt / 33.0), "k--",
          label=r"random walk: 10.6 mHz $\times\sqrt{\tau/33\,s}$ (stated, July estimate)")
ax.set_xlabel("lag tau (s)")
ax.set_ylabel("|Delta total chain offset| (mHz)")
ax.set_title("Look-to-look chain-offset structure function (rim-verified looks)")
ax.legend()
fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
print(f"\nwrote {OUT_PNG}")
