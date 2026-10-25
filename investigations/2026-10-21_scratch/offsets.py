"""Station-to-station and per-look registration offsets for the 2026-10-11 stacks."""
import csv
import os
import sys

import numpy as np
import healpy as hp

sys.path.insert(0, "/home/than/code/moon-radar")
import registration_analysis as ra

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
chan = sys.argv[1] if len(sys.argv) > 1 else "chan1"
step = 0.075
lon_axis = np.arange(-55, 55 + step / 2, step)
lat_axis = np.arange(-55, 55 + step / 2, step)
lo_px, hi_px = 0.3 / step, 2.5 / step
search_px, exclude_px = int(1.5 / step), int(0.5 / step)


def band_of_map(m, mult=None):
    return ra.bandpass(ra.grid_map(m, lon_axis, lat_axis, multiplicity=mult), lo_px, hi_px)


stk = {k: band_of_map(np.load(f"{RUN}/stacked_map_{k}.npy"))
       for k in [chan, f"wsrt_{chan}", f"stockert_wsrt_{chan}"]}

dy, dx, peak, sig = ra.xcorr_offset(stk[chan], stk[f"wsrt_{chan}"], search_px, exclude_px)
print(f"WSRT stack -> Stockert stack: dlon {dx*step:+.3f} dlat {dy*step:+.3f} deg "
      f"(peak {peak:.3f}, signif {sig:.2f})")

ref = stk[f"stockert_wsrt_{chan}"]
for prefix, st in [("registration_runs", "stockert"), ("registration_runs_wsrt", "wsrt")]:
    for r in csv.DictReader(open(f"{RUN}/{prefix}_{chan}.csv")):
        m = np.load(os.path.join(RUN, os.path.basename(r["map_npy"])))
        mult = np.load(os.path.join(RUN, os.path.basename(r["mult_npy"])))
        b = band_of_map(m, mult)
        dy, dx, peak, sig = ra.xcorr_offset(ref, b, search_px, exclude_px)
        # also against each single-station stack
        dy_s, dx_s, _, sig_s = ra.xcorr_offset(stk[chan], b, search_px, exclude_px)
        dy_w, dx_w, _, sig_w = ra.xcorr_offset(stk[f"wsrt_{chan}"], b, search_px, exclude_px)
        print(f"{st:8s} {r['rx_start_utc'][11:19]}  vs combined: dlon {dx*step:+.3f} "
              f"dlat {dy*step:+.3f} (signif {sig:.2f}) | vs Stockert stack "
              f"{dx_s*step:+.3f} {dy_s*step:+.3f} ({sig_s:.2f}) | vs WSRT stack "
              f"{dx_w*step:+.3f} {dy_w*step:+.3f} ({sig_w:.2f})", flush=True)
