"""Are short and long looks registered to each other? Tile offsets between a
short-look stack and a long-look stack (both stations, 09:24-10:16), with
same-size controls drawn within each group."""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, "/tmp/eve_check/blur")
import tiles as T

LONG = ("09:35:24", "09:41:25", "09:46:36", "10:16:10")
looks = T.S + T.W
long_ = [l for l in looks if l[0] in LONG]
short = [l for l in looks if l[0] not in LONG]
print(f"long {len(long_)} looks, short {len(short)} looks")

gL, gS = T.disk_grid(T.stack_of(long_)), T.disk_grid(T.stack_of(short))
# controls: split each group in two by station (independent noise, same look type)
gL_s = T.disk_grid(T.stack_of([l for l in T.S if l[0] in LONG]))
gL_w = T.disk_grid(T.stack_of([l for l in T.W if l[0] in LONG]))
gS_s = T.disk_grid(T.stack_of([l for l in T.S if l[0] not in LONG]))
gS_w = T.disk_grid(T.stack_of([l for l in T.W if l[0] not in LONG]))
for name, a, b in [("long vs short", gL, gS),
                   ("control: long Stockert vs long WSRT", gL_s, gL_w),
                   ("control: short Stockert vs short WSRT", gS_s, gS_w)]:
    T.report(name, T.tile_offsets(a, b))

print("\nrobust spread and position dependence (tiles with signif > 1.3):")
e = np.radians(T.EQ_DEG)
for name, a, b in [("long vs short", gL, gS),
                   ("control: long Stockert vs long WSRT", gL_s, gL_w),
                   ("control: short Stockert vs short WSRT", gS_s, gS_w)]:
    t = T.tile_offsets(a, b)
    t = t[t[:, 5] > 1.3]
    x_al = t[:, 0] * np.cos(e) + t[:, 1] * np.sin(e)      # tile position along the equator
    y_ac = -t[:, 0] * np.sin(e) + t[:, 1] * np.cos(e)
    d_al = t[:, 2] * np.cos(e) + t[:, 3] * np.sin(e)
    d_ac = -t[:, 2] * np.sin(e) + t[:, 3] * np.cos(e)
    mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
    sa = np.polyfit(x_al, d_al, 1)[0] * 1000
    sc = np.polyfit(y_ac, d_ac, 1)[0] * 1000
    print(f"  {name:40s} robust spread along {mad(d_al):4.1f} km, across {mad(d_ac):4.1f} km; "
          f"stretch along {sa:+.1f} km per 1000 km, across {sc:+.1f} km per 1000 km; "
          f"tiles beyond 8 km: {(np.hypot(d_al, d_ac) > 8).sum()}/{len(t)}")
