"""Sharp-look stack (125 s or 50 kchip/s looks, both stations) vs a same-size
stack containing the short 25 kchip/s looks; widths and saved maps."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/eve_check/blur")
import tiles as T
import acwidth as A   # prints its own table first

LONG = ("09:35:24", "09:41:25", "09:46:36", "10:16:10")
C50 = ("09:49:09", "09:49:57")
allk = A.by
sharp = [l for k, l in allk.items() if k[1] in LONG + C50]
short = [l for k, l in allk.items() if k[1] not in LONG + C50]
# same size as sharp, built from all the short 25 kchip/s looks plus sharp ones
mixed = short + sharp[:len(sharp) - len(short)]
print(f"\nsharp set {len(sharp)}: " + " ".join(k[0] + k[1][:5] for k in allk if k[1] in LONG + C50))
print(f"comparison set {len(mixed)} (all {len(short)} short 25 kchip/s looks + {len(mixed) - len(short)} sharp)")
for name, items in [("sharp set", sharp), ("comparison set", mixed)]:
    h = T.stack_of(items)
    np.save(f"/tmp/eve_check/blur/var_chan1_{name.split()[0]}.npy", h)
    a, c = A.ac_widths(T.disk_grid(h))
    print(f"{name:20s} {len(items):2d} looks  width along {a:.1f} km, across {c:.1f} km")
