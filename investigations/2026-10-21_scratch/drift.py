"""Echo delay (timing offset) in successive segments of single looks, both
stations: a sampling-clock rate error shows up as a steady delay drift."""
import gc
import glob
import os
import sys

import numpy as np
import cupy
from astropy import units as au

sys.path.insert(0, "/home/than/code/moon-radar")
import doppler_equator_alignment as dea
import freq_offset_hunt as foh

ROOT = "/home/than/code/moon-radar/data/eve2026_circ"
LOOKS = [("09_41_25", [(4, 34), (34, 64), (64, 94), (94, 123)]),
         ("10_16_10", [(4, 34), (34, 64), (64, 94), (94, 123)]),
         ("09_27_04", [(4, 19), (19, 33)])]
for stamp, segs in LOOKS:
    for st, pre in [("STOCKERT", "stockert_25m"), ("WSRT", "wsrt")]:
        path = glob.glob(f"{ROOT}/2026-10-11/{pre}_moon_2026_10_11_{stamp}_*.chan1.sigmf-meta")[0]
        rx, tx, fs_q, freq, rx_start, _, _ = dea.load_observation(path, ROOT)
        fs = fs_q.to_value(au.Hz)
        txc = foh.compensated_tx(rx, tx, fs_q, freq, rx_start, rx_start + 1.0 * au.s,
                                 tx_name="DWINGELOO", rx_name=st)
        del tx
        out = []
        for a, b in segs:
            i0, i1 = int(a * fs), int(b * fs)
            res = foh.measure_offset(rx[i0:i1], txc[i0:i1], fs, max_shift=60)
            out.append((a, b, res["shift_refined"], res["snr"], res["f_centroid"]))
        del rx, txc
        gc.collect()
        cupy.get_default_memory_pool().free_all_blocks()
        sh = np.array([o[2] for o in out])
        mid = np.array([(o[0] + o[1]) / 2 for o in out])
        slope = np.polyfit(mid, sh, 1)[0] if len(out) > 2 else (sh[-1] - sh[0]) / (mid[-1] - mid[0])
        print(f"{st:8s} {stamp}: " + "  ".join(
            f"[{a:3d}-{b:3d} s] {s:+7.2f} samp (tone {n:3.0f}, {f*1e3:+6.1f} mHz)" for a, b, s, n, f in out)
            + f"   drift {slope * 4.0 * 60:+.2f} us/min", flush=True)
