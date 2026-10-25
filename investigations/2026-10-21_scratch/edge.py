"""Delay response per look: rise of the echo leading edge (sub-radar point)
in the saved DD image, power summed over the central +/-1 Hz of Doppler."""
import csv, glob, os
import numpy as np
RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
for st, pre in [("stockert", "stockert_25m"), ("wsrt", "wsrt")]:
    for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_chan1.csv")):
        d = np.load(os.path.join(RUN, os.path.basename(r["dd_npz"])))
        p = np.exp(2.0 * d["log_A"].astype(np.float64))
        dop, dly = d["doppler_hz"], d["delay_s"]
        noise = p[:, :12].mean(axis=1, keepdims=True)
        sel = np.abs(dop) < 1.0
        prof = (p[sel] / noise[sel]).mean(axis=0) - 1
        k = int(np.argmax(prof))
        pk = prof[k]
        # interpolated crossings of 10% and 90% of the peak before it
        def cross(f):
            j = np.where(prof[:k + 1] < f * pk)[0][-1]
            return dly[j] + (f * pk - prof[j]) / (prof[j + 1] - prof[j]) * (dly[j + 1] - dly[j])
        rise = (cross(0.9) - cross(0.1)) * 1e6
        # width of the peak at half maximum (after the edge)
        j2 = k + int(np.argmax(prof[k:] < 0.5 * pk))
        fwhm = (dly[j2] - cross(0.5)) * 1e6
        code = "50k" if "fzc50000" in r["tx_file"] else "25k"
        n = "4x" if "4x" in r["tx_file"] else "1x"
        print(f"{st:8s} {r['rx_start_utc'][11:19]} {code} {n}  edge rise 10-90% {rise:6.1f} us  "
              f"peak half-width {fwhm:6.1f} us  peak at {(dly[k]-dly[0])*1e6:7.1f} us into window  "
              f"cell {np.median(np.diff(dly))*1e6:.1f} us")
