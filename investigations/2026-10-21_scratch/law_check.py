"""Scattering-law estimators per cos(incidence) bin, on the looks a stack kept:
median (production), mean, and mean with the top and bottom 1 % trimmed.
usage: law_check.py STACK_LOG CHAN RUN_DIR..."""
import csv, re, sys
import numpy as np, scipy.stats
sys.path.insert(0, "/home/than/code/moon-radar")
import stack_maps as sm
import doppler_equator_alignment as dea
log, chan, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
excluded = set(re.findall(r"^  - (\S+): ", open(log).read(), re.M))
law = sm.ScatterLaw(); cache = {}; n = 0
for d in runs:
    for st in ("stockert", "wsrt"):
        try:
            rows = list(csv.DictReader(open(f"{d}/registration_runs_{st}_{chan}.csv")))
        except FileNotFoundError:
            continue
        for r in rows:
            if r["rx_file"] in excluded:
                continue
            for k in ("map_npy", "mult_npy", "area_npy"):
                r[k] = sm.resolve_npy(r[k], d)
            cosi = sm.incidence_cos(dea.csp.str2et(r["rx_start_utc"]), 12 * 400 ** 2, cache, rx_name=st.upper())
            law.collect(sm.look_brightness(r, cosi), cosi); n += 1
s = np.vstack(law.samples)
edges = np.linspace(0.02, 1.0, law.nbins + 1)
idx = np.clip(np.digitize(s[:, 0], edges) - 1, 0, law.nbins - 1)
print(f"{n} looks; per bin: incidence, samples, median, mean, 1%-trimmed mean, fraction <= 0")
for b in list(range(0, 8)) + list(range(8, law.nbins, 4)):
    v = s[idx == b, 1]
    print(f"  {np.degrees(np.arccos((edges[b] + edges[b + 1]) / 2)):5.1f} deg {len(v):7d} "
          f"{np.median(v):9.4f} {v.mean():9.4f} {scipy.stats.trim_mean(v, 0.01):9.4f} {(v <= 0).mean():6.2f}")
