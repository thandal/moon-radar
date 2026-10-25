"""Run directories for stack_maps.py from the nside 400 focusing outputs:
the production runs CSVs, with each long look's map, mapcount, cell area and
noise floor swapped for the focused (or block, no drift correction) ones.
The noise floor is production's, carried over as its offset from the map
median: the prototype's 10 ms blocks cut the Zadoff-Chu code short of a
period, so the strong opposite-sense echo leaks into the delay columns ahead of the
echo where the floor is measured (the same-sense floor agrees with production to 0.01).
Short looks keep their production maps; stack with --min-tx-s 100 so only
the long looks are used, gated exactly as the production long-look stack.

usage: make_focused_run.py focused|alpha0 OUT_DIR [SOURCE_ROOT, default /var/tmp/focus400]"""
import csv, glob, json, os, sys
import numpy as np

RUN = "/home/than/code/moon-radar/results/EVE2026_CIRC"
version, out = sys.argv[1], sys.argv[2]
source = sys.argv[3] if len(sys.argv) > 3 else "/var/tmp/focus400"
os.makedirs(out, exist_ok=True)
swap = {}
for p in glob.glob(f"{source}/*/rows.json"):
    row = json.load(open(p))[version]
    swap[row["rx_file"]] = row


def median_map(map_npy, mult_npy):
    """Median log amplitude over the pixels stack_maps uses."""
    m, mu = np.load(map_npy), np.load(mult_npy)
    ok = (m > -1e30) & np.isfinite(m) & (mu <= 3 * np.median(mu[mu > 0]))
    return float(np.median(m[ok]))


n = 0
for st in ("stockert", "wsrt"):
    for chan in ("chan1", "chan0"):
        rows = list(csv.DictReader(open(f"{RUN}/registration_runs_{st}_{chan}.csv")))
        for r in rows:
            if r["rx_file"] in swap:
                prod_offset = float(r["noise_log_amp"]) - median_map(
                    os.path.join(RUN, os.path.basename(r["map_npy"])),
                    os.path.join(RUN, os.path.basename(r["mult_npy"])))
                for k in ("map_npy", "mult_npy", "area_npy"):
                    r[k] = swap[r["rx_file"]][k]
                r["noise_log_amp"] = median_map(r["map_npy"], r["mult_npy"]) + prod_offset
                n += 1
            else:
                for k in ("map_npy", "mult_npy", "area_npy"):
                    r[k] = os.path.join(RUN, os.path.basename(r[k]))
            if float(r["tx_s"]) > 100:
                assert r["rx_file"] in swap, f"{r['rx_file']} has no {version} map"
        with open(f"{out}/registration_runs_{st}_{chan}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
print(f"{n} looks swapped for {version} -> {out}")
