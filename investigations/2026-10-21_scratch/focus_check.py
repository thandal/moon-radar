import csv, sys, numpy as np, healpy as hp
sys.path.insert(0, "/tmp/eve_check/blur")
import focus_proto as F
F.dea.load_lola_dem()
r = [r for r in csv.DictReader(open("results/EVE2026_CIRC/registration_runs_stockert_chan1.csv")) if r["rx_start_utc"][11:19] == "09:41:25"][0]
rows = F.process_look(r, "/var/tmp/focus/check", 400, 4)
for name, old in (("alpha0", "map_alpha0.npy"), ("focused", "map_focused.npy")):
    a = np.load(f"/var/tmp/focus/check/{name}_map.npy"); b = np.load(f"/var/tmp/focus/stockert_0941_chan1/{old}")
    ok = (a != hp.UNSEEN) & (b != hp.UNSEEN)
    print(f"{name}: same valid pixels {np.array_equal(a == hp.UNSEEN, b == hp.UNSEEN)}, max |diff| {np.abs(a[ok] - b[ok]).max():.2e}")
