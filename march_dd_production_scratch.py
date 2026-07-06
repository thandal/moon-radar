"""March monostatic DD images via the PRODUCTION compute_dd_image (waveform-
agnostic matched filter). ZC vs BPSK, same day/elevation. Scratch only."""
import os, numpy as np, astropy.units as au
from scipy.signal import decimate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as pl
import doppler_equator_alignment as dea

DEC = 4  # 1 Msps -> 250 ksps: 50 kHz chip is untouched (Nyquist 125 kHz); matches
         # the production reference. Delay bins then reflect 4 us sampling (true
         # resolution ~20 us = the 1/chip-rate), not 20x-oversampled 1 us bins.

ROOT = "data.camras.nl/thomas/sdr-eme/"
OUT  = "results/WAVEFORM_COMPARISON/"
PAIR = [("ZC-50k",  "rx-2025-03-04/dwingeloo_eme_2025_03_04_15_47_55_1297.500MHz_1.00Msps_ci16_le.sigmf-meta"),
        ("BPSK-50k","rx-2025-03-04/dwingeloo_eme_2025_03_04_15_42_12_1297.500MHz_1.00Msps_ci16_le.sigmf-meta")]

imgs = []
for label, rel in PAIR:
    rx_file = os.path.join(ROOT, rel)
    rx, tx, sr, freq, rx_start, tx_start, txfn = dea.load_observation(rx_file, ROOT)
    rx = decimate(rx, DEC, ftype="fir").astype("complex64")
    tx = decimate(tx, DEC, ftype="fir").astype("complex64")
    sr = sr / DEC
    print(f"{label}: tx={txfn}  rx_dur={len(rx)/sr.value:.1f}s  sr={sr.value/1e3:.0f}ksps  f={freq.value/1e6:.1f}MHz", flush=True)
    tx_emit = rx_start + 1.0*au.s
    log_A, dlt_shifts, delay_s, lt_min, rate = dea.compute_dd_image(
        rx, tx, sr, freq, rx_start, tx_emit, tx_name="DWINGELOO", rx_name="DWINGELOO")
    dop_span = (dlt_shifts.max()-dlt_shifts.min())*freq.to_value(au.Hz)
    print(f"  log_A {log_A.shape}  Doppler {dop_span:.1f} Hz / {log_A.shape[0]} rows "
          f"({dop_span/log_A.shape[0]*1e3:.0f} mHz/row, res 1/T={1/(len(rx)/sr.value):.2f} Hz); "
          f"delay {np.ptp(delay_s)*1e3:.1f} ms / {log_A.shape[1]} cols ({1e6/sr.value:.0f} us/col, res ~20us)", flush=True)
    imgs.append((label, txfn, log_A))
    np.save(OUT+f"march_prod_{label}.npy", log_A)
    pl.imsave(OUT+f"march_prod_{label}.png", log_A.T,
              vmax=log_A.max()*0.8, vmin=log_A.max()*0.4)
    print(f"  log_A shape={log_A.shape}  max={log_A.max():.2f}", flush=True)

# side-by-side, each on its own stretch (same convention as production log_A.png)
fig, ax = pl.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
for a, (label, txfn, log_A) in zip(ax, imgs):
    a.imshow(log_A.T, origin="lower", aspect="auto", cmap="viridis",
             vmax=log_A.max()*0.8, vmin=log_A.max()*0.4)
    a.set_title(f"{label}  ({txfn.split('.')[0]})")
    a.set_xlabel("Doppler row"); a.set_ylabel("delay sample")
fig.suptitle("March 2025 monostatic Dwingeloo lunar echo via production compute_dd_image "
             "(2025-03-04, el 57°, matched 50 kHz)")
fig.savefig(OUT+"march_prod_zc_vs_bpsk.png", dpi=120)
print("wrote", OUT+"march_prod_zc_vs_bpsk.png")
