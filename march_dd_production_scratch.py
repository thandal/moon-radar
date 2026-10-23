"""March monostatic DD images via the PRODUCTION compute_dd_image (waveform-
agnostic matched filter). ZC vs BPSK, same day (each capture's time and Moon
elevation are computed and printed, and shown in the figure title). Scratch
only."""
import os, numpy as np, astropy.units as au
from scipy.signal import decimate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as pl
import doppler_equator_alignment as dea
from predict_libration_opportunities import get_moon_elevations

DEC = 4  # 1 Msps -> 250 ksps: 50 kHz chip is untouched (Nyquist 125 kHz); matches
         # the production reference. Delay bins then reflect 4 us sampling (true
         # resolution ~20 us = the 1/chip-rate), not 20x-oversampled 1 us bins.

ROOT = "data.camras.nl/thomas/sdr-eme/"
OUT  = "results/WAVEFORM_COMPARISON/"
RX_NAME = dea.STATION_RX["dwingeloo"]   # monostatic: Dwingeloo receives its own echo
PAIR = [("ZC-50k",  "rx-2025-03-04/dwingeloo_eme_2025_03_04_15_47_55_1297.500MHz_1.00Msps_ci16_le.sigmf-meta"),
        ("BPSK-50k","rx-2025-03-04/dwingeloo_eme_2025_03_04_15_42_12_1297.500MHz_1.00Msps_ci16_le.sigmf-meta")]

def db(log_amp):                         # natural-log amplitude -> dB
    return 20 / np.log(10) * log_amp

imgs = []
for label, rel in PAIR:
    rx_file = os.path.join(ROOT, rel)
    rx, tx, sr, freq, rx_start, tx_start, txfn = dea.load_observation(rx_file, ROOT)
    rx = decimate(rx, DEC, ftype="fir").astype("complex64")
    tx = decimate(tx, DEC, ftype="fir").astype("complex64")
    sr = sr / DEC
    print(f"{label}: tx={txfn}  rx_dur={len(rx)/sr.value:.1f}s  sr={sr.value/1e3:.0f}ksps  f={freq.value/1e6:.1f}MHz", flush=True)
    # Moon elevation at the receiver at the capture start (geometric, from
    # the station's topocentric frame).
    _, el = get_moon_elevations(dea.et_from_astropy(rx_start), dea.TX_STATION, RX_NAME)
    print(f"  {rx_start.utc.iso[:19]} UTC, Moon elevation {el:.1f} deg", flush=True)
    # No timing offset is measured for these captures (timing offset 0): the
    # echo's leading edge sits wherever the chain delay puts it.
    log_A, dlt_shifts, delay_s, lt_min, rate, tx_overlap_s = dea.compute_dd_image(
        rx, tx, sr, freq, rx_start, dea.nominal_tx_emit_start(rx_start), 0.0, 0.0,
        dea.TX_STATION, RX_NAME)
    dop_span = (dlt_shifts.max()-dlt_shifts.min())*freq.to_value(au.Hz)
    print(f"  log_A {log_A.shape}  Doppler {dop_span:.1f} Hz / {log_A.shape[0]} rows "
          f"({dop_span/log_A.shape[0]*1e3:.0f} mHz/row, res 1/T={1/tx_overlap_s:.2f} Hz over "
          f"the {tx_overlap_s:.1f} s the TX covers); "
          f"delay {np.ptp(delay_s)*1e3:.1f} ms / {log_A.shape[1]} cols ({1e6/sr.value:.0f} us/col; "
          "chip 20 us = 1/50 kHz, stated)", flush=True)
    # The pipeline's noise floor: the lead delay columns, ahead of the echo
    # only if the echo's leading edge is at lag 0 -- not established here, so
    # compare it with the image median (mostly noise): a floor well above the
    # median would mean echo in the lead columns.
    floor = dea.noise_floor_log_amp(log_A)
    med = float(np.log(np.median(np.exp(log_A))))
    print(f"  noise floor (lead columns) {db(floor):.1f} dB, image median {db(med):.1f} dB, "
          f"peak {db(float(log_A.max())):.1f} dB (log-amplitude units in dB)", flush=True)
    imgs.append((label, txfn, log_A, floor, rx_start, el))
    np.save(OUT+f"march_prod_{label}.npy", log_A)

# Both images on one scale: dB above each image's own noise floor, from 0
# (the floor) to the higher of the two peaks, so a floor or peak difference
# shows as a difference in the picture.
top = max(db(float(log_A.max()) - floor) for _, _, log_A, floor, _, _ in imgs)
for label, txfn, log_A, floor, _, _ in imgs:
    pl.imsave(OUT+f"march_prod_{label}.png", db(log_A.T - floor), vmin=0.0, vmax=top)
fig, ax = pl.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
for a, (label, txfn, log_A, floor, _, _) in zip(ax, imgs):
    im = a.imshow(db(log_A.T - floor), origin="lower", aspect="auto", cmap="viridis",
                  vmin=0.0, vmax=top)
    a.set_title(f"{label}  ({txfn.split('.')[0]})")
    a.set_xlabel("Doppler row"); a.set_ylabel("delay sample")
fig.colorbar(im, ax=ax, label="dB above the image's noise floor")
fig.suptitle("March 2025 monostatic Dwingeloo lunar echo via production compute_dd_image ("
             + ", ".join(f"{label} {t.utc.iso[:16]} UTC el {el:.0f}°"
                         for label, _, _, _, t, el in imgs)
             + "; 50 kHz chips, stated)")
fig.savefig(OUT+"march_prod_zc_vs_bpsk.png", dpi=120)
print("wrote", OUT+"march_prod_zc_vs_bpsk.png")
