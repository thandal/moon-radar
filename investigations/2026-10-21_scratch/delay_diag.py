"""Diagnose the 2025-09-11 large-shift looks: tone SNR vs timing shift, and
where the echo sits in delay in the DD image built with the run's applied
corrections. usage: delay_diag.py HH_MM_SS"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import cupy
from astropy import units as au
import doppler_equator_alignment as dea
import freq_offset_hunt as foh
t = sys.argv[1]
root = "data.camras.nl/lunar-radar"
path = f"{root}/2025-09-10/stockert_radar_2025_09_11_{t}_1299.500MHz_0.25Msps_ci16_le.chan1.sigmf-meta"
row = next(r for r in csv.DictReader(open("results/STOCKERT_2025/registration_runs_stockert_chan1.csv"))
           if r["rx_file"] == os.path.basename(path))
rx, tx, fs_q, f_q, rx_start, _, _ = dea.load_observation(path, root)
fs = fs_q.to_value(au.Hz)
tx_emit = rx_start + 1.0 * au.s
tx_comp = foh.compensated_tx(rx, tx, fs_q, f_q, rx_start, tx_emit, tx_name="DWINGELOO", rx_name="STOCKERT")
decim = int(fs // 5000); n = (len(tx_comp) // decim) * decim
freqs = np.fft.fftfreq(n // decim, decim / fs)
txc = cupy.conj(cupy.asarray(tx_comp[:n])); rxg = cupy.asarray(rx[:len(tx_comp)], dtype=cupy.complex64)
prof = []
for s in range(-60, 61):
    y = (cupy.roll(rxg, -s)[:n] * txc).reshape(-1, decim).sum(axis=1)
    Y = cupy.asnumpy(cupy.abs(cupy.fft.fft(y)))
    sel = np.abs(freqs) <= 2.0
    prof.append((s, float(Y[sel].max()), float(np.median(Y[sel]))))
del txc, rxg, tx_comp
cupy.get_default_memory_pool().free_all_blocks()
shift = float(row["applied_shift_samples"]); df = float(row["applied_df_hz"])
log_A, dlt, delay_s, lt_min, _ = dea.compute_dd_image(
    rx, tx, fs_q, f_q, rx_start, tx_emit, shift / fs, 0.0, "DWINGELOO", "STOCKERT", freq_offset_hz=df)
I = np.exp(2 * log_A.astype(np.float64))
np.savez(f"/var/tmp/diag/{t}.npz", prof=np.array(prof), col_mean=I.mean(axis=0), col_max=I.max(axis=0),
         delay_s=delay_s, shift=shift, noise_log_amp=float(row["noise_log_amp"]))
print(t, "applied shift", shift, "saved")
