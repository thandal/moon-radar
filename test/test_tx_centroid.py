import sigmf
import numpy as np
import scipy.fft
import os

DATA_PREFIX = "data.camras.nl/lunar-radar/2025-06-21/"
# Take any meta file to get the desc
rx_filename = [f for f in os.listdir(DATA_PREFIX) if 'stockert' in f and 'chan0.sigmf-meta' in f][0]
sigmf_file = sigmf.sigmffile.fromfile(DATA_PREFIX + rx_filename, skip_checksum=True)
rx_info = sigmf_file.get_global_info()
tx_filename = rx_info['core:description'].split(';')[0]
tx_sigmf_filename = f"data.camras.nl/lunar-radar/tx_signals/{tx_filename}"

print("TX filename:", tx_sigmf_filename)
tx_sigmf_file = sigmf.sigmffile.fromfile(tx_sigmf_filename, skip_checksum=True)
tx_samples = tx_sigmf_file.read_samples().astype("complex64")

sample_rate = tx_sigmf_file.get_global_info()['core:sample_rate']

# Compute the centroid frequency
f_axis = scipy.fft.fftfreq(len(tx_samples), 1/sample_rate)
fft_tx = np.abs(scipy.fft.fft(tx_samples))**2
centroid = np.sum(f_axis * fft_tx) / np.sum(fft_tx)

print(f"Sample Rate: {sample_rate} Hz")
print(f"Centroid frequency of TX baseband signal: {centroid} Hz")

# What would be the shift due to stretching?
# stretch factor = dlt = ~ -1.5e-6
# for 2025-09-16 maybe dlt is different?
