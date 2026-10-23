"""Diagnose the ZC vs BPSK real-echo peak/median gap that quant_waveform_scratch.py
prints. Is it the BPSK code's self-clutter raising the floor, or a real peak
difference? Scratch."""
import numpy as np, sigmf
from scipy.signal import decimate
import doppler_equator_alignment as dea
OUT = "results/WAVEFORM_COMPARISON/"; ROOT = "data.camras.nl/thomas/sdr-eme/"

def db(log_amp):                         # natural-log amplitude -> dB
    return 20 / np.log(10) * log_amp

def code_clutter(txname):
    tx = sigmf.sigmffile.fromfile(ROOT+"tx/"+txname, skip_checksum=True).read_samples().astype("complex64")
    tx = decimate(tx, 4, ftype="fir").astype(complex); tx /= np.abs(tx).mean()
    X = np.fft.fft(tx); ac = np.abs(np.fft.ifft(X*np.conj(X))); ac /= ac.max()
    # Main lobe: out to the first local minimum on either side of lag 0 (the
    # chip-wide triangle, widened by the decimation filter); sidelobes beyond.
    rise = np.nonzero(np.diff(ac[:len(ac)//2]) > 0)[0]
    if not len(rise): raise ValueError(f"{txname}: autocorrelation never rises again")
    k = int(rise[0])                     # first local minimum
    sl = ac.copy(); sl[:k] = 0; sl[len(ac)-k+1:] = 0
    return k, 20*np.log10(sl.max()), 10*np.log10((sl**2).sum())

print("ACTUAL transmitted-code self-clutter (periodic autocorr, 250 ksps):")
for nm in ["zadoff-chu-50000-100003-1301-0-2-1.sigmf-meta", "bpsk-50000-2-1.sigmf-meta"]:
    k, p, i = code_clutter(nm)
    print(f"  {nm:45} main lobe +-{k-1} samples   PSLR {p:6.1f} dB   ISLR {i:6.1f} dB")

print("\nDD-image decomposition (peak vs global median vs the pipeline's noise floor):")
print("  floor = dea.noise_floor_log_amp: rms amplitude of the lead delay columns, which")
print("  hold only noise if the echo's leading edge is at lag 0 (not measured for these")
print("  captures). For pure noise the median amplitude sits 10*log10(ln 2) = "
      f"{10*np.log10(np.log(2)):.1f} dB below the rms.")
for label in ["ZC-50k", "BPSK-50k"]:
    log_A = np.load(OUT+f"march_prod_{label}.npy")
    floor = dea.noise_floor_log_amp(log_A)
    peak = float(log_A.max()); med = float(np.log(np.median(np.exp(log_A))))
    print(f"  {label:9}  peak/median = {db(peak - med):5.1f} dB | "
          f"peak/floor = {db(peak - floor):5.1f} dB | "
          f"median/floor = {db(med - floor):5.1f} dB")
