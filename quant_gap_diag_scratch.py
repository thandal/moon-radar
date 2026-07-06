"""Diagnose the ZC vs BPSK real-echo peak/median gap (44.5 vs 35.9 dB).
Is it the BPSK code's self-clutter raising the floor, or a real peak difference?"""
import numpy as np, sigmf
from scipy.signal import decimate
OUT = "results/WAVEFORM_COMPARISON/"; ROOT = "data.camras.nl/thomas/sdr-eme/"

def code_clutter(txname):
    tx = sigmf.sigmffile.fromfile(ROOT+"tx/"+txname, skip_checksum=True).read_samples().astype("complex64")
    tx = decimate(tx, 4, ftype="fir").astype(complex); tx /= np.abs(tx).mean()
    X = np.fft.fft(tx); ac = np.abs(np.fft.ifft(X*np.conj(X))); ac /= ac.max()
    sl = ac.copy(); sl[:8] = 0; sl[-8:] = 0          # exclude ~5-sample ZOH mainlobe
    return 20*np.log10(sl.max()), 10*np.log10((sl**2).sum())

print("ACTUAL transmitted-code self-clutter (periodic autocorr, 250 ksps):")
for nm in ["zadoff-chu-50000-100003-1301-0-2-1.sigmf-meta", "bpsk-50000-2-1.sigmf-meta"]:
    p, i = code_clutter(nm); print(f"  {nm:45} PSLR {p:6.1f} dB   ISLR {i:6.1f} dB")

print("\nDD-image decomposition (peak vs global median vs off-disk floor):")
for label in ["ZC-50k", "BPSK-50k"]:
    A = np.exp(np.load(OUT+f"march_prod_{label}.npy")); nrow, ncol = A.shape
    peak, med = A.max(), np.median(A)
    off = np.r_[A[:nrow//4, :300], A[3*nrow//4:, :300]]   # small delay + extreme Doppler = off-parabola
    floor = np.median(off)
    print(f"  {label:9}  peak/median = {20*np.log10(peak/med):5.1f} dB | "
          f"peak/offdisk-floor = {20*np.log10(peak/floor):5.1f} dB | "
          f"median/floor = {20*np.log10(med/floor):4.1f} dB")
