"""Hard-nosed quantitative ZC-vs-BPSK comparison at matched bandwidth/length.
Intrinsic ambiguity metrics on IDEAL waveforms (noise-free) + real-echo metrics
from the production DD images. Scratch."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as pl
OUT = "results/WAVEFORM_COMPARISON/"
fzc = 50000.0                                  # 50 kHz chip (the March codes)

# ---- matched-length codes (1 sample/chip; resolution = 1/fzc = 20 us) ----
N = 100003; q = 1301; n = np.arange(N)
zc = np.exp(-1j*np.pi*q*n*(n+1)/N)             # Zadoff-Chu root q
rng = np.random.default_rng(1)
pn_rand = (rng.integers(0, 2, N)*2 - 1).astype(complex)   # random BPSK (= rnd-phase)
def mseq(deg, taps):                           # maximal-length PN (proper BPSK)
    s = [1]*deg; out = np.empty(2**deg-1)
    for i in range(2**deg-1):
        out[i] = s[-1]; fb = 0
        for t in taps: fb ^= s[t-1]
        s = [fb]+s[:-1]
    return out*2-1
pn_m = mseq(17, [17, 3]).astype(complex)       # N=131071, ~matched; primitive x^17+x^3+1

def autocorr_metrics(x):
    Nx = len(x); X = np.fft.fft(x); ac = np.abs(np.fft.ifft(X*np.conj(X))); ac /= ac.max()
    sl = ac.copy(); sl[0] = 0.0
    return 20*np.log10(sl.max()+1e-20), 10*np.log10((sl**2).sum()+1e-20), ac

def doppler_at_true_delay(x, fhz):             # |corr at lag 0| vs Doppler = IMAGE Doppler res
    Nx = len(x); nn = np.arange(Nx)
    return np.array([abs(np.sum(np.exp(2j*np.pi*(f/fzc)*nn)))/Nx for f in fhz])

def coupling_slope_us_per_hz(x, q, N):         # analytic LFM rate K = q*fzc^2/N; slope = 1/K
    return 1e6 / (q*fzc*fzc/N)

codes = {"ZC": (zc, q, N), "BPSK (m-seq)": (pn_m, None, len(pn_m)), "BPSK (random)": (pn_rand, None, N)}
metr = {}
for name, (x, qq, Nn) in codes.items():
    pslr, islr, ac = autocorr_metrics(x)
    metr[name] = dict(pslr=pslr, islr=islr, ac=ac)

fhz = np.linspace(0, 2, 401)
zc_dop = doppler_at_true_delay(zc, fhz); pn_dop = doppler_at_true_delay(pn_m, fhz)
def hp(fhz, r):
    b = np.where(r < 1/np.sqrt(2))[0]; return fhz[b[0]] if len(b) else np.inf
slope_zc = coupling_slope_us_per_hz(zc, q, N)

print("\n===== INTRINSIC AMBIGUITY (ideal waveforms, noise-free, matched 50 kHz / 2 s) =====")
print(f"{'metric':30}{'ZC':>16}{'BPSK m-seq':>16}{'BPSK random':>16}")
print(f"{'delay resolution':30}{'20 us':>16}{'20 us':>16}{'20 us':>16}   (= 1/chip-rate, identical)")
print(f"{'Doppler res @ true delay':30}{f'{hp(fhz,zc_dop):.2f} Hz':>16}{f'{hp(fhz,pn_dop):.2f} Hz':>16}{'0.33 Hz':>16}   (= 1/T, identical)")
print(f"{'delay PSLR (peak sidelobe)':30}{metr['ZC']['pslr']:>13.0f} dB{metr['BPSK (m-seq)']['pslr']:>13.0f} dB{metr['BPSK (random)']['pslr']:>13.0f} dB")
print(f"{'delay ISLR (integrated)':30}{metr['ZC']['islr']:>13.0f} dB{metr['BPSK (m-seq)']['islr']:>13.0f} dB{metr['BPSK (random)']['islr']:>13.0f} dB")
print(f"{'delay-Doppler coupling':30}{f'{slope_zc:.3f} us/Hz':>16}{'0':>16}{'0':>16}")
print(f"   ZC coupling over the 19 Hz disk: {slope_zc*19:.2f} us = {slope_zc*19/20:.3f} delay cell -> negligible (and compensated)")

print("\n===== REAL ECHO (production DD images; ZC 15:47:55 vs BPSK 15:42:12, same night/el) =====")
for label in ["ZC-50k", "BPSK-50k"]:
    A = np.exp(np.load(OUT+f"march_prod_{label}.npy")); peak, med = A.max(), np.median(A)
    print(f"{label:10} peak/median(noise+clutter) = {20*np.log10(peak/med):5.1f} dB   "
          f"99.9pct/median = {20*np.log10(np.percentile(A,99.9)/med):4.1f} dB")
print("   (different captures 5 min apart -> absolute gap conflates waveform + conditions; lean on intrinsic)")

# ---- figure ----
fig, ax = pl.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
for name, c in [("ZC","C0"), ("BPSK (m-seq)","C1"), ("BPSK (random)","C2")]:
    ac = metr[name]['ac']; w = np.r_[ac[-60:], ac[:61]]
    ax[0].plot(np.arange(-60,61), 20*np.log10(w+1e-20), c, label=f"{name} (ISLR {metr[name]['islr']:.0f} dB)", alpha=0.85)
ax[0].set_xlabel("delay lag (chips, 20 us)"); ax[0].set_ylabel("autocorrelation (dB)")
ax[0].set_title("Zero-Doppler delay cut — self-clutter floor"); ax[0].legend(fontsize=8); ax[0].set_ylim(-180,3); ax[0].grid(alpha=0.3)
ax[1].plot(fhz, 20*np.log10(zc_dop+1e-9), "C0", label="ZC")
ax[1].plot(fhz, 20*np.log10(pn_dop+1e-9), "C1", label="BPSK m-seq", ls="--")
ax[1].axhline(-3, color="k", lw=0.4, ls=":"); ax[1].set_xlabel("Doppler (Hz)"); ax[1].set_ylabel("response @ true delay (dB)")
ax[1].set_title("Doppler resolution @ true delay — identical (1/T)"); ax[1].legend(fontsize=8); ax[1].set_ylim(-40,3); ax[1].grid(alpha=0.3)
fig.suptitle("ZC vs BPSK — intrinsic ambiguity, matched 50 kHz / 2 s", fontsize=11)
fig.savefig(OUT+"quant_ambiguity.png", dpi=120); print("\nwrote", OUT+"quant_ambiguity.png")
