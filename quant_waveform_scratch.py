"""Hard-nosed quantitative ZC-vs-BPSK comparison at matched chip rate.
Intrinsic ambiguity metrics on IDEAL waveforms (noise-free) + real-echo metrics
from the production DD images (march_dd_production_scratch.py). Scratch.
Every printed number is computed here unless marked "(stated)"."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as pl
import recommend_zc as rz
OUT = "results/WAVEFORM_COMPARISON/"
fzc = 50000.0                                  # 50 kHz chip (the March codes)
SPAN_HZ = 19.0                                 # (stated) limb-to-limb lunar Doppler span, recommend_zc's default

# ---- codes at 1 sample/chip (delay resolution = 1/fzc by construction) ----
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
pn_m = mseq(17, [17, 3]).astype(complex)       # N=131071 (31 % longer than the ZC); primitive x^17+x^3+1

def autocorr_metrics(x):
    """Periodic autocorrelation PSLR / ISLR (dB). An ideal ZC has exactly zero
    periodic sidelobes, so its numbers are float64 round-off, not a property."""
    X = np.fft.fft(x); ac = np.abs(np.fft.ifft(X*np.conj(X))); ac /= ac.max()
    sl = ac.copy(); sl[0] = 0.0
    with np.errstate(divide="ignore"):         # an exact zero is -inf dB, shown as such
        return 20*np.log10(sl.max()), 10*np.log10((sl**2).sum()), ac

def doppler_cut_at_true_delay(x, fhz):
    """|zero-lag correlation of x shifted by f Hz with x| / N vs f. For any
    constant-modulus code x*conj(x) = 1, so this is the Dirichlet kernel of the
    code length: it depends on N alone, not on the waveform."""
    xx = x*np.conj(x); nn = np.arange(len(x))
    return np.array([abs(np.sum(xx*np.exp(2j*np.pi*(f/fzc)*nn)))/len(x) for f in fhz])

def zc_coupling_slope_us_per_hz(q, N):         # analytic LFM rate K = q*fzc^2/N; slope = 1/K
    return 1e6 / (q*fzc*fzc/N)

codes = {"ZC": zc, "BPSK (m-seq)": pn_m, "BPSK (random)": pn_rand}
metr = {}
fhz = np.linspace(0, 2, 401)
def half_width_3db(fhz, r):                    # first f where the cut drops below -3 dB
    b = np.where(r < 1/np.sqrt(2))[0]
    if not len(b): raise ValueError("cut stays above -3 dB over the scanned Doppler range")
    return fhz[b[0]]
for name, x in codes.items():
    pslr, islr, ac = autocorr_metrics(x)
    dop = doppler_cut_at_true_delay(x, fhz)
    metr[name] = dict(pslr=pslr, islr=islr, ac=ac, dop=dop, hw=half_width_3db(fhz, dop),
                      T=len(x)/fzc)
slope_zc = zc_coupling_slope_us_per_hz(q, N)
ghost_j, ghost_chips = rz.nearest_doppler_ghost(N, q, fzc, SPAN_HZ)

names = list(codes)
print(f"\n===== INTRINSIC AMBIGUITY (ideal waveforms, noise-free, {fzc/1e3:.0f} kchip/s) =====")
print(f"{'metric':34}" + "".join(f"{nm:>16}" for nm in names))
print(f"{'code length T (s)':34}" + "".join(f"{metr[nm]['T']:>16.3f}" for nm in names))
print(f"{'delay resolution':34}" + "".join(f"{1e6/fzc:>13.0f} us" for nm in names)
      + "   (stated: 1 sample per chip = 1/chip-rate)")
print(f"{'Doppler -3 dB half-width @ delay 0':34}" + "".join(f"{metr[nm]['hw']:>13.3f} Hz" for nm in names)
      + "   (set by T alone: same function for any constant-modulus code)")
print(f"{'delay PSLR (peak sidelobe)':34}" + "".join(f"{metr[nm]['pslr']:>13.0f} dB" for nm in names)
      + "   (ZC: float round-off; ideal is -inf)")
print(f"{'delay ISLR (integrated)':34}" + "".join(f"{metr[nm]['islr']:>13.0f} dB" for nm in names))
print(f"{'delay-Doppler ridge slope':34}{f'{slope_zc:.3f} us/Hz':>16}{'none':>16}{'none':>16}"
      "   (ZC: analytic N/(q fzc^2); BPSK: stated, no chirp structure)")
print(f"   ZC ridge motion over the {SPAN_HZ:g} Hz span: {slope_zc*SPAN_HZ:.2f} us = "
      f"{slope_zc*SPAN_HZ*fzc/1e6:.3f} delay cell (uncompensated estimate)")
print(f"   ZC whole-bin Doppler ghost nearest the true delay within the span: "
      f"{ghost_j} bin(s) ({ghost_j*fzc/N:.2f} Hz) -> {ghost_chips/fzc*1e3:.2f} ms "
      f"(lunar delay depth {rz.LUNAR_DELAY_DEPTH_S*1e3:.2f} ms; BPSK has no such ghost)")

print("\n===== REAL ECHO (production DD images; ZC 15:47:55 vs BPSK 15:42:12 on 2025-03-04, "
      "stated: march_dd_production_scratch.py prints each capture's time and Moon elevation) =====")
for label in ["ZC-50k", "BPSK-50k"]:
    A = np.exp(np.load(OUT+f"march_prod_{label}.npy")); peak, med = A.max(), np.median(A)
    print(f"{label:10} peak/median(noise+clutter) = {20*np.log10(peak/med):5.1f} dB   "
          f"99.9pct/median = {20*np.log10(np.percentile(A,99.9)/med):4.1f} dB")
print("   (different captures ~6 min apart (stated) -> absolute gap conflates waveform + conditions; lean on intrinsic)")

# ---- figure ----
fig, ax = pl.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
for name, c in [("ZC","C0"), ("BPSK (m-seq)","C1"), ("BPSK (random)","C2")]:
    ac = metr[name]['ac']; w = np.r_[ac[-60:], ac[:61]]
    with np.errstate(divide="ignore"):
        ax[0].plot(np.arange(-60,61), 20*np.log10(w), c, label=f"{name} (ISLR {metr[name]['islr']:.0f} dB)", alpha=0.85)
ax[0].set_xlabel(f"delay lag (chips, {1e6/fzc:.0f} us)"); ax[0].set_ylabel("autocorrelation (dB)")
ax[0].set_title("Zero-Doppler delay cut — self-clutter floor (ZC: round-off)"); ax[0].legend(fontsize=8); ax[0].set_ylim(-180,3); ax[0].grid(alpha=0.3)
for name, c, ls in [("ZC","C0","-"), ("BPSK (m-seq)","C1","--")]:
    with np.errstate(divide="ignore"):         # the cut has exact zeros at multiples of 1/T
        ax[1].plot(fhz, 20*np.log10(metr[name]['dop']), c, ls=ls, label=f"{name} (T = {metr[name]['T']:.2f} s)")
ax[1].axhline(-3, color="k", lw=0.4, ls=":"); ax[1].set_xlabel("Doppler (Hz)"); ax[1].set_ylabel("response @ true delay (dB)")
ax[1].set_title("Doppler cut @ true delay — set by code length T"); ax[1].legend(fontsize=8); ax[1].set_ylim(-40,3); ax[1].grid(alpha=0.3)
fig.suptitle(f"ZC vs BPSK — intrinsic ambiguity, {fzc/1e3:.0f} kchip/s", fontsize=11)
fig.savefig(OUT+"quant_ambiguity.png", dpi=120); print("\nwrote", OUT+"quant_ambiguity.png")
