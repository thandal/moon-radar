import numpy as np, json, os, glob
C = 299792458.0
TXDIR = "data.camras.nl/thomas/sdr-eme/tx/"

def load_ci16(path, n=None, off=0):
    raw = np.fromfile(path, dtype=np.int16, count=(2*n if n else -1), offset=off*4)
    return (raw[0::2].astype(np.float64) + 1j*raw[1::2].astype(np.float64))

def meta_sr(path):
    m = path.replace(".sigmf-data",".sigmf-meta")
    try: return json.load(open(m))["global"]["core:sample_rate"]
    except Exception: return 1e6

def analyze(name):
    path = TXDIR + name + ".sigmf-data"
    x = load_ci16(path)
    fs = float(meta_sr(path))
    N = x.size
    # constant modulus
    a = np.abs(x); papr = (a.max()**2)/ (np.mean(a**2))
    cm_cv = a.std()/a.mean()
    # bandwidth via Welch-ish PSD
    seg = 1<<14
    nseg = N//seg
    P = np.zeros(seg)
    w = np.hanning(seg)
    for i in range(min(nseg,120)):
        s = x[i*seg:(i+1)*seg]*w
        P += np.abs(np.fft.fftshift(np.fft.fft(s)))**2
    P/=P.max()
    f = np.fft.fftshift(np.fft.fftfreq(seg,1/fs))
    Pl = 10*np.log10(P+1e-20)
    bw3 = f[Pl>-3]; b3 = bw3.max()-bw3.min() if bw3.size else 0
    # 99% power bandwidth
    order=np.argsort(P)[::-1]; cs=np.cumsum(P[order])/P.sum()
    idx=order[:np.searchsorted(cs,0.99)+1]; b99=f[idx].max()-f[idx].min()
    # APERIODIC autocorrelation of the full sequence (matched filter), normalized
    xz = x/np.sqrt(np.mean(np.abs(x)**2))
    nfft = 1<<int(np.ceil(np.log2(2*N)))
    X = np.fft.fft(xz, nfft)
    ac = np.fft.ifft(X*np.conj(X))
    ac = np.abs(ac); ac/=ac.max()
    # main lobe: first drop below 0.5 going out from lag 0
    half = next((k for k in range(1,200) if ac[k]<0.5), 1)
    res_km = (half/fs)*C/2/1000.0   # one-way range res from main-lobe half-width
    # peak sidelobe: ignore main lobe region (+-5 samples and the mirror at the end)
    mask = np.ones(nfft,bool); mask[:8]=False; mask[-8:]=False
    psl = 20*np.log10(ac[mask].max())
    # delay-Doppler coupling on FULL sequence
    t = np.arange(N)/fs
    coup=[]
    Xfull = np.fft.fft(xz, 1<<int(np.ceil(np.log2(2*N))))
    nf2 = Xfull.size
    base = np.abs(np.fft.ifft(Xfull*np.conj(Xfull))); base_pk=base.max()
    for fd in (1.0,5.0,20.0):
        xd = xz*np.exp(2j*np.pi*fd*t)
        cc = np.abs(np.fft.ifft(np.fft.fft(xd,nf2)*np.conj(Xfull)))
        pk = int(np.argmax(cc)); pk = pk if pk<nf2//2 else pk-nf2
        coup.append((fd, pk, cc.max()/base_pk))
    walk = coup[1][1]/coup[1][0]  # samples per Hz at 5 Hz
    loss5 = coup[1][2]
    return dict(name=name, fs=fs, N=N, papr=papr, cm_cv=cm_cv, b3=b3, b99=b99,
                mainlobe_samp=half, res_km=res_km, psl_db=psl, walk_samp_hz=walk, peak_ret_5hz=loss5,
                coup=coup)

waves = {
 "ZC-50k":   "zadoff-chu-50000-100003-1301-0-2-1",
 "ZC-200k":  "zadoff-chu-200000-400009-307-0-2-1",
 "BPSK-50k": "bpsk-50000-2-1",
 "BPSK-200k":"bpsk-200000-2-1",
 "RND-50k":  "rnd-phase-50000-2-1",
 "CHIRP-50k-1":"chirp-50000-1-0-2-1",
 "CHIRP-50k-10":"chirp-50000-10-0-2-1",
 "CW2TONE":  "cw-2tone-100000-2",
}
res={}
print(f"{'label':13s} {'fs(k)':>6} {'B-3dB(k)':>9} {'B99(k)':>8} {'PAPR':>5} {'|x|cv':>6} {'mainlobe':>8} {'res_km':>7} {'PSL_dB':>7} {'walk_s/Hz':>10} {'ret@5Hz':>8}")
for lab,nm in waves.items():
    try:
        r=analyze(nm); res[lab]=r
        print(f"{lab:13s} {r['fs']/1e3:6.0f} {r['b3']/1e3:9.1f} {r['b99']/1e3:8.1f} {r['papr']:5.2f} {r['cm_cv']:6.3f} {r['mainlobe_samp']:8d} {r['res_km']:7.2f} {r['psl_db']:7.1f} {r['walk_samp_hz']:10.1f} {r['peak_ret_5hz']:8.3f}")
    except Exception as e:
        print(f"{lab:13s} ERROR {e}")
np.save("results/WAVEFORM_COMPARISON/tier1_summary.npy", res, allow_pickle=True)
print("\n# coupling detail (fd Hz, peak-walk samples, peak retained):")
for lab,r in res.items():
    print(f"  {lab:13s}", "  ".join(f"{fd:.0f}Hz:{pk:+d}/{ret:.2f}" for fd,pk,ret in r['coup']))
