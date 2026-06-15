import numpy as np, json, glob, os, re
from scipy.signal import decimate
C=299792458.0
def jload(p):
    try: return json.load(open(p))
    except: return None
def load(p,n=None,off=0):
    raw=np.fromfile(p,dtype=np.int16,count=(2*n if n else -1),offset=off*4); return (raw[0::2]+1j*raw[1::2]).astype(np.complex64)

def find_rx(key):
    for m in sorted(glob.glob("data.camras.nl/thomas/sdr-eme/rx-2025-03-*/*.sigmf-meta")):
        d=jload(m)
        if d and key in d.get("global",{}).get("core:description",""):
            return m
    return None

TXDIR="data.camras.nl/thomas/sdr-eme/tx/"
def echo_test(label, txname, rxkey, dec=10):
    m=find_rx(rxkey); 
    if not m: print(label,"no rx"); return
    g=jload(m)["global"]; sr=g["core:sample_rate"]
    dist=g["tracker:distance"]; dop1=g["tracker:doppler"]; fcap=jload(m)["captures"][0]["core:frequency"]
    ftrk=g["tracker:frequency"]
    rtt=2*dist/C                       # round-trip light time
    dop_rt=2*dop1*(fcap/ftrk)          # round-trip Doppler at the radar freq
    tx=load(TXDIR+txname+".sigmf-data")
    rx=load(m.replace(".sigmf-meta",".sigmf-data"))
    # decimate (signal is <=200 kHz wide, well within 1 MHz/dec)
    tx=decimate(tx,dec,ftype='fir'); rx=decimate(rx,dec,ftype='fir'); fsd=sr/dec
    emit=int(round(1.0*fsd))           # TX emits at rx_start+1.0s
    # search a window of RX around the expected echo and brute Doppler near dop_rt
    t=np.arange(tx.size)/fsd
    nf=1<<int(np.ceil(np.log2(rx.size+tx.size)))
    RX=np.fft.fft(rx,nf)
    exp_delay=emit+int(round(rtt*fsd))
    best=None; grid=np.arange(dop_rt-60,dop_rt+60,0.5)
    prof_at_best=None
    for f in grid:
        TXm=np.fft.fft(tx*np.exp(2j*np.pi*f*t),nf)
        cc=np.abs(np.fft.ifft(RX*np.conj(TXm)))
        # echo region: +/-0.2s around expected delay
        lo,hi=exp_delay-int(0.05*fsd),exp_delay+int(0.30*fsd)
        seg=cc[lo:hi]; pk=seg.max(); 
        if best is None or pk>best[0]: best=(pk,f,lo+int(np.argmax(seg)),cc.copy())
    pk,fbest,kbest,cc=best
    # noise: median of |cc| in a signal-free zone (well after echo)
    noise=np.median(cc[int(6.0*fsd):int(6.8*fsd)]) if cc.size>int(6.8*fsd) else np.median(cc)
    snr=20*np.log10(pk/noise)
    delay_s=(kbest-emit)/fsd
    print(f"{label:12s} rtt={rtt:.3f}s dop_rt~{dop_rt:.0f}Hz | echo: delay={delay_s:.3f}s (exp {rtt:.3f}) dop={fbest:.1f}Hz peakSNR={snr:.1f}dB")
    return dict(label=label,snr=snr,delay=delay_s,rtt=rtt,dop=fbest)

print("# Monostatic Dwingeloo echo detection, matched bandwidth (50 kHz), 2-s code, ~2.4s round trip\n")
echo_test("ZC-50k",  "zadoff-chu-50000-100003-1301-0-2-1","zadoff-chu-50000-100003-1301")
echo_test("BPSK-50k","bpsk-50000-2-1","bpsk-50000-2-1")
echo_test("RND-50k", "rnd-phase-50000-2-1","rnd-phase-50000-2-1")
echo_test("ZC-200k", "zadoff-chu-200000-400009-307-0-2-1","zadoff-chu-200000-400009-307", dec=4)
echo_test("BPSK-200k","bpsk-200000-2-1","bpsk-200000-2-1", dec=4)
