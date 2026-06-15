import numpy as np, json, glob
from scipy.signal import decimate
C=299792458.0
def jload(p):
    try: return json.load(open(p))
    except: return None
def load(p):
    raw=np.fromfile(p,dtype=np.int16); return (raw[0::2]+1j*raw[1::2]).astype(np.complex64)
def find_rx(key):
    for m in sorted(glob.glob("data.camras.nl/thomas/sdr-eme/rx-2025-03-*/*.sigmf-meta")):
        d=jload(m)
        if d and key in d.get("global",{}).get("core:description",""): return m
TXDIR="data.camras.nl/thomas/sdr-eme/tx/"
def profile(label,txname,rxkey,dec=10):
    m=find_rx(rxkey); g=jload(m)["global"]; sr=g["core:sample_rate"]
    dist=g["tracker:distance"]; dop1=g["tracker:doppler"]
    fcap=jload(m)["captures"][0]["core:frequency"]; ftrk=g["tracker:frequency"]
    rtt=2*dist/C; dop_c=2*dop1*(fcap/ftrk)
    tx=decimate(load(TXDIR+txname+".sigmf-data"),dec,ftype='fir')
    rx=decimate(load(m.replace(".sigmf-meta",".sigmf-data")),dec,ftype='fir'); fsd=sr/dec
    emit=int(round(1.0*fsd)); t=np.arange(tx.size)/fsd
    nf=1<<int(np.ceil(np.log2(rx.size+tx.size))); RX=np.fft.fft(rx,nf)
    exp=emit+int(round(rtt*fsd)); lo,hi=exp-int(0.05*fsd),exp+int(0.30*fsd)
    fds=np.arange(dop_c-40,dop_c+40,0.5); prof=[]
    for f in fds:
        cc=np.abs(np.fft.ifft(RX*np.conj(np.fft.fft(tx*np.exp(2j*np.pi*f*t),nf))))
        prof.append(cc[lo:hi].max())
    prof=np.array(prof); noise=np.median(np.abs(np.fft.ifft(RX*np.conj(np.fft.fft(tx*np.exp(2j*np.pi*(dop_c+500)*t),nf))))[lo:hi])
    pk=prof.max(); fwhm=(prof>pk/np.sqrt(2)).sum()*0.5
    # energy integrated over the Doppler spread (incoherent sum of power) vs single peak
    e_int=np.sqrt(np.sum(prof**2)); 
    print(f"{label:10s} peakSNR={20*np.log10(pk/noise):5.1f}dB  Doppler-FWHM={fwhm:4.1f}Hz  "
          f"integ/peak={e_int/pk:4.1f}x  integSNR={20*np.log10(e_int/noise):5.1f}dB")
    return fds-dop_c,prof/noise,label
print("# Echo Doppler profile at matched 50 kHz bandwidth (dec=10, fsd=100k)\n")
for lab,tx,rx in [("ZC-50k","zadoff-chu-50000-100003-1301-0-2-1","zadoff-chu-50000-100003-1301"),
                  ("BPSK-50k","bpsk-50000-2-1","bpsk-50000-2-1")]:
    profile(lab,tx,rx)
