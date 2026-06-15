import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C=299792458.0; TXDIR="data.camras.nl/thomas/sdr-eme/tx/"
def load(path):
    raw=np.fromfile(path,dtype=np.int16); return (raw[0::2]+1j*raw[1::2]).astype(np.complex128)
def sr(path):
    try: return float(json.load(open(path.replace('.sigmf-data','.sigmf-meta')))["global"]["core:sample_rate"])
    except: return 1e6

def chipwidth(x):
    # estimate ZOH chip width = run length of (near-)identical consecutive samples
    d=np.abs(np.diff(x[:200000])); changes=np.where(d>0.5*np.median(np.abs(x)))[0]
    if changes.size<2: return 1
    return int(round(np.median(np.diff(changes))))

def metrics(name):
    p=TXDIR+name+".sigmf-data"; x=load(p); fs=sr(p); N=x.size
    xz=x/np.sqrt(np.mean(np.abs(x)**2))
    cw=max(chipwidth(x),1)
    # APERIODIC autocorr (single-shot matched filter)
    nf=1<<int(np.ceil(np.log2(2*N))); X=np.fft.fft(xz,nf)
    ac=np.abs(np.fft.ifft(X*np.conj(X))); ac/=ac.max()
    half=next((k for k in range(1,400) if ac[k]<0.5),1)
    res_km=(half/fs)*C/2/1e3
    # mask out full main lobe (+-2 chip widths) at both ends
    excl=max(2*cw, half*2)+2
    m=np.ones(nf,bool); m[:excl]=False; m[-excl:]=False
    psl_aper=20*np.log10(ac[m].max())
    # PERIODIC autocorr over the natural sequence length N (cyclic)
    Xp=np.fft.fft(xz); acp=np.abs(np.fft.ifft(Xp*np.conj(Xp))); acp/=acp.max()
    mp=np.ones(N,bool); mp[:excl]=False; mp[-excl:]=False
    psl_per=20*np.log10(acp[mp].max())
    # 2D ambiguity ridge: fine Doppler sweep, record peak delay & height (vs zero-Doppler peak)
    t=np.arange(N)/fs; fds=np.arange(-2.0,2.01,0.1)
    pk_delay=[]; pk_ret=[]
    base=np.abs(np.fft.ifft(np.fft.fft(xz,nf)*np.conj(X))).max()
    for fd in fds:
        cc=np.abs(np.fft.ifft(np.fft.fft(xz*np.exp(2j*np.pi*fd*t),nf)*np.conj(X)))
        k=int(np.argmax(cc)); k=k if k<nf//2 else k-nf
        pk_delay.append(k); pk_ret.append(cc.max()/base)
    pk_delay=np.array(pk_delay); pk_ret=np.array(pk_ret)
    # ridge slope from points where retention>0.5 (coupling); else thumbtack
    good=pk_ret>0.5
    slope=np.polyfit(fds[good],pk_delay[good],1)[0] if good.sum()>=3 else np.nan
    # Doppler tolerance: half-power Doppler width (where retention drops to 0.5)
    halfpow=fds[pk_ret>=0.5]; dop_tol=(halfpow.max()-halfpow.min()) if halfpow.size else 0.0
    return dict(name=name,fs=fs,N=N,cw=cw,res_km=res_km,psl_aper=psl_aper,psl_per=psl_per,
                slope=slope,dop_tol=dop_tol,fds=fds,pk_delay=pk_delay,pk_ret=pk_ret)

waves={"ZC-50k":"zadoff-chu-50000-100003-1301-0-2-1","ZC-200k":"zadoff-chu-200000-400009-307-0-2-1",
 "BPSK-50k":"bpsk-50000-2-1","BPSK-200k":"bpsk-200000-2-1","RND-50k":"rnd-phase-50000-2-1",
 "CHIRP-50k":"chirp-50000-1-0-2-1"}
R={}
print(f"{'label':10s} {'chip(smp)':>9} {'res_km':>7} {'PSL_aper':>9} {'PSL_per':>8} {'ridge_smp/Hz':>13} {'Dop_tol_Hz':>10}")
for lab,nm in waves.items():
    r=metrics(nm); R[lab]=r
    sl=f"{r['slope']:.0f}" if np.isfinite(r['slope']) else "  thumbtack"
    print(f"{lab:10s} {r['cw']:9d} {r['res_km']:7.2f} {r['psl_aper']:9.1f} {r['psl_per']:8.1f} {sl:>13} {r['dop_tol']:10.2f}")

fig,ax=plt.subplots(1,2,figsize=(13,5))
for lab,r in R.items():
    ax[0].plot(r['fds'],r['pk_ret'],marker='.',label=lab)
    ax[1].plot(r['fds'],r['pk_delay'],marker='.',label=lab)
ax[0].set(xlabel='Doppler offset (Hz)',ylabel='peak retained',title='Doppler tolerance (single 2-s matched filter)')
ax[0].axhline(0.5,ls=':',c='k'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].set(xlabel='Doppler offset (Hz)',ylabel='peak delay walk (samples)',title='Delay-Doppler coupling (ridge slope)')
ax[1].set_ylim(-3000,3000); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/WAVEFORM_COMPARISON/ambiguity_compare.png",dpi=110)
print("saved results/WAVEFORM_COMPARISON/ambiguity_compare.png")
np.save("results/WAVEFORM_COMPARISON/ambiguity.npy",R,allow_pickle=True)
