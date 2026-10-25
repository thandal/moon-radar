"""Do the new stack weights help or hinder mirror-ghost cancellation?

The 19 looks of one channel on 2026-10-11 (Stockert + WSRT), with their real
geometry, masks, cell areas, noise floors and the production weights.

A. Geometry: how far each look's Doppler equator is turned (about the line of
   sight) from Stockert 09:35's, and how far apart that puts the two looks'
   ghosts of a point 800 km from the equator.
D. Real data, disagreement between pairs of looks: s = 1/K_i + 1/K_j when the
   two looks see the same surface and the same ghosts (speckle only); more when
   their ghosts fall in different places. Simultaneous Stockert/WSRT pairs share
   the geometry, so they give each look's speckle-only K.
B. Synthetic surface with no mirror symmetry: each look's map = half the
   surface at P + half at its mirror point. Ghost leakage of the stack per
   feature size under each weighting (1 = a single look), and the speckle
   noise each weighting leaves (using the speckle-only K from D).

usage: ghost_weights.py chan1|chan0
"""
import csv
import itertools
import sys

import numpy as np
import healpy as hp

sys.path.insert(0, "/tmp/eve_check/blur")
sys.path.insert(0, "/home/than/code/moon-radar")
import stack_maps as sm
import doppler_equator_alignment as dea
from trace_maps_helpers import geometry, v_pix, NSIDE, RUN
import tiles as T

chan = sys.argv[1]
R_KM = 1737.4
npix = hp.nside2npix(NSIDE)
lo_c, hi_c = np.cos(np.radians(sm.GAIN_INCIDENCE_DEG[::-1]))

rows = []
for st in ("stockert", "wsrt"):
    for r in csv.DictReader(open(f"{RUN}/registration_runs_{st}_{chan}.csv")):
        if float(r.get("pair_snr") or r["tone_snr"]) < 15:
            continue
        r["_rx"] = st.upper()
        for k in ("map_npy", "mult_npy", "area_npy"):
            r[k] = sm.resolve_npy(r[k], RUN)
        rows.append(r)
rows.sort(key=lambda r: (r["rx_start_utc"], r["_rx"]))
names = [f"{r['_rx'][:3].lower()} {r['rx_start_utc'][11:19]}" for r in rows]
N = len(rows)
print(f"{chan}: {N} looks")


def brightness(r, cosi):
    """sm.look_brightness, also returning the old stack's weight on the
    brightness (old intensity = noise + brightness * gain * area)."""
    m = np.load(r["map_npy"]); mult = np.load(r["mult_npy"]); area = np.load(r["area_npy"])
    mask = (m != hp.UNSEEN) & np.isfinite(m) & np.isfinite(area) & (area > 0)
    mask &= mult <= 3.0 * np.median(mult[mult > 0])
    ref = np.median(m[mask])
    noise = np.exp(2.0 * (float(r["noise_log_amp"]) - ref))
    b = np.full(npix, np.nan); b[mask] = (np.exp(2.0 * (m[mask] - ref)) - noise) / area[mask]
    band = mask & (cosi >= lo_c) & (cosi <= hi_c)
    gain = np.median(b[band])
    n = np.full(npix, np.nan); n[mask] = noise / area[mask] / gain
    w_old = np.full(npix, np.nan); w_old[mask] = gain * area[mask]
    return (b / gain).astype(np.float32), n.astype(np.float32), w_old.astype(np.float32)


lmax = 3 * NSIDE - 1
ell = np.arange(lmax + 1)
cl = 1.0 / (ell + 20.0) ** 2            # equal contrast per octave of feature size
cl[:2] = 0
np.random.seed(1)
sigma = hp.synfast(cl, NSIDE, lmax=lmax).astype(np.float32)
sigma /= sigma.std()
sig_m = np.empty((N, npix), np.float32)   # the surface at each look's mirror points (interpolated)
I = np.empty((N, npix), np.float32); Nz = np.empty_like(I); Wold = np.empty_like(I)
band = np.empty((N, npix), bool); mp = np.empty((N, npix), np.int32)
U = []
cache = {}
for i, r in enumerate(rows):
    et = dea.csp.str2et(r["rx_start_utc"])
    cosi = sm.incidence_cos(et, npix, cache, rx_name=r["_rx"])
    I[i], Nz[i], Wold[i] = brightness(r, cosi)
    band[i] = (cosi >= lo_c) & (cosi <= hi_c)
    _, _, u, _ = geometry(r["_rx"], et, float(r["capture_s"]))
    U.append(u)
    vm = v_pix - 2 * (v_pix @ u)[:, None] * u
    mp[i] = hp.vec2pix(NSIDE, *vm.T)
    sig_m[i] = hp.get_interp_val(sigma, *hp.vec2ang(vm))
    del vm
U = np.array(U)

# --- A. geometry ---
ref = names.index("sto 09:35:24")
sgn = np.sign(U @ U[ref])
tilt = np.degrees(np.arccos(np.clip(np.abs(U @ U[ref]), 0, 1)))
print("\nA. Doppler-equator turn from Stockert 09:35 (deg), and the ghost "
      "separation it gives at 800 km from the equator (km)")
for i in range(N):
    print(f"  {names[i]}  tx {float(rows[i]['tx_s']):4.0f} s  {rows[i]['tx_file'][23:31]}"
          f"  turn {tilt[i]:5.2f}  ghost separation {2 * 800 * np.radians(tilt[i]):6.1f}")


def turn(i, j):
    return np.degrees(np.arccos(np.clip(abs(U[i] @ U[j]), 0, 1)))


# --- production weights (as stack_maps.main) ---
acc = sm.Accumulator(npix)
for i in range(N):
    acc.add(I[i].astype(float))
S = sm.smooth_map(acc.mean(min_count=3))
K_prod = np.empty(N)
for i in range(N):
    ok = np.isfinite(I[i]) & (acc.count >= 3)
    others = np.full(npix, np.nan)
    others[ok] = (acc.sum[ok] - I[i][ok]) / (acc.count[ok] - 1)
    cosi_band = np.where(band[i], 0.5 * (lo_c + hi_c), -1.0)     # the band mask only
    K_prod[i], _ = sm.independent_samples(I[i].astype(float), others, cosi_band, Nz[i])

# --- D. pairwise disagreement on real data ---
quiet = (Nz < 0.2 * S[None, :])
s_pair = np.full((N, N), np.nan)
for i, j in itertools.combinations(range(N), 2):
    sel = (np.isfinite(I[i]) & np.isfinite(I[j]) & band[i] & band[j]
           & quiet[i] & quiet[j] & np.isfinite(S))
    a, b = I[i][sel].astype(float), I[j][sel].astype(float)
    rr = np.sum((a - b) ** 2) / np.sum(((a + b) / 2) ** 2)
    s_pair[i, j] = s_pair[j, i] = rr / (1 - rr / 4)

# Speckle-only K: simultaneous Stockert/WSRT pairs (and WSRT 12:08/12:09,
# which have no Stockert partner), each look taken as 2 / s.
K_free = np.full(N, np.nan)
partner = {}
for i in range(N):
    for j in range(N):
        if j != i and (rows[i]["rx_start_utc"] == rows[j]["rx_start_utc"]
                       or {names[i], names[j]} == {"wsr 12:08:02", "wsr 12:09:19"}):
            partner[i] = j
for i, j in partner.items():
    K_free[i] = 2.0 / s_pair[i, j]
print("\nD. Pair disagreement s (= 1/K_i + 1/K_j for speckle alone).")
print("   look            production K   partner (turn deg)   s     K from the pair")
for i in range(N):
    if i in partner:
        j = partner[i]
        print(f"   {names[i]}   {K_prod[i]:6.2f}        {names[j]} ({turn(i, j):4.2f})"
              f"   {s_pair[i, j]:5.2f}   {K_free[i]:5.2f}")
    else:
        print(f"   {names[i]}   {K_prod[i]:6.2f}        no simultaneous partner")
# A look without a partner gets the median of looks with the same
# transmission length and chip rate.
for i in range(N):
    if not np.isfinite(K_free[i]):
        same = [k for k in range(N) if np.isfinite(K_free[k]) and rows[k]["tx_s"] == rows[i]["tx_s"]
                and rows[k]["tx_file"] == rows[i]["tx_file"]]
        K_free[i] = np.median(K_free[same])
        print(f"   {names[i]}: K from similar looks {K_free[i]:.2f}")

print("\n   Excess disagreement over speckle, s - 1/K_i - 1/K_j, by how far apart "
      "the two looks' Doppler equators are turned:")
ex, tu = [], []
for i, j in itertools.combinations(range(N), 2):
    ex.append(s_pair[i, j] - 1 / K_free[i] - 1 / K_free[j]); tu.append(turn(i, j))
ex, tu = np.array(ex), np.array(tu)
for a, b in ((0, 0.3), (0.3, 1), (1, 2), (2, 4), (4, 8), (8, 30)):
    k = (tu >= a) & (tu < b)
    if k.any():
        print(f"     turn {a:3g}-{b:<3g} deg: {k.sum():3d} pairs, excess median {np.median(ex[k]):+.2f}"
              f" (range {ex[k].min():+.2f} to {ex[k].max():+.2f})")


print("\n   Is that excess the ghosts? Correlation of the difference of two looks with "
      "the difference of the other looks' stack read at their two mirror points:")
cr = []
for i, j in itertools.combinations(range(N), 2):
    ok2 = acc.count >= 5
    rest = np.full(npix, np.nan)
    both = ok2 & np.isfinite(I[i]) & np.isfinite(I[j])
    rest[both] = (acc.sum[both] - I[i][both] - I[j][both]) / (acc.count[both] - 2)
    sel = (np.isfinite(I[i]) & np.isfinite(I[j]) & band[i] & band[j] & quiet[i] & quiet[j]
           & np.isfinite(rest[mp[i]]) & np.isfinite(rest[mp[j]]))
    d = I[i][sel].astype(float) - I[j][sel]
    g = rest[mp[i]][sel] - rest[mp[j]][sel]
    cr.append(np.corrcoef(d, g)[0, 1])
cr = np.array(cr)
for a, b in ((0, 0.3), (0.3, 1), (1, 2), (2, 4), (4, 8), (8, 30)):
    k = (tu >= a) & (tu < b)
    if k.any():
        print(f"     turn {a:3g}-{b:<3g} deg: correlation median {np.median(cr[k]):+.2f}"
              f" (range {cr[k].min():+.2f} to {cr[k].max():+.2f})")

# --- B. ghost leakage on a synthetic surface ---
valid = np.isfinite(I) & np.isfinite(S)[None, :]


def stack_terms(W):
    """Real and ghost terms of the stack under weights W[look, pixel]."""
    W = np.where(valid & np.isfinite(W) & (W > 0), W, 0.0).astype(np.float32)
    ws = W.sum(axis=0)
    count = (W > 0).sum(axis=0)
    ok = count >= 3
    ghost = np.full(npix, hp.UNSEEN); real = np.full(npix, hp.UNSEEN)
    g = np.zeros(npix, np.float32)
    for i in range(N):
        g += W[i] * sig_m[i]
    ghost[ok] = g[ok] / ws[ok]
    real[ok] = sigma[ok]
    # speckle variance relative to signal^2: sum W^2 (S + n)^2 / K / (sum W)^2 / S^2
    var = np.zeros(npix)
    for i in range(N):
        var += W[i].astype(float) ** 2 * ((S + np.nan_to_num(Nz[i])) ** 2 / K_free[i])
    sp = np.full(npix, np.nan)
    sp[ok] = np.sqrt(var[ok]) / ws[ok] / S[ok]
    return real, ghost, sp


BANDS = [(150, 75), (75, 40), (40, 20), (20, 12), (12, 8), (8, 6), (6, 4.4)]
nt, stp = int(T.TILE / T.PX), int(T.STEP / T.PX)
TAPER = np.outer(np.hanning(nt), np.hanning(nt))
f = np.fft.fftfreq(nt, d=T.PX)
kr = np.hypot(*np.meshgrid(f, f))
with np.errstate(divide="ignore"):
    WL = np.where(kr > 0, 1 / kr, np.inf)
SEL = [(WL <= hi) & (WL > lo) for hi, lo in BANDS]
ORIGINS = [(i0, j0) for i0 in range(0, len(T.xs) - nt + 1, stp) for j0 in range(0, len(T.xs) - nt + 1, stp)]


def disk_grid(hmap):
    """The healpix map on the 2 km disk grid (no trend removal)."""
    mask = hmap != hp.UNSEEN
    v = hp.get_interp_val(np.where(mask, hmap, 0.0), T.theta, T.phi).reshape(T.X.shape)
    w = hp.get_interp_val(mask.astype(float), T.theta, T.phi).reshape(T.X.shape)
    return np.where((w > 0.99) & T.inside, v / np.maximum(w, 1e-9), np.nan)


def leakage(real, ghost, keep=None):
    if keep is not None:
        real = np.where(keep, real, hp.UNSEEN); ghost = np.where(keep, ghost, hp.UNSEEN)
    a, b = disk_grid(ghost), disk_grid(real)
    pa = np.zeros(len(BANDS)); pb = np.zeros_like(pa); x = np.zeros_like(pa)
    for i0, j0 in ORIGINS:
        ta, tb = a[i0:i0 + nt, j0:j0 + nt], b[i0:i0 + nt, j0:j0 + nt]
        ok = np.isfinite(ta) & np.isfinite(tb)
        if ok.mean() < 0.95:
            continue
        A = np.fft.fft2(np.where(ok, ta - np.nanmean(ta), 0.0) * TAPER)
        B = np.fft.fft2(np.where(ok, tb - np.nanmean(tb), 0.0) * TAPER)
        for k, sl in enumerate(SEL):
            pa[k] += (np.abs(A[sl]) ** 2).sum(); pb[k] += (np.abs(B[sl]) ** 2).sum()
            x[k] += np.real(A[sl] * np.conj(B[sl])).sum()
    # the part of the ghost term that is not a copy of the real surface
    # (near the equator a mirror point is next to its source: blur, not a ghost)
    return (pa - x ** 2 / pb) / pb


k41 = names.index("sto 09:41:25")
S32 = S.astype(np.float32)
ones3 = np.where(valid, 1.0, np.nan).astype(np.float32)
w_prod = (K_prod[:, None].astype(np.float32) / (S32[None, :] + Nz) ** 2)
w_free = (K_free[:, None].astype(np.float32) / (S32[None, :] + Nz) ** 2)
groups = np.digitize(tilt, [2.0, 5.0])     # 09:24-09:49 / 10:16 / 12:08-12:09
w_bal = np.zeros_like(w_free)
okw = valid & np.isfinite(w_free)
wf = np.where(okw, w_free, 0.0)
present = np.zeros(npix)
for g in np.unique(groups):
    present += wf[groups == g].sum(axis=0) > 0
for g in np.unique(groups):
    tot = wf[groups == g].sum(axis=0)
    for i in np.where(groups == g)[0]:
        w_bal[i] = np.where(tot > 0, wf[i] / np.maximum(tot, 1e-30) / np.maximum(present, 1), 0.0)
del wf
schemes = [("old stack (raw intensity, equal weights)", Wold),
           ("per unit area, equal weights", ones3),
           ("per unit area, production weights", w_prod),
           ("per unit area, speckle-only K weights", w_free),
           ("same, each Doppler-equator turn group 1/3", w_bal)]
print("\nB. Ghost leakage of the stack (ghost detail power / real detail power; "
      "a single look = 1), synthetic surface")
print(f"   {'feature size':42s}" + "".join(f"{f'{a:g}-{b:g} km':>10s}" for a, b in BANDS)
      + "   speckle (std / signal)")
# a single look: its ghost is the far hemisphere's surface, as strong as the real one
ghost1 = np.where(valid[k41], sig_m[k41], hp.UNSEEN); real1 = np.where(valid[k41], sigma, hp.UNSEEN)
print(f"   {'one look (Stockert 09:41)':42s}" + "".join(f"{x:10.3f}" for x in leakage(real1, ghost1)))
far = np.abs(v_pix @ U[ref]) * R_KM > 300
far_rows = [("one look (Stockert 09:41)", leakage(real1, ghost1, far))]
for label, Wt in schemes:
    real, ghost, sp = stack_terms(Wt)
    L = leakage(real, ghost)
    far_rows.append((label, leakage(real, ghost, far)))
    print(f"   {label:42s}" + "".join(f"{x:10.3f}" for x in L)
          + f"   {np.nanmedian(sp[np.isfinite(sp)]):.3f}")
print("   Only more than 300 km from the Doppler equator:")
for label, L in far_rows:
    print(f"   {label:42s}" + "".join(f"{x:10.3f}" for x in L))
# how much each look counts overall, per scheme (share of the total weight)
print("\n   Share of the stack each look carries (mean over the disk of its weight / the pixel's total, %):")
for k, sc in enumerate(schemes):
    print(f"     column {k + 1}: {sc[0]}")
print(f"   {'look':14s}" + "".join(f"{k + 1:>16d}" for k in range(len(schemes))))
shares = []
for label, Wt in schemes:
    Wv = np.where(valid & np.isfinite(Wt) & (Wt > 0), Wt, 0.0).astype(np.float32)
    tot = Wv.sum(axis=0)
    okp = (Wv > 0).sum(axis=0) >= 3
    shares.append(100 * (Wv[:, okp] / tot[okp]).mean(axis=1))
    del Wv
for i in range(N):
    print(f"   {names[i]:14s}" + "".join(f"{sh[i]:16.1f}" for sh in shares))
