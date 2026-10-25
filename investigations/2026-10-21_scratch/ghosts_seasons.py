"""How much do the 2025 sessions add to mirror-ghost cancellation?

Synthetic surface with no mirror symmetry; each look's map = half the surface
at P + half at P's mirror across that look's Doppler-equator plane, over the
look's real valid pixels (as stack_maps masks them). Ghost leakage of an
equal-weight stack per feature size = power of the ghost term that is not a
copy of the real surface / power of the real surface (one look = 1). Same
tiles for every group. Looks are those the all-seasons stack kept (its log's
exclusion list). usage: ghosts_seasons.py STACK_LOG CHAN"""
import csv, re, sys
import numpy as np, healpy as hp
sys.path.insert(0, "/tmp/eve_check/blur"); sys.path.insert(0, "/home/than/code/moon-radar")
import stack_maps as sm
import doppler_equator_alignment as dea
from trace_maps_helpers import geometry, v_pix, NSIDE
import tiles as T

log, chan = sys.argv[1], sys.argv[2]
excluded = set(re.findall(r"^  - (\S+): ", open(log).read(), re.M))
RUNS = ["/home/than/code/moon-radar/results/STOCKERT_2025", "/home/than/code/moon-radar/results/EVE2026_CIRC"]
rows = []
for d in RUNS:
    for st in ("stockert", "wsrt"):
        try:
            f = open(f"{d}/registration_runs_{st}_{chan}.csv")
        except FileNotFoundError:
            continue
        for r in csv.DictReader(f):
            if r["rx_file"] in excluded:
                continue
            r["_rx"] = st.upper(); r["_sess"] = sm.session_of(r["rx_file"])
            for k in ("map_npy", "mult_npy", "area_npy"):
                r[k] = sm.resolve_npy(r[k], d)
            rows.append(r)
rows.sort(key=lambda r: r["rx_start_utc"])
npix = hp.nside2npix(NSIDE)
sessions = sorted({r["_sess"] for r in rows})
print(f"{chan}: {len(rows)} looks", {s: sum(r['_sess'] == s for r in rows) for s in sessions}, flush=True)

lmax = 3 * NSIDE - 1
ell = np.arange(lmax + 1)
cl = 1.0 / (ell + 20.0) ** 2            # equal contrast per octave of feature size
cl[:2] = 0
np.random.seed(1)
sigma = hp.synfast(cl, NSIDE, lmax=lmax).astype(np.float32)
sigma /= sigma.std()

groups = {s: [s] for s in sessions}
groups["2025, all three sessions"] = [s for s in sessions if s.startswith("2025")]
groups["all seasons"] = sessions
acc = {g: (np.zeros(npix, np.float32), np.zeros(npix, np.int32)) for g in groups}
one = None
U = {}
for i, r in enumerate(rows):
    m = np.load(r["map_npy"]); mult = np.load(r["mult_npy"]); area = np.load(r["area_npy"])
    mask = (m != hp.UNSEEN) & np.isfinite(m) & np.isfinite(area) & (area > 0)
    mask &= mult <= 3.0 * np.median(mult[mult > 0])
    et = dea.csp.str2et(r["rx_start_utc"])
    _, _, u, _ = geometry(r["_rx"], et, float(r["capture_s"]))
    U.setdefault(r["_sess"], []).append(u)
    vm = v_pix[mask] - 2 * (v_pix[mask] @ u)[:, None] * u
    sm_ = np.zeros(npix, np.float32)
    sm_[mask] = hp.get_interp_val(sigma, *hp.vec2ang(vm))
    for g, ss in groups.items():
        if r["_sess"] in ss:
            acc[g][0][mask] += sm_[mask]; acc[g][1][mask] += 1
    if one is None and r["_sess"] == "2026-10-11":
        one = (np.where(mask, sm_, hp.UNSEEN), np.where(mask, sigma, hp.UNSEEN), r["rx_start_utc"][:19])
    if i % 20 == 0:
        print(f"  {i}/{len(rows)}", flush=True)

# Doppler-equator orientation per session: turn of each look's mirror normal
# from the 2026-10-11 mean, in the lunar body frame
U26 = np.array(U["2026-10-11"])
ref = np.mean(U26 * np.sign(U26 @ U26[0])[:, None], axis=0)
ref /= np.linalg.norm(ref)
print("\nDoppler-equator turn from the 2026-10-11 mean (deg): session min / median / max")
for s in sessions:
    t = np.degrees(np.arccos(np.clip(np.abs(np.array(U[s]) @ ref), 0, 1)))
    print(f"  {s:14s} {t.min():6.1f} {np.median(t):6.1f} {t.max():6.1f}   ({len(t)} looks)")

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
    mask = hmap != hp.UNSEEN
    v = hp.get_interp_val(np.where(mask, hmap, 0.0), T.theta, T.phi).reshape(T.X.shape)
    w = hp.get_interp_val(mask.astype(float), T.theta, T.phi).reshape(T.X.shape)
    return np.where((w > 0.99) & T.inside, v / np.maximum(w, 1e-9), np.nan)


grids = {}
for g, (s, n) in acc.items():
    ok = n >= 3
    grids[g] = (disk_grid(np.where(ok, s / np.maximum(n, 1), hp.UNSEEN)), disk_grid(np.where(ok, sigma, hp.UNSEEN)))
grids["one look (" + one[2] + ")"] = (disk_grid(one[0]), disk_grid(one[1]))
tiles = [o for o in ORIGINS if all(np.isfinite(a[o[0]:o[0] + nt, o[1]:o[1] + nt]).mean() >= 0.95
                                   for a, _ in grids.values())]
print(f"\nGhost leakage (ghost detail power not copying the real surface / real detail power; "
      f"one look = 1), {len(tiles)} common tiles")
print(f"  {'feature size':30s}" + "".join(f"{f'{a:g}-{b:g} km':>10s}" for a, b in BANDS))
for g, (a, b) in sorted(grids.items(), key=lambda kv: not kv[0].startswith("one")):
    pa = np.zeros(len(BANDS)); pb = np.zeros_like(pa); x = np.zeros_like(pa)
    for i0, j0 in tiles:
        ta, tb = a[i0:i0 + nt, j0:j0 + nt], b[i0:i0 + nt, j0:j0 + nt]
        ok = np.isfinite(ta) & np.isfinite(tb)
        A = np.fft.fft2(np.where(ok, ta - np.nanmean(ta), 0.0) * TAPER)
        B = np.fft.fft2(np.where(ok, tb - np.nanmean(tb), 0.0) * TAPER)
        for k, sl in enumerate(SEL):
            pa[k] += (np.abs(A[sl]) ** 2).sum(); pb[k] += (np.abs(B[sl]) ** 2).sum()
            x[k] += np.real(A[sl] * np.conj(B[sl])).sum()
    print(f"  {g:30s}" + "".join(f"{v:10.3f}" for v in (pa - x ** 2 / pb) / pb))
