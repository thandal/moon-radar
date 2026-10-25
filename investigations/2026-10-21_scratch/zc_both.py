"""Roots q that satisfy both rules: ghost margin (nearest full-strength
ghost delay / Moon depth, over Doppler offsets up to the full span) and
ridge motion over 19 Hz <= 0.05 chip."""
import math
import numpy as np

DEPTH = 0.011590685179945387
HALF = 13.0          # Doppler half-span used for ghosts (measured 11.0-12.8 Hz)


def margins(N, chip, qs):
    T = N / chip
    K = int(round(2 * HALF * T))
    k = np.arange(1, K + 1, dtype=np.int64)
    out = np.empty(len(qs))
    for i in range(0, len(qs), 1000):
        qq = qs[i:i + 1000]
        u = np.array([pow(int(q), -1, N) for q in qq], dtype=np.int64)
        d = (u[:, None] * k[None, :]) % N
        d = np.minimum(d, N - d)
        out[i:i + 1000] = d.min(axis=1) / chip / DEPTH
    return out


def ridge(N, q, chip):
    return N * 19.0 / (q * chip)


cases = [("25 kchip/s, 30 s, N=750,000 (theirs)", 750000, 25000, 1351),
         ("50 kchip/s, 30 s, N=1,500,000 (theirs)", 1500000, 50000, 1351),
         ("50 kchip/s, 30 s, N=1,500,007 (ours)", 1500007, 50000, 11401),
         ("250 kchip/s, 30 s, N=7,500,013 (ours)", 7500013, 250000, 11401)]
for name, N, chip, q_ref in cases:
    T = N / chip
    q_min = math.ceil(N * 19.0 / (0.05 * chip))
    cand = np.array([q for q in range(q_min, q_min + 150000) if math.gcd(q, N) == 1])
    m = margins(N, chip, cand)
    best = np.argsort(-m)[:3]
    free = np.array([q for q in range(2, 150000) if math.gcd(q, N) == 1])
    mf = margins(N, chip, free)
    j = int(np.argmax(mf))
    ref_m = margins(N, chip, np.array([q_ref]))[0]
    print(f"{name}")
    print(f"   reference q={q_ref}: ghost margin {ref_m:.2f}, ridge {ridge(N, q_ref, chip):.3f} chip")
    print(f"   best q < 150000, ghost rule only:  q={free[j]} margin {mf[j]:.2f}, ridge {ridge(N, free[j], chip):.3f}")
    for b in best:
        print(f"   both rules (q >= {q_min}):  q={cand[b]} margin {m[b]:.2f}, ridge {ridge(N, cand[b], chip):.3f}")
    print(f"   share of ridge-passing roots with margin >= 1.8: {(m >= 1.8).mean():.1%}, with margin < 1: {(m < 1).mean():.1%}")
