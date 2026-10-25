import numpy as np, healpy as hp
b = "stockert_25m_moon_2026_10_11_09_41_25_1299.650MHz_0.25Msps_ci16_le.chan1.sigmf-meta"
for run in ("results/EVE2026_CIRC", "results/EVE2026_CIRC_NSIDE800"):
    m = np.load(f"{run}/{b}_map.npy"); mu = np.load(f"{run}/{b}_mapcount.npy")
    ok = m != hp.UNSEEN
    med = np.median(mu[mu > 0])
    print(run, "nside", hp.npix2nside(len(m)), "median multiplicity", med,
          "| quantiles 50/75/90/99:", np.quantile(mu[ok], [.5, .75, .9, .99]),
          "| share of seen pixels masked (mult > 3 x median):", f"{(mu[ok] > 3 * med).mean():.3f}")
