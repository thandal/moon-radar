import sys
import numpy as np
import scipy.ndimage
sys.path.insert(0, "/tmp/eve_check/blur")
import tiles as T
R = "/home/than/code/moon-radar/results/EVE2026_CIRC"

def ac_widths(g):
    ok = np.isfinite(g)
    v = np.where(ok, g - np.nanmean(g), 0.0)
    F = lambda a: np.fft.rfft2(a, s=(2 * a.shape[0], 2 * a.shape[1]))
    ac = np.fft.fftshift(np.fft.irfft2(np.abs(F(v)) ** 2))
    nw = np.fft.fftshift(np.fft.irfft2(np.abs(F(ok.astype(float))) ** 2))
    ac = ac / np.maximum(nw, 1)
    cy, cx = np.array(ac.shape) // 2
    ac /= ac[cy, cx]
    out = []
    for ang in (np.radians(T.EQ_DEG), np.radians(T.EQ_DEG) + np.pi / 2):
        r = np.arange(0, 40, 0.25)
        prof = scipy.ndimage.map_coordinates(ac, [cy + r * np.sin(ang), cx + r * np.cos(ang)], order=1)
        k = np.argmax(prof < 0.5)
        out.append(2 * T.PX * (r[k - 1] + (prof[k - 1] - 0.5) / (prof[k - 1] - prof[k]) * (r[k] - r[k - 1])))
    return out

for name, f in [("all looks, opposite sense (19)", "stacked_map_stockert_wsrt_chan1"),
                ("long looks, opposite sense (8)", "stacked_map_stockert_wsrt_txmin100s_chan1"),
                ("all looks, co+cross (38)", "stacked_map_stockert_wsrt_dual"),
                ("long looks, co+cross (16)", "stacked_map_stockert_wsrt_txmin100s_dual")]:
    a, c = ac_widths(T.disk_grid(np.load(f"{R}/{f}.npy")))
    print(f"{name:28s} along {a:5.1f} km   across {c:5.1f} km")
