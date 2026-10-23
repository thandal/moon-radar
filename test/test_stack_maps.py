"""Unit tests for stack_maps' statistics and the timing rule (synthetic data).

  - ScatterLaw: a value in every bin with clear signal, including the end
    bins all the way to their edges (no hole at the specular point); none in
    bins whose mean is noise; linear between neighbouring valid centres.
  - closure_noise_probability: matches a pure-noise simulation of the
    least-squares closure.
  - registration_analysis.log_bandpass: pixels whose own mean is not
    positive do not punch holes where the neighbourhood has signal.
  - look_brightness: a look with no signal above its noise floor is
    rejected, a clear one is accepted with gain ~1 after normalization.
  - freq_offset_hunt.leading_peak: the first peak after the edge, not the
    strongest; unresolved, early-edge and weak profiles are flagged.
"""
import os
import sys
import tempfile

import numpy as np
import healpy as hp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import stack_maps as sm  # noqa: E402
import registration_analysis as ra  # noqa: E402
import freq_offset_hunt as foh  # noqa: E402


def test_scatter_law_bins_and_ends():
    rng = np.random.default_rng(0)
    law = sm.ScatterLaw()
    n = 200000
    for _ in range(6):
        cosi = rng.uniform(0.0, 1.0, n)
        signal = np.where(cosi > 0.3, cosi ** 4, 0.0)        # nothing below cos 0.3
        I = signal + 0.05 * rng.standard_normal(n)          # noise around it
        law.collect(I, cosi)
    none, one, weak = law.fit()
    assert none == 0 and one == 0
    good = np.isfinite(law.law)
    assert good[law.centers > 0.35].all()                   # clear signal: a value
    assert not good[law.centers < 0.28].any()               # noise only: none
    # End bin: pixels between its centre and cos = 1 keep the bin's value.
    v = law.value(np.array([0.99, 0.995, 0.9999]))
    assert np.all(np.isfinite(v)) and np.allclose(v, law.law[-1])
    # Between two valid centres: linear.
    c = law.centers
    mid = 0.5 * (c[-3] + c[-2])
    assert np.isclose(law.value(np.array([mid]))[0], 0.5 * (law.law[-3] + law.law[-2]))
    # In a bin without a value, and below COS_MIN: no value.
    assert np.isnan(law.value(np.array([0.1, 0.01]))).all()


def test_closure_probability_matches_noise_simulation():
    rng = np.random.default_rng(1)
    for n in (3, 4):
        names = [str(i) for i in range(n)]
        pairs = [(names[i], names[j]) for i in range(n) for j in range(i + 1, n)]
        ps = []
        for _ in range(4000):
            meas = {p: tuple(rng.standard_normal(2)) + (0.0, 0.0, False) for p in pairs}
            off = sm.solve_offsets(meas, names, names[0])
            res = sum((m[0] - (off[b][0] - off[a][0])) ** 2 + (m[1] - (off[b][1] - off[a][1])) ** 2
                      for (a, b), m in meas.items())
            raw = sum(m[0] ** 2 + m[1] ** 2 for m in meas.values())
            ps.append(sm.closure_noise_probability(np.sqrt(res / raw), n))
        ps = np.array(ps)
        # Under pure noise the probability is uniform on 0..1.
        assert abs(np.mean(ps < 0.05) - 0.05) < 0.015, (n, np.mean(ps < 0.05))
        assert abs(np.mean(ps < 0.5) - 0.5) < 0.04, (n, np.mean(ps < 0.5))


def test_log_bandpass_keeps_coverage_with_nonpositive_pixels():
    rng = np.random.default_rng(2)
    y, x = np.mgrid[0:300, 0:300]
    truth = 1.0 + 0.3 * np.sin(x / 9.0) * np.cos(y / 13.0)
    grid = truth + 1.5 * rng.standard_normal(truth.shape)   # many pixels <= 0
    assert (grid <= 0).mean() > 0.2
    band = ra.log_bandpass(grid, 4.0, 33.0)
    interior = band[80:220, 80:220]
    assert (interior != 0).mean() > 0.99
    log_band = ra.bandpass(np.where(grid > 0, np.log(np.where(grid > 0, grid, 1)), np.nan),
                           4.0, 33.0)
    assert (log_band[80:220, 80:220] != 0).mean() < 0.5       # the old way: mostly lost


def _look_files(tmp, nside, level, noise_log_amp):
    npix = hp.nside2npix(nside)
    m = np.full(npix, 0.5 * np.log(level), dtype=np.float32)
    paths = {k: os.path.join(tmp, f"{k}.npy") for k in ("map", "mult", "area")}
    np.save(paths["map"], m)
    np.save(paths["mult"], np.ones(npix, dtype=np.float32))
    np.save(paths["area"], np.ones(npix, dtype=np.float32))
    return {"map_npy": paths["map"], "mult_npy": paths["mult"], "area_npy": paths["area"],
            "noise_log_amp": str(noise_log_amp)}


def test_look_brightness_rejects_noise_only_look():
    nside = 64
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    cosi = np.cos(theta)                        # incidence from the north pole
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        # Clear signal: level 2 over a noise floor of 1.
        r = _look_files(tmp, nside, 2.0, 0.0)
        I, gain = sm.look_brightness(r, cosi)
        lo, hi = np.cos(np.radians(sm.GAIN_INCIDENCE_DEG[::-1]))
        band = (cosi >= lo) & (cosi <= hi)
        assert gain > 0 and np.isclose(np.nanmean(I[band]), 1.0)
        # Noise only: pixel intensity scattered around the floor.
        npix = hp.nside2npix(nside)
        m = 0.5 * np.log(np.clip(1.0 + 0.3 * rng.standard_normal(npix), 0.05, None))
        np.save(r["map_npy"], m.astype(np.float32))
        try:
            sm.look_brightness(r, cosi)
        except sm.LookRejected:
            pass
        else:
            raise AssertionError("a noise-only look was accepted")


def test_leading_peak_rules():
    x = np.arange(-20, 41)

    def tri(c, h, w=5):
        return h * np.clip(1 - np.abs(x - c) / w, 0, None)

    T = foh.MIN_TONE_SNR
    # A faded specular peak, a stronger later one: the first wins.
    k, st = foh.leading_peak(4 + tri(7, 26) + tri(35, 66), 5, T)
    assert (x[k], st) == (7, "resolved")
    # Small wiggles on the rising slope are not peaks.
    rng = np.random.default_rng(4)
    noisy = (4 + tri(9, 56)) * (1 + 0.05 * rng.standard_normal(len(x)))
    k, st = foh.leading_peak(noisy, 5, T)
    assert abs(x[k] - 9) <= 1 and st == "resolved"
    # A slow rise with no specular peak is not resolved.
    _, st = foh.leading_peak(4 + 100 * np.exp(-((x - 25) / 12.0) ** 2), 5, T)
    assert st == "peak_not_resolved"
    # Above threshold at the first shift: the edge may lie outside the search.
    _, st = foh.leading_peak(20 + tri(-18, 40), 5, T)
    assert st == "edge_at_search_start"
    # Never reaching the threshold.
    _, st = foh.leading_peak(4 + tri(5, 8), 5, T)
    assert st == "below_threshold"
