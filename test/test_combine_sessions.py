"""Unit tests for combine_sessions' building blocks (synthetic data).

  - band: one band alone equals that band of the full split, and the bands
    of a map add back up to it.
  - level_plan: no mask (the default), per-band cell-area cuts (finest
    band 2x the look's median, none above 75 km), or the single
    multiplicity limit.
  - half_of_capture: alternate captures in time within each session, both
    channels of a capture in the same half.
  - combine: weights zero or positive, summing to one where any session is
    present; a lone session gets all the weight; the less noisy of two
    otherwise equal sessions gets more.
  - stack_maps.law_key: one scattering law per channel and season.
"""
import os
import sys

import numpy as np
import healpy as hp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import combine_sessions as cs  # noqa: E402
import stack_maps as sm  # noqa: E402


def test_band_matches_full_split():
    nside = 32
    rng = np.random.default_rng(0)
    m = rng.standard_normal(hp.nside2npix(nside)).astype(np.float32)
    mask = np.ones(len(m), bool)
    mask[:200] = False
    sm_ = cs.Smoother(nside)
    full = cs.bands(m, mask, "k", sm_)
    total = np.nansum(full, axis=0)
    ok = np.all(np.isfinite(full), axis=0)
    assert np.allclose(total[ok], m[ok], atol=1e-4)
    for b in range(len(cs.BAND_NAMES)):
        one = cs.band(m, mask, "k", sm_, b)
        both = np.isfinite(one) & np.isfinite(full[b])
        assert np.array_equal(np.isfinite(one), np.isfinite(full[b]))
        assert np.allclose(one[both], full[b][both], atol=1e-6)


def test_level_plan():
    assert cs.level_plan() == cs.level_plan("none")
    levels, lob, ref = cs.level_plan("cuts")
    assert [levels[k][1] for k in lob] == list(cs.CUT_X_MEDIAN)
    assert levels[lob[0]][1] == 2.0 and np.isinf(levels[lob[-1]][1])
    assert all(np.isinf(lv[0]) for lv in levels)
    assert ref not in levels
    levels, lob, ref = cs.level_plan("limit")
    assert levels == [ref] and lob == [0] * len(cs.BAND_NAMES)
    levels, lob, ref = cs.level_plan("none")
    assert levels == [(np.inf, np.inf)] and lob == [0] * len(cs.BAND_NAMES)


def test_half_of_capture():
    rows = []
    for t in ("10_00_00", "10_05_00", "10_10_00", "10_15_00"):
        for chan in ("chan1", "chan0"):
            rows.append({"rx_file": f"stockert_radar_2025_09_16_{t}_x.{chan}.sigmf-meta",
                         "rx_start_utc": f"2025-09-16T{t.replace('_', ':')}"})
    half = cs.half_of_capture(rows)
    assert [half[f"stockert_radar_2025_09_16_{t}_x"] for t in
            ("10_00_00", "10_05_00", "10_10_00", "10_15_00")] == [0, 1, 0, 1]


def test_combine_weights():
    npix = 4
    B = [np.array([1.0, 2.0, np.nan, 1.0]), np.array([3.0, np.nan, np.nan, 1.0])]
    present = [np.isfinite(b) for b in B]
    z = np.zeros(npix)
    G = [[z, z], [z, z]]
    N = [np.full(npix, 1.0), np.array([1.0, 1.0, 1.0, 4.0])]
    comb, W, VAR = cs.combine(B, G, N, present)
    assert np.all(W >= 0)
    assert np.allclose(W[:, 0].sum(), 1.0) and np.isclose(comb[0], 2.0)    # equal noise: mean
    assert np.allclose(W[:, 1], [1.0, 0.0]) and np.isclose(comb[1], 2.0)   # lone session
    assert np.isnan(comb[2]) and np.all(W[:, 2] == 0)                      # nobody there
    assert W[0, 3] > W[1, 3]                                               # less noise, more weight


def test_law_key_by_season():
    r25 = {"_chan": "chan1", "rx_file": "stockert_radar_2025_09_16_13_00_00_x.chan1.sigmf-meta"}
    r26 = {"_chan": "chan1", "rx_file": "wsrt_moon_2026_10_11_09_24_58_x.chan1.sigmf-meta"}
    assert sm.law_key(r25) == ("chan1", "2025")
    assert sm.law_key(r26) == ("chan1", "2026")
