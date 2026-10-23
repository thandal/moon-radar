import ast
import os

import numpy as np

import recommend_zc as rz

# The DD window's lead at the 0.25 Msps the captures are recorded at.
LEAD_S = rz.dd_lead_s(250_000.0)


def _module_constants(path, names):
    """Numeric values assigned to names at the top level of a module's
    source (a plain number, or a number times a unit), read without
    importing it."""
    out = {}
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            value = node.value
            if isinstance(value, ast.BinOp):     # e.g. 1_737_400.0 * au.m
                value = value.left
            out[node.targets[0].id] = ast.literal_eval(value)
    return out


def test_constants_match_the_pipeline():
    """recommend_zc's copies of the DD window's lead and the lunar radius
    equal doppler_equator_alignment's (read from its source: importing it
    loads CuPy and SPICE)."""
    path = os.path.join(os.path.dirname(__file__), "..", "doppler_equator_alignment.py")
    dea = _module_constants(path, {"LEAD_SAMPLES", "MOON_RADIUS"})
    assert dea == {"LEAD_SAMPLES": rz.DD_LEAD_SAMPLES, "MOON_RADIUS": rz.MOON_RADIUS_M}


def test_primality_and_nearby_search():
    assert rz.is_prime(7_500_013)
    assert not rz.is_prime(7_500_000)
    values = rz.nearby_primes(100, 2, 10)
    assert values == [101, 97]


def test_ghost_lag_matches_direct_correlation():
    """An echo offset by j whole Doppler bins correlates at full strength at
    doppler_ghost_lag(n, q, j), and not at all at lag 0, with the
    correlation convention of compute_dd_image (positive lag = later)."""
    n, q = 1009, 7
    m = np.arange(n)
    x = np.exp(-1j * np.pi * q * m * (m + 1) / n)
    for j in (1, 2, -3, 5, 40):
        echo = x * np.exp(2j * np.pi * j * m / n)
        # c[l] = sum_m echo[m] conj(x[m - l]), circular
        c = np.abs(np.fft.ifft(np.fft.fft(echo) * np.conj(np.fft.fft(x))))
        lag = rz.doppler_ghost_lag(n, q, j)
        assert int(np.argmax(c)) == lag
        assert np.isclose(c[lag], n)
        assert c[0] < 1e-6 * n
        # The reverse Doppler sign mirrors the lag.
        assert rz.doppler_ghost_lag(n, q, -j) == (n - lag) % n


def _root_with_first_ghost_at(n, chips):
    """q whose 1-bin Doppler ghost sits `chips` chips from the true delay
    (q^-1 = chips mod n, so q = chips^-1 mod n)."""
    return pow(chips, -1, n)


def test_ghost_check_fails_when_a_ghost_lands_inside_the_moon():
    n, rate, span, limit = 7_500_013, 250_000.0, 19.0, 0.05
    # The 1-bin ghost 100 chips (0.4 ms) from the true delay: inside the
    # 11.6 ms lunar delay depth. The ridge check passes for this q (it is
    # large), so the failure comes from the ghost check alone.
    q = _root_with_first_ghost_at(n, 100)
    result = rz.evaluate_root(n, q, rate, span, limit, lead_s=LEAD_S)
    assert result["coupling_ok"]
    assert (result["ghost_j"], result["ghost_chips"]) == (1, 100)
    assert result["ghost_s"] < rz.LUNAR_DELAY_DEPTH_S
    assert not result["ghost_ok"]
    assert not result["passed"]


def test_ghost_check_passes_when_every_ghost_is_beyond_the_moon():
    n, rate, span, limit = 7_500_013, 250_000.0, 19.0, 0.05
    # Ghosts at j * 5000 chips for j = 1..571 (the 19 Hz span in 1/30 Hz
    # bins): 20 ms and beyond, all short of n/2, so all outside 11.6 ms.
    q = _root_with_first_ghost_at(n, 5000)
    result = rz.evaluate_root(n, q, rate, span, limit, lead_s=LEAD_S)
    assert (result["ghost_j"], result["ghost_chips"]) == (1, 5000)
    assert result["ghost_s"] > rz.LUNAR_DELAY_DEPTH_S
    assert result["passed"]


def test_ghost_in_the_lead_fails():
    """A ghost just beyond the lunar delay depth but within depth + lead
    lands, for echo from near the limb, in the lead columns ahead of lag 0
    (where the noise floor is read): it fails with the lead counted and
    would pass without it."""
    n, rate, span, limit = 7_500_013, 250_000.0, 19.0, 0.05
    depth_chips = rz.LUNAR_DELAY_DEPTH_S * rate     # ~2897.7
    lead_chips = LEAD_S * rate                      # 20 at 1 sample per chip
    chips = int(np.ceil(depth_chips + lead_chips / 2))
    assert depth_chips < chips < depth_chips + lead_chips
    q = _root_with_first_ghost_at(n, chips)
    with_lead = rz.evaluate_root(n, q, rate, span, limit, lead_s=LEAD_S)
    without_lead = rz.evaluate_root(n, q, rate, span, limit, lead_s=0.0)
    assert with_lead["coupling_ok"]
    assert (with_lead["ghost_j"], with_lead["ghost_chips"]) == (1, chips)
    assert np.isclose(with_lead["ghost_limit_s"], rz.LUNAR_DELAY_DEPTH_S + LEAD_S)
    assert not with_lead["ghost_ok"] and not with_lead["passed"]
    assert without_lead["ghost_ok"]


def test_nearest_ghost_is_found_by_brute_force_on_a_short_code():
    """nearest_doppler_ghost agrees with direct correlation over every
    whole-bin offset in the span."""
    n, q, rate, span = 1009, 7, 1009.0, 12.0   # 1 Hz bins, |j| <= 12
    m = np.arange(n)
    x = np.exp(-1j * np.pi * q * m * (m + 1) / n)
    fx = np.conj(np.fft.fft(x))
    best = None
    for j in range(-12, 13):
        if j == 0:
            continue
        c = np.abs(np.fft.ifft(np.fft.fft(x * np.exp(2j * np.pi * j * m / n)) * fx))
        k = int(np.argmax(c))
        d = min(k, n - k)
        if best is None or d < best[1]:
            best = (abs(j), d)
    assert rz.nearest_doppler_ghost(n, q, rate, span)[1] == best[1]
