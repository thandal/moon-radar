"""Unit tests for the terrain models (doppler_equator.terrain_model):

  - names: True/"True"/"lola", False/"False"/"sphere" and "figure" are
    recognized, anything else is refused;
  - the smooth shape (MOON_FIGURE_KM): its mean over the sphere is its mean
    term, and moon_surface_points puts each point at 1737.4 km plus its
    height (the sphere at 1737.4 km);
  - the height at the specular point under the smooth shape is the shape's
    height there.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spice_setup import furnsh_kernels  # noqa: E402
furnsh_kernels()
import doppler_equator as de  # noqa: E402


def test_terrain_model_names():
    for v in (True, "True", "lola"):
        assert de.terrain_model(v) == "lola"
    for v in (False, "False", "sphere"):
        assert de.terrain_model(v) == "sphere"
    assert de.terrain_model("figure") == "figure"
    with pytest.raises(ValueError):
        de.terrain_model("ellipsoid")


def test_figure_surface():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((200000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    h = de.figure_height_km(u)
    assert abs(h.mean() - de.MOON_FIGURE_KM[0]) < 0.02          # other terms average out
    r_fig = np.linalg.norm(de.moon_surface_points(u[:1000], use_dem="figure"), axis=1)
    assert np.allclose(r_fig, 1737.4 + h[:1000], atol=1e-9)
    r_sph = np.linalg.norm(de.moon_surface_points(u[:1000], use_dem="sphere"), axis=1)
    assert np.allclose(r_sph, 1737.4, atol=1e-9)
    # The Earth-facing point (x = 1): mean + x term + (x^2 - y^2) term - (3z^2 - 1) term.
    c = de.MOON_FIGURE_KM
    assert np.isclose(de.figure_height_km(np.array([1.0, 0.0, 0.0])), c[0] + c[1] + c[7] - c[8])


def test_srp_height_on_figure():
    et = de.csp.str2et("2026-10-24T18:30:42")
    elev, delay = de.extract_srp_elevation(et, "DWINGELOO", "STOCKERT", use_dem="figure")
    srp = de.specular_point_bck(et, "DWINGELOO", "STOCKERT")
    assert np.isclose(elev, de.figure_height_km(srp / np.linalg.norm(srp)))
    assert np.isclose(delay, 2 * elev / de.csp.clight())
