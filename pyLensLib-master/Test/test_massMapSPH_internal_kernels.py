import numpy as np
import pytest

import pyLensLib.cluster as cluster_module
from pyLensLib.cluster import (
    _SPH_KERNEL_CUBIC_SPLINE,
    _SPH_KERNEL_CUBIC_SPLINE_GAMMA,
    _SPH_KERNEL_WENDLAND_C2,
    _SPH_KERNEL_WENDLAND_C2_GAMMA,
    _kernel_cubic_spline_2d_py,
    _kernel_wendland_c2_2d_py,
    _mass_map_sph_native,
    _sph_kernel_gamma,
    _sph_kernel_id,
)


def _sample_particles():
    rng = np.random.default_rng(42)
    pos = rng.normal(0.0, 0.25, size=(48, 3))
    mass = rng.uniform(0.5, 2.0, size=pos.shape[0])
    return pos, mass


def test_sph_kernel_selector_accepts_expected_names():
    assert _sph_kernel_id("cubic_spline") == _SPH_KERNEL_CUBIC_SPLINE
    assert _sph_kernel_id("cubic") == _SPH_KERNEL_CUBIC_SPLINE
    assert _sph_kernel_id("wendland_c2") == _SPH_KERNEL_WENDLAND_C2
    assert _sph_kernel_id("wendland") == _SPH_KERNEL_WENDLAND_C2

    with pytest.raises(ValueError, match="Unsupported SPH kernel"):
        _sph_kernel_id("gaussian")


def test_sph_kernel_gamma_defaults_match_kernel_conventions():
    assert _sph_kernel_gamma(_SPH_KERNEL_CUBIC_SPLINE) == _SPH_KERNEL_CUBIC_SPLINE_GAMMA
    assert _sph_kernel_gamma(_SPH_KERNEL_WENDLAND_C2) == _SPH_KERNEL_WENDLAND_C2_GAMMA
    assert _sph_kernel_gamma(_SPH_KERNEL_WENDLAND_C2, kernel_gamma=2.0) == 2.0

    with pytest.raises(ValueError, match="kernel_gamma must be positive"):
        _sph_kernel_gamma(_SPH_KERNEL_WENDLAND_C2, kernel_gamma=0.0)


@pytest.mark.parametrize(
    "kernel,kernel_gamma,qmax",
    [
        (_kernel_cubic_spline_2d_py, _SPH_KERNEL_CUBIC_SPLINE_GAMMA, 2.0),
        (_kernel_wendland_c2_2d_py, _SPH_KERNEL_WENDLAND_C2_GAMMA, _SPH_KERNEL_WENDLAND_C2_GAMMA),
        (_kernel_wendland_c2_2d_py, 2.0, 2.0),
    ],
)
def test_sph_kernel_normalization_is_unity_in_2d(kernel, kernel_gamma, qmax):
    h = 0.37
    q = np.linspace(0.0, qmax, 20001)
    if kernel is _kernel_wendland_c2_2d_py:
        w = np.array([kernel(float(qi), h, kernel_gamma) for qi in q])
    else:
        w = np.array([kernel(float(qi), h) for qi in q])
    integrate = getattr(np, "trapezoid", np.trapz)
    integral = 2.0 * np.pi * h * h * integrate(q * w, q)

    assert np.isclose(integral, 1.0, rtol=0.0, atol=2.0e-8)


@pytest.mark.parametrize("kernel", ["cubic_spline", "wendland_c2"])
@pytest.mark.parametrize("local_mass_conservation", [False, True])
def test_mass_map_sph_native_conserves_mass_for_supported_kernels(
    kernel, local_mass_conservation
):
    pos, mass = _sample_particles()
    img = _mass_map_sph_native(
        pos,
        mass,
        npix=64,
        xmin=-1.0,
        xmax=1.0,
        ymin=-1.0,
        ymax=1.0,
        nb=12,
        min_hsml=0.03,
        max_hsml=0.3,
        use_numba=False,
        local_mass_conservation=local_mass_conservation,
        kernel=kernel,
    )

    assert img.shape == (64, 64)
    assert np.all(np.isfinite(img))
    assert np.all(img >= 0.0)
    np.testing.assert_allclose(img.sum(), mass.sum(), rtol=0.0, atol=1.0e-10)


def test_wendland_c2_changes_mass_map_shape_but_not_total_mass():
    pos, mass = _sample_particles()
    kwargs = dict(
        npix=64,
        xmin=-1.0,
        xmax=1.0,
        ymin=-1.0,
        ymax=1.0,
        nb=12,
        min_hsml=0.03,
        max_hsml=0.3,
        use_numba=False,
        local_mass_conservation=True,
    )

    cubic = _mass_map_sph_native(pos, mass, kernel="cubic_spline", **kwargs)
    wendland = _mass_map_sph_native(pos, mass, kernel="wendland_c2", **kwargs)
    wendland_gamma2 = _mass_map_sph_native(
        pos, mass, kernel="wendland_c2", kernel_gamma=2.0, **kwargs
    )

    np.testing.assert_allclose(cubic.sum(), mass.sum(), rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(wendland.sum(), mass.sum(), rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(wendland_gamma2.sum(), mass.sum(), rtol=0.0, atol=1.0e-10)
    assert np.max(np.abs(cubic - wendland)) > 0.0
    assert np.max(np.abs(wendland - wendland_gamma2)) > 0.0


@pytest.mark.skipif(not cluster_module._NUMBA_AVAILABLE, reason="numba is not available")
@pytest.mark.parametrize("kernel", ["cubic_spline", "wendland_c2"])
@pytest.mark.parametrize("local_mass_conservation", [False, True])
def test_mass_map_sph_native_numba_matches_python(kernel, local_mass_conservation):
    pos, mass = _sample_particles()
    kwargs = dict(
        npix=48,
        xmin=-1.0,
        xmax=1.0,
        ymin=-1.0,
        ymax=1.0,
        nb=10,
        min_hsml=0.03,
        max_hsml=0.35,
        local_mass_conservation=local_mass_conservation,
        kernel=kernel,
    )

    img_python = _mass_map_sph_native(pos, mass, use_numba=False, **kwargs)
    img_numba = _mass_map_sph_native(pos, mass, use_numba=True, **kwargs)

    np.testing.assert_allclose(img_numba, img_python, rtol=0.0, atol=2.0e-12)
