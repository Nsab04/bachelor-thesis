import importlib
import numpy as np
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import FlatLambdaCDM

from pyLensLib.convergence_integrals import (
    deflection_from_kappa_grid,
    deflection_from_kappa_grid_adaptive,
    deflection_from_surface_density_grid,
)
from pyLensLib.raytracer import raytracer


def _gaussian_kappa(n=65, pixel_scale=0.1, sigma=0.5):
    axis = (np.arange(n) - (n - 1) / 2.0) * pixel_scale
    x, y = np.meshgrid(axis, axis)
    return np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))


def _sigma_crit(co, zl, zs):
    dl = co.angular_diameter_distance(zl)
    ds = co.angular_diameter_distance(zs)
    dls = co.angular_diameter_distance_z1z2(zl, zs)
    return ((c ** 2 / G).to(u.Msun / u.Mpc) / (4.0 * np.pi) * (ds / dl / dls)).value


def test_deflection_fft_recovers_kappa_from_divergence():
    pixel_scale = 0.1
    kappa = _gaussian_kappa(n=65, pixel_scale=pixel_scale, sigma=0.5)
    a1, a2 = deflection_from_kappa_grid(kappa, pixel_scale)

    da1_dy, da1_dx = np.gradient(a1, pixel_scale, pixel_scale)
    da2_dy, da2_dx = np.gradient(a2, pixel_scale, pixel_scale)
    kappa_from_alpha = 0.5 * (da1_dx + da2_dy)

    inner = (slice(20, 45), slice(20, 45))
    assert np.max(np.abs(kappa_from_alpha[inner] - kappa[inner])) < 2.0e-2
    assert np.max(np.abs(da1_dy[inner] - da2_dx[inner])) < 2.0e-2


def test_deflection_fft_matches_lenstronomy_when_available():
    try:
        lenstronomy_integrals = importlib.import_module(
            "lenstronomy.LensModel.convergence_integrals"
        )
    except Exception:
        return
    rng = np.random.default_rng(12)
    kappa = rng.random((32, 32))

    a1, a2 = deflection_from_kappa_grid(kappa, 0.2)
    a1_len, a2_len = lenstronomy_integrals.deflection_from_kappa_grid(kappa, 0.2)

    np.testing.assert_allclose(a1, a1_len, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(a2, a2_len, rtol=0.0, atol=1.0e-12)


def test_surface_density_wrapper_matches_kappa_input():
    kappa = _gaussian_kappa(n=33, pixel_scale=0.2, sigma=0.8)
    sigma_crit = 3.5e15
    surface_density = kappa * sigma_crit

    a1, a2 = deflection_from_kappa_grid(kappa, 0.2)
    s1, s2 = deflection_from_surface_density_grid(surface_density, sigma_crit, 0.2)

    np.testing.assert_allclose(s1, a1, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(s2, a2, rtol=0.0, atol=1.0e-12)


def test_adaptive_deflection_is_close_to_direct_fft():
    kappa = _gaussian_kappa(n=64, pixel_scale=0.1, sigma=0.6)

    a1, a2 = deflection_from_kappa_grid(kappa, 0.1)
    b1, b2 = deflection_from_kappa_grid_adaptive(
        kappa,
        0.1,
        low_res_factor=4,
        high_res_kernel_size=9,
        return_high_res=True,
    )

    assert b1.shape == kappa.shape
    assert b2.shape == kappa.shape
    assert np.all(np.isfinite(b1))
    assert np.all(np.isfinite(b2))
    assert np.mean(np.abs(b1 - a1)) < 6.0e-3
    assert np.mean(np.abs(b2 - a2)) < 6.0e-3


def test_raytracer_deflection_fft_matches_convergence_integral():
    n = 64
    fov = 20.0
    zl = 0.3
    zs = 2.0
    co = FlatLambdaCDM(H0=70.0, Om0=0.3)
    pixel_scale = fov / (n - 1)
    kappa = _gaussian_kappa(n=n, pixel_scale=pixel_scale, sigma=2.0)

    fov_mpc = fov * co.angular_diameter_distance(zl).value * np.pi / 180.0 / 3600.0
    dx_mpc = fov_mpc / (n - 1)
    mass_map = kappa * dx_mpc ** 2 * _sigma_crit(co, zl, zs)

    rt = raytracer(
        co,
        mass_map,
        Nray=n,
        FOVray=fov,
        fromfile=False,
        zl=zl,
        zs=zs,
        fov=fov,
        method="deflection_fft",
        alpha_interpolation_order=1,
    )
    a1, a2 = deflection_from_kappa_grid(kappa, pixel_scale)

    np.testing.assert_allclose(rt.kappa_map, kappa, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(rt.a1, a1, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(rt.a2, a2, rtol=0.0, atol=2.0e-12)


def test_raytracer_legacy_potential_fft_padding_uses_public_numpy_api():
    n = 16
    fov = 20.0
    zl = 0.3
    zs = 2.0
    co = FlatLambdaCDM(H0=70.0, Om0=0.3)
    mass_map = np.zeros((n, n), dtype=float)
    mass_map[n // 2, n // 2] = 1.0e13

    rt = raytracer(
        co,
        mass_map,
        Nray=n,
        FOVray=fov,
        fromfile=False,
        zl=zl,
        zs=zs,
        fov=fov,
        method="potential_fft",
    )

    assert rt.kappa.shape == (5 * n, 5 * n)
    assert rt.a1.shape == (n, n)
    assert rt.a2.shape == (n, n)
    assert np.all(np.isfinite(rt.a1))
    assert np.all(np.isfinite(rt.a2))
