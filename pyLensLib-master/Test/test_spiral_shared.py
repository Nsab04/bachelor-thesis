import numpy as np

from pyLensLib.sersic import sersic
from pyLensLib.spiral import normalize_modifier, spiral, spiral_modifier_from_grid


def _build_disc():
    return sersic(
        Npix=7,
        gl=None,
        save_unlensed=False,
        rmaxf=10.0,
        sizex=[-1.0, 1.0],
        sizey=[-1.0, 1.0],
        n=1.0,
        q=1.0,
        ys1=0.1,
        ys2=-0.2,
        pa=np.deg2rad(30.0),
        re=0.4,
        flux=1.0,
        zs=1.0,
    )


def test_spiral_grid_helper_matches_class_formula():
    se = _build_disc()
    sp = spiral(se, A=0.4, Na=2, Phid=0.3, alpha=1.7, phi=0.2)

    helper = spiral_modifier_from_grid(
        se.y1,
        se.y2,
        se.ys1,
        se.ys2,
        se.pa,
        sp.phi,
        sp.A,
        sp.Na,
        sp.Phid,
        sp.alpha,
        se.re,
    )

    x = np.cos(se.pa) * (se.y1 - se.ys1) + np.sin(se.pa) * (se.y2 - se.ys2)
    y = (-np.sin(se.pa) * (se.y1 - se.ys1) + np.cos(se.pa) * (se.y2 - se.ys2)) / (np.cos(sp.phi) + 1.0e-2)
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    expected = normalize_modifier(sp.spiral_modifier(r, theta))

    assert np.allclose(helper, expected)


def test_normalize_modifier_handles_flat_fields():
    flat = np.ones((3, 3))
    normalized = normalize_modifier(flat)
    assert np.allclose(normalized, np.ones_like(flat))
