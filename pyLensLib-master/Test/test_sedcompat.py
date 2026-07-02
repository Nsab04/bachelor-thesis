import numpy as np

import pyLensLib.sedcompat as ss


def test_ab_zero_point():
    # Simple top-hat filter.
    wa = np.linspace(5000.0, 7000.0, 512)
    tr = np.ones_like(wa)
    pb = ss.Passband(wa=wa, tr=tr)

    # Build an AB=0 SED (constant f_nu = 3631 Jy converted to f_lambda).
    fnu0 = 3631.0e-23
    flambda = ss.fnu_to_flambda(wa, fnu0)
    sed = ss.SED(wa=wa, fl=flambda, z=0.0)

    mag = sed.calc_mag(pb, system="AB")
    assert np.isfinite(mag)
    assert abs(mag) < 1e-3, f"AB zero-point mismatch: got mag={mag}"


def test_normalization_roundtrip():
    wa = np.linspace(4500.0, 8500.0, 1024)
    tr = np.exp(-0.5 * ((wa - 6500.0) / 500.0) ** 2)
    pb = ss.Passband(wa=wa, tr=tr)
    sed = ss.SED(wa=wa, fl=np.ones_like(wa), z=0.0)

    target_mag = 25.3
    sed.normalise_to_mag(target_mag, pb, system="AB")
    mag = sed.calc_mag(pb, system="AB")
    assert np.isfinite(mag)
    assert abs(mag - target_mag) < 1e-6, f"Normalization failed: {mag} vs {target_mag}"


def test_redshift_scaling():
    wa = np.linspace(3000.0, 10000.0, 256)
    fl = np.linspace(1.0, 2.0, 256)
    sed = ss.SED(wa=wa, fl=fl, z=0.5)
    sed2 = sed.copy()
    sed2.redshift_to(2.0)

    scale = (1.0 + 2.0) / (1.0 + 0.5)
    assert np.allclose(sed2.wa, wa * scale)
    assert np.allclose(sed2.fl, fl / scale)


if __name__ == "__main__":
    test_ab_zero_point()
    test_normalization_roundtrip()
    test_redshift_scaling()
    print("test_sedcompat.py: all tests passed")
