import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyLensLib.sersic_numba import sersic


def make_source(size, npix):
    return sersic(
        size=size,
        Npix=npix,
        gl=None,
        save_unlensed=True,
        save_unlensed_recenter=True,
        size_unlensed=1.0,
        npix_unlensed=401,
        y1_unlensed=0.0,
        y2_unlensed=0.0,
        rmaxf=100.0,
        n=2.0,
        q=0.9,
        ys1=0.0,
        ys2=0.0,
        pa=0.3,
        re=0.05,
        flux=1.0,
        zs=1.0,
    )


def test_recentered_unlensed_flux_uses_unlensed_pixel_area():
    coarse_primary_grid = make_source(size=5.0, npix=101)
    fine_primary_grid = make_source(size=10.0, npix=401)

    coarse_flux = coarse_primary_grid.image_unlensed.sum()
    fine_flux = fine_primary_grid.image_unlensed.sum()

    assert abs(coarse_flux - fine_flux) / fine_flux < 1.0e-12


if __name__ == "__main__":
    test_recentered_unlensed_flux_uses_unlensed_pixel_area()
    print("sersic_numba recentered image_unlensed pixel-area test passed")
