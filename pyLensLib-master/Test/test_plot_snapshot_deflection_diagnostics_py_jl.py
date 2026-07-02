import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplcfg')
Path('/tmp/mplcfg').mkdir(parents=True, exist_ok=True)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from test_compare_snapshot_crosssections_py_jl import build_total_mass_map, build_deflector_from_mass_map


def radial_profile(arr, fov_arcsec, nbins=60):
    ny, nx = arr.shape
    x = np.linspace(-fov_arcsec / 2.0, fov_arcsec / 2.0, nx)
    y = np.linspace(-fov_arcsec / 2.0, fov_arcsec / 2.0, ny)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    bins = np.linspace(0.0, rr.max(), nbins + 1)
    rc = 0.5 * (bins[:-1] + bins[1:])
    prof = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (rr >= bins[i]) & (rr < bins[i + 1])
        if np.any(m):
            prof[i] = np.nanmean(arr[m])
    return rc, prof


def main():
    snapshot = '/Users/maxmen3/projects/pyLensLib/Test/snap_058'
    subfind_file = '/Users/maxmen3/projects/pyLensLib/Test/sub_058.0'
    prefix = Path('/Users/maxmen3/projects/pyLensLib/Test/artifacts_phase5/snapshot058_deflection_diagnostics_zs3')
    zs = 3.0
    fov = 200.0
    npix = 256

    cl_ref, mass_map, _ = build_total_mass_map(snapshot, subfind_file, [0, 1, 4], fov, npix, 10.0, nb=32, min_hsml=1e-6, max_hsml=0.1)
    df_py = build_deflector_from_mass_map(cl_ref.co, cl_ref.zl, zs, mass_map, fov, npix, fov)

    h5path = Path('/Users/maxmen3/projects/pyLensLib/Test/artifacts_phase5/snapshot058_critical_lines_zs3.jl_geometry.h5')
    with h5py.File(h5path, 'r') as h5:
        a1_jl = h5['z1']['a1'][...]
        a2_jl = h5['z1']['a2'][...]
        kappa_jl = h5['z1']['kappa'][...]

    a1_py = df_py.a1
    a2_py = df_py.a2
    alpha_py = np.hypot(a1_py, a2_py)
    alpha_jl = np.hypot(a1_jl, a2_jl)
    da1 = a1_py - a1_jl
    da2 = a2_py - a2_jl
    dalpha = alpha_py - alpha_jl

    extent = [-fov/2, fov/2, -fov/2, fov/2]
    fig = plt.figure(figsize=(12.5, 9.5))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1, 0.9], wspace=0.12, hspace=0.22)
    ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[0,1], sharex=ax1, sharey=ax1); cax1 = fig.add_subplot(gs[0,2])
    ax3 = fig.add_subplot(gs[1,0], sharex=ax1, sharey=ax1); ax4 = fig.add_subplot(gs[1,1], sharex=ax1, sharey=ax1); cax2 = fig.add_subplot(gs[1,2])
    axp = fig.add_subplot(gs[2,0:2])

    vmin = min(np.nanpercentile(alpha_py, 5), np.nanpercentile(alpha_jl, 5))
    vmax = max(np.nanpercentile(alpha_py, 99), np.nanpercentile(alpha_jl, 99))
    im1 = ax1.imshow(alpha_py, origin='lower', extent=extent, cmap='magma', vmin=vmin, vmax=vmax)
    ax2.imshow(alpha_jl, origin='lower', extent=extent, cmap='magma', vmin=vmin, vmax=vmax)
    fig.colorbar(im1, cax=cax1, label=r'$|\alpha|$ [arcsec]')
    ax1.set_title('pyLensLib |alpha|')
    ax2.set_title('jlLensLib |alpha|')

    dv = np.nanpercentile(np.abs(dalpha), 99)
    im2 = ax3.imshow(dalpha, origin='lower', extent=extent, cmap='coolwarm', vmin=-dv, vmax=dv)
    ax4.imshow(kappa_jl - df_py.ka, origin='lower', extent=extent, cmap='coolwarm', vmin=-np.nanpercentile(np.abs(kappa_jl-df_py.ka),99), vmax=np.nanpercentile(np.abs(kappa_jl-df_py.ka),99))
    fig.colorbar(im2, cax=cax2, label=r'$\Delta|\alpha|$ [arcsec]')
    ax3.set_title('py - jl |alpha|')
    ax4.set_title('jl - py kappa')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_aspect('equal')
        ax.set_xlabel(r'$\theta_1$ [arcsec]')
    ax1.set_ylabel(r'$\theta_2$ [arcsec]')
    ax3.set_ylabel(r'$\theta_2$ [arcsec]')

    r, p_apy = radial_profile(alpha_py, fov)
    _, p_ajl = radial_profile(alpha_jl, fov)
    _, p_kpy = radial_profile(df_py.ka, fov)
    _, p_kjl = radial_profile(kappa_jl, fov)
    axp.plot(r, p_apy, color='tab:red', lw=1.8, label='py |alpha|')
    axp.plot(r, p_ajl, color='goldenrod', lw=1.8, label='jl |alpha|')
    axp.plot(r, p_kpy, color='tab:blue', lw=1.5, ls='--', label='py kappa')
    axp.plot(r, p_kjl, color='tab:green', lw=1.5, ls='--', label='jl kappa')
    axp.set_yscale('log')
    axp.set_xlabel('R [arcsec]')
    axp.set_ylabel('Azimuthal mean')
    axp.set_title('Radial deflection and convergence profiles')
    axp.grid(alpha=0.25)
    axp.legend(frameon=False, ncol=2)

    out = prefix.with_suffix('.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)
    print('a1_rms', np.sqrt(np.mean(da1**2)), 'a2_rms', np.sqrt(np.mean(da2**2)), 'alpha_rms', np.sqrt(np.mean(dalpha**2)))


if __name__ == '__main__':
    main()
