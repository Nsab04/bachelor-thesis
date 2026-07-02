"""
Test script: giant arc / Einstein-ring simulation with optional subhalos.

This script performs the following tasks:
1) Creates a main SIE lens at zl=0.5, calibrated to an Einstein mass ~1e12 Msun.
2) Creates a Sersic source with re=0.2 arcsec at zs=2.5.
3) Produces an Einstein-ring-like configuration by placing source and lens centers in alignment.
4) Simulates a JWST-like observation with S/N=10 and an assumed spatial resolution FWHM=0.06 arcsec.
5) Repeats the experiment by adding one subhalo with masses 1e6, 1e7, 1e8 Msun.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy import units as u
from astropy.constants import c, G
from astropy.cosmology import FlatLambdaCDM

from pyLensLib.sie import sie
from pyLensLib.nfwell import nfwell
from pyLensLib.sersic import sersic
from pyLensLib.observation import observation


def sigma0_from_einstein_mass(co, zl, zs, m_ein_msun, q=1.0):
    """
    Convert target Einstein enclosed mass to SIE velocity dispersion.
    """
    dl = co.angular_diameter_distance(zl)
    ds = co.angular_diameter_distance(zs)
    dls = co.angular_diameter_distance_z1z2(zl, zs)

    sigma_crit = (c**2 / (4.0 * np.pi * G) * ds / (dl * dls)).to(u.Msun / u.Mpc**2)
    m_ein = m_ein_msun * u.Msun
    theta_e_rad = np.sqrt((m_ein / (np.pi * sigma_crit * dl**2)).decompose().value)

    sigma0 = (c * np.sqrt(theta_e_rad * (ds / dls).value * np.sqrt(q) / (4.0 * np.pi))).to(u.km / u.s).value
    theta_e_arcsec = np.rad2deg(theta_e_rad) * 3600.0
    return sigma0, theta_e_arcsec


def gaussian_psf(fwhm_arcsec, pixel_scale_arcsec, npix=65):
    """
    Build a normalized Gaussian PSF image.
    """
    ax = np.arange(npix) - npix // 2
    x, y = np.meshgrid(ax, ax)
    sigma_pix = fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0))) / pixel_scale_arcsec
    psf = np.exp(-0.5 * (x * x + y * y) / (sigma_pix * sigma_pix))
    psf /= psf.sum()
    return psf


def build_main_lens(co, theta, zl, zs, m_ein_msun):
    """
    Build a circular SIE lens and map it on the provided grid.
    """
    sigma0, theta_e = sigma0_from_einstein_mass(co, zl, zs, m_ein_msun=m_ein_msun, q=1.0)
    lens = sie(
        co,
        zl=zl,
        zs=zs,
        sigma0=sigma0,
        q=1.0,
        pa=0.0,
        theta_c=1e-4,
        x1=0.0,
        x2=0.0,
    )
    lens.setGrid(theta=theta)
    return lens, sigma0, theta_e


def build_subhalo(co, theta, zl, zs, mass_msun, x1, x2):
    """
    Build an NFW subhalo lens map and return it.
    """
    sh = nfwell(
        co=co,
        zl=zl,
        zs=zs,
        mass=mass_msun,
        conc=155.0,
        q=1.0,
        pa=0.0,
        x1=x1,
        x2=x2,
    )
    sh.setGrid(theta=theta)
    return sh


if __name__ == "__main__":
    np.random.seed(42)

    # Geometry / cosmology
    co = FlatLambdaCDM(H0=70.0, Om0=0.3)
    zl = 0.5
    zs = 2.5
    if zs <= zl:
        raise ValueError("Source redshift must be larger than lens redshift.")

    # Requested lens scale
    m_main_ein = 1e12  # Msun (interpreted as Einstein enclosed mass scale)

    # Field setup
    fov = 8.0  # arcsec
    # Deflector grid is 10x finer than the rendering grid used for observations.
    pixel_scale_obs = 0.05  # arcsec/pixel (rendered image scale)
    pixel_scale_deflector = pixel_scale_obs / 10.0  # arcsec/pixel (lens grid scale)
    npix_obs = int(round(fov / pixel_scale_obs)) + 1
    npix_deflector = int(round(fov / pixel_scale_deflector)) + 1
    theta_deflector = np.linspace(-fov / 2.0, fov / 2.0, npix_deflector)

    # JWST-like observation assumptions
    # User asked for S/N=10 and "spatial resolution of ..."; assume 0.06 arcsec FWHM.
    jwst_fwhm = 0.01  # arcsec
    ob = observation(
        size=fov,
        Npix=npix_obs,
        zp=28.0,
        texp=10000.0,
        mlim=27.0,
        rap=0.10,
        sn=10.0,
    )
    psf_image = gaussian_psf(fwhm_arcsec=jwst_fwhm, pixel_scale_arcsec=pixel_scale_obs, npix=65)

    # Source + lens light setup
    flux_src = ob.mag2counts(24.5)
    flux_lens = ob.mag2counts(20.5)

    src_kwargs = {
        "n": 1.0,
        "re": 0.1,  # requested effective radius
        "q": 0.75,
        "pa": np.pi / 4.0,
        "ys1": 0.0,  # aligned to produce Einstein ring
        "ys2": 0.0,
        "flux": flux_src,
        "zs": zs,
    }
    lens_light_kwargs = {
        "n": 4.0,
        "re": 0.45,
        "q": 0.85,
        "pa": 0.0,
        "ys1": 0.0,
        "ys2": 0.0,
        "flux": flux_lens,
        "zs": zl,
    }

    # Build lens light once (not lensed)
    lens_light = sersic(size=fov, Npix=npix_obs, gl=None, save_unlensed=False, **lens_light_kwargs)

    cases = [("No subhalo", None), ("Subhalo 1e7 Msun", 1e7), ("Subhalo 1e8 Msun", 1e8), ("Subhalo 1e9 Msun", 1e9)]
    outputs = []

    for label, msub in cases:
        main_lens, sigma0, theta_e = build_main_lens(
            co, theta_deflector, zl=zl, zs=zs, m_ein_msun=m_main_ein
        )
        perturber_pos = None

        if msub is not None:
            # Place the subhalo on the Einstein ring.
            phi_sub = np.pi / 4.0
            x_sub = theta_e * np.cos(phi_sub)
            y_sub = theta_e * np.sin(phi_sub)
            sub = build_subhalo(
                co, theta_deflector, zl=zl, zs=zs, mass_msun=msub, x1=x_sub, x2=y_sub
            )
            main_lens.combinewith(sub)
            perturber_pos = (x_sub, y_sub)

        tan_lines = [main_lens.getCritPoints(cl) for cl in main_lens.tancl()]
        rad_lines = [main_lens.getCritPoints(cl) for cl in main_lens.radcl()]

        src = sersic(size=fov, Npix=npix_obs, gl=main_lens, save_unlensed=True, **src_kwargs)
        model = src.image + lens_light.image
        model_conv = ob.convolve_psf2(model, psf_image, psf_scale=pixel_scale_obs)
        noise = ob.makeNoise(model_conv)
        mock_obs = model_conv + noise

        outputs.append(
            {
                "label": label,
                "sigma0": sigma0,
                "theta_e": theta_e,
                "kappa": main_lens.ka.copy(),
                "tan_lines": tan_lines,
                "rad_lines": rad_lines,
                "unlensed": src.image_unlensed,
                "lensed": src.image,
                "model": model,
                "model_conv": model_conv,
                "obs": mock_obs,
                "perturber_pos": perturber_pos,
            }
        )

    ref_conv = outputs[0]["model_conv"]
    residual_scales = [np.percentile(np.abs(out["model_conv"] - ref_conv), 99.5) for out in outputs[1:]]
    residual_vmax = max(residual_scales) if len(residual_scales) > 0 else 1.0
    if residual_vmax <= 0.0:
        residual_vmax = 1.0

    # Display and save summary figure
    fig, ax = plt.subplots(len(outputs), 4, figsize=(20, 4 * len(outputs)), sharex=True, sharey=True)
    if len(outputs) == 1:
        ax = np.array([ax])

    extent = [-fov / 2.0, fov / 2.0, -fov / 2.0, fov / 2.0]
    for i, out in enumerate(outputs):
        residual_map = out["model_conv"] - ref_conv
        kappa = out["kappa"]
        kappa_vmin = np.percentile(kappa, 5.0)
        kappa_vmax = np.percentile(kappa, 99.8)
        vmax_lensed = np.percentile(out["lensed"], 99.8)
        vmax_obs = np.percentile(out["obs"], 99.8)

        ax[i, 0].imshow(kappa, origin="lower", extent=extent, cmap="cividis", vmin=kappa_vmin, vmax=kappa_vmax)
        ax[i, 1].imshow(out["lensed"], origin="lower", extent=extent, cmap="magma", vmax=vmax_lensed)
        ax[i, 2].imshow(out["obs"], origin="lower", extent=extent, cmap="magma", vmax=vmax_obs)
        ax[i, 3].imshow(
            residual_map,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-residual_vmax,
            vmax=residual_vmax,
        )

        for il, (xcl, ycl) in enumerate(out["tan_lines"]):
            label_t = "Tangential CL" if (i == 0 and il == 0) else None
            ax[i, 0].plot(xcl, ycl, "-", color="white", linewidth=1.1, alpha=0.95, label=label_t)
        for il, (xcl, ycl) in enumerate(out["rad_lines"]):
            label_r = "Radial CL" if (i == 0 and il == 0) else None
            ax[i, 0].plot(xcl, ycl, "--", color="orange", linewidth=1.0, alpha=0.95, label=label_r)

        if out["perturber_pos"] is not None:
            x_sub, y_sub = out["perturber_pos"]
            for j in (0, 1, 2, 3):
                ax[i, j].add_patch(
                    Circle((x_sub, y_sub), radius=0.10, fill=False, edgecolor="cyan", linewidth=1.8)
                )

        ax[i, 0].set_ylabel(f"{out['label']}\narcsec")
        ax[i, 0].set_title("Convergence κ")
        ax[i, 1].set_title("Lensed Arc / Ring")
        ax[i, 2].set_title("E-ELT-like Mock")
        ax[i, 3].set_title("Residual vs No Perturber")

    ax[0, 0].legend(loc="upper right", fontsize=8, frameon=True)

    for j in range(4):
        ax[-1, j].set_xlabel("arcsec")

    fig.suptitle(
        (
            f"Giant Arc Test (zl={zl}, zs={zs}) | S/N=10 | "
            f"Assumed JWST resolution FWHM={jwst_fwhm:.2f}\""
        ),
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("test_pert_einsteinring_summary.png", dpi=180)

    print("Saved figure: test_pert_einsteinring_summary.png")
    for out in outputs:
        print(
            f"{out['label']}: sigma0={out['sigma0']:.2f} km/s, "
            f"theta_E~{out['theta_e']:.3f} arcsec"
        )
