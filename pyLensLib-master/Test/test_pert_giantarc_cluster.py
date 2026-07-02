"""
Cluster-scale giant-arc perturbation test.

Workflow:
1) Build a PIEMD cluster lens (q=0.7, sigma0=1500 km/s, zl=0.5, zs=3.0).
2) Define a Sersic source similar to test_pert_einsteinring.
3) Place source near a cusp of the tangential caustic.
4) Produce mock lensed image.
5) Place an NFW subhalo along the generated arc (estimated with point-source images).
6) Repeat for subhalo masses 1e7, 1e8, 1e9 Msun.
7) Produce summary figure with convergence+critical lines, lensed image,
   mock observation, and residuals with respect to the no-perturber case.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy.cosmology import FlatLambdaCDM

from pyLensLib.piemd import piemd
from pyLensLib.nfwell import nfwell
from pyLensLib.sersic import sersic
from pyLensLib.pointsrc import pointsrc
from pyLensLib.observation import observation


def gaussian_psf(fwhm_arcsec, pixel_scale_arcsec, npix=65):
    ax = np.arange(npix) - npix // 2
    x, y = np.meshgrid(ax, ax)
    sigma_pix = fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0))) / pixel_scale_arcsec
    psf = np.exp(-0.5 * (x * x + y * y) / (sigma_pix * sigma_pix))
    psf /= psf.sum()
    return psf


def build_cluster_lens(co, thetax, thetay, zl, zs, sigma0=1500.0, q=0.7):
    kwargs = {
        "zl": zl,
        "zs": zs,
        "sigma0": sigma0,
        "q": q,
        "pa": 0.0,
        "theta_c": 2.0,
        "theta_t": 300.0,
        "x1": 0.0,
        "x2": 0.0,
    }
    lens = piemd(co, **kwargs)
    lens.setGrid(thetax=thetax, thetay=thetay)
    return lens


def build_subhalo(co, thetax, thetay, zl, zs, mass_msun, x1, x2):
    sub = nfwell(
        co=co,
        zl=zl,
        zs=zs,
        mass=mass_msun,
        conc=25.0,
        q=1.0,
        pa=0.0,
        x1=x1,
        x2=x2,
    )
    sub.setGrid(thetax=thetax, thetay=thetay)
    return sub


def select_source_near_cusp(lens, zs, probe_fov, probe_npix=401):
    tan = lens.tancl(size_principale=0.0)
    if len(tan) == 0:
        raise RuntimeError("No tangential critical lines found.")

    caustics = lens.getCaustics(tan)
    ic = np.argmax([c.getArea() for c in caustics])
    x_cau, y_cau = lens.getCausticPoints(caustics[ic], pixel_units=False)

    cx, cy = np.mean(x_cau), np.mean(y_cau)
    rr = (x_cau - cx) ** 2 + (y_cau - cy) ** 2
    i_cusp = np.argmax(rr)
    cusp_x, cusp_y = x_cau[i_cusp], y_cau[i_cusp]

    best = None
    for frac in np.linspace(0.02, 0.25, 16):
        ys1 = cusp_x + frac * (cx - cusp_x)
        ys2 = cusp_y + frac * (cy - cusp_y)
        ps = pointsrc(size=probe_fov, Npix=probe_npix, gl=lens, ys1=ys1, ys2=ys2, flux=1.0, zs=zs)
        xi, yi, mu = ps.xi1, ps.xi2, ps.mui
        if mu.size == 0:
            continue
        score = np.max(np.abs(mu))
        nimg = xi.size
        if best is None or (nimg >= 2 and score > best["score"]):
            best = {"ys1": ys1, "ys2": ys2, "score": score, "nimg": nimg}

    if best is None:
        raise RuntimeError("Could not identify a source position near cusp.")

    return best["ys1"], best["ys2"], (cusp_x, cusp_y), (cx, cy)


def estimate_arc_anchor(lens, ys1, ys2, probe_fov, probe_npix=401):
    ps = pointsrc(size=probe_fov, Npix=probe_npix, gl=lens, ys1=ys1, ys2=ys2, flux=1.0, zs=lens.zs)
    xi, yi, mu = ps.xi1, ps.xi2, ps.mui
    if xi.size == 0:
        raise RuntimeError("No point-source images found to anchor arc position.")

    idx_sorted = np.argsort(np.abs(mu))[::-1]
    idx = idx_sorted[0]
    return xi[idx], yi[idx], mu[idx]


if __name__ == "__main__":
    np.random.seed(7)

    co = FlatLambdaCDM(H0=70.0, Om0=0.3)
    zl = 0.5
    zs = 3.0

    # Same observation setup used in the previous perturbation test.
    pixel_scale_obs = 0.004
    pixel_scale_deflector = pixel_scale_obs / 1.0
    jwst_fwhm = 0.01

    # Wide grid for cusp/arc localization.
    fov_search = 120.0
    pixel_scale_search_def = 0.05
    npix_search_def = int(round(fov_search / pixel_scale_search_def)) + 1
    theta_search = np.linspace(-fov_search / 2.0, fov_search / 2.0, npix_search_def)

    lens_search = build_cluster_lens(
        co, theta_search, theta_search, zl=zl, zs=zs, sigma0=1500.0, q=0.7
    )

    ys1, ys2, cusp, caustic_center = select_source_near_cusp(
        lens_search, zs=zs, probe_fov=fov_search, probe_npix=401
    )
    arc_x, arc_y, arc_mu = estimate_arc_anchor(
        lens_search, ys1=ys1, ys2=ys2, probe_fov=fov_search, probe_npix=401
    )

    # Zoomed field around the arc for detailed rendering.
    fov_zoom = 20.0
    half_zoom = fov_zoom / 2.0
    x_bounds = [arc_x - half_zoom, arc_x + half_zoom]
    y_bounds = [arc_y - half_zoom, arc_y + half_zoom]

    npix_obs = int(round(fov_zoom / pixel_scale_obs)) + 1
    npix_def = int(round(fov_zoom / pixel_scale_deflector)) + 1
    thetax = np.linspace(x_bounds[0], x_bounds[1], npix_def)
    thetay = np.linspace(y_bounds[0], y_bounds[1], npix_def)

    ob = observation(
        size=fov_zoom,
        Npix=npix_obs,
        zp=28.0,
        texp=10000.0,
        mlim=27.0,
        rap=0.10,
        sn=10.0,
    )
    psf_image = gaussian_psf(fwhm_arcsec=jwst_fwhm, pixel_scale_arcsec=pixel_scale_obs, npix=65)

    flux_src = ob.mag2counts(24.5)
    flux_lens = ob.mag2counts(20.5)

    src_kwargs = {
        "n": 1.0,
        "re": 0.1,
        "q": 0.75,
        "pa": np.pi / 4.0,
        "ys1": ys1,
        "ys2": ys2,
        "flux": flux_src,
        "zs": zs,
    }

    lens_light_kwargs = {
        "n": 4.0,
        "re": 6.0,
        "q": 0.8,
        "pa": 0.0,
        "ys1": 0.0,
        "ys2": 0.0,
        "flux": flux_lens,
        "zs": zl,
    }
    lens_light = sersic(
        sizex=x_bounds, sizey=y_bounds, Npix=npix_obs, gl=None, save_unlensed=False, **lens_light_kwargs
    )

    cases = [
        ("No subhalo", None),
        ("Subhalo 1e7 Msun", 1e7),
        ("Subhalo 1e8 Msun", 1e8),
        ("Subhalo 1e9 Msun", 1e9),
    ]
    outputs = []

    for label, msub in cases:
        lens = build_cluster_lens(co, thetax, thetay, zl=zl, zs=zs, sigma0=1500.0, q=0.7)
        perturber_pos = None

        if msub is not None:
            sub = build_subhalo(
                co,
                thetax,
                thetay,
                zl=zl,
                zs=zs,
                mass_msun=msub,
                x1=arc_x,
                x2=arc_y,
            )
            lens.combinewith(sub)
            perturber_pos = (arc_x, arc_y)

        tan_lines = [lens.getCritPoints(cl) for cl in lens.tancl()]
        rad_lines = [lens.getCritPoints(cl) for cl in lens.radcl()]

        src = sersic(
            sizex=x_bounds, sizey=y_bounds, Npix=npix_obs, gl=lens, save_unlensed=True, **src_kwargs
        )
        model = src.image + lens_light.image
        model_conv = ob.convolve_psf2(model, psf_image, psf_scale=pixel_scale_obs)
        noise = ob.makeNoise(model_conv)
        mock_obs = model_conv + noise

        outputs.append(
            {
                "label": label,
                "kappa": lens.ka.copy(),
                "tan_lines": tan_lines,
                "rad_lines": rad_lines,
                "lensed": src.image,
                "obs": mock_obs,
                "model_conv": model_conv,
                "perturber_pos": perturber_pos,
            }
        )

    ref_conv = outputs[0]["model_conv"]
    residual_scales = [np.percentile(np.abs(out["model_conv"] - ref_conv), 99.5) for out in outputs[1:]]
    residual_vmax = max(residual_scales) if len(residual_scales) > 0 else 1.0
    if residual_vmax <= 0.0:
        residual_vmax = 1.0

    fig, ax = plt.subplots(len(outputs), 4, figsize=(20, 4 * len(outputs)), sharex=True, sharey=True)
    if len(outputs) == 1:
        ax = np.array([ax])

    extent = [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]]
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
                    Circle((x_sub, y_sub), radius=0.25, fill=False, edgecolor="cyan", linewidth=1.8)
                )

        ax[i, 0].set_ylabel(f"{out['label']}\narcsec")
        ax[i, 0].set_title("Convergence κ")
        ax[i, 1].set_title("Lensed Arc")
        ax[i, 2].set_title("E-ELT-like Mock")
        ax[i, 3].set_title("Residual vs No Perturber")

    ax[0, 0].legend(loc="upper right", fontsize=8, frameon=True)

    for j in range(4):
        ax[-1, j].set_xlabel("arcsec")

    fig.suptitle(
        (
            "Cluster PIEMD Giant-Arc Perturbation Test "
            f"(zl={zl}, zs={zs}, sigma0=1500 km/s, q=0.7)\n"
            f"Source near cusp at ({ys1:.3f}, {ys2:.3f}) arcsec; "
            f"arc anchor ({arc_x:.3f}, {arc_y:.3f}) arcsec, mu~{arc_mu:.2f}"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig("test_pert_giantarc_cluster_summary.png", dpi=180)

    print("Saved figure: test_pert_giantarc_cluster_summary.png")
    print(f"Selected cusp location: ({cusp[0]:.4f}, {cusp[1]:.4f}) arcsec")
    print(f"Caustic centroid: ({caustic_center[0]:.4f}, {caustic_center[1]:.4f}) arcsec")
    print(f"Chosen source position: ({ys1:.4f}, {ys2:.4f}) arcsec")
    print(f"Arc anchor from point-source images: ({arc_x:.4f}, {arc_y:.4f}) arcsec")
