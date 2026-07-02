"""
Create per-galaxy summary figures for problematic M0416 galaxies.

For each selected galaxy:
1) Extract an HST F814W cutout from the input drz image.
2) Reproduce the galaxy light with pyLensLib.photocatalogs (Sersic model).
3) Fit projected PIEMD surface density with projected tNFW + k*Sigma_Sersic.
4) Save a summary plot with cutout, Sersic model, residual, and surface-density profiles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM

from pyLensLib import photocatalogs
from pyLensLib.observation import observation
from pyLensLib.lenstool import readLenstoolBlock, selectPotentielByType, findInBlock


G_KPC = 4.302e-6  # kpc (km/s)^2 / Msun

DEFAULT_SELECTED = "/Users/maxmen3/projects/pyLensLib/Test/M0416_top10_problematic_galaxies_selected.csv"
DEFAULT_DRZ = "/Users/maxmen3/stiva/HFF/M0416/hlsp_frontier_hst_acs-30mas-selfcal_macs0416_f814w_v1.0_drz.fits"
DEFAULT_TSV = "/Users/maxmen3/projects/pyLensLib/Test/M0416.tsv"
DEFAULT_PAR = "/Users/maxmen3/stiva/pietro_models/M0416_B22.par"
DEFAULT_OUTDIR = "/Users/maxmen3/projects/pyLensLib/Test/problematic_galaxy_summaries"


def piemd_sigma_2d(r, r_core, r_cut, sigma_v):
    """
    Circularized PIEMD projected surface density (Msun/kpc^2).
    """
    r = np.asarray(r, dtype=float)
    return (
        (sigma_v**2)
        / (2.0 * G_KPC)
        * (r_cut / (r_cut - r_core))
        * (1.0 / np.sqrt(r_core**2 + r**2) - 1.0 / np.sqrt(r_cut**2 + r**2))
    )


def trunc_nfw_rho(r, rs, rt, rho0):
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    tau = rt / rs
    trunc = tau**2 / (x**2 + tau**2)
    return trunc * (rho0 / (x * (1.0 + x) ** 2))


def sigma_tnfw_projected(r, rs, rt, rho0, n_u=240):
    """
    Projected tNFW surface density from LOS integration of the 3D tNFW profile.

    Sigma(R) = 2 * integral_0^inf rho_tNFW(sqrt(R^2 + z^2)) dz
    """
    r_in = np.asarray(r, dtype=float)
    r_vec = np.atleast_1d(r_in)
    if rs <= 0.0 or rt <= 0.0 or rho0 <= 0.0:
        raise ValueError("rs, rt, rho0 must be > 0.")

    x = np.maximum(r_vec / rs, np.finfo(float).eps)
    tau = rt / rs
    umax = max(800.0, 50.0 * tau, 20.0 * float(np.nanmax(x)))
    u = np.logspace(-6, np.log10(umax), int(n_u))
    u = np.concatenate(([0.0], u))

    s = np.sqrt(x[:, None] ** 2 + u[None, :] ** 2)
    integrand = (tau**2 / (s**2 + tau**2)) / (s * (1.0 + s) ** 2)
    trapz = getattr(np, "trapezoid", np.trapz)
    i_los = trapz(integrand, u, axis=1)
    sigma = 2.0 * rho0 * rs * i_los

    if np.ndim(r_in) == 0:
        return float(sigma[0])
    return sigma


def b_n_sersic(n):
    n = float(n)
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)


def sersic_sigma_2d(r, re_kpc, n, mu_e):
    r = np.asarray(r, dtype=float)
    re_kpc = float(re_kpc)
    n = float(n)
    mu_e = float(mu_e)
    rr = np.maximum(r, np.finfo(float).eps) / re_kpc
    bn = b_n_sersic(n)
    i_e = 10.0 ** (-0.4 * mu_e)  # arbitrary linear scale
    return i_e * np.exp(-bn * (np.power(rr, 1.0 / n) - 1.0))


def fit_tnfw_plus_sersic_2d(r, sigma_target, sigma_sersic, rs_guess, rt_guess, rho0_guess, k_guess=None, rmin_fit=0.05):
    r = np.asarray(r, dtype=float)
    sigma_target = np.asarray(sigma_target, dtype=float)
    sigma_sersic = np.asarray(sigma_sersic, dtype=float)

    valid = (
        np.isfinite(r)
        & np.isfinite(sigma_target)
        & np.isfinite(sigma_sersic)
        & (r >= rmin_fit)
        & (sigma_target > 0.0)
        & (sigma_sersic > 0.0)
    )
    r_fit = r[valid]
    y_fit = sigma_target[valid]
    s_fit = sigma_sersic[valid]

    if r_fit.size < 20:
        raise ValueError("Not enough valid radial points for fit.")

    # Keep fit runtime bounded while preserving radial leverage.
    if r_fit.size > 280:
        idx = np.unique(np.geomspace(1, r_fit.size, num=280).astype(int) - 1)
        r_fit = r_fit[idx]
        y_fit = y_fit[idx]
        s_fit = s_fit[idx]

    if k_guess is None:
        k_guess = np.nanmedian(y_fit / s_fit)
    rs_guess = max(float(rs_guess), 1.0e-6)
    rt_guess = max(float(rt_guess), 1.001 * rs_guess)
    rho0_guess = max(float(rho0_guess), 1.0e-12)
    k_guess = max(float(k_guess), 1.0e-20)

    theta0 = np.array(
        [
            np.log(rs_guess),
            np.log(max(rt_guess / rs_guess - 1.0, 1.0e-6)),
            np.log(rho0_guess),
            np.log(k_guess),
        ],
        dtype=float,
    )

    def model(theta):
        rs = np.exp(theta[0])
        rt = rs * (1.0 + np.exp(theta[1]))
        rho0 = np.exp(theta[2])
        k = np.exp(theta[3])
        return sigma_tnfw_projected(r_fit, rs, rt, rho0) + k * s_fit

    def residuals(theta):
        y_model = model(theta)
        res = np.log(y_model) - np.log(y_fit)
        return res[np.isfinite(res)]

    lower = np.array(
        [np.log(max(rmin_fit, 1.0e-3)), np.log(1.0e-6), np.log(1.0e2), np.log(1.0e-40)],
        dtype=float,
    )
    upper = np.array(
        [np.log(max(5.0 * np.nanmax(r_fit), 1.0)), np.log(1.0e4), np.log(1.0e16), np.log(1.0e40)],
        dtype=float,
    )
    theta0 = np.clip(theta0, lower + 1.0e-6, upper - 1.0e-6)

    sol = least_squares(
        residuals,
        theta0,
        method="trf",
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=5000,
    )

    rs = float(np.exp(sol.x[0]))
    rt = float(rs * (1.0 + np.exp(sol.x[1])))
    rho0 = float(np.exp(sol.x[2]))
    k = float(np.exp(sol.x[3]))

    y_model = sigma_tnfw_projected(r_fit, rs, rt, rho0) + k * s_fit
    res_log = np.log(y_model) - np.log(y_fit)
    rmse_log = float(np.sqrt(np.mean(res_log**2)))
    ylog = np.log(y_fit)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum((np.log(y_model) - ylog) ** 2) / sst) if sst > 0 else np.nan

    return rs, rt, rho0, k, {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(r_fit.size), "success": bool(sol.success)}


def robust_linear_scale(observed, model):
    valid = np.isfinite(observed) & np.isfinite(model)
    if np.count_nonzero(valid) < 10:
        return 0.0
    # Remove a simple background estimate from the observed cutout.
    edge = np.concatenate([observed[0, :], observed[-1, :], observed[:, 0], observed[:, -1]])
    bkg = np.nanmedian(edge[np.isfinite(edge)])
    obs = observed - bkg
    m = model
    denom = np.nansum((m[valid]) ** 2)
    if denom <= 0.0:
        return 0.0
    alpha = np.nansum(obs[valid] * m[valid]) / denom
    return max(float(alpha), 0.0)


def build_single_sersic_model(tsv_row, size_arcsec, npix, zp):
    df_one = pd.DataFrame([tsv_row]).copy().reset_index(drop=True)
    ra0 = float(df_one.iloc[0]["RAJ2000"])
    dec0 = float(df_one.iloc[0]["DEJ2000"])
    ob = observation(size=float(size_arcsec), Npix=int(npix), zp=float(zp), texp=1.0, bkg=0.0, bkg_counts_in=True)
    sersic_list = photocatalogs.create_sersic_from_photo_cat(
        df=df_one,
        ob=ob,
        zs=1.0,
        band="F814W",
        RA_ref=ra0,
        DEC_ref=dec0,
        Npix=int(npix),
        size=float(size_arcsec),
        gl=None,
        sizex=None,
        sizey=None,
        pcx=None,
        pcy=None,
        mask=None,
        save_unlensed=False,
        rmaxf=100,
    )
    if isinstance(sersic_list, str):
        raise RuntimeError(f"photocatalogs.create_sersic_from_photo_cat failed: {sersic_list}")
    if len(sersic_list) != 1:
        raise RuntimeError("Expected one Sersic model in output list.")
    return np.asarray(sersic_list[0].image, dtype=float)


def main():
    parser = argparse.ArgumentParser(description="Build summary figures for problematic M0416 galaxies.")
    parser.add_argument("--selected", default=DEFAULT_SELECTED, help="CSV with selected problematic galaxies.")
    parser.add_argument("--drz", default=DEFAULT_DRZ, help="Input drz FITS image.")
    parser.add_argument("--tsv", default=DEFAULT_TSV, help="M0416 TSV photo-catalog used by photocatalogs.py.")
    parser.add_argument("--parfile", default=DEFAULT_PAR, help="LensTool .par file.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--zp-f814w", type=float, default=25.94, help="AB zeropoint used for Sersic model counts scaling.")
    parser.add_argument("--max-galaxies", type=int, default=None, help="If set, process only first N selected rows.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(args.selected)
    if args.max_galaxies is not None:
        selected = selected.head(int(args.max_galaxies)).copy()

    # Catalog for image-domain Sersic rendering (arcsec-based Re values).
    df_tsv = photocatalogs.read_photo_cat(args.tsv, skiprows=40)
    if isinstance(df_tsv, str):
        raise RuntimeError(f"Failed to read TSV catalog: {df_tsv}")

    # PIEMD parameters from LensTool model.
    potentiel_all = readLenstoolBlock(best_par=args.parfile, block_name="potentiel")
    potentiel = selectPotentielByType(potentiel_all, ptype="gal")
    cosmologie = readLenstoolBlock(best_par=args.parfile, block_name="cosmologie")
    cosmo = FlatLambdaCDM(H0=cosmologie["H0"], Om0=cosmologie["omegaM"])

    # Open image once.
    with fits.open(args.drz, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        wcs = WCS(hdul[0].header)
        pixscale_arcsec = abs(hdul[0].header.get("CD2_2", 0.0)) * 3600.0
        if pixscale_arcsec <= 0:
            pixscale_arcsec = abs(hdul[0].header.get("CDELT2", 8.3333333333333e-06)) * 3600.0

        summary_rows = []
        r_prof = np.logspace(-2, 2.6, 360)  # kpc

        for _, srow in selected.iterrows():
            lens_idx = int(srow["lens_index"])
            cat_idx = int(srow["cat_index"])
            ra = float(srow["ra_lens"])
            dec = float(srow["dec_lens"])
            f_bad = float(srow["problematic_sample_fraction"])

            tsv_row = df_tsv.iloc[cat_idx]
            re_f814 = float(tsv_row["ReF814W"])
            size_arcsec = float(np.clip(6.0 * re_f814, 8.0, 24.0))
            npix = int(np.clip(np.round(size_arcsec / pixscale_arcsec), 150, 900))

            # Observed cutout (HST image).
            pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            cut = Cutout2D(
                data=data,
                position=pos,
                size=u.Quantity((size_arcsec, size_arcsec), u.arcsec),
                wcs=wcs,
                mode="partial",
                fill_value=np.nan,
            )
            obs = np.asarray(cut.data, dtype=float)
            npix = int(obs.shape[0])
            size_arcsec = float(npix * pixscale_arcsec)

            # Sersic model from photocatalogs.py.
            model = build_single_sersic_model(tsv_row=tsv_row, size_arcsec=size_arcsec, npix=npix, zp=args.zp_f814w)

            # Scale model amplitude to observed cutout and compute residual.
            alpha = robust_linear_scale(obs, model)
            edge = np.concatenate([obs[0, :], obs[-1, :], obs[:, 0], obs[:, -1]])
            bkg = np.nanmedian(edge[np.isfinite(edge)])
            obs_sub = obs - bkg
            model_scaled = alpha * model
            resid = obs_sub - model_scaled
            resid_rms = float(np.nanstd(resid))

            # PIEMD projected profile and best-fit projected tNFW + k*Sigma_Sersic profile.
            pblock = potentiel[lens_idx]
            z_lens = float(findInBlock(pblock, "z_lens"))
            sigma0 = float(findInBlock(pblock, "v_disp")) * np.sqrt(3.0 / 2.0)
            r_core = float(findInBlock(pblock, "core_radius_kpc"))
            r_cut = float(findInBlock(pblock, "cut_radius_kpc"))
            sigma_piemd = piemd_sigma_2d(r_prof, r_core, r_cut, sigma0)

            # Use the same TSV/F814W Sersic parameters for both image and profile fits.
            kpc_per_arcsec = cosmo.angular_diameter_distance(z_lens).to_value("kpc") * np.deg2rad(1.0 / 3600.0)
            re_kpc = float(tsv_row["ReF814W"]) * kpc_per_arcsec
            n_kpc = float(tsv_row["NF814W"])
            mu_kpc = float(tsv_row["muF814W"])
            sigma_ser2d = sersic_sigma_2d(r_prof, re_kpc, n_kpc, mu_kpc)

            rs_guess = max(1.0, 0.2 * r_cut)
            rt_guess = max(r_cut, 2.0 * rs_guess)
            rho0_guess = max(np.nanmax(sigma_piemd) * 1.0e-2, 1.0e2)
            k_guess = max(float(np.nanmedian(sigma_piemd / np.maximum(sigma_ser2d, np.finfo(float).eps))), 1.0e-20)
            rs_tnfw, rt_tnfw, rho0_tnfw, k_ser, diag = fit_tnfw_plus_sersic_2d(
                r_prof,
                sigma_piemd,
                sigma_ser2d,
                rs_guess=rs_guess,
                rt_guess=rt_guess,
                rho0_guess=rho0_guess,
                k_guess=k_guess,
                rmin_fit=0.05,
            )
            sigma_tnfw = sigma_tnfw_projected(r_prof, rs_tnfw, rt_tnfw, rho0_tnfw)
            sigma_stellar = k_ser * sigma_ser2d
            sigma_combo = sigma_tnfw + sigma_stellar

            # Plot summary.
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
            ax_obs, ax_mod = axes[0, 0], axes[0, 1]
            ax_res, ax_prof = axes[1, 0], axes[1, 1]

            finite_obs = obs_sub[np.isfinite(obs_sub)]
            if finite_obs.size > 0:
                v1, v2 = np.nanpercentile(finite_obs, [5.0, 99.5])
            else:
                v1, v2 = 0.0, 1.0

            im0 = ax_obs.imshow(obs_sub, origin="lower", cmap="gray", vmin=v1, vmax=v2)
            ax_obs.set_title("Observed cutout (F814W, bkg-subtracted)")
            plt.colorbar(im0, ax=ax_obs, fraction=0.046, pad=0.04)

            finite_mod = model_scaled[np.isfinite(model_scaled)]
            if finite_mod.size > 0:
                m1, m2 = np.nanpercentile(finite_mod, [5.0, 99.5])
            else:
                m1, m2 = 0.0, 1.0
            im1 = ax_mod.imshow(model_scaled, origin="lower", cmap="magma", vmin=m1, vmax=m2)
            ax_mod.set_title("Sersic model from photocatalogs.py (scaled)")
            plt.colorbar(im1, ax=ax_mod, fraction=0.046, pad=0.04)

            finite_res = resid[np.isfinite(resid)]
            if finite_res.size > 0:
                rmax = np.nanpercentile(np.abs(finite_res), 99.0)
            else:
                rmax = 1.0
            im2 = ax_res.imshow(resid, origin="lower", cmap="coolwarm", vmin=-rmax, vmax=rmax)
            ax_res.set_title(f"Residual: observed - model (RMS={resid_rms:.3e})")
            plt.colorbar(im2, ax=ax_res, fraction=0.046, pad=0.04)

            ax_prof.loglog(r_prof, sigma_piemd, color="black", lw=2.0, label="PIEMD projected")
            ax_prof.loglog(r_prof, sigma_tnfw, color="tab:orange", lw=1.6, label="tNFW projected")
            ax_prof.loglog(r_prof, sigma_stellar, color="tab:blue", lw=1.6, label=r"$k\Sigma_{Sersic}$")
            ax_prof.loglog(r_prof, sigma_combo, color="tab:green", ls="--", lw=1.8, label=r"tNFW + $k\Sigma_{Sersic}$")
            ax_prof.axvline(re_kpc, color="tab:blue", lw=1.0, ls=":", label=f"Re(F814W)={re_kpc:.2f} kpc")
            ax_prof.set_xlabel("r [kpc]")
            ax_prof.set_ylabel(r"$\Sigma(R)$ [arb./Msun kpc$^{-2}$]")
            ax_prof.set_title(f"Surface-density fit (rmse_log={diag['rmse_log']:.3f}, r2_log={diag['r2_log']:.3f})")
            ax_prof.legend(fontsize=8)

            fig.suptitle(
                f"M0416 lens_index={lens_idx} cat_index={cat_idx} | f_bad={f_bad:.2f}\n"
                f"RA={ra:.6f} deg, Dec={dec:.6f} deg, size={size_arcsec:.2f}\"",
                fontsize=12,
            )

            out_png = outdir / f"summary_lens{lens_idx:03d}_cat{cat_idx:03d}.png"
            fig.savefig(out_png, dpi=160)
            plt.close(fig)

            summary_rows.append(
                {
                    "lens_index": lens_idx,
                    "cat_index": cat_idx,
                    "ra_lens_deg": ra,
                    "dec_lens_deg": dec,
                    "size_arcsec": size_arcsec,
                    "npix": npix,
                    "problematic_sample_fraction": f_bad,
                    "scale_alpha_obs_model": alpha,
                    "residual_rms": resid_rms,
                    "sigma0_km_s": sigma0,
                    "z_lens": z_lens,
                    "core_radius_kpc": r_core,
                    "cut_radius_kpc": r_cut,
                    "ReF814W_arcsec": float(tsv_row["ReF814W"]),
                    "NF814W": float(tsv_row["NF814W"]),
                    "muF814W": float(tsv_row["muF814W"]),
                    "ReF814W_kpc": re_kpc,
                    "NF814W_sigma": n_kpc,
                    "muF814W_sigma": mu_kpc,
                    "rs_tnfw_kpc": rs_tnfw,
                    "rt_tnfw_kpc": rt_tnfw,
                    "rho0_tnfw_msun_kpc3": rho0_tnfw,
                    "k_sigma_sersic2d": k_ser,
                    "fit_rmse_log": diag["rmse_log"],
                    "fit_r2_log": diag["r2_log"],
                    "fit_npts": diag["npts"],
                    "fit_success": diag["success"],
                    "summary_png": str(out_png),
                }
            )

            print(
                f"[ok] lens={lens_idx:3d} cat={cat_idx:3d} "
                f"size={size_arcsec:5.2f}\" npix={npix:4d} "
                f"rmse={diag['rmse_log']:.3f} r2={diag['r2_log']:.3f} -> {out_png.name}"
            )

    out_csv = outdir / "problematic_galaxy_summary_table.csv"
    pd.DataFrame(summary_rows).sort_values("lens_index").to_csv(out_csv, index=False)
    print(f"Saved summary table: {out_csv}")


if __name__ == "__main__":
    main()
