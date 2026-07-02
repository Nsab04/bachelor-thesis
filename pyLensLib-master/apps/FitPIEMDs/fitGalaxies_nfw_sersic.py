"""
Fit LensTool PIEMD total density profiles with:

    rho_tot(r) = rho_tNFW(r) + k * rho_Sersic,3D(r)

where Sersic parameters are taken directly from the input photo-morphology catalog
(e.g. ReF160W, NF160W, muF160W), without converting magnitudes to stellar mass.

Notes
-----
- rho_tNFW is a multiplicatively truncated NFW 3D density (Msun/kpc^3):
      rho_tNFW(r) = rho0 / [x (1+x)^2] * [tau^2 / (x^2 + tau^2)],
      with x=r/rs and tau=rt/rs.
- rho_Sersic,3D is the deprojected Sersic approximation (Prugniel-Simien):
      rho(r) is proportional to (r/Re)^(-p) * exp[-b_n * ((r/Re)^(1/n) - 1)],
      p(n) = 1 - 0.6097/n + 0.05463/n^2.
  The scale factor k absorbs unit conversion and acts as a fitted M/L-like factor.
- This script mirrors the matching logic used in fitGalaxies_mcmc_h5.py.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brentq
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

from pyLensLib.lenstool import (
    findInBlock,
    getRADECfromXY,
    getRef_RA_DEC,
    getXYfromPotentiel,
    readLenstoolBlock,
    selectPotentielByType,
)


G_KPC = 4.302e-6  # kpc (km/s)^2 / Msun
DEFAULT_PAR = "/Users/maxmen3/stiva/pietro_models/M0416_B22.par"
DEFAULT_CATALOG = "/Users/maxmen3/projects/Pierpaoli/m0416.dat"
DEFAULT_README = "/Users/maxmen3/projects/Pierpaoli/ReadMe.txt"


def piemd_rho(r, r_core, r_cut, sigma_v):
    """Spherical PIEMD 3D density (Msun/kpc^3)."""
    r = np.asarray(r, dtype=float)
    return (
        (sigma_v**2)
        / (2.0 * np.pi * G_KPC)
        * (r_cut + r_core)
        / (r_core**2 * r_cut)
        / (1.0 + (r / r_core) ** 2)
        / (1.0 + (r / r_cut) ** 2)
    )


def trunc_nfw_rho(r, rs, rt, rho0):
    """Multiplicatively truncated NFW 3D density (Msun/kpc^3)."""
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    tau = rt / rs
    trunc = tau**2 / (x**2 + tau**2)
    return trunc * (rho0 / (x * (1.0 + x) ** 2))


def m_tnfw_enclosed(r, rs, rt, rho0):
    """
    Enclosed mass for multiplicatively truncated NFW.
    Closed-form primitive used in fitGalaxies_mcmc_h5.py.
    """
    r = np.asarray(r, dtype=float)
    x = r / rs
    tau = rt / rs
    tau2 = tau**2
    denom = (tau2 + 1.0) ** 2
    a = tau2 * (tau2 - 1.0) / denom
    b = -tau2 / (tau2 + 1.0)
    d = 2.0 * tau2**2 / denom
    term1 = a * (np.log1p(x) - 0.5 * np.log1p((x / tau) ** 2))
    term2 = b * (x / (1.0 + x))
    term3 = (d / tau) * np.arctan(x / tau)
    f = term1 + term2 + term3
    return 4.0 * np.pi * rho0 * rs**3 * f


def rdelta_tnfw(rs, rt, rho0, rho_ref, delta=200.0):
    """
    Solve R_delta where <rho>(<R) = delta * rho_ref for truncated NFW.
    """
    if rs <= 0.0 or rt <= 0.0 or rho0 <= 0.0 or rho_ref <= 0.0:
        raise ValueError("rs, rt, rho0 and rho_ref must be > 0.")

    target_coeff = (4.0 / 3.0) * np.pi * delta * rho_ref

    def g(rad):
        return m_tnfw_enclosed(rad, rs, rt, rho0) - target_coeff * rad**3

    r_lo = 1.0e-8 * rs
    while g(r_lo) <= 0.0:
        r_lo *= 0.1
        if r_lo < 1.0e-30 * rs:
            break

    m_tot_guess = m_tnfw_enclosed(rt, rs, rt, rho0)
    r_est = max((m_tot_guess / target_coeff) ** (1.0 / 3.0), rt)
    r_hi = 2.0 * r_est
    while g(r_hi) > 0.0:
        r_hi *= 2.0
        if r_hi > 1.0e8 * max(rs, rt):
            raise RuntimeError("Failed to bracket R_delta for truncated NFW.")

    return brentq(g, r_lo, r_hi)


def b_n_sersic(n):
    """
    Sersic b_n approximation (Ciotti & Bertin style series).
    Accurate for n >= ~0.36, and robust in the range seen in the catalogs.
    """
    n = float(n)
    return 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)


def p_n_prugniel_simien(n):
    """Inner power-law index for deprojected Sersic approximation."""
    n = float(n)
    return 1.0 - 0.6097 / n + 0.05463 / (n**2)


def sersic_sigma(r, re_kpc, n, mu_e):
    """
    Circularized Sersic surface-brightness profile in arbitrary linear units.

    Sigma(r) = I_e * exp[-b_n ((r/Re)^(1/n) - 1)]
    with I_e proportional to 10^(-0.4 * mu_e).
    """
    r = np.asarray(r, dtype=float)
    re_kpc = float(re_kpc)
    n = float(n)
    mu_e = float(mu_e)
    if re_kpc <= 0.0 or n <= 0.0:
        raise ValueError("Sersic parameters require Re>0 and n>0.")

    bn = b_n_sersic(n)
    ie = 10.0 ** (-0.4 * mu_e)  # arbitrary linear units
    return ie * np.exp(-bn * ((np.maximum(r, np.finfo(float).eps) / re_kpc) ** (1.0 / n) - 1.0))


def sersic_rho_3d(r, re_kpc, n, mu_e):
    """
    Deprojected Sersic 3D density using the Prugniel-Simien approximation.

    rho_3D(r) = rho_e * (r/Re)^(-p_n) * exp[-b_n ((r/Re)^(1/n) - 1)],
    where rho_e is taken proportional to I_e = 10^(-0.4 mu_e).
    """
    r = np.asarray(r, dtype=float)
    re_kpc = float(re_kpc)
    n = float(n)
    mu_e = float(mu_e)
    if re_kpc <= 0.0 or n <= 0.0:
        raise ValueError("Sersic parameters require Re>0 and n>0.")

    rr = np.maximum(r, np.finfo(float).eps) / re_kpc
    bn = b_n_sersic(n)
    pn = p_n_prugniel_simien(n)
    rho_e = 10.0 ** (-0.4 * mu_e)  # arbitrary linear scale from surface brightness
    return rho_e * np.power(rr, -pn) * np.exp(-bn * (np.power(rr, 1.0 / n) - 1.0))


def fit_tnfw_plus_sersic(
    r, rho_target, rho_ser3d, rs_guess, rt_guess, rho0_guess, k_guess=None, rmin_fit=0.05
):
    """
    Fit rho_target with rho_tNFW + k*rho_Sersic,3D using robust log-space residuals.
    """
    r = np.asarray(r, dtype=float)
    rho_target = np.asarray(rho_target, dtype=float)
    rho_ser3d = np.asarray(rho_ser3d, dtype=float)

    valid = (
        np.isfinite(r)
        & np.isfinite(rho_target)
        & np.isfinite(rho_ser3d)
        & (r >= rmin_fit)
        & (rho_target > 0.0)
        & (rho_ser3d > 0.0)
    )
    r_fit = r[valid]
    y_fit = rho_target[valid]
    s_fit = rho_ser3d[valid]

    if r_fit.size < 20:
        raise ValueError("Not enough valid radial points for fit.")

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
        return trunc_nfw_rho(r_fit, rs, rt, rho0) + k * s_fit

    def residuals(theta):
        y_model = model(theta)
        # logarithmic residuals are appropriate for multi-decade profiles
        res = np.log(y_model) - np.log(y_fit)
        return res[np.isfinite(res)]

    # Broad but physical positivity bounds to avoid pathological collapsed scales.
    lower = np.array(
        [np.log(max(rmin_fit, 1.0e-3)), np.log(1.0e-6), np.log(1.0e2), np.log(1.0e-40)], dtype=float
    )
    upper = np.array(
        [np.log(max(5.0 * np.nanmax(r_fit), 1.0)), np.log(1.0e4), np.log(1.0e16), np.log(1.0e40)], dtype=float
    )
    theta0 = np.clip(theta0, lower + 1.0e-6, upper - 1.0e-6)

    sol = least_squares(
        residuals,
        theta0,
        method="trf",
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=20000,
    )

    rs = float(np.exp(sol.x[0]))
    rt = float(rs * (1.0 + np.exp(sol.x[1])))
    rho0 = float(np.exp(sol.x[2]))
    k = float(np.exp(sol.x[3]))

    y_model = trunc_nfw_rho(r_fit, rs, rt, rho0) + k * s_fit
    res_log = np.log(y_model) - np.log(y_fit)
    rmse_log = float(np.sqrt(np.mean(res_log**2)))
    ylog = np.log(y_fit)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum((np.log(y_model) - ylog) ** 2) / sst) if sst > 0 else np.nan

    return rs, rt, rho0, k, {
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "npts": int(r_fit.size),
        "success": bool(sol.success),
        "message": str(sol.message),
    }


def _parse_vizier_bytes_table(readme_text: str, data_basename: str):
    """Return (colspecs, names) from a VizieR ReadMe for a given data file."""
    sec_pat = re.compile(r"Byte-by-byte Description of file:\s*(.*)")
    lines = readme_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        m = sec_pat.search(line)
        if m and data_basename in m.group(1):
            start_idx = i
            break
    if start_idx is None:
        for i, line in enumerate(lines):
            if sec_pat.search(line):
                start_idx = i
                break
    if start_idx is None:
        raise ValueError("No 'Byte-by-byte' section found in ReadMe.")

    row_pat = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s+[A-Za-z0-9.]+\s+\S+\s+([A-Za-z0-9_]+)")
    colspecs, names = [], []
    for line in lines[start_idx + 1 :]:
        if line.strip().startswith("Byte-by-byte Description of file:"):
            break
        m = row_pat.match(line)
        if m:
            a, b, label = m.groups()
            colspecs.append((int(a) - 1, int(b)))
            names.append(label)
        elif colspecs and not line.strip():
            break
    if not colspecs:
        raise ValueError("Could not parse column specs from ReadMe.")
    return colspecs, names


def load_cluster(readme_path, data_path):
    with open(readme_path, encoding="utf-8", errors="replace") as f:
        readme_text = f.read()
    colspecs, names = _parse_vizier_bytes_table(readme_text, os.path.basename(data_path))
    return pd.read_fwf(data_path, colspecs=colspecs, names=names)


def pick_sersic_columns(df_cat, band):
    """
    Return catalog column names for a selected band token (e.g. 'F160W').
    Required columns: Re{band}, N{band}, mu{band}.
    Optional columns: ar{band}, PA{band}, {band} (magnitude).
    """
    re_col = f"Re{band}"
    n_col = f"N{band}"
    mu_col = f"mu{band}"
    ar_col = f"ar{band}"
    pa_col = f"PA{band}"
    mag_col = band

    required = [re_col, n_col, mu_col]
    missing = [c for c in required if c not in df_cat.columns]
    if missing:
        raise KeyError(
            f"Missing required Sersic columns for band={band}: {missing}. "
            f"Available columns include: {list(df_cat.columns)[:30]} ..."
        )
    return {
        "re": re_col,
        "n": n_col,
        "mu": mu_col,
        "ar": ar_col if ar_col in df_cat.columns else None,
        "pa": pa_col if pa_col in df_cat.columns else None,
        "mag": mag_col if mag_col in df_cat.columns else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit PIEMD total profiles with rho_tNFW + k*rho_Sersic,3D using catalog Sersic parameters."
    )
    parser.add_argument("--parfile", default=DEFAULT_PAR, help="LensTool parameter file.")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, help="Photo-morphology catalog file.")
    parser.add_argument("--explain", default=DEFAULT_README, help="VizieR ReadMe file for fixed-width parsing.")
    parser.add_argument("--band", default="F160W", help="Band token for Sersic parameters (e.g. F814W, F160W).")
    parser.add_argument("--tolerance-arcsec", type=float, default=0.1, help="Sky-match tolerance in arcsec.")
    parser.add_argument("--outcsv", default="lens_galaxy_fits_nfw_sersic.csv", help="Output CSV filename.")
    parser.add_argument("--rmin-fit-kpc", type=float, default=0.05, help="Minimum radius used in the fit [kpc].")
    parser.add_argument("--max-galaxies", type=int, default=None, help="If set, stop after this many fitted galaxies.")
    parser.add_argument("--plot-dir", default="fit_nfw_sersic_profiles", help="Directory for optional per-galaxy plots.")
    parser.add_argument("--max-plots", type=int, default=0, help="Number of profile plots to save (0 disables plotting).")
    args = parser.parse_args()

    # Lens model and cosmology
    potentiel_all = readLenstoolBlock(best_par=args.parfile, block_name="potentiel")
    potentiel = selectPotentielByType(potentiel_all, ptype="gal")
    cosmologie = readLenstoolBlock(best_par=args.parfile, block_name="cosmologie")
    cosmo = FlatLambdaCDM(H0=cosmologie["H0"], Om0=cosmologie["omegaM"])

    # Catalog
    df_cat = load_cluster(readme_path=args.explain, data_path=args.catalog)
    scol = pick_sersic_columns(df_cat, args.band)
    print(f"Using Sersic columns for band={args.band}: {scol}")

    # Coordinates + matching
    x, y = getXYfromPotentiel(potentiel)
    ra_ref, dec_ref = getRef_RA_DEC(args.parfile)
    ra_lens, dec_lens = getRADECfromXY(x, y, ra_ref, dec_ref)

    coords_lens = SkyCoord(ra=ra_lens * u.deg, dec=dec_lens * u.deg)
    coords_cat = SkyCoord(ra=df_cat["RAdeg"].values * u.deg, dec=df_cat["DEdeg"].values * u.deg)
    idx_cat, d2d, _ = coords_lens.match_to_catalog_sky(coords_cat)
    matched = d2d < (args.tolerance_arcsec * u.arcsec)

    lens_idx = np.where(matched)[0]
    cat_idx = idx_cat[matched]
    match_map = dict(zip(lens_idx, cat_idx))
    print(f"Matched {len(match_map)} lens galaxies within {args.tolerance_arcsec:.3f}\".")

    r = np.logspace(-2, 3, 1000)  # kpc
    rows = []
    nplot = 0
    if args.max_plots > 0:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    for j in range(len(potentiel)):
        if j not in match_map:
            continue

        j_cat = int(match_map[j])
        crow = df_cat.iloc[j_cat]

        re_kpc = float(crow[scol["re"]])
        n_ser = float(crow[scol["n"]])
        mu_e = float(crow[scol["mu"]])
        if not np.isfinite(re_kpc) or not np.isfinite(n_ser) or not np.isfinite(mu_e):
            continue
        if re_kpc <= 0.0 or n_ser <= 0.0:
            continue

        ar_ser = float(crow[scol["ar"]]) if scol["ar"] is not None else np.nan
        pa_ser = float(crow[scol["pa"]]) if scol["pa"] is not None else np.nan
        mag_ser = float(crow[scol["mag"]]) if scol["mag"] is not None else np.nan

        # Lens parameters from .par block
        v_disp = float(findInBlock(potentiel[j], "v_disp")) * np.sqrt(3.0 / 2.0)  # convert LensTool sigma_LT -> sigma0
        z_lens = float(findInBlock(potentiel[j], "z_lens"))
        r_cut = float(findInBlock(potentiel[j], "cut_radius_kpc"))
        r_core = float(findInBlock(potentiel[j], "core_radius_kpc"))
        if r_cut <= 0.0 or r_core <= 0.0 or v_disp <= 0.0:
            continue

        rho_target = piemd_rho(r, r_core, r_cut, v_disp)
        rho_ser3d = sersic_rho_3d(r, re_kpc, n_ser, mu_e)

        # Simple initial guesses
        rs_guess = max(1.0, 0.2 * r_cut)
        rt_guess = max(r_cut, 2.0 * rs_guess)
        rho0_guess = np.nanmax(rho_target) * 1.0e-3
        k_guess = np.nanmedian(rho_target / np.maximum(rho_ser3d, np.finfo(float).eps))

        try:
            rs_tnfw, rt_tnfw, rho0_tnfw, k_ser, diag = fit_tnfw_plus_sersic(
                r, rho_target, rho_ser3d, rs_guess, rt_guess, rho0_guess, k_guess, rmin_fit=args.rmin_fit_kpc
            )
        except Exception as exc:
            print(f"[skip] lens_index={j}: fit failed ({exc})")
            continue

        # tNFW-derived summary quantities
        rho_crit = cosmo.critical_density(z_lens).to_value("Msun/kpc3")
        try:
            r200 = rdelta_tnfw(rs_tnfw, rt_tnfw, rho0_tnfw, rho_crit, delta=200.0)
            c200 = r200 / rs_tnfw
            m200 = float(m_tnfw_enclosed(r200, rs_tnfw, rt_tnfw, rho0_tnfw))
        except Exception:
            c200, r200, m200 = np.nan, np.nan, np.nan

        rows.append(
            {
                "lens_index": int(j),
                "cat_index": int(j_cat),
                "ra_lens": float(ra_lens[j]),
                "dec_lens": float(dec_lens[j]),
                "z_lens": z_lens,
                "band": args.band,
                "mag_band": mag_ser,
                "Re_band_kpc": re_kpc,
                "N_band": n_ser,
                "mu_band": mu_e,
                "ar_band": ar_ser,
                "PA_band_deg": pa_ser,
                "v_disp_sigma0_km_s": v_disp,
                "core_radius_kpc": r_core,
                "cut_radius_kpc": r_cut,
                "rs_tnfw_kpc": rs_tnfw,
                "rt_tnfw_kpc": rt_tnfw,
                "rho0_tnfw_msun_kpc3": rho0_tnfw,
                "k_rho_sersic3d": k_ser,
                "c200_tnfw": c200,
                "r200_kpc": r200,
                "m200_tnfw_msun": m200,
                "fit_rmse_log": diag["rmse_log"],
                "fit_r2_log": diag["r2_log"],
                "fit_npts": diag["npts"],
                "fit_success": diag["success"],
                "fit_message": diag["message"],
            }
        )

        if args.max_plots > 0 and nplot < args.max_plots:
            rho_n = trunc_nfw_rho(r, rs_tnfw, rt_tnfw, rho0_tnfw)
            rho_s = k_ser * rho_ser3d
            rho_m = rho_n + rho_s

            fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0), constrained_layout=True)
            ax.loglog(r, rho_target, color="black", lw=2.0, label="PIEMD total (target)")
            ax.loglog(r, rho_n, color="tab:orange", lw=1.8, label="tNFW component")
            ax.loglog(r, rho_s, color="tab:blue", lw=1.8, label=r"$k \cdot \rho_{Sersic,3D}$ component")
            ax.loglog(r, rho_m, color="tab:green", lw=1.8, ls="--", label=r"tNFW + $k\rho_{Sersic,3D}$")
            ax.axvline(re_kpc, color="tab:blue", lw=1.0, ls=":", label="Re (catalog)")
            ax.axvline(rs_tnfw, color="tab:orange", lw=1.0, ls=":", label="rs (fit)")
            ax.axvline(rt_tnfw, color="tab:brown", lw=1.0, ls=":", label="rt (fit)")
            ax.set_xlabel("r [kpc]")
            ax.set_ylabel(r"Profile amplitude")
            ax.set_title(
                f"Lens #{j} / Cat #{j_cat} | {args.band}: n={n_ser:.2f}, Re={re_kpc:.2f} kpc\n"
                f"rmse_log={diag['rmse_log']:.3f}, r2_log={diag['r2_log']:.3f}"
            )
            ax.legend(fontsize=8)
            out_png = Path(args.plot_dir) / f"fit_nfw_sersic_lens{j:03d}.png"
            fig.savefig(out_png, dpi=160)
            plt.close(fig)
            nplot += 1

        print(
            f"[fit] lens={j:3d} cat={j_cat:3d} "
            f"rs={rs_tnfw:9.3e} rt={rt_tnfw:9.3e} kpc rho0={rho0_tnfw:9.3e} k={k_ser:9.3e} "
            f"rmse={diag['rmse_log']:.3f}"
        )

        if args.max_galaxies is not None and len(rows) >= args.max_galaxies:
            break

    if len(rows) == 0:
        raise RuntimeError("No successful fits were produced.")

    df_out = pd.DataFrame(rows).sort_values("lens_index").reset_index(drop=True)
    df_out.to_csv(args.outcsv, index=False)
    print(f"Saved {len(df_out)} fitted galaxies to {args.outcsv}")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
