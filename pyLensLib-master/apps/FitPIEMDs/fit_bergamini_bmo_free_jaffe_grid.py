#!/usr/bin/env python3
"""
Grid test: PIEMD profiles from LensTool .par-derived r_cut(sigma0), then BMO+free-Jaffe fits.

Tasks:
1) Derive r_cut normalization from .par galaxies, then use r_cut = rcut_ref * (sigma0 / 220)^2
2) Build sigma0 grid from 10 to 250 km/s
3) For each sigma0, generate a PIEMD density profile at z_lens=0.396
4) Fit with BMO + free Jaffe
5) Plot concentration-mass relation over all mock galaxies

Core-radius assumption:
- By default, adopts the Bergamini et al. (2019) choice for cluster members:
  vanishing core radius (r_core = 0).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import least_squares


G_KPC = 4.302e-6  # kpc (km/s)^2 / Msun
SIGMA0_REF_KMS = 220.0


def _load_reference_fit_module():
    """Load fitGalaxies_tnfw_jaffe_dual.py as the reference fitting backend."""
    ref_path = Path(__file__).resolve().with_name("fitGalaxies_tnfw_jaffe_dual.py")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference fit module not found: {ref_path}")

    # Ensure local pyLensLib package is importable when this script is run directly.
    project_root = ref_path.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    spec = importlib.util.spec_from_file_location("fitgalaxies_tnfw_jaffe_dual_ref", str(ref_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {ref_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_REF_FITMOD = _load_reference_fit_module()


def par_derived_rcut_kpc(
    sigma0_kms: float | np.ndarray,
    rcut_ref_kpc: float,
    sigma0_ref_kms: float = SIGMA0_REF_KMS,
    slope: float = 2.0,
) -> float | np.ndarray:
    """r_cut(sigma0) relation derived from LensTool .par scaling-member population."""
    sigma0_kms = np.asarray(sigma0_kms, dtype=float)
    return float(rcut_ref_kpc) * (sigma0_kms / float(sigma0_ref_kms)) ** float(slope)


def piemd_rho(r_kpc: np.ndarray, r_core_kpc: float, r_cut_kpc: float, sigma0_kms: float) -> np.ndarray:
    """PIEMD 3D density [Msun/kpc^3], including singular-core limit."""
    r = np.asarray(r_kpc, dtype=float)
    rr = np.maximum(r, np.finfo(float).eps)
    if r_core_kpc <= 0.0:
        # Singular (vanishing-core) dPIE / PIEMD limit:
        # rho = (sigma0^2 / 2piG) * rcut^2 / [r^2 (r^2 + rcut^2)]
        return (
            (sigma0_kms**2)
            / (2.0 * np.pi * G_KPC)
            * (r_cut_kpc**2)
            / (rr**2 * (rr**2 + r_cut_kpc**2))
        )
    return (
        (sigma0_kms**2)
        / (2.0 * np.pi * G_KPC)
        * (r_cut_kpc + r_core_kpc)
        / (r_core_kpc**2 * r_cut_kpc)
        / (1.0 + r**2 / r_core_kpc**2)
        / (1.0 + r**2 / r_cut_kpc**2)
    )


def jaffe_rho_rj(r_kpc: np.ndarray, mstar_msun: float, rj_kpc: float) -> np.ndarray:
    """Jaffe 3D density [Msun/kpc^3] with explicit scale radius rJ."""
    return _REF_FITMOD.jaffe_rho_rj(r_kpc, mstar_msun, rj_kpc)


def bmo_rho(r_kpc: np.ndarray, rs_kpc: float, rt_kpc: float, rho0_msun_kpc3: float) -> np.ndarray:
    """BMO n=0.5 density profile [Msun/kpc^3]."""
    return _REF_FITMOD.bmo_rho(r_kpc, rs_kpc, rt_kpc, rho0_msun_kpc3)


def fit_bmo_plus_free_jaffe(
    r_kpc: np.ndarray,
    rho_target_msun_kpc3: np.ndarray,
    rs_guess_kpc: float,
    rt_guess_kpc: float,
    rho0_guess_msun_kpc3: float,
    mstar_guess_msun: float,
    re_kpc: float,
    rmin_fit_kpc: float = 0.05,
    rmax_fit_kpc: float | None = None,
    penalty_weight: float = 100.0,
    inner_radius_fraction_of_re: float = 0.3,
    inner_penalty_weight: float = 100.0,
    enforce_inner_stellar_dominance: bool = False,
    inner_scale_kpc: float | None = None,
    max_nfev: int = 8000,
) -> tuple[float, float, float, float, float, dict]:
    """
    Fit rho_target = rho_BMO(rs, rt, rho0) + rho_Jaffe(Mstar, rJ).
    """
    # Keep a valid unit-Jaffe shape for API compatibility; the reference fitter
    # currently rebuilds Jaffe internally from free (Mstar, rJ).
    rho_jaffe_shape = _REF_FITMOD.jaffe_rho_rj(np.asarray(r_kpc, dtype=float), 1.0, 1.0)
    rs, rt, rho0, mstar, rj, diag = _REF_FITMOD.fit_bmo_plus_free_jaffe(
        np.asarray(r_kpc, dtype=float),
        np.asarray(rho_target_msun_kpc3, dtype=float),
        rho_jaffe_shape,
        float(rs_guess_kpc),
        float(rt_guess_kpc),
        float(rho0_guess_msun_kpc3),
        mstar_guess=float(mstar_guess_msun),
        rmin_fit=float(rmin_fit_kpc),
        rmax_fit=None if rmax_fit_kpc is None else float(rmax_fit_kpc),
        # Re-regularization disabled: with free rJ, solution should be Re-neutral.
        re_kpc=None,
        penalty_weight=float(penalty_weight),
    )
    diag = dict(diag)
    inner_scale = float(inner_scale_kpc) if inner_scale_kpc is not None else float(re_kpc)
    inner_rmax = max(float(rmin_fit_kpc), float(inner_radius_fraction_of_re) * max(inner_scale, 1.0e-6))

    if enforce_inner_stellar_dominance:
        r = np.asarray(r_kpc, dtype=float)
        y = np.asarray(rho_target_msun_kpc3, dtype=float)
        mask = np.isfinite(r) & np.isfinite(y) & (r >= float(rmin_fit_kpc))
        if rmax_fit_kpc is not None:
            mask &= (r <= float(rmax_fit_kpc))
        rr, yy = r[mask], y[mask]
        if rr.size >= 10:
            inner_mask = rr <= inner_rmax
            if np.sum(inner_mask) < 5:
                inner_mask = np.arange(rr.size) < min(20, rr.size)

            # Start from the reference optimum and refine with an explicit inner penalty.
            theta0 = np.array(
                [
                    np.log(max(rs, 1.0e-8)),
                    np.log(max(rt / max(rs, 1.0e-8) - 1.0, 1.0e-8)),
                    np.log(max(rho0, 1.0e-30)),
                    np.log(max(mstar, 1.0e-20)),
                    np.log(max(rj, 1.0e-6)),
                ],
                dtype=float,
            )
            lower = np.array(
                [
                    np.log(max(float(rmin_fit_kpc), 1.0e-4)),
                    np.log(1.0e-8),
                    np.log(1.0e-30),
                    np.log(1.0e-12),
                    np.log(1.0e-6),
                ],
                dtype=float,
            )
            upper = np.array(
                [
                    np.log(max(5.0 * np.nanmax(rr), 1.0)),
                    np.log(1.0e6),
                    np.log(1.0e30),
                    np.log(1.0e18),
                    np.log(max(5.0 * np.nanmax(rr), 10.0)),
                ],
                dtype=float,
            )
            theta0 = np.clip(theta0, lower + 1.0e-8, upper - 1.0e-8)

            def residuals_inner(theta: np.ndarray) -> np.ndarray:
                rs_ = float(np.exp(theta[0]))
                rt_ = float(rs_ * (1.0 + np.exp(theta[1])))
                rho0_ = float(np.exp(theta[2]))
                mstar_ = float(np.exp(theta[3]))
                rj_ = float(np.exp(theta[4]))
                rho_dm_ = bmo_rho(rr, rs_, rt_, rho0_)
                rho_star_ = jaffe_rho_rj(rr, mstar_, rj_)
                ymod_ = rho_dm_ + rho_star_
                res_ = np.log(np.maximum(ymod_, np.finfo(float).tiny)) - np.log(np.maximum(yy, np.finfo(float).tiny))
                res_ = res_[np.isfinite(res_)]
                if np.any(inner_mask):
                    vio = np.maximum(rho_dm_[inner_mask] - rho_star_[inner_mask], 0.0)
                    pen = float(inner_penalty_weight) * vio / np.maximum(rho_dm_[inner_mask], np.finfo(float).tiny)
                    res_ = np.concatenate([res_, pen])
                return res_

            sol_in = least_squares(
                residuals_inner,
                theta0,
                method="trf",
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=0.1,
                max_nfev=int(max_nfev),
            )
            rs = float(np.exp(sol_in.x[0]))
            rt = float(rs * (1.0 + np.exp(sol_in.x[1])))
            rho0 = float(np.exp(sol_in.x[2]))
            mstar = float(np.exp(sol_in.x[3]))
            rj = float(np.exp(sol_in.x[4]))

            rho_dm = bmo_rho(rr, rs, rt, rho0)
            rho_star = jaffe_rho_rj(rr, mstar, rj)
            ymod = rho_dm + rho_star
            reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(np.maximum(yy, np.finfo(float).tiny))
            rmse_log = float(np.sqrt(np.mean(reslog**2)))
            ylog = np.log(np.maximum(yy, np.finfo(float).tiny))
            sst = np.sum((ylog - ylog.mean()) ** 2)
            r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan
            diag["success"] = bool(diag.get("success", True)) and bool(sol_in.success)
            diag["message"] = f"{diag.get('message', '')} | inner-penalty refine: {sol_in.message}"
            diag["rmse_log"] = rmse_log
            diag["r2_log"] = r2_log
            diag["npts"] = int(rr.size)

            diag["stellar_dominates_inner"] = bool(np.all(rho_star[inner_mask] >= rho_dm[inner_mask])) if np.any(inner_mask) else True
            diag["inner_penalty_effective"] = float(inner_penalty_weight)
        else:
            diag["stellar_dominates_inner"] = False
            diag["inner_penalty_effective"] = float(inner_penalty_weight)
    else:
        diag.setdefault("stellar_dominates_inner", False)
        diag.setdefault("inner_penalty_effective", np.nan)

    diag["uses_reference_fit_module"] = True
    diag["re_regularization_used"] = False
    diag.setdefault("inner_rmax_kpc", float(inner_rmax))
    return rs, rt, rho0, mstar, rj, diag


def compute_bmo_derived(
    rs_kpc: float,
    rt_kpc: float,
    rho0_msun_kpc3: float,
    rho_crit_msun_kpc3: float,
    npts: int = 10000,
) -> dict:
    """Compute c200, R200, M200 from a BMO profile by numerical integration."""
    return _REF_FITMOD.compute_bmo_derived(rs_kpc, rt_kpc, rho0_msun_kpc3, rho_crit_msun_kpc3, npts=int(npts))


def load_par_galaxy_sigma0_rcut(parfile: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (sigma0, r_cut) for galaxy-scale components from a LensTool .par file."""
    pot_all = _REF_FITMOD.readLenstoolBlock(best_par=str(parfile), block_name="potentiel")
    pot_gal = _REF_FITMOD.selectPotentielByType(pot_all, ptype="gal")
    sigma0_vals: list[float] = []
    rcut_vals: list[float] = []
    for pot in pot_gal:
        v_disp = _REF_FITMOD.findInBlock(pot, "v_disp")
        r_cut_kpc = _REF_FITMOD.findInBlock(pot, "cut_radius_kpc")
        if v_disp is None or r_cut_kpc is None:
            continue
        sigma0 = float(v_disp) * np.sqrt(3.0 / 2.0)
        rcut = float(r_cut_kpc)
        if np.isfinite(sigma0) and np.isfinite(rcut) and sigma0 > 0.0 and rcut > 0.0:
            sigma0_vals.append(sigma0)
            rcut_vals.append(rcut)
    return np.asarray(sigma0_vals, dtype=float), np.asarray(rcut_vals, dtype=float)


def fit_rcut_scaling_from_par(
    sigma0_kms: np.ndarray,
    rcut_kpc: np.ndarray,
    sigma0_ref_kms: float = SIGMA0_REF_KMS,
) -> tuple[float, float]:
    """
    Fit both normalization and slope in:
        r_cut = rcut_ref * (sigma0/sigma0_ref)^slope
    using linear regression in log space.
    """
    s = np.asarray(sigma0_kms, dtype=float)
    r = np.asarray(rcut_kpc, dtype=float)
    m = np.isfinite(s) & np.isfinite(r) & (s > 0.0) & (r > 0.0)
    if np.sum(m) < 2:
        raise ValueError("Not enough valid galaxies to fit rcut scaling.")
    x = np.log(s[m] / float(sigma0_ref_kms))
    y = np.log(r[m])
    slope, intercept = np.polyfit(x, y, 1)
    rcut_ref = float(np.exp(intercept))
    if not (np.isfinite(rcut_ref) and rcut_ref > 0 and np.isfinite(slope)):
        raise ValueError("Non-finite rcut scaling fit.")
    return rcut_ref, float(slope)


def _load_dual_csv(dual_csv_path: Path) -> pd.DataFrame | None:
    if dual_csv_path.exists():
        try:
            return pd.read_csv(dual_csv_path)
        except Exception as exc:
            print(f"Warning: could not read dual-fit CSV: {exc}")
            return None
    print(f"Warning: dual-fit CSV not found: {dual_csv_path}")
    return None


def _build_dual_templates(df_dual: pd.DataFrame | None) -> dict:
    out = {"sigma": np.array([], dtype=float), "re": np.array([], dtype=float), "mstar": np.array([], dtype=float)}
    if df_dual is None:
        return out
    required = {"v_disp_sigma0_km_s", "ReF160W_kpc", "Mstar_from_F160W_Msun"}
    if not required.issubset(set(df_dual.columns)):
        return out
    m = (
        np.isfinite(df_dual["v_disp_sigma0_km_s"].values)
        & np.isfinite(df_dual["ReF160W_kpc"].values)
        & np.isfinite(df_dual["Mstar_from_F160W_Msun"].values)
        & (df_dual["v_disp_sigma0_km_s"].values > 0.0)
        & (df_dual["ReF160W_kpc"].values > 0.0)
        & (df_dual["Mstar_from_F160W_Msun"].values > 0.0)
    )
    d = df_dual.loc[m].copy()
    if len(d) == 0:
        return out
    out["sigma"] = d["v_disp_sigma0_km_s"].to_numpy(dtype=float)
    out["re"] = d["ReF160W_kpc"].to_numpy(dtype=float)
    out["mstar"] = d["Mstar_from_F160W_Msun"].to_numpy(dtype=float)
    return out


def _nearest_template_value(sigma0: float, sigma_ref: np.ndarray, value_ref: np.ndarray, fallback: float) -> float:
    if sigma_ref.size == 0 or value_ref.size == 0:
        return float(fallback)
    idx = int(np.argmin(np.abs(sigma_ref - sigma0)))
    return float(value_ref[idx])


def _cm_logfit(logM: np.ndarray, c200: np.ndarray) -> dict:
    """Fit log10(c200) = a * log10(M200) + b and return diagnostics."""
    x = np.asarray(logM, dtype=float)
    c = np.asarray(c200, dtype=float)
    m = np.isfinite(x) & np.isfinite(c) & (c > 0.0)
    xx = x[m]
    yy = np.log10(c[m])
    out = {"n": int(xx.size), "a": np.nan, "b": np.nan, "scatter_dex": np.nan}
    if xx.size < 3:
        return out
    a, b = np.polyfit(xx, yy, 1)
    yfit = a * xx + b
    out["a"] = float(a)
    out["b"] = float(b)
    out["scatter_dex"] = float(np.sqrt(np.mean((yy - yfit) ** 2)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIEMD grid from .par-derived r_cut(sigma0), fitted with BMO+free Jaffe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sigma-min", type=float, default=10.0, help="Minimum sigma0 [km/s].")
    parser.add_argument("--sigma-max", type=float, default=250.0, help="Maximum sigma0 [km/s].")
    parser.add_argument("--sigma-step", type=float, default=1.0, help="Step in sigma0 [km/s].")
    parser.add_argument(
        "--rcut-ref-kpc",
        type=float,
        default=None,
        help=(
            "Optional normalization for r_cut(sigma0): r_cut = rcut_ref*(sigma0/220)^slope. "
            "Must be provided together with --rcut-slope; otherwise both are fit from --parfile."
        ),
    )
    parser.add_argument(
        "--rcut-slope",
        type=float,
        default=None,
        help=(
            "Optional power-law slope in r_cut = rcut_ref*(sigma0/220)^slope. "
            "Must be provided together with --rcut-ref-kpc; otherwise both are fit from --parfile."
        ),
    )
    parser.add_argument(
        "--parfile",
        type=str,
        default=str(Path(__file__).resolve().with_name("M0416.par")),
        help="LensTool .par file used to overplot observed galaxy sigma0-r_cut points.",
    )
    parser.add_argument("--z-lens", type=float, default=0.396, help="Lens redshift.")
    parser.add_argument(
        "--re-kpc",
        type=float,
        default=2.0,
        help="Deprecated/ignored in free-Jaffe mode (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--r-core-kpc",
        type=float,
        default=0.0,
        help="PIEMD core radius in kpc. Use 0 for Bergamini+2019 (vanishing core).",
    )
    parser.add_argument("--rmin-fit-kpc", type=float, default=0.05, help="Minimum radius for the fit [kpc].")
    parser.add_argument(
        "--enforce-inner-stellar-dominance",
        action="store_true",
        help="If set, add an explicit penalty in the fit when rho_DM exceeds rho_star in inner points.",
    )
    parser.add_argument(
        "--inner-stellar-radius-factor",
        type=float,
        default=0.3,
        help="Inner radius = factor * r_cut used for the optional inner stellar-dominance penalty.",
    )
    parser.add_argument(
        "--inner-stellar-penalty",
        type=float,
        default=120.0,
        help="Penalty strength for violating rho_star >= rho_DM in the inner region (when enabled).",
    )
    parser.add_argument(
        "--rmax-re-factor",
        type=float,
        default=10.0,
        help="Deprecated/ignored (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--rmax-fit-kpc",
        type=float,
        default=None,
        help="Maximum radius used in the fit [kpc]. If omitted, no upper radial cut is applied.",
    )
    parser.add_argument("--max-nfev", type=int, default=8000, help="Maximum LSQ function evaluations.")
    parser.add_argument("--mstar-guess", type=float, default=1.0e10, help="Initial guess for free stellar mass [Msun].")
    parser.add_argument(
        "--mstar-guess-mode",
        choices=["fixed", "dual-nearest-sigma", "dual-median"],
        default="fixed",
        help="How to set stellar-mass initial guess: fixed --mstar-guess, or inferred from dual CSV.",
    )
    parser.add_argument("--out-csv", type=str, default="bergamini_bmo_free_jaffe_grid.csv", help="Output CSV file.")
    parser.add_argument(
        "--out-fig",
        type=str,
        default="bergamini_bmo_free_jaffe_c200_vs_M200.png",
        help="Output concentration-mass figure.",
    )
    parser.add_argument(
        "--out-rcut",
        type=str,
        default="bergamini_sigma0_vs_rcut.png",
        help="Output sigma0-r_cut relation figure.",
    )
    parser.add_argument(
        "--out-mosaic",
        type=str,
        default="bergamini_bmo_free_jaffe_profiles_mosaic_4x4.png",
        help="Output 4x4 density-profile mosaic.",
    )
    parser.add_argument(
        "--dual-csv",
        type=str,
        default=str(Path(__file__).resolve().with_name("lens_galaxy_fits_bmo_jaffe_dual.csv")),
        help="CSV from fitGalaxies_tnfw_jaffe_dual.py used to overplot c200-M200 points.",
    )
    args = parser.parse_args()

    if args.sigma_step <= 0:
        raise ValueError("--sigma-step must be > 0.")
    if args.sigma_max < args.sigma_min:
        raise ValueError("--sigma-max must be >= --sigma-min.")
    if args.rcut_ref_kpc is not None and args.rcut_ref_kpc <= 0:
        raise ValueError("--rcut-ref-kpc must be > 0 when provided.")
    if args.rcut_slope is not None and not np.isfinite(args.rcut_slope):
        raise ValueError("--rcut-slope must be finite when provided.")
    if (args.rcut_ref_kpc is None) != (args.rcut_slope is None):
        raise ValueError("Provide both --rcut-ref-kpc and --rcut-slope, or provide neither.")
    if args.rmax_fit_kpc is not None and args.rmax_fit_kpc <= 0:
        raise ValueError("--rmax-fit-kpc must be > 0 when provided.")
    if args.enforce_inner_stellar_dominance and args.inner_stellar_penalty <= 0:
        raise ValueError("--inner-stellar-penalty must be > 0 when --enforce-inner-stellar-dominance is set.")
    if args.enforce_inner_stellar_dominance and args.inner_stellar_radius_factor <= 0:
        raise ValueError("--inner-stellar-radius-factor must be > 0 when --enforce-inner-stellar-dominance is set.")

    sigma_grid = np.arange(args.sigma_min, args.sigma_max + 0.5 * args.sigma_step, args.sigma_step, dtype=float)
    #r_eval = np.logspace(-2.0, 4.0, 10000)  # kpc

    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    rho_crit = cosmo.critical_density(args.z_lens).to_value("Msun/kpc3")

    parfile_path = Path(args.parfile).expanduser().resolve()
    sigma0_par = np.array([], dtype=float)
    rcut_par = np.array([], dtype=float)
    rcut_ref_source = "input"
    rcut_slope_source = "input"
    need_par_for_scaling = (args.rcut_ref_kpc is None)
    need_par_for_overlay = args.parfile is not None
    if need_par_for_scaling:
        if not parfile_path.exists():
            raise FileNotFoundError(
                f"parfile not found: {parfile_path}. "
                "Provide both --rcut-ref-kpc and --rcut-slope to override scaling."
            )
        sigma0_par, rcut_par = load_par_galaxy_sigma0_rcut(parfile_path)
    elif need_par_for_overlay:
        if parfile_path.exists():
            try:
                sigma0_par, rcut_par = load_par_galaxy_sigma0_rcut(parfile_path)
            except Exception as exc:
                print(f"Warning: could not parse parfile for sigma0-r_cut overlay: {exc}")
        else:
            print(f"Warning: parfile not found, skipping sigma0-r_cut overlay: {parfile_path}")

    if args.rcut_ref_kpc is None:
        rcut_ref_kpc, rcut_slope = fit_rcut_scaling_from_par(
            sigma0_par,
            rcut_par,
            sigma0_ref_kms=SIGMA0_REF_KMS,
        )
        rcut_ref_source = "derived_from_par_fit"
        rcut_slope_source = "derived_from_par_fit"
    else:
        rcut_ref_kpc = float(args.rcut_ref_kpc)
        rcut_slope = float(args.rcut_slope)
        rcut_ref_source = "input"
        rcut_slope_source = "input"


    dual_csv_path = Path(args.dual_csv).expanduser().resolve()
    df_dual = _load_dual_csv(dual_csv_path)
    dual_tpl = _build_dual_templates(df_dual)
    dual_has_tpl = dual_tpl["sigma"].size > 0

    rows: list[dict] = []
    for sigma0 in sigma_grid:
        r_cut = float(
            par_derived_rcut_kpc(
                sigma0,
                rcut_ref_kpc=rcut_ref_kpc,
                sigma0_ref_kms=SIGMA0_REF_KMS,
                slope=rcut_slope,
            )
        )
        if not np.isfinite(r_cut) or r_cut <= 0:
            continue

        r_core = float(args.r_core_kpc)
        if r_core < 0:
            r_core = 0.0
        if r_core >= r_cut:
            r_core = max(min(0.2 * r_cut, r_cut - 1.0e-6), 0.0)
        # Global user-defined upper radius, or no upper cut if omitted.
        rmax_fit_kpc = float(args.rmax_fit_kpc) if args.rmax_fit_kpc is not None else None
        r_eval = np.logspace(np.log10(r_cut*0.01), np.log10(r_cut*1000), 10000)

        rho_piemd = piemd_rho(r_eval, r_core, r_cut, sigma0)

        rs_guess = max(1.0, 0.2 * r_cut)
        rt_guess = max(r_cut, 2.0 * rs_guess)
        rho0_guess = max(float(np.nanmax(rho_piemd) * 1.0e-3), 1.0e-20)

        re_fit_kpc = float(args.re_kpc)

        if args.mstar_guess_mode == "fixed":
            mstar_guess_fit = float(args.mstar_guess)
        elif args.mstar_guess_mode == "dual-median":
            mstar_guess_fit = float(np.median(dual_tpl["mstar"])) if dual_has_tpl else float(args.mstar_guess)
        else:
            mstar_guess_fit = _nearest_template_value(
                sigma0, dual_tpl["sigma"], dual_tpl["mstar"], float(args.mstar_guess)
            )

        try:
            rs, rt, rho0, mstar, rj, diag = fit_bmo_plus_free_jaffe(
                r_kpc=r_eval,
                rho_target_msun_kpc3=rho_piemd,
                rs_guess_kpc=rs_guess,
                rt_guess_kpc=rt_guess,
                rho0_guess_msun_kpc3=rho0_guess,
                mstar_guess_msun=mstar_guess_fit,
                re_kpc=re_fit_kpc,
                rmin_fit_kpc=args.rmin_fit_kpc,
                rmax_fit_kpc=rmax_fit_kpc,
                inner_radius_fraction_of_re=args.inner_stellar_radius_factor,
                inner_penalty_weight=args.inner_stellar_penalty,
                enforce_inner_stellar_dominance=bool(args.enforce_inner_stellar_dominance),
                inner_scale_kpc=r_cut,
                max_nfev=args.max_nfev,
            )
            derived = compute_bmo_derived(rs, rt, rho0, rho_crit_msun_kpc3=rho_crit)
        except Exception as exc:
            rs = rt = rho0 = mstar = rj = np.nan
            diag = {
                "success": False,
                "message": str(exc),
                "npts": 0,
                "rmse_log": np.nan,
                "r2_log": np.nan,
                "mstar_max_constraint": np.nan,
            }
            derived = {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}

        rows.append(
            {
                "sigma0_km_s": sigma0,
                "r_cut_kpc": r_cut,
                "r_core_kpc": r_core,
                "z_lens": args.z_lens,
                "rs_bmo_kpc": rs,
                "rt_bmo_kpc": rt,
                "rho0_bmo_msun_kpc3": rho0,
                "Mstar_free_Msun": mstar,
                "rJ_free_kpc": rj,
                "Re_fit_kpc": re_fit_kpc,
                "Mstar_guess_fit_Msun": mstar_guess_fit,
                "c200": derived["c200"],
                "R200_kpc": derived["R200_kpc"],
                "M200_Msun": derived["M200_Msun"],
                "fit_success": bool(diag["success"]),
                "fit_message": str(diag["message"]),
                "fit_npts": int(diag["npts"]),
                "fit_rmse_log": diag["rmse_log"],
                "fit_r2_log": diag["r2_log"],
                "mstar_max_constraint_Msun": diag["mstar_max_constraint"],
                "inner_rmax_kpc": diag.get("inner_rmax_kpc", np.nan),
                "stellar_dominates_inner": bool(diag.get("stellar_dominates_inner", False)),
                "inner_penalty_effective": diag.get("inner_penalty_effective", np.nan),
                "rmax_fit_kpc": (rmax_fit_kpc if rmax_fit_kpc is not None else np.nan),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv).resolve()
    df.to_csv(out_csv, index=False)

    if df_dual is None:
        print(f"Warning: dual-fit CSV not available for c200-M200 overlay: {dual_csv_path}")

    # Plot sigma0 - r_cut relation.
    out_rcut = Path(args.out_rcut).resolve()
    fig0, ax0 = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    # Explicit analytical scaling curve.
    if sigma0_par.size > 0:
        smin = min(float(np.nanmin(df["sigma0_km_s"].values)), float(np.nanmin(sigma0_par)))
        smax = max(float(np.nanmax(df["sigma0_km_s"].values)), float(np.nanmax(sigma0_par)))
    else:
        smin = float(np.nanmin(df["sigma0_km_s"].values))
        smax = float(np.nanmax(df["sigma0_km_s"].values))
    s_curve = np.linspace(max(1.0, smin), smax, 400)
    ax0.plot(
        s_curve,
        par_derived_rcut_kpc(
            s_curve,
            rcut_ref_kpc=rcut_ref_kpc,
            sigma0_ref_kms=SIGMA0_REF_KMS,
            slope=rcut_slope,
        ),
        color="tab:blue",
        lw=2.0,
        label=(
            r"Relation: $r_{\rm cut}="
            + f"{rcut_ref_kpc:.4f}"
            + r"(\sigma_0/"
            + f"{SIGMA0_REF_KMS:.0f}"
            + r")^{"
            + f"{rcut_slope:.3f}"
            + r"}$"
        ),
    )
    # Grid galaxies generated in this script.
    ax0.scatter(
        df["sigma0_km_s"].values,
        df["r_cut_kpc"].values,
        s=16,
        alpha=0.70,
        color="tab:orange",
        edgecolors="none",
        label=f"Grid galaxies ({len(df)})",
    )
    if sigma0_par.size > 0:
        par_label = Path(args.parfile).name
        ax0.scatter(
            sigma0_par,
            rcut_par,
            s=14,
            alpha=0.65,
            color="black",
            edgecolors="none",
            label=f"{par_label} galaxies ({sigma0_par.size})",
        )
    ax0.set_xlabel(r"$\sigma_0$ [km s$^{-1}$]")
    ax0.set_ylabel(r"$r_{\rm cut}$ [kpc]")
    ax0.set_title(".par-derived scaling for r_cut(sigma0)")
    ax0.legend(fontsize=9)
    fig0.savefig(out_rcut, dpi=180, bbox_inches="tight")
    plt.close(fig0)

    # Plot c200-M200 relation from all successful fits.
    ok = (
        df["fit_success"].astype(bool)
        & np.isfinite(df["c200"])
        & np.isfinite(df["M200_Msun"])
        & (df["c200"] > 0)
        & (df["M200_Msun"] > 0)
    )
    dplot = df.loc[ok].copy()

    out_fig = Path(args.out_fig).resolve()
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    x_grid = np.array([], dtype=float)
    c_grid = np.array([], dtype=float)
    sigma_grid = np.array([], dtype=float)
    x_dual = np.array([], dtype=float)
    c_dual = np.array([], dtype=float)
    if len(dplot) > 0:
        x = np.log10(dplot["M200_Msun"].values)
        y = dplot["c200"].values
        x_grid = np.asarray(x, dtype=float)
        c_grid = np.asarray(y, dtype=float)
        sigma_grid = np.asarray(dplot["sigma0_km_s"].values, dtype=float)
        sc = ax.scatter(
            x,
            y,
            c=dplot["sigma0_km_s"].values,
            cmap="viridis",
            s=18,
            alpha=0.85,
            edgecolors="none",
            label=f"Mock galaxies ({len(dplot)})",
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$\sigma_0$ [km s$^{-1}$]")

        # Median concentration in mass bins.
        nbins = 10
        edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
        xb, yb = [], []
        for i in range(nbins):
            if i < nbins - 1:
                m = (x >= edges[i]) & (x < edges[i + 1])
            else:
                m = (x >= edges[i]) & (x <= edges[i + 1])
            if np.sum(m) < 3:
                continue
            xb.append(np.nanmedian(x[m]))
            yb.append(np.nanmedian(y[m]))
        if len(xb) > 0:
            ax.plot(xb, yb, color="crimson", lw=2.0, marker="o", ms=4, label="Binned median")

    if df_dual is not None:
        required_cols = {"c200_free", "M200_bmo_free_Msun"}
        if required_cols.issubset(set(df_dual.columns)):
            ok_dual = (
                np.isfinite(df_dual["c200_free"].values)
                & np.isfinite(df_dual["M200_bmo_free_Msun"].values)
                & (df_dual["c200_free"].values > 0.0)
                & (df_dual["M200_bmo_free_Msun"].values > 0.0)
            )
            if "fit_free_success" in df_dual.columns:
                ok_dual &= df_dual["fit_free_success"].astype(bool).values
            ddual = df_dual.loc[ok_dual].copy()
            if len(ddual) > 0:
                x_dual = np.log10(ddual["M200_bmo_free_Msun"].values)
                c_dual = ddual["c200_free"].values
                ax.scatter(
                    x_dual,
                    c_dual,
                    color="black",
                    s=14,
                    alpha=0.45,
                    edgecolors="none",
                    label=f"Dual-fit galaxies ({len(ddual)})",
                )
        else:
            print(
                "Warning: dual-fit CSV missing required columns for c200-M200 overlay: "
                "expected c200_free, M200_bmo_free_Msun."
            )

    # Diagnostic fits: log10(c200) = a*log10(M200) + b.
    diag_grid = _cm_logfit(x_grid, c_grid)
    diag_dual = _cm_logfit(x_dual, c_dual)
    diag_grid_overlap = {"n": 0, "a": np.nan, "b": np.nan, "scatter_dex": np.nan}
    if x_grid.size > 0 and sigma_grid.size == x_grid.size and df_dual is not None and x_dual.size > 0:
        if "v_disp_sigma0_km_s" in df_dual.columns:
            s_dual = df_dual["v_disp_sigma0_km_s"].to_numpy(dtype=float)
            s_dual = s_dual[np.isfinite(s_dual) & (s_dual > 0.0)]
            if s_dual.size > 0:
                smin, smax = float(np.min(s_dual)), float(np.max(s_dual))
                mov = (sigma_grid >= smin) & (sigma_grid <= smax)
                diag_grid_overlap = _cm_logfit(x_grid[mov], c_grid[mov])
            else:
                mov = np.zeros_like(sigma_grid, dtype=bool)
        else:
            mov = np.zeros_like(sigma_grid, dtype=bool)
    else:
        mov = np.zeros_like(sigma_grid, dtype=bool)

    if np.isfinite(diag_grid["a"]):
        xx = np.linspace(np.nanmin(x_grid), np.nanmax(x_grid), 300)
        yy = 10.0 ** (diag_grid["a"] * xx + diag_grid["b"])
        ax.plot(xx, yy, color="tab:blue", lw=1.2, ls=":", alpha=0.9, label="Grid log-fit (all)")
    if np.isfinite(diag_grid_overlap["a"]):
        xx = np.linspace(np.nanmin(x_grid[mov]), np.nanmax(x_grid[mov]), 300)
        yy = 10.0 ** (diag_grid_overlap["a"] * xx + diag_grid_overlap["b"])
        ax.plot(xx, yy, color="tab:blue", lw=2.0, ls="-", alpha=0.95, label="Grid log-fit (dual-sigma overlap)")
    if np.isfinite(diag_dual["a"]):
        xx = np.linspace(np.nanmin(x_dual), np.nanmax(x_dual), 300)
        yy = 10.0 ** (diag_dual["a"] * xx + diag_dual["b"])
        ax.plot(xx, yy, color="black", lw=1.4, ls="--", alpha=0.9, label="Dual log-fit")

    diag_lines = [
        r"Diagnostics: $\log_{10}c_{200}=a\,\log_{10}M_{200}+b$",
        (
            f"Grid: N={diag_grid['n']}, a={diag_grid['a']:.4f}, "
            f"b={diag_grid['b']:.4f}, rms={diag_grid['scatter_dex']:.4f} dex"
            if np.isfinite(diag_grid["a"])
            else f"Grid: N={diag_grid['n']} (insufficient points)"
        ),
        (
            f"Grid overlap: N={diag_grid_overlap['n']}, a={diag_grid_overlap['a']:.4f}, "
            f"b={diag_grid_overlap['b']:.4f}, rms={diag_grid_overlap['scatter_dex']:.4f} dex"
            if np.isfinite(diag_grid_overlap["a"])
            else "Grid overlap: unavailable"
        ),
        (
            f"Dual: N={diag_dual['n']}, a={diag_dual['a']:.4f}, "
            f"b={diag_dual['b']:.4f}, rms={diag_dual['scatter_dex']:.4f} dex"
            if np.isfinite(diag_dual["a"])
            else f"Dual: N={diag_dual['n']} (insufficient points)"
        ),
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(diag_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="0.5"),
    )

    ax.set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$")
    ax.set_ylabel(r"$c_{200}$")
    ax.set_title("Concentration-mass relation from PIEMD -> BMO+free Jaffe fits")
    ax.legend(fontsize=9)
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4x4 mosaic: PIEMD target profile and best-fit decomposition.
    out_mosaic = Path(args.out_mosaic).resolve()
    figm, axm = plt.subplots(4, 4, figsize=(16, 16), sharex=True, sharey=True, constrained_layout=True)
    axs = axm.ravel()
    for axi in axs:
        axi.set_visible(False)

    if len(dplot) > 0:
        dsel = dplot.sort_values("sigma0_km_s").reset_index(drop=True)
        nsel = min(16, len(dsel))
        if len(dsel) <= 16:
            idx_sel = np.arange(len(dsel))
        else:
            idx_sel = np.unique(np.round(np.linspace(0, len(dsel) - 1, 16)).astype(int))

        for ip, irow in enumerate(idx_sel[:16]):
            row = dsel.iloc[int(irow)]
            axi = axs[ip]
            axi.set_visible(True)

            sigma0 = float(row["sigma0_km_s"])
            r_cut = float(row["r_cut_kpc"])
            r_core = float(row["r_core_kpc"])
            rs = float(row["rs_bmo_kpc"])
            rt = float(row["rt_bmo_kpc"])
            rho0 = float(row["rho0_bmo_msun_kpc3"])
            mstar = float(row["Mstar_free_Msun"])
            rj = float(row["rJ_free_kpc"])
            c200 = float(row["c200"])
            m200 = float(row["M200_Msun"])
            rmse = float(row["fit_rmse_log"])

            rho_targ = piemd_rho(r_eval, r_core, r_cut, sigma0)
            rho_dm = bmo_rho(r_eval, rs, rt, rho0)
            rho_star = jaffe_rho_rj(r_eval, mstar, rj)
            rho_tot = rho_dm + rho_star

            axi.loglog(r_eval, rho_targ, color="black", lw=1.8, label="PIEMD")
            axi.loglog(r_eval, rho_tot, color="crimson", lw=1.5, label="BMO+Jaffe")
            axi.loglog(r_eval, rho_dm, color="tab:orange", lw=1.0, ls="--", alpha=0.9, label="BMO")
            axi.loglog(r_eval, rho_star, color="tab:blue", lw=1.0, ls=":", alpha=0.9, label="Jaffe")

            title = (
                rf"$\sigma_0={sigma0:.0f}$, $r_{{cut}}={r_cut:.2f}$ kpc" + "\n"
                + rf"$c_{{200}}={c200:.1f}$, $\log_{{10}}M_{{200}}={np.log10(m200):.2f}$, RMSE={rmse:.3f}"
            )
            axi.set_title(title, fontsize=9)
            axi.grid(alpha=0.15, which="both")

        # Legend on first visible panel.
        for axi in axs:
            if axi.get_visible():
                axi.legend(fontsize=7, loc="lower left")
                break

    for i in range(12, 16):
        axs[i].set_xlabel("r [kpc]")
    for i in [0, 4, 8, 12]:
        axs[i].set_ylabel(r"$\rho(r)$ [M$_\odot$/kpc$^3$]")
    figm.savefig(out_mosaic, dpi=180, bbox_inches="tight")
    plt.close(figm)

    n_ok = int(ok.sum())
    n_inner_dom = int(df["stellar_dominates_inner"].sum()) if "stellar_dominates_inner" in df.columns else 0
    print(f"Mock galaxies processed: {len(df)}")
    print(f"Successful fits with finite c200/M200: {n_ok}")
    if args.enforce_inner_stellar_dominance:
        print(f"Fits satisfying enforced inner stellar dominance: {n_inner_dom}/{len(df)}")
    else:
        print(f"Inner-dominance flag count (penalty disabled): {n_inner_dom}/{len(df)}")
    print(f"Fit upper radius: {args.rmax_fit_kpc if args.rmax_fit_kpc is not None else 'None'} kpc")
    print(f"Mstar guess mode: {args.mstar_guess_mode}")
    print(f"Inner stellar-dominance penalty enabled: {bool(args.enforce_inner_stellar_dominance)}")
    print("Re handling: ignored in optimizer regularization (free rJ fit).")
    print(
        f"Derived r_cut normalization: rcut_ref={rcut_ref_kpc:.6f} kpc at "
        f"sigma0_ref={SIGMA0_REF_KMS:.1f} km/s "
        f"(slope={rcut_slope:.6f}, source_ref={rcut_ref_source}, source_slope={rcut_slope_source})"
    )
    if rcut_ref_source == "input" and rcut_slope_source == "input":
        print("r_cut(sigma0) relation is set by CLI inputs; par galaxies are plotted only as reference.")
    if args.mstar_guess_mode != "fixed" and not dual_has_tpl:
        print("Warning: dual-template modes requested but dual CSV templates unavailable; used fixed fallbacks.")
    print(f"r_core assumption: {args.r_core_kpc:.6g} kpc (Bergamini-like if 0)")
    print(f"Saved table: {out_csv}")
    print(f"Saved c-M figure: {out_fig}")
    if np.isfinite(diag_grid["a"]):
        print(
            "Grid c-M log-fit: "
            f"N={diag_grid['n']}, a={diag_grid['a']:.6f}, b={diag_grid['b']:.6f}, "
            f"rms={diag_grid['scatter_dex']:.6f} dex"
        )
    if np.isfinite(diag_grid_overlap["a"]):
        print(
            "Grid-overlap c-M log-fit: "
            f"N={diag_grid_overlap['n']}, a={diag_grid_overlap['a']:.6f}, b={diag_grid_overlap['b']:.6f}, "
            f"rms={diag_grid_overlap['scatter_dex']:.6f} dex"
        )
    if np.isfinite(diag_dual["a"]):
        print(
            "Dual c-M log-fit: "
            f"N={diag_dual['n']}, a={diag_dual['a']:.6f}, b={diag_dual['b']:.6f}, "
            f"rms={diag_dual['scatter_dex']:.6f} dex"
        )
    print(f"Saved sigma-r_cut figure: {out_rcut}")
    print(f"Saved 4x4 profile mosaic: {out_mosaic}")


if __name__ == "__main__":
    main()
