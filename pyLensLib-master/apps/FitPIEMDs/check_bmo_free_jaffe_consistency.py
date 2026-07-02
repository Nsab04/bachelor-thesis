#!/usr/bin/env python3
"""
Consistency check for BMO+free-Jaffe fits.

For the first N galaxies in lens_galaxy_fits_bmo_jaffe_dual.csv:
1) Rebuild PIEMD density profiles from M0416.par
2) Refit with BMO+free Jaffe using fitGalaxies_tnfw_jaffe_dual.py functions
3) Compare best-fit parameters and derived c200/M200 against CSV values
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

# Ensure local project package is importable when running script directly.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyLensLib.lenstool import findInBlock, readLenstoolBlock, selectPotentielByType


def _load_fit_module(path: Path):
    spec = importlib.util.spec_from_file_location("fitmod", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _rel_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-30)


def _build_jaffe_shape(fitmod, r: np.ndarray, re_kpc: float) -> np.ndarray:
    """Return unit-mass Jaffe shape for module-specific API."""
    if hasattr(fitmod, "jaffe_rho"):
        return fitmod.jaffe_rho(r, 1.0, re_kpc)
    if hasattr(fitmod, "jaffe_rho_rj"):
        return fitmod.jaffe_rho_rj(r, 1.0, re_kpc)
    raise AttributeError("Fit module lacks jaffe_rho/jaffe_rho_rj.")


def _call_fit_bmo_free(
    fitmod,
    r: np.ndarray,
    rho_piemd: np.ndarray,
    rho_j_shape: np.ndarray,
    rs_guess: float,
    rt_guess: float,
    rho0_guess: float,
    mstar_phot: float,
    re_kpc: float,
    rmax_fit: float | None,
    inner_stellar_radius_factor: float,
    inner_stellar_penalty: float,
):
    """
    Call fit_bmo_plus_free_jaffe with either fitGalaxies_tnfw_jaffe_dual.py API
    or fit_bergamini_bmo_free_jaffe_grid.py API.
    """
    f = fitmod.fit_bmo_plus_free_jaffe
    p = inspect.signature(f).parameters

    if "mstar_guess" in p:
        # fitGalaxies_tnfw_jaffe_dual.py API
        return f(
            r,
            rho_piemd,
            rho_j_shape,
            rs_guess,
            rt_guess,
            rho0_guess,
            mstar_guess=mstar_phot,
            rmin_fit=0.05,
            rmax_fit=rmax_fit,
            re_kpc=re_kpc,
        )

    if "mstar_guess_msun" in p:
        # fit_bergamini_bmo_free_jaffe_grid.py API
        return f(
            r_kpc=r,
            rho_target_msun_kpc3=rho_piemd,
            rs_guess_kpc=rs_guess,
            rt_guess_kpc=rt_guess,
            rho0_guess_msun_kpc3=rho0_guess,
            mstar_guess_msun=mstar_phot,
            re_kpc=re_kpc,
            rmin_fit_kpc=0.05,
            rmax_fit_kpc=rmax_fit,
            inner_radius_fraction_of_re=inner_stellar_radius_factor,
            inner_penalty_weight=inner_stellar_penalty,
        )

    raise TypeError("Unsupported fit_bmo_plus_free_jaffe signature.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce and compare BMO+free-Jaffe fits for first N galaxies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default="/Users/maxmen3/projects/pyLensLib/Test/lens_galaxy_fits_bmo_jaffe_dual.csv",
        help="Input CSV from fitGalaxies_tnfw_jaffe_dual.py --use-bmo run.",
    )
    parser.add_argument(
        "--parfile",
        default="/Users/maxmen3/projects/pyLensLib/Test/M0416.par",
        help="LensTool .par file used to rebuild PIEMD profiles.",
    )
    parser.add_argument(
        "--fit-script",
        default="/Users/maxmen3/projects/pyLensLib/Test/fitGalaxies_tnfw_jaffe_dual.py",
        help="Path to the fit script providing BMO+free-Jaffe functions.",
    )
    parser.add_argument(
        "--fit-window",
        choices=["csv", "2rcut", "none"],
        default="csv",
        help="Upper fit radius: from CSV, enforced 2*r_cut, or unlimited.",
    )
    parser.add_argument(
        "--inner-stellar-radius-factor",
        type=float,
        default=0.3,
        help="Used only for fit_bergamini API: enforce Jaffe>=BMO for r<=factor*Re.",
    )
    parser.add_argument(
        "--inner-stellar-penalty",
        type=float,
        default=120.0,
        help="Used only for fit_bergamini API: inner-dominance penalty strength.",
    )
    parser.add_argument("--n", type=int, default=5, help="Number of CSV rows to test.")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for consistency flag on each quantity.",
    )
    parser.add_argument(
        "--out-csv",
        default="/Users/maxmen3/projects/pyLensLib/Test/consistency_bmo_free_jaffe_first5.csv",
        help="Output table with fitted vs reference parameters and relative differences.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    par_path = Path(args.parfile)
    fit_script_path = Path(args.fit_script)
    out_path = Path(args.out_csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not par_path.exists():
        raise FileNotFoundError(f".par file not found: {par_path}")
    if not fit_script_path.exists():
        raise FileNotFoundError(f"Fit script not found: {fit_script_path}")

    fitmod = _load_fit_module(fit_script_path)
    df = pd.read_csv(csv_path).head(int(args.n)).copy()

    pot_all = readLenstoolBlock(str(par_path), "potentiel")
    pot_gal = selectPotentielByType(pot_all, ptype="gal")
    r = np.logspace(-2, np.log10(1000.0), 1000)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)

    rows = []
    for _, row in df.iterrows():
        lens_index = int(row["lens_index"])
        pot = pot_gal[lens_index]

        sigma0 = float(findInBlock(pot, "v_disp")) * np.sqrt(3.0 / 2.0)
        r_core = float(findInBlock(pot, "core_radius_kpc"))
        r_cut = float(findInBlock(pot, "cut_radius_kpc"))
        z_lens = float(row["z_lens"])
        re_kpc = float(row["ReF160W_kpc"])
        mstar_phot = float(row["Mstar_from_F160W_Msun"])
        if args.fit_window == "2rcut":
            rmax_fit = 2.0 * r_cut
        elif args.fit_window == "none":
            rmax_fit = None
        else:
            rmax_fit = float(row["rmax_fit_kpc"]) if np.isfinite(row["rmax_fit_kpc"]) else None

        rho_piemd = fitmod.piemd_rho(r, r_core, r_cut, sigma0)
        rho_j_shape = _build_jaffe_shape(fitmod, r, re_kpc)
        rs_guess = max(1.0, 0.2 * r_cut)
        rt_guess = max(r_cut, 2.0 * rs_guess)
        rho0_guess = max(float(np.nanmax(rho_piemd) * 1.0e-3), 1.0e-20)

        rs, rt, rho0, mstar_fit, rj_fit, diag = _call_fit_bmo_free(
            fitmod=fitmod,
            r=r,
            rho_piemd=rho_piemd,
            rho_j_shape=rho_j_shape,
            rs_guess=rs_guess,
            rt_guess=rt_guess,
            rho0_guess=rho0_guess,
            mstar_phot=mstar_phot,
            re_kpc=re_kpc,
            rmax_fit=rmax_fit,
            inner_stellar_radius_factor=float(args.inner_stellar_radius_factor),
            inner_stellar_penalty=float(args.inner_stellar_penalty),
        )

        rho_crit = cosmo.critical_density(z_lens).to_value("Msun/kpc3")
        derived = fitmod.compute_bmo_derived(rs, rt, rho0, rho_crit)

        rows.append(
            {
                "lens_index": lens_index,
                "fit_success_recomputed": bool(diag.get("success", True)),
                "rs_fit_kpc": rs,
                "rs_csv_kpc": float(row["rs_bmo_free_kpc"]),
                "rt_fit_kpc": rt,
                "rt_csv_kpc": float(row["rt_bmo_free_kpc"]),
                "rho0_fit_msun_kpc3": rho0,
                "rho0_csv_msun_kpc3": float(row["rho0_bmo_free_msun_kpc3"]),
                "mstar_fit_msun": mstar_fit,
                "mstar_csv_msun": float(row["Mstar_free_Msun"]),
                "rJ_fit_kpc": rj_fit,
                "rJ_csv_kpc": float(row["rJ_free_kpc"]),
                "c200_fit": float(derived["c200"]),
                "c200_csv": float(row["c200_free"]),
                "M200_fit_msun": float(derived["M200_Msun"]),
                "M200_csv_msun": float(row["M200_bmo_free_Msun"]),
            }
        )

    out = pd.DataFrame(rows)
    for key in ["rs", "rt", "rho0", "mstar", "rJ", "c200", "M200"]:
        out[f"{key}_rel_diff"] = _rel_diff(out[f"{key}_fit" + ("_kpc" if key in ("rs", "rt", "rJ") else "_msun_kpc3" if key == "rho0" else "_msun" if key in ("mstar", "M200") else "")].values,
                                           out[f"{key}_csv" + ("_kpc" if key in ("rs", "rt", "rJ") else "_msun_kpc3" if key == "rho0" else "_msun" if key in ("mstar", "M200") else "")].values)

    rel_cols = [
        "rs_rel_diff",
        "rt_rel_diff",
        "rho0_rel_diff",
        "mstar_rel_diff",
        "rJ_rel_diff",
        "c200_rel_diff",
        "M200_rel_diff",
    ]
    out["all_consistent"] = (out[rel_cols].max(axis=1) <= float(args.rtol))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Checked rows: {len(out)}")
    print(f"Tolerance: rtol = {args.rtol:g}")
    for c in rel_cols:
        print(f"{c:>16s}  max={out[c].max():.3e}  median={out[c].median():.3e}")
    n_ok = int(out["all_consistent"].sum())
    print(f"Rows consistent within tolerance: {n_ok}/{len(out)}")
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
