#!/usr/bin/env python
"""
Sanity checks on the outputs of fitGalaxies_tnfw_jaffe_dual.py.

Validates:
  1. CSV structure and column completeness (auto-detects DM profile type)
  2. Physical plausibility of fit parameters
  3. Constraint enforcement (Jaffe <= PIEMD, DM > Jaffe at r > Re)
  4. Consistency between fixed-Jaffe and free-Jaffe fits
  5. Stellar mass mismatch columns
  6. HDF5 structure, shapes, and NaN fractions (if present)
  7. MCMC percentile consistency (if present)
  8. Density profile reconstruction: DM+Jaffe vs PIEMD residuals

Supports tNFW, BMO (n=0.5), NFW, Einasto, and gNFW DM profile types (auto-detected from the
``dm_profile`` column in the CSV).

Usage:
    python validate_dual_fit_outputs.py --csv results.csv [--h5 results.h5] [--plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Density profile functions (copied from fitGalaxies_tnfw_jaffe_dual.py so
# that this script is self-contained).
# ---------------------------------------------------------------------------
G_KPC = 4.302e-6
JAFFE_Re_OVER_rJ = 1.0


def jaffe_rho(r, mstar, re):
    rj = float(re) / JAFFE_Re_OVER_rJ
    rr = np.maximum(r, np.finfo(float).eps)
    return (float(mstar) * rj) / (4.0 * np.pi * rr**2 * (rr + rj) ** 2)


def jaffe_rho_rj(r, mstar, rj):
    rr = np.maximum(r, np.finfo(float).eps)
    rj = np.maximum(float(rj), np.finfo(float).eps)
    return (float(mstar) * rj) / (4.0 * np.pi * rr**2 * (rr + rj) ** 2)


def get_free_rj_kpc(row):
    """Return free-fit Jaffe scale radius, with Re fallback for legacy outputs."""
    rj = row.get("rJ_free_kpc", np.nan)
    if np.isfinite(rj) and rj > 0:
        return float(rj)
    re = row.get("ReF160W_kpc", np.nan)
    if np.isfinite(re) and re > 0:
        return float(re) / JAFFE_Re_OVER_rJ
    return np.nan


def piemd_rho(r, r_core, r_cut, sigma_v):
    return (
        (sigma_v**2)
        / (2.0 * np.pi * G_KPC)
        * (r_cut + r_core)
        / (r_core**2 * r_cut)
        / (1.0 + r**2 / r_core**2)
        / (1.0 + r**2 / r_cut**2)
    )


def trunc_nfw_rho(r, rs, rt, rho0):
    x = np.maximum(r / rs, np.finfo(float).eps)
    tau = rt / rs
    trunc = tau**2 / (x**2 + tau**2)
    return trunc * (rho0 / (x * (1.0 + x) ** 2))


def einasto_rho(r, rs, rho_s, n):
    r = np.asarray(r, dtype=float)
    s = np.maximum(r / rs, np.finfo(float).eps)
    alpha = 1.0 / n
    dn = 2.0 * n
    return rho_s * np.exp(-dn * (s**alpha - 1.0))


def gnfw_rho(r, rs, rho_s, gamma):
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    return rho_s / (x**gamma * (1.0 + x) ** (3.0 - gamma))


def nfw_rho(r, rs, rho0):
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    return rho0 / (x * (1.0 + x) ** 2)


def bmo_rho(r, rs, rt, rho0):
    """BMO smoothly truncated NFW density profile (n=0.5)."""
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    trunc = np.sqrt(rt**2 / (r**2 + rt**2))
    return trunc * (rho0 / (x * (1.0 + x) ** 2))


def sersic_surface_brightness(r, re, n=4.0):
    """Normalized projected Sersic profile I(R), with I(Re)=1."""
    r = np.asarray(r, dtype=float)
    rr = np.maximum(r, np.finfo(float).eps)
    n = float(n)
    if not np.isfinite(n) or n <= 0:
        n = 4.0
    re = max(float(re), np.finfo(float).eps)
    # Ciotti & Bertin approximation for b_n
    b_n = 2.0 * n - 1.0 / 3.0 + 0.009876 / n
    return np.exp(-b_n * ((rr / re) ** (1.0 / n) - 1.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_n_pass = 0
_n_fail = 0
_n_warn = 0


def _pass(msg):
    global _n_pass
    _n_pass += 1
    print(f"  ✓ {msg}")


def _fail(msg):
    global _n_fail
    _n_fail += 1
    print(f"  ✗ FAIL: {msg}")


def _warn(msg):
    global _n_warn
    _n_warn += 1
    print(f"  ⚠ WARN: {msg}")


def _section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. CSV structure — profile-aware column definitions
# ---------------------------------------------------------------------------
_COMMON_COLUMNS = [
    "lens_index", "cat_index", "ra_lens", "dec_lens", "z_lens",
    "F160W", "ReF160W_kpc", "rmax_fit_kpc", "v_disp_sigma0_km_s",
    "core_radius_kpc", "cut_radius_kpc",
    "Mstar_from_F160W_Msun", "negative_center_fixedJaffe",
    "dm_profile",
    "fit_fix_rmse_log", "fit_fix_r2_log", "fit_fix_npts",
    "fit_fix_success", "fit_fix_message", "fit_fix_jaffe_clamped",
    "Mstar_free_Msun",
    "fit_free_rmse_log", "fit_free_r2_log", "fit_free_npts",
    "fit_free_success", "fit_free_message",
    "Mstar_max_constraint_Msun",
    "Mstar_ratio_free_over_phot",
    "Mstar_log_ratio_free_over_phot",
    "Mstar_ratio_phot_over_max",
]

_TNFW_COLUMNS = [
    "rs_fix_kpc", "rt_fix_kpc", "rho0_fix_msun_kpc3",
    "c200_fix", "R200_fix_kpc", "M200_fix_Msun", "Mtot_tNFW_fix_Msun",
    "rs_free_kpc", "rt_free_kpc", "rho0_free_msun_kpc3",
    "c200_free", "R200_free_kpc", "M200_free_Msun", "Mtot_tNFW_free_Msun",
]

_EINASTO_COLUMNS = [
    "rs_ein_fix_kpc", "rhos_ein_fix_msun_kpc3", "n_ein_fix",
    "R200_ein_fix_kpc", "M200_ein_fix_Msun",
    "rs_ein_free_kpc", "rhos_ein_free_msun_kpc3", "n_ein_free",
    "R200_ein_free_kpc", "M200_ein_free_Msun",
]

_GNFW_COLUMNS = [
    "rs_gnfw_fix_kpc", "rhos_gnfw_fix_msun_kpc3", "gamma_gnfw_fix",
    "R200_gnfw_fix_kpc", "M200_gnfw_fix_Msun",
    "rs_gnfw_free_kpc", "rhos_gnfw_free_msun_kpc3", "gamma_gnfw_free",
    "R200_gnfw_free_kpc", "M200_gnfw_free_Msun",
]

_NFW_COLUMNS = [
    "rs_nfw_fix_kpc", "rho0_nfw_fix_msun_kpc3",
    "c200_fix", "R200_nfw_fix_kpc", "M200_nfw_fix_Msun",
    "rs_nfw_free_kpc", "rho0_nfw_free_msun_kpc3",
    "c200_free", "R200_nfw_free_kpc", "M200_nfw_free_Msun",
]

_BMO_COLUMNS = [
    "rs_bmo_fix_kpc", "rt_bmo_fix_kpc", "rho0_bmo_fix_msun_kpc3",
    "c200_fix", "R200_bmo_fix_kpc", "M200_bmo_fix_Msun",
    "rs_bmo_free_kpc", "rt_bmo_free_kpc", "rho0_bmo_free_msun_kpc3",
    "c200_free", "R200_bmo_free_kpc", "M200_bmo_free_Msun",
]

_MCMC_COMMON_COLUMNS = [
    "mcmc_nsamp_total", "mcmc_nsamp_ok_fix", "mcmc_nsamp_ok_free",
    "Mstar_free_p16", "Mstar_free_p50", "Mstar_free_p84",
]

_MCMC_TNFW_COLUMNS = [
    "rs_fix_p16", "rs_fix_p50", "rs_fix_p84",
    "rt_fix_p16", "rt_fix_p50", "rt_fix_p84",
    "rho0_fix_p16", "rho0_fix_p50", "rho0_fix_p84",
    "rs_free_p16", "rs_free_p50", "rs_free_p84",
    "rt_free_p16", "rt_free_p50", "rt_free_p84",
    "rho0_free_p16", "rho0_free_p50", "rho0_free_p84",
    "c200_fix_p16", "c200_fix_p50", "c200_fix_p84",
    "M200_fix_p16", "M200_fix_p50", "M200_fix_p84",
    "c200_free_p16", "c200_free_p50", "c200_free_p84",
    "M200_free_p16", "M200_free_p50", "M200_free_p84",
]

_MCMC_EINASTO_COLUMNS = [
    "rs_ein_fix_p16", "rs_ein_fix_p50", "rs_ein_fix_p84",
    "rhos_ein_fix_p16", "rhos_ein_fix_p50", "rhos_ein_fix_p84",
    "n_ein_fix_p16", "n_ein_fix_p50", "n_ein_fix_p84",
    "rs_ein_free_p16", "rs_ein_free_p50", "rs_ein_free_p84",
    "rhos_ein_free_p16", "rhos_ein_free_p50", "rhos_ein_free_p84",
    "n_ein_free_p16", "n_ein_free_p50", "n_ein_free_p84",
]

_MCMC_GNFW_COLUMNS = [
    "rs_gnfw_fix_p16", "rs_gnfw_fix_p50", "rs_gnfw_fix_p84",
    "rhos_gnfw_fix_p16", "rhos_gnfw_fix_p50", "rhos_gnfw_fix_p84",
    "gamma_gnfw_fix_p16", "gamma_gnfw_fix_p50", "gamma_gnfw_fix_p84",
    "rs_gnfw_free_p16", "rs_gnfw_free_p50", "rs_gnfw_free_p84",
    "rhos_gnfw_free_p16", "rhos_gnfw_free_p50", "rhos_gnfw_free_p84",
    "gamma_gnfw_free_p16", "gamma_gnfw_free_p50", "gamma_gnfw_free_p84",
]

_MCMC_NFW_COLUMNS = [
    "rs_nfw_fix_p16", "rs_nfw_fix_p50", "rs_nfw_fix_p84",
    "rho0_nfw_fix_p16", "rho0_nfw_fix_p50", "rho0_nfw_fix_p84",
    "rs_nfw_free_p16", "rs_nfw_free_p50", "rs_nfw_free_p84",
    "rho0_nfw_free_p16", "rho0_nfw_free_p50", "rho0_nfw_free_p84",
]

_MCMC_BMO_COLUMNS = [
    "rs_bmo_fix_p16", "rs_bmo_fix_p50", "rs_bmo_fix_p84",
    "rt_bmo_fix_p16", "rt_bmo_fix_p50", "rt_bmo_fix_p84",
    "rho0_bmo_fix_p16", "rho0_bmo_fix_p50", "rho0_bmo_fix_p84",
    "rs_bmo_free_p16", "rs_bmo_free_p50", "rs_bmo_free_p84",
    "rt_bmo_free_p16", "rt_bmo_free_p50", "rt_bmo_free_p84",
    "rho0_bmo_free_p16", "rho0_bmo_free_p50", "rho0_bmo_free_p84",
    "c200_fix_p16", "c200_fix_p50", "c200_fix_p84",
    "M200_fix_p16", "M200_fix_p50", "M200_fix_p84",
    "c200_free_p16", "c200_free_p50", "c200_free_p84",
    "M200_free_p16", "M200_free_p50", "M200_free_p84",
]


def detect_dm_profile(df):
    """Auto-detect the DM profile type from the CSV."""
    if "dm_profile" in df.columns:
        vals = df["dm_profile"].dropna().unique()
        if len(vals) == 1:
            return str(vals[0]).strip().lower()
    # Fallback: detect from column names
    if "rs_gnfw_fix_kpc" in df.columns:
        return "gnfw"
    if "rs_ein_fix_kpc" in df.columns:
        return "einasto"
    if "rs_nfw_fix_kpc" in df.columns:
        return "nfw"
    if "rs_bmo_fix_kpc" in df.columns:
        return "bmo"
    return "tnfw"


def get_required_columns(profile):
    if profile == "einasto":
        return _COMMON_COLUMNS + _EINASTO_COLUMNS
    elif profile == "gnfw":
        return _COMMON_COLUMNS + _GNFW_COLUMNS
    elif profile == "nfw":
        return _COMMON_COLUMNS + _NFW_COLUMNS
    elif profile == "bmo":
        return _COMMON_COLUMNS + _BMO_COLUMNS
    else:
        return _COMMON_COLUMNS + _TNFW_COLUMNS


def get_mcmc_columns(profile):
    if profile == "einasto":
        return _MCMC_COMMON_COLUMNS + _MCMC_EINASTO_COLUMNS
    elif profile == "gnfw":
        return _MCMC_COMMON_COLUMNS + _MCMC_GNFW_COLUMNS
    elif profile == "nfw":
        return _MCMC_COMMON_COLUMNS + _MCMC_NFW_COLUMNS
    elif profile == "bmo":
        return _MCMC_COMMON_COLUMNS + _MCMC_BMO_COLUMNS
    else:
        return _MCMC_COMMON_COLUMNS + _MCMC_TNFW_COLUMNS


def get_rs_col(profile, mode):
    """Return column name for scale radius given profile and fit mode."""
    if profile == "einasto":
        return f"rs_ein_{mode}_kpc"
    elif profile == "gnfw":
        return f"rs_gnfw_{mode}_kpc"
    elif profile == "nfw":
        return f"rs_nfw_{mode}_kpc"
    elif profile == "bmo":
        return f"rs_bmo_{mode}_kpc"
    else:
        return f"rs_{mode}_kpc"


def check_csv_structure(df, profile):
    _section("1. CSV structure & column completeness")
    _pass(f"Detected DM profile type: {profile}")
    required = get_required_columns(profile)
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail(f"Missing required columns: {missing}")
    else:
        _pass(f"All {len(required)} required columns present")

    # Optional newer free-Jaffe radius columns (present in outputs with free rJ fitting)
    if "rJ_free_kpc" in df.columns:
        _pass("Detected rJ_free_kpc column (free-Jaffe radius is persisted)")
    else:
        _warn("rJ_free_kpc column missing (legacy output format without persisted free Jaffe radius)")

    if len(df) == 0:
        _fail("CSV has 0 rows")
        return False

    _pass(f"CSV has {len(df)} galaxies")

    # Check for duplicate lens indices
    dups = df["lens_index"].duplicated().sum()
    if dups > 0:
        _fail(f"{dups} duplicate lens_index values")
    else:
        _pass("No duplicate lens_index values")

    # Check MCMC columns
    has_mcmc = "mcmc_nsamp_total" in df.columns
    if has_mcmc:
        mcmc_cols = get_mcmc_columns(profile)
        mcmc_missing = [c for c in mcmc_cols if c not in df.columns]
        if mcmc_missing:
            _fail(f"Missing MCMC columns: {mcmc_missing}")
        else:
            _pass(f"All {len(mcmc_cols)} MCMC columns present")
        if "rJ_free_p50" in df.columns:
            _pass("Detected rJ_free percentile columns in MCMC summary")
        else:
            _warn("rJ_free percentile columns missing in MCMC summary (legacy output format)")
    else:
        print("  (no MCMC columns found — skipping MCMC checks in CSV)")

    return True


# ---------------------------------------------------------------------------
# 2. Physical plausibility
# ---------------------------------------------------------------------------
def check_physical_plausibility(df, profile):
    _section("2. Physical plausibility of fit parameters")
    n = len(df)

    # --- Input parameters ---
    bad_z = ((df["z_lens"] <= 0) | (df["z_lens"] > 3)).sum()
    if bad_z:
        _fail(f"{bad_z}/{n} galaxies with implausible z_lens (<=0 or >3)")
    else:
        _pass("All z_lens in (0, 3]")

    bad_sigma = ((df["v_disp_sigma0_km_s"] <= 0) | (df["v_disp_sigma0_km_s"] > 1000)).sum()
    if bad_sigma:
        _fail(f"{bad_sigma}/{n} galaxies with implausible sigma0 (<=0 or >1000 km/s)")
    else:
        _pass("All sigma0 in (0, 1000] km/s")

    bad_re = ((df["ReF160W_kpc"] <= 0) | (df["ReF160W_kpc"] > 100)).sum()
    if bad_re:
        _fail(f"{bad_re}/{n} galaxies with implausible Re (<=0 or >100 kpc)")
    else:
        _pass("All Re in (0, 100] kpc")

    # --- Fixed fit parameters ---
    ok_fix = df["fit_fix_success"].astype(bool)
    n_ok_fix = ok_fix.sum()
    _pass(f"Fixed fit converged for {n_ok_fix}/{n} galaxies ({100*n_ok_fix/n:.0f}%)")
    if n_ok_fix < 0.5 * n:
        _warn(f"Less than 50% of fixed fits converged")

    fix_valid = df.loc[ok_fix]
    rs_col_fix = get_rs_col(profile, "fix")

    bad_rs = ((fix_valid[rs_col_fix] <= 0) | (fix_valid[rs_col_fix] > 1e4)).sum()
    if bad_rs:
        _warn(f"{bad_rs} converged fixed fits with implausible rs (<=0 or >10^4 kpc)")
    else:
        _pass("All converged fixed-fit rs in plausible range")

    if profile == "tnfw":
        bad_rt = ((fix_valid["rt_fix_kpc"] <= 0) | (fix_valid["rt_fix_kpc"] > 1e5)).sum()
        bad_rho = ((fix_valid["rho0_fix_msun_kpc3"] <= 0)).sum()
        if bad_rt:
            _warn(f"{bad_rt} converged fixed fits with implausible rt")
        else:
            _pass("All converged fixed-fit rt in plausible range")
        if bad_rho:
            _warn(f"{bad_rho} converged fixed fits with rho0 <= 0")
        else:
            _pass("All converged fixed-fit rho0 > 0")
        # rt > rs (truncation radius should exceed scale radius)
        bad_rt_rs = (fix_valid["rt_fix_kpc"] <= fix_valid["rs_fix_kpc"]).sum()
        if bad_rt_rs:
            _warn(f"{bad_rt_rs} fixed fits with rt <= rs (unusual but possible)")
        else:
            _pass("All fixed fits have rt > rs")
    elif profile == "einasto":
        rhos_col = "rhos_ein_fix_msun_kpc3"
        n_col = "n_ein_fix"
        bad_rhos = (fix_valid[rhos_col] <= 0).sum()
        if bad_rhos:
            _warn(f"{bad_rhos} converged fixed fits with rho_s <= 0")
        else:
            _pass("All converged fixed-fit rho_s > 0")
        bad_n = ((fix_valid[n_col] <= 0) | (fix_valid[n_col] > 20)).sum()
        if bad_n:
            _warn(f"{bad_n} converged fixed fits with implausible Einasto n (<=0 or >20)")
        else:
            _pass("All converged fixed-fit Einasto n in plausible range")
    elif profile == "gnfw":
        rhos_col = "rhos_gnfw_fix_msun_kpc3"
        gamma_col = "gamma_gnfw_fix"
        bad_rhos = (fix_valid[rhos_col] <= 0).sum()
        if bad_rhos:
            _warn(f"{bad_rhos} converged fixed fits with rho_s <= 0")
        else:
            _pass("All converged fixed-fit rho_s > 0")
        bad_gamma = ((fix_valid[gamma_col] <= 0) | (fix_valid[gamma_col] >= 3)).sum()
        if bad_gamma:
            _warn(f"{bad_gamma} converged fixed fits with implausible gamma (<=0 or >=3)")
        else:
            _pass("All converged fixed-fit gNFW gamma in (0, 3)")
    elif profile == "nfw":
        bad_rho = (fix_valid["rho0_nfw_fix_msun_kpc3"] <= 0).sum()
        if bad_rho:
            _warn(f"{bad_rho} converged fixed fits with rho0 <= 0")
        else:
            _pass("All converged fixed-fit rho0 > 0")
    elif profile == "bmo":
        bad_rt = ((fix_valid["rt_bmo_fix_kpc"] <= 0) | (fix_valid["rt_bmo_fix_kpc"] > 1e5)).sum()
        bad_rho = (fix_valid["rho0_bmo_fix_msun_kpc3"] <= 0).sum()
        if bad_rt:
            _warn(f"{bad_rt} converged fixed fits with implausible rt")
        else:
            _pass("All converged fixed-fit rt in plausible range")
        if bad_rho:
            _warn(f"{bad_rho} converged fixed fits with rho0 <= 0")
        else:
            _pass("All converged fixed-fit rho0 > 0")
        bad_rt_rs = (fix_valid["rt_bmo_fix_kpc"] <= fix_valid["rs_bmo_fix_kpc"]).sum()
        if bad_rt_rs:
            _warn(f"{bad_rt_rs} fixed fits with rt <= rs (unusual but possible)")
        else:
            _pass("All fixed fits have rt > rs")

    # --- Free fit parameters ---
    ok_free = df["fit_free_success"].astype(bool)
    n_ok_free = ok_free.sum()
    _pass(f"Free fit converged for {n_ok_free}/{n} galaxies ({100*n_ok_free/n:.0f}%)")
    if n_ok_free < 0.5 * n:
        _warn(f"Less than 50% of free fits converged")

    free_valid = df.loc[ok_free]
    bad_mstar = ((free_valid["Mstar_free_Msun"] <= 0)).sum()
    if bad_mstar:
        _warn(f"{bad_mstar} converged free fits with Mstar_free <= 0")
    else:
        _pass("All converged free-fit Mstar > 0")

    if "rJ_free_kpc" in free_valid.columns:
        bad_rj = ((free_valid["rJ_free_kpc"] <= 0) | (free_valid["rJ_free_kpc"] > 1e4)).sum()
        if bad_rj:
            _warn(f"{bad_rj} converged free fits with implausible rJ_free (<=0 or >10^4 kpc)")
        else:
            _pass("All converged free-fit rJ_free in plausible range")

    # --- Fit quality ---
    rmse_fix_med = fix_valid["fit_fix_rmse_log"].median()
    rmse_free_med = free_valid["fit_free_rmse_log"].median()
    _pass(f"Median RMSE(log): fixed={rmse_fix_med:.4f}, free={rmse_free_med:.4f}")

    bad_rmse_fix = (fix_valid["fit_fix_rmse_log"] > 1.0).sum()
    bad_rmse_free = (free_valid["fit_free_rmse_log"] > 1.0).sum()
    if bad_rmse_fix:
        _warn(f"{bad_rmse_fix} fixed fits with RMSE_log > 1.0 (poor fit)")
    if bad_rmse_free:
        _warn(f"{bad_rmse_free} free fits with RMSE_log > 1.0 (poor fit)")

    # --- Derived quantities (c200, M200) — common to all profiles ---
    for mode, ok_mask in [("fix", ok_fix), ("free", ok_free)]:
        sub = df.loc[ok_mask]
        c_col = f"c200_{mode}"
        if profile == "einasto":
            m_col = f"M200_ein_{mode}_Msun"
        elif profile == "gnfw":
            m_col = f"M200_gnfw_{mode}_Msun"
        elif profile == "nfw":
            m_col = f"M200_nfw_{mode}_Msun"
        elif profile == "bmo":
            m_col = f"M200_bmo_{mode}_Msun"
        else:
            m_col = f"M200_{mode}_Msun"

        if c_col in sub.columns:
            c_valid = sub[c_col][np.isfinite(sub[c_col])]
            if len(c_valid) > 0:
                bad_c = ((c_valid <= 0) | (c_valid > 500)).sum()
                if bad_c:
                    _warn(f"{bad_c} {mode} fits with implausible c200 (<=0 or >500)")
                else:
                    _pass(f"All {mode} c200 in plausible range (0, 500]")
                _pass(f"c200_{mode} distribution: median={np.median(c_valid):.1f}, "
                      f"16-84th=[{np.percentile(c_valid, 16):.1f}, {np.percentile(c_valid, 84):.1f}]")
            else:
                _warn(f"No finite c200_{mode} values")
        else:
            print(f"  (c200_{mode} column not found — skipping c200 checks)")

        if m_col in sub.columns:
            m_valid = sub[m_col][np.isfinite(sub[m_col])]
            if len(m_valid) > 0:
                bad_m = (m_valid <= 0).sum()
                if bad_m:
                    _warn(f"{bad_m} {mode} fits with M200 <= 0")
                else:
                    _pass(f"All {mode} M200 > 0")
                _pass(f"log10(M200_{mode}/Msun) distribution: "
                      f"median={np.median(np.log10(m_valid[m_valid > 0])):.2f}, "
                      f"16-84th=[{np.percentile(np.log10(m_valid[m_valid > 0]), 16):.2f}, "
                      f"{np.percentile(np.log10(m_valid[m_valid > 0]), 84):.2f}]")
        else:
            print(f"  (M200_{mode} column not found — skipping M200 checks)")

        # Mtot check only applicable to tNFW
        if profile == "tnfw":
            mt_col = f"Mtot_tNFW_{mode}_Msun"
            mt_valid = sub[mt_col][np.isfinite(sub[mt_col]) & np.isfinite(sub[m_col])]
            m200_for_mt = sub[m_col][np.isfinite(sub[mt_col]) & np.isfinite(sub[m_col])]
            if len(mt_valid) > 0:
                bad_mt = (mt_valid < m200_for_mt * 0.99).sum()
                if bad_mt:
                    _warn(f"{bad_mt} {mode} fits with Mtot < M200 (should be Mtot >= M200)")
                else:
                    _pass(f"All {mode} Mtot >= M200 (as expected)")


# ---------------------------------------------------------------------------
# 3. Constraint enforcement
# ---------------------------------------------------------------------------
def check_constraints(df, profile):
    _section("3. Constraint enforcement")

    ok = df["fit_fix_success"].astype(bool) & df["fit_free_success"].astype(bool)
    n_ok = ok.sum()
    if n_ok == 0:
        _warn("No galaxies with both fits converged — skipping constraint checks")
        return

    r = np.logspace(-2, 3, 500)

    n_jaffe_exceeds_piemd = 0
    n_dm_below_jaffe_outer = 0
    n_checked = 0
    dm_label = {"tnfw": "tNFW", "einasto": "Einasto", "gnfw": "gNFW", "nfw": "NFW", "bmo": "BMO"}.get(profile, profile)
    rs_free_col = get_rs_col(profile, "free")

    for _, row in df.loc[ok].iterrows():
        re = row["ReF160W_kpc"]
        rho_piemd = piemd_rho(r, row["core_radius_kpc"], row["cut_radius_kpc"],
                              row["v_disp_sigma0_km_s"])

        # Restrict to fitted radial range if rmax_fit_kpc is available
        rmax = row.get("rmax_fit_kpc", np.nan)
        if np.isfinite(rmax) and rmax > 0:
            r_mask = r <= rmax
        else:
            r_mask = np.ones_like(r, dtype=bool)

        # Fixed fit: Jaffe should not exceed PIEMD
        mstar_mag = row["Mstar_from_F160W_Msun"]
        rho_jaffe = jaffe_rho(r, mstar_mag, re)
        frac_exceed = np.sum((rho_jaffe > rho_piemd * 1.001) & r_mask) / r_mask.sum()
        if frac_exceed > 0.3:
            n_jaffe_exceeds_piemd += 1

        # Free fit: at r > Re, DM should exceed Jaffe
        rj_free = get_free_rj_kpc(row)
        if np.isfinite(row[rs_free_col]) and np.isfinite(row["Mstar_free_Msun"]) and np.isfinite(rj_free):
            if profile == "tnfw":
                rho_dm_free = trunc_nfw_rho(r, row["rs_free_kpc"], row["rt_free_kpc"],
                                            row["rho0_free_msun_kpc3"])
            elif profile == "nfw":
                rho_dm_free = nfw_rho(r, row["rs_nfw_free_kpc"],
                                      row["rho0_nfw_free_msun_kpc3"])
            elif profile == "einasto":
                rho_dm_free = einasto_rho(r, row["rs_ein_free_kpc"],
                                          row["rhos_ein_free_msun_kpc3"],
                                          row["n_ein_free"])
            elif profile == "bmo":
                rho_dm_free = bmo_rho(r, row["rs_bmo_free_kpc"],
                                      row["rt_bmo_free_kpc"],
                                      row["rho0_bmo_free_msun_kpc3"])
            else:  # gnfw
                rho_dm_free = gnfw_rho(r, row["rs_gnfw_free_kpc"],
                                       row["rhos_gnfw_free_msun_kpc3"],
                                       row["gamma_gnfw_free"])
            rho_jaffe_free = jaffe_rho_rj(r, row["Mstar_free_Msun"], rj_free)
            outer = (r > re) & r_mask
            if np.any(outer):
                violation = np.sum(rho_jaffe_free[outer] > rho_dm_free[outer] * 1.01)
                if violation > 0.1 * np.sum(outer):
                    n_dm_below_jaffe_outer += 1
        n_checked += 1

    if n_jaffe_exceeds_piemd:
        _warn(f"{n_jaffe_exceeds_piemd}/{n_checked} galaxies where unclamped Jaffe exceeds PIEMD "
              f"over >30% of radial range (clamping expected in fixed fit)")
    else:
        _pass("Unclamped Jaffe does not dominate PIEMD for any galaxy")

    clamped_count = df.loc[ok, "fit_fix_jaffe_clamped"].sum()
    _pass(f"Jaffe clamping applied in {clamped_count}/{n_checked} galaxies")

    if n_dm_below_jaffe_outer:
        _warn(f"{n_dm_below_jaffe_outer}/{n_checked} free fits where {dm_label} < Jaffe at r > Re "
              f"over >10% of outer points (penalty may not be strong enough)")
    else:
        _pass(f"{dm_label} dominates Jaffe at r > Re in all free fits (constraint satisfied)")


# ---------------------------------------------------------------------------
# 4. Consistency between fixed and free fits
# ---------------------------------------------------------------------------
def check_fix_vs_free_consistency(df, profile):
    _section("4. Fixed vs free fit consistency")

    ok = df["fit_fix_success"].astype(bool) & df["fit_free_success"].astype(bool)
    both = df.loc[ok].copy()
    n_both = len(both)
    if n_both == 0:
        _warn("No galaxies with both fits converged")
        return

    rs_fix_col = get_rs_col(profile, "fix")
    rs_free_col = get_rs_col(profile, "free")

    # rs should be broadly correlated
    rs_ratio = both[rs_free_col] / both[rs_fix_col]
    rs_ratio = rs_ratio[np.isfinite(rs_ratio) & (rs_ratio > 0)]
    if len(rs_ratio) > 0:
        med = np.median(rs_ratio)
        _pass(f"Median rs_free/rs_fix = {med:.2f} ({len(rs_ratio)} galaxies)")
        outliers = ((rs_ratio < 0.01) | (rs_ratio > 100)).sum()
        if outliers > 0:
            _warn(f"{outliers} galaxies with >100x discrepancy in rs between modes")
        else:
            _pass("No extreme rs discrepancies between fixed and free modes")

    # Free fit should generally have equal or lower RMSE (more degrees of freedom)
    better_free = (both["fit_free_rmse_log"] <= both["fit_fix_rmse_log"] * 1.05).sum()
    _pass(f"Free fit RMSE <= 1.05 × fixed RMSE in {better_free}/{n_both} galaxies "
          f"({100*better_free/n_both:.0f}%)")


# ---------------------------------------------------------------------------
# 5. Stellar mass mismatch columns
# ---------------------------------------------------------------------------
def check_mstar_mismatch(df):
    _section("5. Stellar mass mismatch columns")

    ok = df["fit_free_success"].astype(bool)
    sub = df.loc[ok].copy()
    n = len(sub)
    if n == 0:
        _warn("No converged free fits")
        return

    # Mstar_ratio_free_over_phot should be recomputable
    recomp = sub["Mstar_free_Msun"] / sub["Mstar_from_F160W_Msun"]
    stored = sub["Mstar_ratio_free_over_phot"]
    valid = np.isfinite(recomp) & np.isfinite(stored) & (stored > 0)
    if valid.sum() > 0:
        max_err = np.abs(recomp[valid] - stored[valid]).max()
        if max_err < 1e-6:
            _pass(f"Mstar_ratio_free_over_phot consistent (max err = {max_err:.2e})")
        else:
            _fail(f"Mstar_ratio_free_over_phot inconsistent (max err = {max_err:.2e})")

    # Mstar_log_ratio_free_over_phot
    log_recomp = np.log10(recomp)
    log_stored = sub["Mstar_log_ratio_free_over_phot"]
    valid2 = np.isfinite(log_recomp) & np.isfinite(log_stored)
    if valid2.sum() > 0:
        max_err2 = np.abs(log_recomp[valid2] - log_stored[valid2]).max()
        if max_err2 < 1e-6:
            _pass(f"Mstar_log_ratio_free_over_phot consistent (max err = {max_err2:.2e})")
        else:
            _fail(f"Mstar_log_ratio_free_over_phot inconsistent (max err = {max_err2:.2e})")

    # Mstar_ratio_phot_over_max
    mstar_max = sub["Mstar_max_constraint_Msun"]
    ratio_recomp = sub["Mstar_from_F160W_Msun"] / mstar_max
    ratio_stored = sub["Mstar_ratio_phot_over_max"]
    valid3 = np.isfinite(ratio_recomp) & np.isfinite(ratio_stored) & (ratio_stored > 0)
    if valid3.sum() > 0:
        max_err3 = np.abs(ratio_recomp[valid3] - ratio_stored[valid3]).max()
        if max_err3 < 1e-6:
            _pass(f"Mstar_ratio_phot_over_max consistent (max err = {max_err3:.2e})")
        else:
            _fail(f"Mstar_ratio_phot_over_max inconsistent (max err = {max_err3:.2e})")

    # Mstar_free <= Mstar_max_constraint
    exceed = (sub["Mstar_free_Msun"] > sub["Mstar_max_constraint_Msun"] * 1.01)
    exceed_valid = exceed[np.isfinite(sub["Mstar_max_constraint_Msun"])]
    if exceed_valid.sum() > 0:
        _warn(f"{exceed_valid.sum()}/{len(exceed_valid)} galaxies where Mstar_free > Mstar_max "
              f"(should be bounded by the optimizer)")
    else:
        _pass("Mstar_free <= Mstar_max constraint satisfied for all galaxies")

    # Distribution summary
    log_ratio = sub["Mstar_log_ratio_free_over_phot"]
    lr_valid = log_ratio[np.isfinite(log_ratio)]
    if len(lr_valid) > 0:
        _pass(f"log10(Mstar_free/Mstar_phot) distribution: "
              f"median={np.median(lr_valid):.3f} dex, "
              f"16-84th=[{np.percentile(lr_valid, 16):.3f}, {np.percentile(lr_valid, 84):.3f}] dex")


# ---------------------------------------------------------------------------
# 6. HDF5 structure
# ---------------------------------------------------------------------------
def check_h5(h5_path, df, profile):
    _section("6. HDF5 structure & data integrity")

    try:
        import h5py
    except ImportError:
        _warn("h5py not installed — skipping HDF5 checks")
        return

    if not Path(h5_path).exists():
        _fail(f"HDF5 file not found: {h5_path}")
        return

    with h5py.File(h5_path, "r") as f:
        # Required datasets
        req_datasets = ["lens_index", "cat_index", "ra_lens_deg", "dec_lens_deg",
                        "sample_id", "fix_param_names", "fix_params",
                        "free_param_names", "free_params",
                        "piemd_param_names", "piemd_params"]
        missing_ds = [d for d in req_datasets if d not in f]
        if missing_ds:
            _fail(f"Missing HDF5 datasets: {missing_ds}")
        else:
            _pass(f"All {len(req_datasets)} required datasets present")

        n_gal = f["lens_index"].shape[0]
        n_samp = f["sample_id"].shape[0]
        _pass(f"HDF5: {n_gal} galaxies × {n_samp} MCMC samples")

        # Shape checks
        fix_shape = f["fix_params"].shape
        free_shape = f["free_params"].shape
        piemd_shape = f["piemd_params"].shape
        expected_fix_dim = 2 if profile == "nfw" else 3
        expected_free_dim = 4 if profile == "nfw" else 5
        legacy_free_dim = expected_free_dim - 1

        if fix_shape != (n_gal, n_samp, expected_fix_dim):
            _fail(f"fix_params shape {fix_shape} != expected ({n_gal}, {n_samp}, {expected_fix_dim})")
        else:
            _pass(f"fix_params shape correct: {fix_shape}")

        if free_shape == (n_gal, n_samp, expected_free_dim):
            _pass(f"free_params shape correct: {free_shape}")
            free_has_rj = True
        elif free_shape == (n_gal, n_samp, legacy_free_dim):
            _warn(
                f"free_params shape is legacy ({free_shape}); expected new format "
                f"({n_gal}, {n_samp}, {expected_free_dim}) including rJ."
            )
            free_has_rj = False
        else:
            _fail(
                f"free_params shape {free_shape} != expected "
                f"({n_gal}, {n_samp}, {expected_free_dim}) or legacy ({n_gal}, {n_samp}, {legacy_free_dim})"
            )
            free_has_rj = False

        if piemd_shape != (n_gal, n_samp, 2):
            _fail(f"piemd_params shape {piemd_shape} != expected ({n_gal}, {n_samp}, 2)")
        else:
            _pass(f"piemd_params shape correct: {piemd_shape}")

        # NaN fractions
        fix_nan = np.isnan(f["fix_params"][:]).mean()
        free_nan = np.isnan(f["free_params"][:]).mean()
        piemd_nan = np.isnan(f["piemd_params"][:]).mean()
        _pass(f"NaN fractions: fix={fix_nan:.1%}, free={free_nan:.1%}, piemd={piemd_nan:.1%}")
        if fix_nan > 0.5:
            _warn(f"fix_params NaN fraction {fix_nan:.1%} > 50% — most samples failed")
        if free_nan > 0.5:
            _warn(f"free_params NaN fraction {free_nan:.1%} > 50% — most samples failed")

        # Lens indices should match CSV
        h5_lens = f["lens_index"][:]
        csv_lens = df["lens_index"].values
        if len(h5_lens) != len(csv_lens):
            _warn(f"HDF5 has {len(h5_lens)} galaxies but CSV has {len(csv_lens)}")
        else:
            if np.array_equal(h5_lens, csv_lens):
                _pass("lens_index arrays match between CSV and HDF5")
            else:
                _fail("lens_index mismatch between CSV and HDF5")

        # Parameter names
        fix_names = [s.decode() if isinstance(s, bytes) else s for s in f["fix_param_names"][:]]
        free_names = [s.decode() if isinstance(s, bytes) else s for s in f["free_param_names"][:]]
        _pass(f"fix_param_names: {fix_names}")
        _pass(f"free_param_names: {free_names}")
        if len(fix_names) != expected_fix_dim:
            _fail(f"fix_param_names length {len(fix_names)} != expected {expected_fix_dim}")
        else:
            _pass("fix_param_names length matches fix_params last dimension")
        if len(free_names) == expected_free_dim:
            _pass("free_param_names length matches free_params last dimension")
        elif len(free_names) == legacy_free_dim:
            _warn("free_param_names length indicates legacy free-parameter format without rJ.")
        else:
            _fail(
                f"free_param_names length {len(free_names)} != expected {expected_free_dim} "
                f"or legacy {legacy_free_dim}"
            )

        # Spot-check: all valid samples should have physically sensible values.
        fix_data = f["fix_params"][:]
        valid_fix = np.isfinite(fix_data[:, :, 0])
        if valid_fix.sum() > 0:
            rs_valid = fix_data[:, :, 0][valid_fix]
            bad_rs = (rs_valid <= 0).sum()
            if bad_rs:
                _fail(f"{bad_rs} fix_params samples with rs <= 0")
            else:
                _pass("All valid fix_params samples have rs > 0")
            if profile == "nfw":
                rho0_valid = fix_data[:, :, 1][valid_fix]
                bad_rho = (rho0_valid <= 0).sum()
                if bad_rho:
                    _fail(f"{bad_rho} fix_params NFW samples with rho0 <= 0")
                else:
                    _pass("All valid fix_params NFW samples have rho0 > 0")
            elif profile in ("tnfw", "bmo"):
                rt_valid = fix_data[:, :, 1][valid_fix]
                rho0_valid = fix_data[:, :, 2][valid_fix]
                bad_rt = (rt_valid <= rs_valid).sum()
                bad_rho = (rho0_valid <= 0).sum()
                if bad_rt:
                    _warn(f"{bad_rt} fix_params samples with rt <= rs")
                else:
                    _pass("All valid fix_params samples have rt > rs")
                if bad_rho:
                    _fail(f"{bad_rho} fix_params samples with rho0 <= 0")
                else:
                    _pass("All valid fix_params samples have rho0 > 0")
            elif profile == "einasto":
                rhos_valid = fix_data[:, :, 1][valid_fix]
                n_valid = fix_data[:, :, 2][valid_fix]
                bad_rhos = (rhos_valid <= 0).sum()
                bad_n = (n_valid <= 0).sum()
                if bad_rhos:
                    _fail(f"{bad_rhos} fix_params Einasto samples with rho_s <= 0")
                else:
                    _pass("All valid fix_params Einasto samples have rho_s > 0")
                if bad_n:
                    _warn(f"{bad_n} fix_params Einasto samples with n <= 0")
                else:
                    _pass("All valid fix_params Einasto samples have n > 0")
            elif profile == "gnfw":
                rhos_valid = fix_data[:, :, 1][valid_fix]
                gamma_valid = fix_data[:, :, 2][valid_fix]
                bad_rhos = (rhos_valid <= 0).sum()
                bad_gamma = ((gamma_valid <= 0) | (gamma_valid >= 3)).sum()
                if bad_rhos:
                    _fail(f"{bad_rhos} fix_params gNFW samples with rho_s <= 0")
                else:
                    _pass("All valid fix_params gNFW samples have rho_s > 0")
                if bad_gamma:
                    _warn(f"{bad_gamma} fix_params gNFW samples with gamma outside (0, 3)")
                else:
                    _pass("All valid fix_params gNFW samples have gamma in (0, 3)")

        free_data = f["free_params"][:]
        valid_free = np.isfinite(free_data[:, :, 0])
        if valid_free.sum() > 0:
            rs_valid = free_data[:, :, 0][valid_free]
            mstar_idx = -2 if free_has_rj else -1
            mstar_valid = free_data[:, :, mstar_idx][valid_free]
            bad_rs = (rs_valid <= 0).sum()
            bad_mstar = (mstar_valid <= 0).sum()
            if bad_rs:
                _fail(f"{bad_rs} free_params samples with rs <= 0")
            else:
                _pass("All valid free_params samples have rs > 0")
            if bad_mstar:
                _warn(f"{bad_mstar} free_params samples with Mstar <= 0")
            else:
                _pass("All valid free_params samples have Mstar > 0")
            if free_has_rj:
                rj_valid = free_data[:, :, -1][valid_free]
                bad_rj = (rj_valid <= 0).sum()
                if bad_rj:
                    _warn(f"{bad_rj} free_params samples with rJ <= 0")
                else:
                    _pass("All valid free_params samples have rJ > 0")
            else:
                _warn("No explicit rJ in free_params (legacy format); skipping rJ validity check.")
            if profile in ("tnfw", "bmo"):
                rt_valid = free_data[:, :, 1][valid_free]
                rho0_valid = free_data[:, :, 2][valid_free]
                bad_rt = (rt_valid <= rs_valid).sum()
                bad_rho = (rho0_valid <= 0).sum()
                if bad_rt:
                    _warn(f"{bad_rt} free_params samples with rt <= rs")
                else:
                    _pass("All valid free_params samples have rt > rs")
                if bad_rho:
                    _fail(f"{bad_rho} free_params samples with rho0 <= 0")
                else:
                    _pass("All valid free_params samples have rho0 > 0")
            elif profile == "nfw":
                rho0_valid = free_data[:, :, 1][valid_free]
                bad_rho = (rho0_valid <= 0).sum()
                if bad_rho:
                    _fail(f"{bad_rho} free_params NFW samples with rho0 <= 0")
                else:
                    _pass("All valid free_params NFW samples have rho0 > 0")
            elif profile == "einasto":
                rhos_valid = free_data[:, :, 1][valid_free]
                n_valid = free_data[:, :, 2][valid_free]
                bad_rhos = (rhos_valid <= 0).sum()
                bad_n = (n_valid <= 0).sum()
                if bad_rhos:
                    _fail(f"{bad_rhos} free_params Einasto samples with rho_s <= 0")
                else:
                    _pass("All valid free_params Einasto samples have rho_s > 0")
                if bad_n:
                    _warn(f"{bad_n} free_params Einasto samples with n <= 0")
                else:
                    _pass("All valid free_params Einasto samples have n > 0")
            elif profile == "gnfw":
                rhos_valid = free_data[:, :, 1][valid_free]
                gamma_valid = free_data[:, :, 2][valid_free]
                bad_rhos = (rhos_valid <= 0).sum()
                bad_gamma = ((gamma_valid <= 0) | (gamma_valid >= 3)).sum()
                if bad_rhos:
                    _fail(f"{bad_rhos} free_params gNFW samples with rho_s <= 0")
                else:
                    _pass("All valid free_params gNFW samples have rho_s > 0")
                if bad_gamma:
                    _warn(f"{bad_gamma} free_params gNFW samples with gamma outside (0, 3)")
                else:
                    _pass("All valid free_params gNFW samples have gamma in (0, 3)")

        # Metadata attributes
        req_attrs = ["bayesfile", "parfile", "catalog", "mref_F160W", "alpha", "beta_cut"]
        missing_attrs = [a for a in req_attrs if a not in f.attrs]
        if missing_attrs:
            _warn(f"Missing HDF5 attributes: {missing_attrs}")
        else:
            _pass(f"All metadata attributes present: {list(f.attrs.keys())}")

        # Report rmax_re_factor if present
        rmax_factor = f.attrs.get("rmax_re_factor", None)
        if rmax_factor is not None:
            if rmax_factor < 0:
                _pass("rmax_re_factor: not set (full radial grid used)")
            else:
                _pass(f"rmax_re_factor: {rmax_factor:.1f} (fit up to {rmax_factor:.1f} × Re per galaxy)")


# ---------------------------------------------------------------------------
# 7. MCMC percentile consistency
# ---------------------------------------------------------------------------
def check_mcmc_percentiles(df, h5_path, profile):
    _section("7. MCMC percentile consistency (CSV vs HDF5)")

    if "mcmc_nsamp_total" not in df.columns:
        print("  (no MCMC columns in CSV — skipping)")
        return

    try:
        import h5py
    except ImportError:
        _warn("h5py not installed — skipping HDF5 cross-check")
        return

    if h5_path is None or not Path(h5_path).exists():
        print("  (no HDF5 file — skipping cross-check)")
        # Still check internal consistency of MCMC columns
        ok_fix = df["mcmc_nsamp_ok_fix"]
        ok_free = df["mcmc_nsamp_ok_free"]
        total = df["mcmc_nsamp_total"]
        if (ok_fix > total).any():
            _fail("mcmc_nsamp_ok_fix > mcmc_nsamp_total for some galaxies")
        else:
            _pass("mcmc_nsamp_ok_fix <= mcmc_nsamp_total")
        if (ok_free > total).any():
            _fail("mcmc_nsamp_ok_free > mcmc_nsamp_total for some galaxies")
        else:
            _pass("mcmc_nsamp_ok_free <= mcmc_nsamp_total")

        # Percentile ordering: p16 <= p50 <= p84
        # Build list of prefixes to check based on profile
        if profile == "nfw":
            fix_prefixes = ["rs_nfw_fix", "rho0_nfw_fix"]
            free_prefixes = ["rs_nfw_free", "rho0_nfw_free", "Mstar_free", "rJ_free"]
        elif profile == "einasto":
            fix_prefixes = ["rs_ein_fix", "rhos_ein_fix", "n_ein_fix"]
            free_prefixes = ["rs_ein_free", "rhos_ein_free", "n_ein_free", "Mstar_free", "rJ_free"]
        elif profile == "gnfw":
            fix_prefixes = ["rs_gnfw_fix", "rhos_gnfw_fix", "gamma_gnfw_fix"]
            free_prefixes = ["rs_gnfw_free", "rhos_gnfw_free", "gamma_gnfw_free", "Mstar_free", "rJ_free"]
        elif profile == "bmo":
            fix_prefixes = ["rs_bmo_fix", "rt_bmo_fix", "rho0_bmo_fix"]
            free_prefixes = ["rs_bmo_free", "rt_bmo_free", "rho0_bmo_free", "Mstar_free", "rJ_free"]
        else:
            fix_prefixes = ["rs_fix", "rt_fix", "rho0_fix"]
            free_prefixes = ["rs_free", "rt_free", "rho0_free", "Mstar_free", "rJ_free"]
        derived_prefixes = ["c200_fix", "M200_fix", "c200_free", "M200_free"]

        for prefix in fix_prefixes + free_prefixes + derived_prefixes:
            p16_col, p50_col, p84_col = f"{prefix}_p16", f"{prefix}_p50", f"{prefix}_p84"
            if p16_col not in df.columns:
                continue
            p16 = df[p16_col]
            p50 = df[p50_col]
            p84 = df[p84_col]
            valid = np.isfinite(p16) & np.isfinite(p50) & np.isfinite(p84)
            if valid.sum() == 0:
                continue
            bad = ((p16[valid] > p50[valid] * 1.001) | (p50[valid] > p84[valid] * 1.001)).sum()
            if bad:
                _fail(f"{prefix}: {bad} galaxies with p16 > p50 or p50 > p84")
            else:
                _pass(f"{prefix}: percentile ordering p16 <= p50 <= p84 OK")
        return

    # Cross-check CSV percentiles against HDF5 raw samples
    with h5py.File(h5_path, "r") as f:
        fix_data = f["fix_params"][:]    # (Ngal, Nsamp, 2 or 3)
        free_data = f["free_params"][:]  # (Ngal, Nsamp, 4 or 5)

    if profile == "nfw":
        fix_pnames = ["rs_nfw_fix", "rho0_nfw_fix"]
        free_pnames = ["rs_nfw_free", "rho0_nfw_free", "Mstar_free", "rJ_free"]
    elif profile == "einasto":
        fix_pnames = ["rs_ein_fix", "rhos_ein_fix", "n_ein_fix"]
        free_pnames = ["rs_ein_free", "rhos_ein_free", "n_ein_free", "Mstar_free", "rJ_free"]
    elif profile == "gnfw":
        fix_pnames = ["rs_gnfw_fix", "rhos_gnfw_fix", "gamma_gnfw_fix"]
        free_pnames = ["rs_gnfw_free", "rhos_gnfw_free", "gamma_gnfw_free", "Mstar_free", "rJ_free"]
    elif profile == "bmo":
        fix_pnames = ["rs_bmo_fix", "rt_bmo_fix", "rho0_bmo_fix"]
        free_pnames = ["rs_bmo_free", "rt_bmo_free", "rho0_bmo_free", "Mstar_free", "rJ_free"]
    else:
        fix_pnames = ["rs_fix", "rt_fix", "rho0_fix"]
        free_pnames = ["rs_free", "rt_free", "rho0_free", "Mstar_free", "rJ_free"]

    n_gal = min(len(df), fix_data.shape[0])
    max_err_all = 0.0
    n_checked = 0
    n_free_cols = free_data.shape[2]
    if n_free_cols < len(free_pnames):
        _warn(
            f"HDF5 free_params has {n_free_cols} columns but validator expects "
            f"{len(free_pnames)} for full free-rJ format; only common columns will be cross-checked."
        )

    for igal in range(n_gal):
        row = df.iloc[igal]
        for ic, pname in enumerate(fix_pnames):
            vals = fix_data[igal, :, ic]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue
            p50_h5 = np.median(vals)
            p50_csv = row.get(f"{pname}_p50", np.nan)
            if np.isfinite(p50_csv) and np.isfinite(p50_h5) and p50_h5 > 0:
                err = abs(p50_csv - p50_h5) / p50_h5
                max_err_all = max(max_err_all, err)
                n_checked += 1

        for ic, pname in enumerate(free_pnames[:n_free_cols]):
            vals = free_data[igal, :, ic]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue
            p50_h5 = np.median(vals)
            p50_csv = row.get(f"{pname}_p50", np.nan)
            if np.isfinite(p50_csv) and np.isfinite(p50_h5) and p50_h5 > 0:
                err = abs(p50_csv - p50_h5) / p50_h5
                max_err_all = max(max_err_all, err)
                n_checked += 1

    if n_checked > 0:
        if max_err_all < 0.01:
            _pass(f"CSV percentiles match HDF5 recomputed values (max rel err = {max_err_all:.2e}, "
                  f"{n_checked} checks)")
        else:
            _fail(f"CSV percentiles vs HDF5 mismatch: max rel err = {max_err_all:.2e}")
    else:
        _warn("No valid percentile comparisons could be made")


# ---------------------------------------------------------------------------
# 8. Profile reconstruction
# ---------------------------------------------------------------------------
def check_profile_reconstruction(df, profile, n_sample=5):
    _section("8. Density profile reconstruction (spot check)")

    ok = df["fit_fix_success"].astype(bool) & df["fit_free_success"].astype(bool)
    sub = df.loc[ok]
    if len(sub) == 0:
        _warn("No galaxies with both fits converged — skipping reconstruction")
        return

    indices = sub.index[:n_sample]
    r = np.logspace(-1, 2.5, 300)
    dm_label = {"tnfw": "tNFW", "einasto": "Einasto", "gnfw": "gNFW", "nfw": "NFW", "bmo": "BMO"}.get(profile, profile)

    for idx in indices:
        row = sub.loc[idx]
        rho_piemd = piemd_rho(r, row["core_radius_kpc"], row["cut_radius_kpc"],
                              row["v_disp_sigma0_km_s"])

        # Restrict residuals to fitted radial range if rmax_fit_kpc is available
        rmax = row.get("rmax_fit_kpc", np.nan)
        if np.isfinite(rmax) and rmax > 0:
            r_mask = r <= rmax
        else:
            r_mask = np.ones_like(r, dtype=bool)

        # Fixed fit reconstruction
        if profile == "tnfw":
            rho_dm_fix = trunc_nfw_rho(r, row["rs_fix_kpc"], row["rt_fix_kpc"],
                                       row["rho0_fix_msun_kpc3"])
        elif profile == "nfw":
            rho_dm_fix = nfw_rho(r, row["rs_nfw_fix_kpc"],
                                  row["rho0_nfw_fix_msun_kpc3"])
        elif profile == "einasto":
            rho_dm_fix = einasto_rho(r, row["rs_ein_fix_kpc"],
                                     row["rhos_ein_fix_msun_kpc3"], row["n_ein_fix"])
        elif profile == "bmo":
            rho_dm_fix = bmo_rho(r, row["rs_bmo_fix_kpc"], row["rt_bmo_fix_kpc"],
                                 row["rho0_bmo_fix_msun_kpc3"])
        else:
            rho_dm_fix = gnfw_rho(r, row["rs_gnfw_fix_kpc"],
                                  row["rhos_gnfw_fix_msun_kpc3"], row["gamma_gnfw_fix"])

        rho_jaffe_fix = jaffe_rho(r, row["Mstar_from_F160W_Msun"], row["ReF160W_kpc"])
        rho_jaffe_clamped = np.minimum(rho_jaffe_fix, rho_piemd)
        rho_total_fix = rho_dm_fix + rho_jaffe_clamped

        mask = (rho_piemd > 0) & r_mask
        rel_res_fix = np.abs(rho_total_fix[mask] - rho_piemd[mask]) / rho_piemd[mask]
        med_res_fix = np.median(rel_res_fix)

        # Free fit reconstruction
        if profile == "tnfw":
            rho_dm_free = trunc_nfw_rho(r, row["rs_free_kpc"], row["rt_free_kpc"],
                                        row["rho0_free_msun_kpc3"])
        elif profile == "nfw":
            rho_dm_free = nfw_rho(r, row["rs_nfw_free_kpc"],
                                   row["rho0_nfw_free_msun_kpc3"])
        elif profile == "einasto":
            rho_dm_free = einasto_rho(r, row["rs_ein_free_kpc"],
                                      row["rhos_ein_free_msun_kpc3"], row["n_ein_free"])
        elif profile == "bmo":
            rho_dm_free = bmo_rho(r, row["rs_bmo_free_kpc"], row["rt_bmo_free_kpc"],
                                 row["rho0_bmo_free_msun_kpc3"])
        else:
            rho_dm_free = gnfw_rho(r, row["rs_gnfw_free_kpc"],
                                   row["rhos_gnfw_free_msun_kpc3"], row["gamma_gnfw_free"])

        rho_jaffe_free = jaffe_rho_rj(r, row["Mstar_free_Msun"], get_free_rj_kpc(row))
        rho_total_free = rho_dm_free + rho_jaffe_free
        rel_res_free = np.abs(rho_total_free[mask] - rho_piemd[mask]) / rho_piemd[mask]
        med_res_free = np.median(rel_res_free)

        lens_id = int(row["lens_index"])
        rmax_str = f", rmax={rmax:.1f} kpc" if np.isfinite(rmax) and rmax > 0 else ""
        status = "✓" if max(med_res_fix, med_res_free) < 0.5 else "⚠"
        print(f"  {status} Lens {lens_id} ({dm_label}{rmax_str}): median |Δρ/ρ| fix={med_res_fix:.3f}, free={med_res_free:.3f}")


# ---------------------------------------------------------------------------
# 9. Summary plot (optional)
# ---------------------------------------------------------------------------
def make_summary_plots(df, h5_path, outdir, profile, n_examples=1):
    _section("9. Summary diagnostic plots")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ok_fix = df["fit_fix_success"].astype(bool)
    ok_free = df["fit_free_success"].astype(bool)
    rs_fix_col = get_rs_col(profile, "fix")
    rs_free_col = get_rs_col(profile, "free")
    dm_label = {"tnfw": "tNFW", "einasto": "Einasto", "gnfw": "gNFW", "nfw": "NFW", "bmo": "BMO"}.get(profile, profile)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # (0,0) RMSE distribution
    ax = axes[0, 0]
    if ok_fix.sum():
        ax.hist(df.loc[ok_fix, "fit_fix_rmse_log"], bins=30, alpha=0.6, label="fixed", color="tab:blue")
    if ok_free.sum():
        ax.hist(df.loc[ok_free, "fit_free_rmse_log"], bins=30, alpha=0.6, label="free", color="tab:orange")
    ax.set_xlabel("RMSE (log)")
    ax.set_ylabel("N galaxies")
    ax.set_title(f"Fit quality: RMSE distribution ({dm_label})")
    ax.legend()

    # (0,1) rs distribution
    ax = axes[0, 1]
    if ok_fix.sum():
        ax.hist(np.log10(df.loc[ok_fix, rs_fix_col].clip(1e-3)), bins=30, alpha=0.6, label="rs fix", color="tab:blue")
    if ok_free.sum():
        ax.hist(np.log10(df.loc[ok_free, rs_free_col].clip(1e-3)), bins=30, alpha=0.6, label="rs free", color="tab:orange")
    ax.set_xlabel("log10(rs / kpc)")
    ax.set_ylabel("N galaxies")
    ax.set_title("Scale radius distribution")
    ax.legend()

    # (0,2) profile-specific third parameter
    ax = axes[0, 2]
    if profile == "tnfw":
        if ok_fix.sum():
            ratio_fix = df.loc[ok_fix, "rt_fix_kpc"] / df.loc[ok_fix, "rs_fix_kpc"]
            ax.hist(np.log10(ratio_fix.clip(0.1)), bins=30, alpha=0.6, label="fix", color="tab:blue")
        if ok_free.sum():
            ratio_free = df.loc[ok_free, "rt_free_kpc"] / df.loc[ok_free, "rs_free_kpc"]
            ax.hist(np.log10(ratio_free.clip(0.1)), bins=30, alpha=0.6, label="free", color="tab:orange")
        ax.set_xlabel("log10(rt/rs)")
        ax.set_title("Truncation ratio rt/rs")
    elif profile == "nfw":
        if ok_fix.sum():
            rho_vals_fix = df.loc[ok_fix, "rho0_nfw_fix_msun_kpc3"].dropna()
            ax.hist(np.log10(rho_vals_fix[rho_vals_fix > 0]), bins=30, alpha=0.6, label="fix", color="tab:blue")
        if ok_free.sum():
            rho_vals_free = df.loc[ok_free, "rho0_nfw_free_msun_kpc3"].dropna()
            ax.hist(np.log10(rho_vals_free[rho_vals_free > 0]), bins=30, alpha=0.6, label="free", color="tab:orange")
        ax.set_xlabel(r"log10($\rho_0$ / M$_\odot$ kpc$^{-3}$)")
        ax.set_title("NFW scale density distribution")
    elif profile == "einasto":
        if ok_fix.sum():
            ax.hist(df.loc[ok_fix, "n_ein_fix"].dropna(), bins=30, alpha=0.6, label="fix", color="tab:blue")
        if ok_free.sum():
            ax.hist(df.loc[ok_free, "n_ein_free"].dropna(), bins=30, alpha=0.6, label="free", color="tab:orange")
        ax.set_xlabel("Einasto n")
        ax.set_title("Einasto index distribution")
    elif profile == "bmo":
        if ok_fix.sum():
            ax.hist(df.loc[ok_fix, "rt_bmo_fix_kpc"].dropna(), bins=30, alpha=0.6, label="fix", color="tab:blue")
        if ok_free.sum():
            ax.hist(df.loc[ok_free, "rt_bmo_free_kpc"].dropna(), bins=30, alpha=0.6, label="free", color="tab:orange")
        ax.set_xlabel("Truncation radius rt [kpc]")
        ax.set_title("BMO truncation radius distribution")
    else:  # gnfw
        if ok_fix.sum():
            ax.hist(df.loc[ok_fix, "gamma_gnfw_fix"].dropna(), bins=30, alpha=0.6, label="fix", color="tab:blue")
        if ok_free.sum():
            ax.hist(df.loc[ok_free, "gamma_gnfw_free"].dropna(), bins=30, alpha=0.6, label="free", color="tab:orange")
        ax.set_xlabel(r"$\gamma$ (inner slope)")
        ax.set_title("gNFW inner slope distribution")
    ax.set_ylabel("N galaxies")
    ax.legend()

    # (1,0) Mstar_free vs Mstar_phot
    ax = axes[1, 0]
    both = ok_fix & ok_free
    if both.sum():
        mp = df.loc[both, "Mstar_from_F160W_Msun"]
        mf = df.loc[both, "Mstar_free_Msun"]
        valid = (mp > 0) & (mf > 0)
        if valid.sum():
            ax.scatter(np.log10(mp[valid]), np.log10(mf[valid]), s=8, alpha=0.5)
            lims = [min(np.log10(mp[valid]).min(), np.log10(mf[valid]).min()) - 0.3,
                    max(np.log10(mp[valid]).max(), np.log10(mf[valid]).max()) + 0.3]
            ax.plot(lims, lims, "k--", lw=0.8)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
    ax.set_xlabel("log10(M* phot / Msun)")
    ax.set_ylabel("log10(M* free / Msun)")
    ax.set_title("Stellar mass: photometry vs free fit")
    ax.set_aspect("equal", adjustable="box")

    # (1,1) Mstar_log_ratio histogram
    ax = axes[1, 1]
    if ok_free.sum():
        lr = df.loc[ok_free, "Mstar_log_ratio_free_over_phot"]
        lr_valid = lr[np.isfinite(lr)]
        if len(lr_valid):
            ax.hist(lr_valid, bins=30, alpha=0.7, color="tab:green")
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.axvline(np.median(lr_valid), color="tab:red", ls="-", lw=1.5,
                       label=f"median={np.median(lr_valid):.3f}")
    ax.set_xlabel("log10(M*_free / M*_phot)")
    ax.set_ylabel("N galaxies")
    ax.set_title("Stellar mass mismatch (dex)")
    ax.legend()

    # (1,2) RMSE fix vs free
    ax = axes[1, 2]
    if both.sum():
        ax.scatter(df.loc[both, "fit_fix_rmse_log"], df.loc[both, "fit_free_rmse_log"],
                   s=8, alpha=0.5)
        lims = [0, max(df.loc[both, "fit_fix_rmse_log"].max(),
                       df.loc[both, "fit_free_rmse_log"].max()) * 1.1]
        ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xlabel("RMSE fixed")
    ax.set_ylabel("RMSE free")
    ax.set_title("Fit quality: fixed vs free")
    ax.set_aspect("equal", adjustable="box")

    out_png = outdir / "validation_summary.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _pass(f"Saved summary plot to {out_png}")

    # --- MCMC posterior widths (if available) ---
    if "mcmc_nsamp_total" in df.columns and ok_free.sum() > 0:
        if profile == "nfw":
            mcmc_prefixes = ["rs_nfw_free", "rho0_nfw_free", "Mstar_free"]
            mcmc_labels = [r"$r_s$", r"$\rho_0$", r"$M_\star$"]
        elif profile == "einasto":
            mcmc_prefixes = ["rs_ein_free", "n_ein_free", "Mstar_free"]
            mcmc_labels = [r"$r_s$", r"$n$", r"$M_\star$"]
        elif profile == "gnfw":
            mcmc_prefixes = ["rs_gnfw_free", "gamma_gnfw_free", "Mstar_free"]
            mcmc_labels = [r"$r_s$", r"$\gamma$", r"$M_\star$"]
        elif profile == "bmo":
            mcmc_prefixes = ["rs_bmo_free", "rt_bmo_free", "Mstar_free"]
            mcmc_labels = [r"$r_s$", r"$r_t$", r"$M_\star$"]
        else:
            mcmc_prefixes = ["rs_free", "rt_free", "Mstar_free"]
            mcmc_labels = [r"$r_s$", r"$r_t$", r"$M_\star$"]

        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for ax, prefix, label in zip(axes2, mcmc_prefixes, mcmc_labels):
            p16_col = f"{prefix}_p16"
            p50_col = f"{prefix}_p50"
            p84_col = f"{prefix}_p84"
            if p16_col not in df.columns:
                continue
            p16 = df.loc[ok_free, p16_col]
            p50 = df.loc[ok_free, p50_col]
            p84 = df.loc[ok_free, p84_col]
            valid = np.isfinite(p16) & np.isfinite(p50) & np.isfinite(p84) & (p50 > 0)
            if valid.sum() == 0:
                continue
            frac_unc = ((p84[valid] - p16[valid]) / (2 * p50[valid])).values
            ax.hist(frac_unc, bins=30, alpha=0.7, color="steelblue")
            ax.axvline(np.median(frac_unc), color="tab:red", lw=1.5,
                       label=f"median={np.median(frac_unc):.2f}")
            ax.set_xlabel(f"Fractional 68% width ({label})")
            ax.set_ylabel("N galaxies")
            ax.set_title(f"MCMC posterior width: {label}")
            ax.legend(fontsize=9)

        out_png2 = outdir / "validation_mcmc_widths.png"
        fig2.savefig(out_png2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        _pass(f"Saved MCMC posterior width plot to {out_png2}")

    # --- c200 histogram(s) ---
    c_fix = df.loc[ok_fix, "c200_fix"] if ok_fix.sum() else pd.Series(dtype=float)
    c_free = df.loc[ok_free, "c200_free"] if ok_free.sum() else pd.Series(dtype=float)
    c_fix = c_fix[np.isfinite(c_fix) & (c_fix > 0)]
    c_free = c_free[np.isfinite(c_free) & (c_free > 0)]
    if len(c_fix) > 0 or len(c_free) > 0:
        figc, axc = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

        # Linear-space histogram
        if len(c_fix):
            axc[0].hist(c_fix, bins=30, alpha=0.6, label="fix", color="tab:blue")
            axc[0].axvline(np.median(c_fix), color="tab:blue", ls="--", lw=1.2,
                           label=f"fix med={np.median(c_fix):.1f}")
        if len(c_free):
            axc[0].hist(c_free, bins=30, alpha=0.6, label="free", color="tab:orange")
            axc[0].axvline(np.median(c_free), color="tab:orange", ls="--", lw=1.2,
                           label=f"free med={np.median(c_free):.1f}")
        axc[0].set_xlabel(r"$c_{200}$")
        axc[0].set_ylabel("N galaxies")
        axc[0].set_title(f"Concentration histogram ({dm_label})")
        axc[0].legend(fontsize=8)

        # Log-space histogram
        if len(c_fix):
            axc[1].hist(np.log10(c_fix), bins=30, alpha=0.6, label="fix", color="tab:blue")
            axc[1].axvline(np.log10(np.median(c_fix)), color="tab:blue", ls="--", lw=1.2)
        if len(c_free):
            axc[1].hist(np.log10(c_free), bins=30, alpha=0.6, label="free", color="tab:orange")
            axc[1].axvline(np.log10(np.median(c_free)), color="tab:orange", ls="--", lw=1.2)
        axc[1].set_xlabel(r"$\log_{10}(c_{200})$")
        axc[1].set_ylabel("N galaxies")
        axc[1].set_title(f"Concentration histogram in log-space ({dm_label})")
        axc[1].legend(fontsize=8)

        out_c200 = outdir / "validation_c200_histograms.png"
        figc.savefig(out_c200, dpi=150, bbox_inches="tight")
        plt.close(figc)
        _pass(f"Saved c200 histogram plot to {out_c200}")

    # --- M200 histogram(s) ---
    if profile == "einasto":
        m_fix_col, m_free_col = "M200_ein_fix_Msun", "M200_ein_free_Msun"
    elif profile == "gnfw":
        m_fix_col, m_free_col = "M200_gnfw_fix_Msun", "M200_gnfw_free_Msun"
    elif profile == "nfw":
        m_fix_col, m_free_col = "M200_nfw_fix_Msun", "M200_nfw_free_Msun"
    elif profile == "bmo":
        m_fix_col, m_free_col = "M200_bmo_fix_Msun", "M200_bmo_free_Msun"
    else:
        m_fix_col, m_free_col = "M200_fix_Msun", "M200_free_Msun"

    m_fix = df.loc[ok_fix, m_fix_col] if (ok_fix.sum() and m_fix_col in df.columns) else pd.Series(dtype=float)
    m_free = df.loc[ok_free, m_free_col] if (ok_free.sum() and m_free_col in df.columns) else pd.Series(dtype=float)
    m_fix = m_fix[np.isfinite(m_fix) & (m_fix > 0)]
    m_free = m_free[np.isfinite(m_free) & (m_free > 0)]
    if len(m_fix) > 0 or len(m_free) > 0:
        figm, axm = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

        # Linear-space histogram
        if len(m_fix):
            axm[0].hist(m_fix, bins=30, alpha=0.6, label="fix", color="tab:blue")
            axm[0].axvline(np.median(m_fix), color="tab:blue", ls="--", lw=1.2,
                           label=f"fix med={np.median(m_fix):.2e}")
        if len(m_free):
            axm[0].hist(m_free, bins=30, alpha=0.6, label="free", color="tab:orange")
            axm[0].axvline(np.median(m_free), color="tab:orange", ls="--", lw=1.2,
                           label=f"free med={np.median(m_free):.2e}")
        axm[0].set_xlabel(r"$M_{200}$ [M$_\odot$]")
        axm[0].set_ylabel("N galaxies")
        axm[0].set_title(f"$M_{{200}}$ histogram ({dm_label})")
        axm[0].legend(fontsize=8)

        # Log-space histogram
        if len(m_fix):
            axm[1].hist(np.log10(m_fix), bins=30, alpha=0.6, label="fix", color="tab:blue")
            axm[1].axvline(np.log10(np.median(m_fix)), color="tab:blue", ls="--", lw=1.2)
        if len(m_free):
            axm[1].hist(np.log10(m_free), bins=30, alpha=0.6, label="free", color="tab:orange")
            axm[1].axvline(np.log10(np.median(m_free)), color="tab:orange", ls="--", lw=1.2)
        axm[1].set_xlabel(r"$\log_{10}(M_{200}/M_\odot)$")
        axm[1].set_ylabel("N galaxies")
        axm[1].set_title(f"$M_{{200}}$ histogram in log-space ({dm_label})")
        axm[1].legend(fontsize=8)

        out_m200 = outdir / "validation_M200_histograms.png"
        figm.savefig(out_m200, dpi=150, bbox_inches="tight")
        plt.close(figm)
        _pass(f"Saved M200 histogram plot to {out_m200}")

    # --- Concentration–mass relation scatter plot ---
    _plot_concentration_mass(df, outdir, plt, profile)

    # --- Single-galaxy posterior distributions ---
    has_mcmc = "mcmc_nsamp_total" in df.columns
    if has_mcmc and h5_path is not None and Path(h5_path).exists():
        _plot_single_galaxy_posteriors(df, h5_path, outdir, plt, profile, n_examples=n_examples)


def _plot_concentration_mass(df, outdir, plt, profile):
    """Scatter plot of concentration c200 vs M200 for both fit modes, with MCMC error bars.

    Also overlays a theoretical LCDM c(M,z) relation (Dutton & Maccio 2014, Planck)
    and writes an outlier report based on robust residuals in log(c)-log(M) space.
    """
    from matplotlib.ticker import FuncFormatter
    has_mcmc = "c200_free_p50" in df.columns
    dm_label = {"tnfw": "tNFW", "einasto": "Einasto", "gnfw": "gNFW", "nfw": "NFW", "bmo": "BMO"}.get(profile, profile)

    def _lcdm_c200_dutton_maccio(m200_msun, z, h=0.7):
        """LCDM c200(M200,z) relation from Dutton & Maccio (2014), Planck cosmology.

        log10 c200 = a + b log10(M200 / (1e12 h^-1 Msun))
        with a(z) = 0.520 + (0.905 - 0.520) * exp(-0.617 z^1.21)
             b(z) = -0.101 + 0.026 z
        """
        m = np.asarray(m200_msun, dtype=float)
        zz = max(float(z), 0.0)
        a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * (zz**1.21))
        b = -0.101 + 0.026 * zz
        x = np.log10((m * h) / 1.0e12)
        return 10.0 ** (a + b * x)

    def _robust_trend_outliers(logm, logc, n_sigma=3.0, n_iter=6):
        """Iterative sigma-clipped polynomial trend fit in logc(logm), return outlier mask."""
        logm = np.asarray(logm, dtype=float)
        logc = np.asarray(logc, dtype=float)
        mask = np.isfinite(logm) & np.isfinite(logc)
        if mask.sum() < 8:
            res = np.full_like(logm, np.nan, dtype=float)
            return res, np.nan, np.zeros_like(mask, dtype=bool), np.array([np.nan, np.nan]), np.nan, 1

        good = mask.copy()
        coef = np.array([np.nan, np.nan], dtype=float)
        med_res = np.nan
        sigma = np.nan
        deg = 2 if mask.sum() >= 20 else 1
        for _ in range(n_iter):
            if good.sum() < (deg + 3):
                break
            coef = np.polyfit(logm[good], logc[good], deg)
            pred = np.polyval(coef, logm)
            res = logc - pred
            med_res = np.nanmedian(res[good])
            mad = np.nanmedian(np.abs(res[good] - med_res))
            sigma = 1.4826 * mad if mad > 0 else np.nanstd(res[good])
            if not np.isfinite(sigma) or sigma <= 0:
                break
            new_good = good & (np.abs(res - med_res) <= n_sigma * sigma)
            if np.array_equal(new_good, good):
                break
            good = new_good

        pred = np.polyval(coef, logm) if np.all(np.isfinite(coef)) else np.full_like(logm, np.nan)
        res = logc - pred
        # Avoid over-flagging due to extremely tiny intrinsic scatter.
        if np.isfinite(sigma):
            sigma = max(float(sigma), 0.02)
        if not np.isfinite(sigma) or sigma <= 0:
            out = np.zeros_like(mask, dtype=bool)
        else:
            out = mask & (np.abs(res - med_res) > n_sigma * sigma)
        return res, sigma, out, coef, med_res, deg

    def _binned_median_relation(logm, cvals, nbins=8, min_per_bin=5):
        """Return bin centers and median c200 in log10(M200) bins."""
        logm = np.asarray(logm, dtype=float)
        cvals = np.asarray(cvals, dtype=float)
        mask = np.isfinite(logm) & np.isfinite(cvals) & (cvals > 0)
        if mask.sum() < max(min_per_bin, 3):
            return np.array([]), np.array([]), np.array([])
        x = logm[mask]
        y = cvals[mask]
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            return np.array([]), np.array([]), np.array([])
        edges = np.linspace(xmin, xmax, nbins + 1)
        xc, yc, yerr = [], [], []
        for i in range(nbins):
            if i < nbins - 1:
                m = (x >= edges[i]) & (x < edges[i + 1])
            else:
                m = (x >= edges[i]) & (x <= edges[i + 1])
            if m.sum() < min_per_bin:
                continue
            xb = x[m]
            yb = y[m]
            xc.append(np.nanmedian(xb))
            p16, p50, p84 = np.nanpercentile(yb, [16, 50, 84])
            yc.append(p50)
            yerr.append(0.5 * (p84 - p16))
        return np.asarray(xc), np.asarray(yc), np.asarray(yerr)

    def _m200_col(mode):
        if profile == "einasto":
            return f"M200_ein_{mode}_Msun"
        elif profile == "gnfw":
            return f"M200_gnfw_{mode}_Msun"
        elif profile == "nfw":
            return f"M200_nfw_{mode}_Msun"
        elif profile == "bmo":
            return f"M200_bmo_{mode}_Msun"
        return f"M200_{mode}_Msun"

    outlier_rows = []
    mode_centers = {}

    for mode, color, marker in [("free", "tab:blue", "o"), ("fix", "tab:orange", "s")]:
        c_col = f"c200_{mode}"
        m_col = _m200_col(mode)
        ok_col = "fit_free_success" if mode == "free" else "fit_fix_success"
        ok = df[ok_col].astype(bool)
        sub = df.loc[ok].copy()

        valid = (
            np.isfinite(sub[c_col]) & (sub[c_col] > 0)
            & np.isfinite(sub[m_col]) & (sub[m_col] > 0)
        )
        sub = sub.loc[valid]
        if len(sub) == 0:
            continue

        fig_cm, ax_cm = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

        log_m = np.log10(sub[m_col].values)
        c_val = sub[c_col].values

        # Default plotted centers are best-fit values.
        plot_logm = log_m.copy()
        plot_c = c_val.copy()
        has_err = np.zeros(len(sub), dtype=bool)

        # Per-point asymmetric uncertainties (for points with MCMC percentiles)
        xerr_lo = np.full(len(sub), np.nan, dtype=float)
        xerr_hi = np.full(len(sub), np.nan, dtype=float)
        yerr_lo = np.full(len(sub), np.nan, dtype=float)
        yerr_hi = np.full(len(sub), np.nan, dtype=float)

        # Plot error bars from MCMC percentiles if available
        if has_mcmc:
            c_p16 = sub.get(f"c200_{mode}_p16", pd.Series(dtype=float))
            c_p50 = sub.get(f"c200_{mode}_p50", pd.Series(dtype=float))
            c_p84 = sub.get(f"c200_{mode}_p84", pd.Series(dtype=float))
            m_p16 = sub.get(f"M200_{mode}_p16", pd.Series(dtype=float))
            m_p50 = sub.get(f"M200_{mode}_p50", pd.Series(dtype=float))
            m_p84 = sub.get(f"M200_{mode}_p84", pd.Series(dtype=float))

            has_err = (
                np.isfinite(c_p16) & np.isfinite(c_p50) & np.isfinite(c_p84)
                & np.isfinite(m_p16) & np.isfinite(m_p50) & np.isfinite(m_p84)
                & (c_p16 > 0) & (c_p50 > 0) & (c_p84 > 0)
                & (m_p16 > 0) & (m_p50 > 0) & (m_p84 > 0)
            ).values

            if np.sum(has_err) > 0:
                # For points with percentiles, center on posterior medians.
                se = sub.loc[has_err]
                # Use posterior medians as central values when plotting percentile error bars.
                log_m_e = np.log10(se[f"M200_{mode}_p50"].values)
                c_e = se[f"c200_{mode}_p50"].values
                plot_logm[has_err] = log_m_e
                plot_c[has_err] = c_e

                # Asymmetric error bars in log10(M200) space
                log_m_lo = log_m_e - np.log10(se[f"M200_{mode}_p16"].values)
                log_m_hi = np.log10(se[f"M200_{mode}_p84"].values) - log_m_e
                xerr = np.array([log_m_lo, log_m_hi])
                xerr = np.clip(xerr, 0, None)
                xerr_lo[has_err] = xerr[0]
                xerr_hi[has_err] = xerr[1]

                # Asymmetric error bars in c200 space
                c_lo = c_e - se[f"c200_{mode}_p16"].values
                c_hi = se[f"c200_{mode}_p84"].values - c_e
                yerr = np.array([c_lo, c_hi])
                yerr = np.clip(yerr, 0, None)
                yerr_lo[has_err] = yerr[0]
                yerr_hi[has_err] = yerr[1]

                ax_cm.errorbar(
                    log_m_e, c_e, xerr=xerr, yerr=yerr,
                    fmt="none", ecolor=color, elinewidth=0.5, alpha=0.3,
                    capsize=0, zorder=1,
                )

        # Scatter central values.
        label = f"{mode} central values"
        if np.sum(has_err) > 0:
            label += f" ({np.sum(has_err)} posterior medians + {len(sub)-np.sum(has_err)} best-fit)"
        ax_cm.scatter(
            plot_logm, plot_c, s=18, c=color, marker=marker, alpha=0.65,
            edgecolors="none", label=label, zorder=2,
        )

        # Binned median relation c200(M200) in log-mass bins.
        bx, by, byerr = _binned_median_relation(plot_logm, plot_c, nbins=8, min_per_bin=5)
        if len(bx) > 0:
            ax_cm.plot(
                bx, by, color=color, lw=2.0, alpha=0.95, zorder=3,
                label=f"{mode} binned median",
            )
            ax_cm.errorbar(
                bx, by, yerr=byerr, fmt="none",
                ecolor=color, elinewidth=1.0, alpha=0.7, capsize=2, zorder=3,
            )

        # Overlay theoretical LCDM relation at representative lens redshift.
        z_ref = float(np.nanmedian(sub["z_lens"])) if ("z_lens" in sub.columns and np.any(np.isfinite(sub["z_lens"]))) else 0.0
        mline = np.logspace(np.nanmin(plot_logm), np.nanmax(plot_logm), 200)
        cline = _lcdm_c200_dutton_maccio(mline, z_ref)
        ax_cm.plot(
            np.log10(mline), cline, color="black", lw=1.4, ls="-",
            label=f"LCDM (Dutton+Maccio14, z={z_ref:.2f})", zorder=1,
        )

        # Robust outlier detection in plotted logc-logM space.
        plot_logc = np.log10(plot_c)
        res, sigma, out_mask, coef, med_res, trend_deg = _robust_trend_outliers(plot_logm, plot_logc, n_sigma=3.0)
        if np.any(out_mask):
            ax_cm.scatter(
                plot_logm[out_mask], plot_c[out_mask],
                s=70, facecolors="none", edgecolors="red", linewidths=1.1,
                marker="o", label=f"outliers (>3σ trend): {np.sum(out_mask)}", zorder=4,
            )
            out = sub.loc[out_mask].copy()
            out["mode"] = mode
            out["plot_log10_M200"] = plot_logm[out_mask]
            out["plot_log10_c200"] = plot_logc[out_mask]
            out["residual_empirical_dex"] = res[out_mask]
            if np.isfinite(sigma) and sigma > 0:
                out["residual_empirical_sigma"] = np.abs((res[out_mask] - med_res) / sigma)
            else:
                out["residual_empirical_sigma"] = np.nan
            c_lcdm = _lcdm_c200_dutton_maccio(10.0 ** out["plot_log10_M200"].values, z_ref)
            out["residual_lcdm_dex"] = out["plot_log10_c200"].values - np.log10(c_lcdm)
            out["trend_degree"] = trend_deg
            out["trend_coefficients"] = ",".join([f"{v:.8e}" for v in coef])
            out["trend_sigma_dex"] = sigma
            out["z_ref_lcdm"] = z_ref
            outlier_rows.append(out)

        mode_centers[mode] = {
            "logm": plot_logm,
            "c": plot_c,
            "z_ref": z_ref,
            "has_err": has_err,
            "xerr_lo": xerr_lo,
            "xerr_hi": xerr_hi,
            "yerr_lo": yerr_lo,
            "yerr_hi": yerr_hi,
        }

        ax_cm.set_xlabel(r"$\log_{10}(M_{200} / M_\odot)$")
        ax_cm.set_ylabel(r"$c_{200}$")
        ax_cm.set_title(f"Concentration–mass relation ({mode} fit)")
        ax_cm.legend(fontsize=9)
        ax_cm.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))

        out_cm = outdir / f"validation_c200_vs_M200_{mode}.png"
        fig_cm.savefig(out_cm, dpi=150, bbox_inches="tight")
        plt.close(fig_cm)
        _pass(f"Saved c200–M200 plot ({mode}) to {out_cm}")

    # --- Combined panel: both modes overlaid ---
    fig_both, ax_b = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    for mode, color, marker, label in [
        ("free", "tab:blue", "o", "free-Jaffe"),
        ("fix", "tab:orange", "s", "fixed-Jaffe"),
    ]:
        if mode not in mode_centers:
            continue
        mdat = mode_centers[mode]
        logm_mode = mdat["logm"]
        c_mode = mdat["c"]
        ax_b.scatter(
            logm_mode, c_mode,
            s=12, c=color, marker=marker, alpha=0.5,
            edgecolors="none", label=f"{label} ({len(logm_mode)})",
        )
        has_err = mdat["has_err"]
        if np.any(has_err):
            xerr = np.vstack((mdat["xerr_lo"][has_err], mdat["xerr_hi"][has_err]))
            yerr = np.vstack((mdat["yerr_lo"][has_err], mdat["yerr_hi"][has_err]))
            ax_b.errorbar(
                logm_mode[has_err], c_mode[has_err], xerr=xerr, yerr=yerr,
                fmt="none", ecolor=color, elinewidth=0.45, alpha=0.28,
                capsize=0,
            )
        bx, by, byerr = _binned_median_relation(logm_mode, c_mode, nbins=8, min_per_bin=5)
        if len(bx) > 0:
            ax_b.plot(
                bx, by, color=color, lw=2.0, alpha=0.95,
                label=f"{label} median relation",
            )
            ax_b.errorbar(
                bx, by, yerr=byerr, fmt="none",
                ecolor=color, elinewidth=1.0, alpha=0.7, capsize=2,
            )

    # Single LCDM reference line for combined panel at median lens redshift.
    z_ref_all = float(np.nanmedian(df["z_lens"])) if ("z_lens" in df.columns and np.any(np.isfinite(df["z_lens"]))) else 0.0
    all_logm = np.concatenate([vals["logm"] for vals in mode_centers.values()]) if len(mode_centers) else np.array([])
    if len(all_logm):
        mline = np.logspace(np.nanmin(all_logm), np.nanmax(all_logm), 200)
        cline = _lcdm_c200_dutton_maccio(mline, z_ref_all)
        ax_b.plot(
            np.log10(mline), cline, color="black", lw=1.5,
            label=f"LCDM (Dutton+Maccio14, z={z_ref_all:.2f})",
        )
    ax_b.set_xlabel(r"$\log_{10}(M_{200} / M_\odot)$")
    ax_b.set_ylabel(r"$c_{200}$")
    ax_b.set_title("Concentration–mass relation (both modes)")
    ax_b.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
    ax_b.legend(fontsize=9)
    out_both = outdir / "validation_c200_vs_M200_combined.png"
    fig_both.savefig(out_both, dpi=150, bbox_inches="tight")
    plt.close(fig_both)
    _pass(f"Saved combined c200–M200 plot to {out_both}")

    # Save outlier report if any outliers were identified.
    if len(outlier_rows) > 0:
        outliers = pd.concat(outlier_rows, axis=0, ignore_index=True)
        outliers = outliers.sort_values("residual_empirical_sigma", ascending=False)
        preferred_cols = [
            "mode", "lens_index", "cat_index", "z_lens",
            "plot_log10_M200", "plot_log10_c200",
            "residual_empirical_dex", "residual_empirical_sigma",
            "residual_lcdm_dex",
            "fit_fix_rmse_log", "fit_free_rmse_log",
            "Mstar_ratio_free_over_phot", "Mstar_log_ratio_free_over_phot",
            "rJ_free_kpc",
        ]
        profile_cols = []
        if profile == "bmo":
            profile_cols = [
                "rs_bmo_fix_kpc", "rt_bmo_fix_kpc", "rho0_bmo_fix_msun_kpc3",
                "rs_bmo_free_kpc", "rt_bmo_free_kpc", "rho0_bmo_free_msun_kpc3",
                "c200_fix", "c200_free", "M200_bmo_fix_Msun", "M200_bmo_free_Msun",
            ]
        elif profile == "nfw":
            profile_cols = [
                "rs_nfw_fix_kpc", "rho0_nfw_fix_msun_kpc3",
                "rs_nfw_free_kpc", "rho0_nfw_free_msun_kpc3",
                "c200_fix", "c200_free", "M200_nfw_fix_Msun", "M200_nfw_free_Msun",
            ]
        elif profile == "einasto":
            profile_cols = [
                "rs_ein_fix_kpc", "rhos_ein_fix_msun_kpc3", "n_ein_fix",
                "rs_ein_free_kpc", "rhos_ein_free_msun_kpc3", "n_ein_free",
                "c200_fix", "c200_free", "M200_ein_fix_Msun", "M200_ein_free_Msun",
            ]
        elif profile == "gnfw":
            profile_cols = [
                "rs_gnfw_fix_kpc", "rhos_gnfw_fix_msun_kpc3", "gamma_gnfw_fix",
                "rs_gnfw_free_kpc", "rhos_gnfw_free_msun_kpc3", "gamma_gnfw_free",
                "c200_fix", "c200_free", "M200_gnfw_fix_Msun", "M200_gnfw_free_Msun",
            ]
        else:
            profile_cols = [
                "rs_fix_kpc", "rt_fix_kpc", "rho0_fix_msun_kpc3",
                "rs_free_kpc", "rt_free_kpc", "rho0_free_msun_kpc3",
                "c200_fix", "c200_free", "M200_fix_Msun", "M200_free_Msun",
            ]
        keep = [c for c in preferred_cols + profile_cols if c in outliers.columns]
        outliers = outliers[keep]
        out_csv = outdir / "validation_c200_M200_outliers.csv"
        outliers.to_csv(out_csv, index=False)

        counts = outliers["mode"].value_counts().to_dict() if "mode" in outliers.columns else {}
        out_txt = outdir / "validation_c200_M200_outliers.txt"
        with out_txt.open("w") as f:
            f.write("c200-M200 outlier report (|residual| > 3 sigma from robust trend in log-space)\\n")
            f.write(f"Total outliers: {len(outliers)}\\n")
            f.write(f"By mode: {counts}\\n\\n")
            f.write(outliers.head(30).to_string(index=False))
            f.write("\\n")
        _warn(f"Detected {len(outliers)} c200-M200 outlier points; report saved to {out_csv}")
        _pass(f"Saved outlier summary text to {out_txt}")

        # Save detailed density-fit plots for outliers, with Sersic inset.
        outlier_plot_dir = outdir / "outlier_profiles"
        outlier_plot_dir.mkdir(parents=True, exist_ok=True)
        out_unique = outliers.drop_duplicates(subset=["mode", "lens_index"]).copy()
        for _, o in out_unique.iterrows():
            lid = int(o["lens_index"])
            mode = str(o.get("mode", "unknown"))
            full_rows = df[df["lens_index"] == lid]
            if len(full_rows) == 0:
                _warn(f"Could not find full row for outlier lens {lid}; skipping outlier profile plot")
                continue
            row_full = full_rows.iloc[0]
            z = float(row_full["z_lens"]) if np.isfinite(row_full.get("z_lens", np.nan)) else np.nan
            _plot_single_galaxy_profile(
                row_full, lid, z, profile, outlier_plot_dir, plt,
                out_suffix=f"_outlier_{mode}",
                add_sersic_inset=False,
                outlier_mode=mode,
            )
        _pass(f"Saved outlier profile plots to {outlier_plot_dir}")


def _plot_single_galaxy_posteriors(df, h5_path, outdir, plt, profile, n_examples=1):
    """Plot MCMC posterior distributions of M*, c200, and a third profile-specific
    parameter for the top n_examples galaxies."""
    try:
        import h5py
    except ImportError:
        _warn("h5py not installed — skipping single-galaxy posterior plot")
        return

    if not Path(h5_path).exists():
        _warn(f"HDF5 not found: {h5_path} — skipping single-galaxy posterior plot")
        return

    if "mcmc_nsamp_ok_free" not in df.columns:
        return

    # Select top n_examples galaxies by number of valid MCMC free-fit samples
    ranked = df.nlargest(n_examples, "mcmc_nsamp_ok_free")

    with h5py.File(h5_path, "r") as f:
        h5_lens = f["lens_index"][:]

        for _, best_row in ranked.iterrows():
            lens_id = int(best_row["lens_index"])
            matches = np.where(h5_lens == lens_id)[0]
            if len(matches) == 0:
                _warn(f"Lens {lens_id} not found in HDF5 — skipping")
                continue
            igal = int(matches[0])
            free_samp = f["free_params"][igal, :, :]  # (Nsamp, 4 or 5)

            valid_mask = np.isfinite(free_samp[:, 0])
            samp = free_samp[valid_mask]
            if len(samp) < 10:
                _warn(f"Lens {lens_id}: only {len(samp)} valid MCMC samples — skipping")
                continue

            z_lens = float(best_row["z_lens"])

            _plot_one_galaxy_posterior(
                best_row, lens_id, z_lens, samp, profile, outdir, plt,
            )
            _plot_single_galaxy_profile(best_row, lens_id, z_lens, profile, outdir, plt)


def _plot_one_galaxy_posterior(row, lens_id, z_lens, samp, profile, outdir, plt):
    """Posterior histograms of M*, c200, and a third parameter for one galaxy."""
    # NFW: samp = [rs, rho0, mstar, rJ] — shape (Nsamp, 4)
    # tNFW: samp = [rs, rt, rho0, mstar, rJ] — shape (Nsamp, 5)
    # Einasto: samp = [rs, rho_s, n, mstar, rJ] — shape (Nsamp, 5)
    # gNFW: samp = [rs, rho_s, gamma, mstar, rJ] — shape (Nsamp, 5)
    # BMO: samp = [rs, rt, rho0, mstar, rJ] — shape (Nsamp, 5)
    rs_samp = samp[:, 0]
    mstar_samp = samp[:, -2]

    # c200 = R200/rs — for tNFW we need to compute R200 from rho0;
    # for simplicity, just use the best-fit c200 from CSV for the marker
    # and show the rs distribution directly as the third panel.
    bf_mstar = row.get("Mstar_free_Msun", np.nan)
    bf_c200 = row.get("c200_free", np.nan)

    if profile == "nfw":
        third_samp = samp[:, 1]  # rho0
        bf_third = row.get("rho0_nfw_free_msun_kpc3", np.nan)
        third_label = r"$\rho_0$ [M$_\odot$/kpc$^3$]"
        third_title = "NFW scale density"
    elif profile == "tnfw":
        third_samp = samp[:, 1]  # rt
        bf_third = row.get("rt_free_kpc", np.nan)
        third_label = r"$r_t$ [kpc]"
        third_title = "Truncation radius"
    elif profile == "einasto":
        third_samp = samp[:, 2]  # n
        bf_third = row.get("n_ein_free", np.nan)
        third_label = r"$n$ (Einasto index)"
        third_title = "Einasto index"
    elif profile == "bmo":
        third_samp = samp[:, 1]  # rt
        bf_third = row.get("rt_bmo_free_kpc", np.nan)
        third_label = r"$r_t$ [kpc]"
        third_title = "BMO truncation radius"
    else:  # gnfw
        third_samp = samp[:, 2]  # gamma
        bf_third = row.get("gamma_gnfw_free", np.nan)
        third_label = r"$\gamma$ (inner slope)"
        third_title = "gNFW inner slope"

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)

    panels = [
        (mstar_samp, bf_mstar, r"$M_\star$ [M$_\odot$]", "Stellar mass", True),
        (rs_samp, row.get(get_rs_col(profile, "free"), np.nan), r"$r_s$ [kpc]", "Scale radius", False),
        (third_samp, bf_third, third_label, third_title, False),
    ]

    for ax, (samples, bf_val, xlabel, title, use_log) in zip(axes, panels):
        s = samples[np.isfinite(samples)]
        if len(s) == 0:
            ax.set_title(f"{title} (no data)")
            continue

        if use_log and np.all(s > 0):
            s = np.log10(s)
            if np.isfinite(bf_val) and bf_val > 0:
                bf_val = np.log10(bf_val)
            xlabel = r"$\log_{10}($" + xlabel.strip("$") + r"$)$"

        p16, p50, p84 = np.percentile(s, [16, 50, 84])

        ax.hist(s, bins=min(40, max(10, len(s) // 20)), alpha=0.7,
                color="steelblue", edgecolor="white", linewidth=0.3,
                density=True, label="MCMC posterior")

        # Median and 68% CI
        ax.axvline(p50, color="tab:red", lw=1.5, ls="-",
                   label=f"median = {p50:.3g}")
        ax.axvspan(p16, p84, alpha=0.15, color="tab:red",
                   label=f"68% CI [{p16:.3g}, {p84:.3g}]")

        # Best-fit value
        if np.isfinite(bf_val):
            ax.axvline(bf_val, color="black", lw=1.8, ls="--",
                       label=f"best fit = {bf_val:.3g}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        f"Lens {lens_id} (z={z_lens:.3f}, F160W={row.get('F160W', np.nan):.2f}) — "
        f"{len(samp)} MCMC samples",
        fontsize=11,
    )

    out_post = outdir / f"validation_posterior_lens{lens_id:03d}.png"
    fig.savefig(out_post, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _pass(f"Saved single-galaxy posterior plot (lens {lens_id}) to {out_post}")


def _plot_single_galaxy_profile(
    row, lens_id, z_lens, profile, outdir, plt,
    out_suffix="", add_sersic_inset=False, outlier_mode=None,
):
    """Density profile plot for one galaxy: PIEMD vs DM+Jaffe (fixed & free).

    Upper panel: density profiles on log-log scale.
    Lower panel: fractional residuals (model - PIEMD) / PIEMD.
    """
    r = np.logspace(-2, 3, 500)
    dm_label = {"tnfw": "tNFW", "einasto": "Einasto", "gnfw": "gNFW", "nfw": "NFW", "bmo": "BMO"}.get(profile, profile)
    rs_fix_col = get_rs_col(profile, "fix")
    rs_free_col = get_rs_col(profile, "free")

    # --- PIEMD total density ---
    r_core = row["core_radius_kpc"]
    r_cut = row["cut_radius_kpc"]
    sigma0 = row["v_disp_sigma0_km_s"]
    rho_piemd = piemd_rho(r, r_core, r_cut, sigma0)

    mstar_phot = row["Mstar_from_F160W_Msun"]
    re_kpc = row["ReF160W_kpc"]

    # --- Fixed-Jaffe mode ---
    rs_fix = row[rs_fix_col]
    ok_fix = np.isfinite(rs_fix)
    rho_dm_fix = rho_jaffe_fix_clamped = rho_total_fix = None
    if ok_fix:
        if profile == "tnfw":
            ok_fix = ok_fix and np.isfinite(row["rt_fix_kpc"]) and np.isfinite(row["rho0_fix_msun_kpc3"])
            if ok_fix:
                rho_dm_fix = trunc_nfw_rho(r, rs_fix, row["rt_fix_kpc"], row["rho0_fix_msun_kpc3"])
        elif profile == "nfw":
            ok_fix = ok_fix and np.isfinite(row["rho0_nfw_fix_msun_kpc3"])
            if ok_fix:
                rho_dm_fix = nfw_rho(r, rs_fix, row["rho0_nfw_fix_msun_kpc3"])
        elif profile == "einasto":
            ok_fix = ok_fix and np.isfinite(row["rhos_ein_fix_msun_kpc3"]) and np.isfinite(row["n_ein_fix"])
            if ok_fix:
                rho_dm_fix = einasto_rho(r, rs_fix, row["rhos_ein_fix_msun_kpc3"], row["n_ein_fix"])
        elif profile == "bmo":
            ok_fix = ok_fix and np.isfinite(row["rho0_bmo_fix_msun_kpc3"]) and np.isfinite(row["rt_bmo_fix_kpc"])
            if ok_fix:
                rho_dm_fix = bmo_rho(r, rs_fix, row["rt_bmo_fix_kpc"], row["rho0_bmo_fix_msun_kpc3"])
        else:
            ok_fix = ok_fix and np.isfinite(row["rhos_gnfw_fix_msun_kpc3"]) and np.isfinite(row["gamma_gnfw_fix"])
            if ok_fix:
                rho_dm_fix = gnfw_rho(r, rs_fix, row["rhos_gnfw_fix_msun_kpc3"], row["gamma_gnfw_fix"])
        if ok_fix and rho_dm_fix is not None:
            rho_jaffe_fix = jaffe_rho(r, mstar_phot, re_kpc)
            rho_jaffe_fix_clamped = np.minimum(rho_jaffe_fix, rho_piemd)
            rho_total_fix = rho_dm_fix + rho_jaffe_fix_clamped

    # --- Free-Jaffe mode ---
    mstar_free = row["Mstar_free_Msun"]
    rj_free = get_free_rj_kpc(row)
    rs_free = row[rs_free_col]
    ok_free = np.isfinite(rs_free) and np.isfinite(mstar_free) and np.isfinite(rj_free)
    rho_dm_free = rho_jaffe_free = rho_total_free = None
    if ok_free:
        if profile == "tnfw":
            ok_free = ok_free and np.isfinite(row["rt_free_kpc"]) and np.isfinite(row["rho0_free_msun_kpc3"])
            if ok_free:
                rho_dm_free = trunc_nfw_rho(r, rs_free, row["rt_free_kpc"], row["rho0_free_msun_kpc3"])
        elif profile == "nfw":
            ok_free = ok_free and np.isfinite(row["rho0_nfw_free_msun_kpc3"])
            if ok_free:
                rho_dm_free = nfw_rho(r, rs_free, row["rho0_nfw_free_msun_kpc3"])
        elif profile == "einasto":
            ok_free = ok_free and np.isfinite(row["rhos_ein_free_msun_kpc3"]) and np.isfinite(row["n_ein_free"])
            if ok_free:
                rho_dm_free = einasto_rho(r, rs_free, row["rhos_ein_free_msun_kpc3"], row["n_ein_free"])
        elif profile == "bmo":
            ok_free = ok_free and np.isfinite(row["rho0_bmo_free_msun_kpc3"]) and np.isfinite(row["rt_bmo_free_kpc"])
            if ok_free:
                rho_dm_free = bmo_rho(r, rs_free, row["rt_bmo_free_kpc"], row["rho0_bmo_free_msun_kpc3"])
        else:
            ok_free = ok_free and np.isfinite(row["rhos_gnfw_free_msun_kpc3"]) and np.isfinite(row["gamma_gnfw_free"])
            if ok_free:
                rho_dm_free = gnfw_rho(r, rs_free, row["rhos_gnfw_free_msun_kpc3"], row["gamma_gnfw_free"])
        if ok_free and rho_dm_free is not None:
            rho_jaffe_free = jaffe_rho_rj(r, mstar_free, rj_free)
            rho_total_free = rho_dm_free + rho_jaffe_free

    if not ok_fix and not ok_free:
        _warn(f"Lens {lens_id}: neither fit converged — skipping profile plot")
        return

    # --- Figure with two panels ---
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        constrained_layout=True,
    )

    # ---- Upper panel: density profiles ----
    ax_top.loglog(r, rho_piemd, color="black", lw=2.2, label="PIEMD total", zorder=5)

    if ok_fix and rho_dm_fix is not None:
        ax_top.loglog(r, rho_dm_fix, color="tab:orange", lw=1.0, ls=":",
                      label=f"{dm_label} (fixed), rs={rs_fix:.1f} kpc")
        ax_top.loglog(r, rho_jaffe_fix_clamped, color="tab:cyan", lw=1.0, ls=":",
                      label=f"Jaffe (phot), M*={mstar_phot:.2e}")
        ax_top.loglog(r, rho_total_fix, color="tab:green", lw=1.8, ls="--",
                      label=f"{dm_label} + Jaffe (fixed)")

    if ok_free and rho_dm_free is not None:
        ax_top.loglog(r, rho_dm_free, color="tab:purple", lw=1.0, ls=":",
                      label=f"{dm_label} (free), rs={rs_free:.1f} kpc")
        ax_top.loglog(r, rho_jaffe_free, color="tab:pink", lw=1.0, ls=":",
                      label=f"Jaffe (free), M*={mstar_free:.2e}, rJ={rj_free:.2f} kpc")
        ax_top.loglog(r, rho_total_free, color="tab:red", lw=1.8, ls="-.",
                      label=f"{dm_label} + Jaffe (free)")

    ax_top.axvline(re_kpc, color="gray", lw=0.8, ls=":", alpha=0.7,
                   label=f"Re = {re_kpc:.2f} kpc")

    # Show the maximum fitting radius if set
    rmax = row.get("rmax_fit_kpc", np.nan)
    if np.isfinite(rmax) and rmax > 0:
        ax_top.axvline(rmax, color="red", lw=1.0, ls="--", alpha=0.6,
                       label=f"rmax_fit = {rmax:.1f} kpc")

    ax_top.set_ylabel(r"$\rho(r)$ [M$_\odot$ / kpc$^3$]")
    mode_note = f", outlier mode={outlier_mode}" if outlier_mode is not None else ""
    ax_top.set_title(
        f"Lens {lens_id}  (z={z_lens:.3f}, F160W={row.get('F160W', np.nan):.2f}, "
        f"$\\sigma_0$={sigma0:.0f} km/s, DM={dm_label}{mode_note})",
        fontsize=10,
    )
    ax_top.legend(fontsize=7, loc="upper right", ncol=2)
    ax_top.set_xlim(r[0], r[-1])

    if add_sersic_inset:
        # Use available Sersic index if present; fallback to n=4 (de Vaucouleurs).
        n_candidates = ["sersic_n", "n_sersic", "nF160W", "n_ser", "n"]
        n_ser = np.nan
        for cname in n_candidates:
            if cname in row.index:
                val = row.get(cname, np.nan)
                if np.isfinite(val) and val > 0:
                    n_ser = float(val)
                    break
        if not np.isfinite(n_ser):
            n_ser = 4.0

        rmin_i = max(r[0], re_kpc / 30.0 if np.isfinite(re_kpc) and re_kpc > 0 else r[0])
        rmax_i = min(r[-1], re_kpc * 40.0 if np.isfinite(re_kpc) and re_kpc > 0 else r[-1])
        if rmax_i <= rmin_i:
            rmax_i = rmin_i * 50.0
        r_ser = np.logspace(np.log10(rmin_i), np.log10(rmax_i), 300)
        i_ser = sersic_surface_brightness(r_ser, re_kpc, n_ser)
        ax_ins = ax_top.inset_axes([0.06, 0.57, 0.35, 0.35])
        ax_ins.loglog(r_ser, i_ser, color="tab:blue", lw=1.2)
        ax_ins.axvline(re_kpc, color="red", ls="--", lw=0.9, label=r"$R_e$")
        ax_ins.set_title(f"Sersic I(R), n={n_ser:.1f}", fontsize=7)
        ax_ins.set_xlabel("R [kpc]", fontsize=7)
        ax_ins.set_ylabel("I/Ie", fontsize=7)
        ax_ins.tick_params(axis="both", which="both", labelsize=7)
        ax_ins.legend(fontsize=6, loc="lower left")

    # ---- Lower panel: fractional residuals ----
    mask = rho_piemd > 0
    r_m = r[mask]
    rho_p = rho_piemd[mask]

    if ok_fix and rho_total_fix is not None:
        res_fix = (rho_total_fix[mask] - rho_p) / rho_p
        ax_bot.semilogx(r_m, res_fix, color="tab:green", lw=1.5, ls="--",
                        label="fixed")

    if ok_free and rho_total_free is not None:
        res_free = (rho_total_free[mask] - rho_p) / rho_p
        ax_bot.semilogx(r_m, res_free, color="tab:red", lw=1.5, ls="-.",
                        label="free")

    ax_bot.axhline(0, color="black", lw=0.8, ls="-")
    ax_bot.axhline(0.1, color="gray", lw=0.5, ls=":")
    ax_bot.axhline(-0.1, color="gray", lw=0.5, ls=":")
    ax_bot.set_xlabel("r [kpc]")
    ax_bot.set_ylabel(r"$(\rho_{\rm model} - \rho_{\rm PIEMD}) / \rho_{\rm PIEMD}$")
    ax_bot.set_xlim(r[0], r[-1])
    ax_bot.set_ylim(-1, 1)
    ax_bot.legend(fontsize=8, loc="upper right")

    suffix = out_suffix if out_suffix else ""
    out_prof = outdir / f"validation_profile_lens{lens_id:03d}{suffix}.png"
    fig.savefig(out_prof, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _pass(f"Saved density profile plot (lens {lens_id}) to {out_prof}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sanity checks on outputs of fitGalaxies_tnfw_jaffe_dual.py")
    parser.add_argument("--csv", required=True, help="Path to the output CSV file.")
    parser.add_argument("--h5", default=None, help="Path to the output HDF5 file (optional).")
    parser.add_argument("--plot", action="store_true", help="Generate summary diagnostic plots.")
    parser.add_argument("--plot-dir", default="validation_plots", help="Directory for plots.")
    parser.add_argument("--n-examples", type=int, default=1,
                        help="Number of example galaxies to show in posterior and profile plots (default: 1).")
    args = parser.parse_args()

    print(f"Validating: {args.csv}")
    if args.h5:
        print(f"HDF5 file: {args.h5}")

    df = pd.read_csv(args.csv)
    profile = detect_dm_profile(df)

    ok = check_csv_structure(df, profile)
    if not ok:
        print("\nCSV structure check failed — aborting.")
        sys.exit(1)

    check_physical_plausibility(df, profile)
    check_constraints(df, profile)
    check_fix_vs_free_consistency(df, profile)
    check_mstar_mismatch(df)

    if args.h5:
        check_h5(args.h5, df, profile)
        check_mcmc_percentiles(df, args.h5, profile)
    else:
        # Still check internal MCMC consistency if columns exist
        check_mcmc_percentiles(df, None, profile)

    check_profile_reconstruction(df, profile)

    if args.plot:
        make_summary_plots(df, args.h5, args.plot_dir, profile, n_examples=args.n_examples)

    # --- Final summary ---
    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  ✓ Passed: {_n_pass}")
    print(f"  ⚠ Warnings: {_n_warn}")
    print(f"  ✗ Failures: {_n_fail}")
    if _n_fail == 0:
        print(f"\n  All checks passed! ✓")
    else:
        print(f"\n  {_n_fail} check(s) FAILED — review output above.")
    print()

    sys.exit(1 if _n_fail > 0 else 0)


if __name__ == "__main__":
    main()
