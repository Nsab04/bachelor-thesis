#!/usr/bin/env python
"""
Sanity checks on the outputs of fitGalaxies_tnfw_jaffe_dual.py.

Validates:
  1. CSV structure and column completeness
  2. Physical plausibility of fit parameters
  3. Constraint enforcement (Jaffe <= PIEMD, tNFW > Jaffe at r > Re)
  4. Consistency between fixed-Jaffe and free-Jaffe fits
  5. Stellar mass mismatch columns
  6. HDF5 structure, shapes, and NaN fractions (if present)
  7. MCMC percentile consistency (if present)
  8. Density profile reconstruction: tNFW+Jaffe vs PIEMD residuals

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
# 1. CSV structure
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "lens_index", "cat_index", "ra_lens", "dec_lens", "z_lens",
    "F160W", "ReF160W_kpc", "v_disp_sigma0_km_s",
    "core_radius_kpc", "cut_radius_kpc",
    "Mstar_from_F160W_Msun", "negative_center_fixedJaffe",
    "rs_fix_kpc", "rt_fix_kpc", "rho0_fix_msun_kpc3",
    "fit_fix_rmse_log", "fit_fix_r2_log", "fit_fix_npts",
    "fit_fix_success", "fit_fix_message", "fit_fix_jaffe_clamped",
    "c200_fix", "R200_fix_kpc", "M200_fix_Msun", "Mtot_tNFW_fix_Msun",
    "rs_free_kpc", "rt_free_kpc", "rho0_free_msun_kpc3",
    "Mstar_free_Msun",
    "fit_free_rmse_log", "fit_free_r2_log", "fit_free_npts",
    "fit_free_success", "fit_free_message",
    "c200_free", "R200_free_kpc", "M200_free_Msun", "Mtot_tNFW_free_Msun",
    "Mstar_max_constraint_Msun",
    "Mstar_ratio_free_over_phot",
    "Mstar_log_ratio_free_over_phot",
    "Mstar_ratio_phot_over_max",
]

MCMC_COLUMNS = [
    "mcmc_nsamp_total", "mcmc_nsamp_ok_fix", "mcmc_nsamp_ok_free",
    "rs_fix_p16", "rs_fix_p50", "rs_fix_p84",
    "rt_fix_p16", "rt_fix_p50", "rt_fix_p84",
    "rho0_fix_p16", "rho0_fix_p50", "rho0_fix_p84",
    "rs_free_p16", "rs_free_p50", "rs_free_p84",
    "rt_free_p16", "rt_free_p50", "rt_free_p84",
    "rho0_free_p16", "rho0_free_p50", "rho0_free_p84",
    "Mstar_free_p16", "Mstar_free_p50", "Mstar_free_p84",
    "c200_fix_p16", "c200_fix_p50", "c200_fix_p84",
    "M200_fix_p16", "M200_fix_p50", "M200_fix_p84",
    "c200_free_p16", "c200_free_p50", "c200_free_p84",
    "M200_free_p16", "M200_free_p50", "M200_free_p84",
]


def check_csv_structure(df):
    _section("1. CSV structure & column completeness")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        _fail(f"Missing required columns: {missing}")
    else:
        _pass(f"All {len(REQUIRED_COLUMNS)} required columns present")

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
        mcmc_missing = [c for c in MCMC_COLUMNS if c not in df.columns]
        if mcmc_missing:
            _fail(f"Missing MCMC columns: {mcmc_missing}")
        else:
            _pass(f"All {len(MCMC_COLUMNS)} MCMC columns present")
    else:
        print("  (no MCMC columns found — skipping MCMC checks in CSV)")

    return True


# ---------------------------------------------------------------------------
# 2. Physical plausibility
# ---------------------------------------------------------------------------
def check_physical_plausibility(df):
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
    bad_rs = ((fix_valid["rs_fix_kpc"] <= 0) | (fix_valid["rs_fix_kpc"] > 1e4)).sum()
    bad_rt = ((fix_valid["rt_fix_kpc"] <= 0) | (fix_valid["rt_fix_kpc"] > 1e5)).sum()
    bad_rho = ((fix_valid["rho0_fix_msun_kpc3"] <= 0)).sum()
    if bad_rs:
        _warn(f"{bad_rs} converged fixed fits with implausible rs (<=0 or >10^4 kpc)")
    else:
        _pass("All converged fixed-fit rs in plausible range")
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

    # --- NFW-derived quantities ---
    for mode, ok_mask in [("fix", ok_fix), ("free", ok_free)]:
        sub = df.loc[ok_mask]
        c_col = f"c200_{mode}"
        m_col = f"M200_{mode}_Msun"
        mt_col = f"Mtot_tNFW_{mode}_Msun"
        r_col = f"R200_{mode}_kpc"

        c_valid = sub[c_col][np.isfinite(sub[c_col])]
        m_valid = sub[m_col][np.isfinite(sub[m_col])]

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

        # Mtot should be >= M200 (total mass includes everything beyond R200)
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
def check_constraints(df):
    _section("3. Constraint enforcement")

    ok = df["fit_fix_success"].astype(bool) & df["fit_free_success"].astype(bool)
    n_ok = ok.sum()
    if n_ok == 0:
        _warn("No galaxies with both fits converged — skipping constraint checks")
        return

    r = np.logspace(-2, 3, 500)

    n_jaffe_exceeds_piemd = 0
    n_tnfw_below_jaffe_outer = 0
    n_checked = 0

    for _, row in df.loc[ok].iterrows():
        re = row["ReF160W_kpc"]
        rho_piemd = piemd_rho(r, row["core_radius_kpc"], row["cut_radius_kpc"],
                              row["v_disp_sigma0_km_s"])
        # Fixed fit: Jaffe should not exceed PIEMD
        mstar_mag = row["Mstar_from_F160W_Msun"]
        rho_jaffe = jaffe_rho(r, mstar_mag, re)
        # After clamping, the effective Jaffe used is min(jaffe, piemd)
        # So DM = piemd - min(jaffe, piemd) >= 0 by construction.
        # But check that the *original* unclamped Jaffe vs PIEMD:
        frac_exceed = np.sum(rho_jaffe > rho_piemd * 1.001) / len(r)
        if frac_exceed > 0.3:
            n_jaffe_exceeds_piemd += 1

        # Free fit: at r > Re, tNFW should exceed Jaffe
        if np.isfinite(row["rs_free_kpc"]) and np.isfinite(row["Mstar_free_Msun"]):
            rho_tnfw_free = trunc_nfw_rho(r, row["rs_free_kpc"], row["rt_free_kpc"],
                                          row["rho0_free_msun_kpc3"])
            rho_jaffe_free = jaffe_rho(r, row["Mstar_free_Msun"], re)
            outer = r > re
            if np.any(outer):
                violation = np.sum(rho_jaffe_free[outer] > rho_tnfw_free[outer] * 1.01)
                if violation > 0.1 * np.sum(outer):
                    n_tnfw_below_jaffe_outer += 1
        n_checked += 1

    if n_jaffe_exceeds_piemd:
        _warn(f"{n_jaffe_exceeds_piemd}/{n_checked} galaxies where unclamped Jaffe exceeds PIEMD "
              f"over >30% of radial range (clamping expected)")
    else:
        _pass("Unclamped Jaffe does not dominate PIEMD for any galaxy")

    clamped_count = df.loc[ok, "fit_fix_jaffe_clamped"].sum()
    _pass(f"Jaffe clamping applied in {clamped_count}/{n_checked} galaxies")

    if n_tnfw_below_jaffe_outer:
        _warn(f"{n_tnfw_below_jaffe_outer}/{n_checked} free fits where tNFW < Jaffe at r > Re "
              f"over >10% of outer points (penalty may not be strong enough)")
    else:
        _pass("tNFW dominates Jaffe at r > Re in all free fits (constraint satisfied)")


# ---------------------------------------------------------------------------
# 4. Consistency between fixed and free fits
# ---------------------------------------------------------------------------
def check_fix_vs_free_consistency(df):
    _section("4. Fixed vs free fit consistency")

    ok = df["fit_fix_success"].astype(bool) & df["fit_free_success"].astype(bool)
    both = df.loc[ok].copy()
    n_both = len(both)
    if n_both == 0:
        _warn("No galaxies with both fits converged")
        return

    # rs should be broadly correlated
    rs_ratio = both["rs_free_kpc"] / both["rs_fix_kpc"]
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
def check_h5(h5_path, df):
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

        if fix_shape != (n_gal, n_samp, 3):
            _fail(f"fix_params shape {fix_shape} != expected ({n_gal}, {n_samp}, 3)")
        else:
            _pass(f"fix_params shape correct: {fix_shape}")

        if free_shape != (n_gal, n_samp, 4):
            _fail(f"free_params shape {free_shape} != expected ({n_gal}, {n_samp}, 4)")
        else:
            _pass(f"free_params shape correct: {free_shape}")

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

        # Spot-check: all valid samples should have rs > 0, rt > rs, rho0 > 0
        fix_data = f["fix_params"][:]
        valid_fix = np.isfinite(fix_data[:, :, 0])
        if valid_fix.sum() > 0:
            rs_valid = fix_data[:, :, 0][valid_fix]
            rt_valid = fix_data[:, :, 1][valid_fix]
            rho0_valid = fix_data[:, :, 2][valid_fix]
            bad_rs = (rs_valid <= 0).sum()
            bad_rt = (rt_valid <= rs_valid).sum()
            bad_rho = (rho0_valid <= 0).sum()
            if bad_rs:
                _fail(f"{bad_rs} fix_params samples with rs <= 0")
            else:
                _pass("All valid fix_params samples have rs > 0")
            if bad_rt:
                _warn(f"{bad_rt} fix_params samples with rt <= rs")
            else:
                _pass("All valid fix_params samples have rt > rs")
            if bad_rho:
                _fail(f"{bad_rho} fix_params samples with rho0 <= 0")
            else:
                _pass("All valid fix_params samples have rho0 > 0")

        # Metadata attributes
        req_attrs = ["bayesfile", "parfile", "catalog", "mref_F160W", "alpha", "beta_cut"]
        missing_attrs = [a for a in req_attrs if a not in f.attrs]
        if missing_attrs:
            _warn(f"Missing HDF5 attributes: {missing_attrs}")
        else:
            _pass(f"All metadata attributes present: {list(f.attrs.keys())}")


# ---------------------------------------------------------------------------
# 7. MCMC percentile consistency
# ---------------------------------------------------------------------------
def check_mcmc_percentiles(df, h5_path):
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
        for prefix in ["rs_fix", "rt_fix", "rho0_fix", "rs_free", "rt_free", "rho0_free", "Mstar_free",
                        "c200_fix", "M200_fix", "c200_free", "M200_free"]:
            p16 = df[f"{prefix}_p16"]
            p50 = df[f"{prefix}_p50"]
            p84 = df[f"{prefix}_p84"]
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
        fix_data = f["fix_params"][:]    # (Ngal, Nsamp, 3)
        free_data = f["free_params"][:]  # (Ngal, Nsamp, 4)

    n_gal = min(len(df), fix_data.shape[0])
    max_err_all = 0.0
    n_checked = 0

    for igal in range(n_gal):
        row = df.iloc[igal]
        # Fixed fit percentiles
        for ic, pname in enumerate(["rs_fix", "rt_fix", "rho0_fix"]):
            vals = fix_data[igal, :, ic]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue
            p16_h5, p50_h5, p84_h5 = np.percentile(vals, [16, 50, 84])
            p16_csv = row.get(f"{pname}_p16", np.nan)
            p50_csv = row.get(f"{pname}_p50", np.nan)
            if np.isfinite(p50_csv) and np.isfinite(p50_h5) and p50_h5 > 0:
                err = abs(p50_csv - p50_h5) / p50_h5
                max_err_all = max(max_err_all, err)
                n_checked += 1

        # Free fit percentiles
        for ic, pname in enumerate(["rs_free", "rt_free", "rho0_free", "Mstar_free"]):
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
def check_profile_reconstruction(df, n_sample=5):
    _section("8. Density profile reconstruction (spot check)")

    ok = df["fit_fix_success"].astype(bool) & df["fit_free_success"].astype(bool)
    sub = df.loc[ok]
    if len(sub) == 0:
        _warn("No galaxies with both fits converged — skipping reconstruction")
        return

    indices = sub.index[:n_sample]
    r = np.logspace(-1, 2.5, 300)

    for idx in indices:
        row = sub.loc[idx]
        rho_piemd = piemd_rho(r, row["core_radius_kpc"], row["cut_radius_kpc"],
                              row["v_disp_sigma0_km_s"])

        # Fixed fit reconstruction
        rho_tnfw_fix = trunc_nfw_rho(r, row["rs_fix_kpc"], row["rt_fix_kpc"],
                                     row["rho0_fix_msun_kpc3"])
        rho_jaffe_fix = jaffe_rho(r, row["Mstar_from_F160W_Msun"], row["ReF160W_kpc"])
        rho_jaffe_clamped = np.minimum(rho_jaffe_fix, rho_piemd)
        rho_total_fix = rho_tnfw_fix + rho_jaffe_clamped

        # Relative residual
        mask = rho_piemd > 0
        rel_res_fix = np.abs(rho_total_fix[mask] - rho_piemd[mask]) / rho_piemd[mask]
        med_res_fix = np.median(rel_res_fix)

        # Free fit reconstruction
        rho_tnfw_free = trunc_nfw_rho(r, row["rs_free_kpc"], row["rt_free_kpc"],
                                      row["rho0_free_msun_kpc3"])
        rho_jaffe_free = jaffe_rho(r, row["Mstar_free_Msun"], row["ReF160W_kpc"])
        rho_total_free = rho_tnfw_free + rho_jaffe_free
        rel_res_free = np.abs(rho_total_free[mask] - rho_piemd[mask]) / rho_piemd[mask]
        med_res_free = np.median(rel_res_free)

        lens_id = int(row["lens_index"])
        status = "✓" if max(med_res_fix, med_res_free) < 0.5 else "⚠"
        print(f"  {status} Lens {lens_id}: median |Δρ/ρ| fix={med_res_fix:.3f}, free={med_res_free:.3f}")


# ---------------------------------------------------------------------------
# 9. Summary plot (optional)
# ---------------------------------------------------------------------------
def make_summary_plots(df, h5_path, outdir, n_examples=1):
    _section("9. Summary diagnostic plots")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ok_fix = df["fit_fix_success"].astype(bool)
    ok_free = df["fit_free_success"].astype(bool)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # (0,0) RMSE distribution
    ax = axes[0, 0]
    if ok_fix.sum():
        ax.hist(df.loc[ok_fix, "fit_fix_rmse_log"], bins=30, alpha=0.6, label="fixed", color="tab:blue")
    if ok_free.sum():
        ax.hist(df.loc[ok_free, "fit_free_rmse_log"], bins=30, alpha=0.6, label="free", color="tab:orange")
    ax.set_xlabel("RMSE (log)")
    ax.set_ylabel("N galaxies")
    ax.set_title("Fit quality: RMSE distribution")
    ax.legend()

    # (0,1) rs distribution
    ax = axes[0, 1]
    if ok_fix.sum():
        ax.hist(np.log10(df.loc[ok_fix, "rs_fix_kpc"].clip(1e-3)), bins=30, alpha=0.6, label="rs fix", color="tab:blue")
    if ok_free.sum():
        ax.hist(np.log10(df.loc[ok_free, "rs_free_kpc"].clip(1e-3)), bins=30, alpha=0.6, label="rs free", color="tab:orange")
    ax.set_xlabel("log10(rs / kpc)")
    ax.set_ylabel("N galaxies")
    ax.set_title("Scale radius distribution")
    ax.legend()

    # (0,2) rt/rs ratio
    ax = axes[0, 2]
    if ok_fix.sum():
        ratio_fix = df.loc[ok_fix, "rt_fix_kpc"] / df.loc[ok_fix, "rs_fix_kpc"]
        ax.hist(np.log10(ratio_fix.clip(0.1)), bins=30, alpha=0.6, label="fix", color="tab:blue")
    if ok_free.sum():
        ratio_free = df.loc[ok_free, "rt_free_kpc"] / df.loc[ok_free, "rs_free_kpc"]
        ax.hist(np.log10(ratio_free.clip(0.1)), bins=30, alpha=0.6, label="free", color="tab:orange")
    ax.set_xlabel("log10(rt/rs)")
    ax.set_ylabel("N galaxies")
    ax.set_title("Truncation ratio rt/rs")
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
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for ax, prefix, label in zip(
            axes2,
            ["rs_free", "rt_free", "Mstar_free"],
            [r"$r_s$", r"$r_t$", r"$M_\star$"],
        ):
            p16 = df.loc[ok_free, f"{prefix}_p16"]
            p50 = df.loc[ok_free, f"{prefix}_p50"]
            p84 = df.loc[ok_free, f"{prefix}_p84"]
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

    # --- Concentration–mass relation scatter plot ---
    _plot_concentration_mass(df, outdir, plt)

    # --- Single-galaxy posterior distributions ---
    has_mcmc = "mcmc_nsamp_total" in df.columns
    if has_mcmc and h5_path is not None and Path(h5_path).exists():
        _plot_single_galaxy_posteriors(df, h5_path, outdir, plt, n_examples=n_examples)


def _plot_concentration_mass(df, outdir, plt):
    """Scatter plot of NFW concentration vs M200 for both fit modes, with MCMC error bars."""
    has_mcmc = "c200_free_p50" in df.columns

    for mode, color, marker in [("free", "tab:blue", "o"), ("fix", "tab:orange", "s")]:
        c_col = f"c200_{mode}"
        m_col = f"M200_{mode}_Msun"
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

        # Plot error bars from MCMC percentiles if available
        if has_mcmc:
            c_p16 = sub.get(f"c200_{mode}_p16", pd.Series(dtype=float))
            c_p50 = sub.get(f"c200_{mode}_p50", pd.Series(dtype=float))
            c_p84 = sub.get(f"c200_{mode}_p84", pd.Series(dtype=float))
            m_p16 = sub.get(f"M200_{mode}_p16", pd.Series(dtype=float))
            m_p50 = sub.get(f"M200_{mode}_p50", pd.Series(dtype=float))
            m_p84 = sub.get(f"M200_{mode}_p84", pd.Series(dtype=float))

            has_err = (
                np.isfinite(c_p16) & np.isfinite(c_p84)
                & np.isfinite(m_p16) & np.isfinite(m_p84)
                & (c_p16 > 0) & (m_p16 > 0) & (m_p84 > 0)
            )

            if has_err.sum() > 0:
                # Galaxies with MCMC error bars
                se = sub.loc[has_err]
                log_m_e = np.log10(se[m_col].values)
                c_e = se[c_col].values

                # Asymmetric error bars in log10(M200) space
                log_m_lo = log_m_e - np.log10(se[f"M200_{mode}_p16"].values)
                log_m_hi = np.log10(se[f"M200_{mode}_p84"].values) - log_m_e
                xerr = np.array([log_m_lo, log_m_hi])
                xerr = np.clip(xerr, 0, None)

                # Asymmetric error bars in c200 space
                c_lo = c_e - se[f"c200_{mode}_p16"].values
                c_hi = se[f"c200_{mode}_p84"].values - c_e
                yerr = np.array([c_lo, c_hi])
                yerr = np.clip(yerr, 0, None)

                ax_cm.errorbar(
                    log_m_e, c_e, xerr=xerr, yerr=yerr,
                    fmt="none", ecolor=color, elinewidth=0.5, alpha=0.3,
                    capsize=0, zorder=1,
                )

                # Galaxies without error bars
                no_err = ~has_err
                if no_err.sum() > 0:
                    sn = sub.loc[no_err]
                    ax_cm.scatter(
                        np.log10(sn[m_col].values), sn[c_col].values,
                        s=15, c=color, marker=marker, alpha=0.4,
                        edgecolors="none", zorder=2, label=f"{mode} (no MCMC)",
                    )

                ax_cm.scatter(
                    log_m_e, c_e,
                    s=20, c=color, marker=marker, alpha=0.7,
                    edgecolors="none", zorder=3,
                    label=f"{mode} ({has_err.sum()} gal, 68% CI)",
                )
            else:
                ax_cm.scatter(
                    log_m, c_val, s=15, c=color, marker=marker, alpha=0.6,
                    edgecolors="none", label=f"{mode} (best fit)",
                )
        else:
            ax_cm.scatter(
                log_m, c_val, s=15, c=color, marker=marker, alpha=0.6,
                edgecolors="none", label=f"{mode} (best fit)",
            )

        ax_cm.set_xlabel(r"$\log_{10}(M_{200} / M_\odot)$")
        ax_cm.set_ylabel(r"$c_{200}$")
        ax_cm.set_title(f"Concentration–mass relation ({mode} fit)")
        ax_cm.legend(fontsize=9)
        ax_cm.set_yscale("log")

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
        c_col = f"c200_{mode}"
        m_col = f"M200_{mode}_Msun"
        ok_col = "fit_free_success" if mode == "free" else "fit_fix_success"
        ok = df[ok_col].astype(bool)
        sub = df.loc[ok]
        valid = (
            np.isfinite(sub[c_col]) & (sub[c_col] > 0)
            & np.isfinite(sub[m_col]) & (sub[m_col] > 0)
        )
        sub = sub.loc[valid]
        if len(sub) == 0:
            continue
        ax_b.scatter(
            np.log10(sub[m_col].values), sub[c_col].values,
            s=12, c=color, marker=marker, alpha=0.5,
            edgecolors="none", label=f"{label} ({len(sub)})",
        )
    ax_b.set_xlabel(r"$\log_{10}(M_{200} / M_\odot)$")
    ax_b.set_ylabel(r"$c_{200}$")
    ax_b.set_title("Concentration–mass relation (both modes)")
    ax_b.set_yscale("log")
    ax_b.legend(fontsize=9)
    out_both = outdir / "validation_c200_vs_M200_combined.png"
    fig_both.savefig(out_both, dpi=150, bbox_inches="tight")
    plt.close(fig_both)
    _pass(f"Saved combined c200–M200 plot to {out_both}")


def _plot_single_galaxy_posteriors(df, h5_path, outdir, plt, n_examples=1):
    """Plot MCMC posterior distributions of M*, c200, rt and density profiles
    for the top n_examples galaxies (ranked by number of valid MCMC free-fit samples)."""
    from scipy.optimize import brentq

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

    # Cosmology for c200 computation (read from HDF5 attrs if available)
    from astropy.cosmology import FlatLambdaCDM
    try:
        with h5py.File(h5_path, "r") as f:
            H0 = f.attrs.get("H0", 70.0)
            Om0 = f.attrs.get("Om0", 0.3)
    except Exception:
        H0, Om0 = 70.0, 0.3
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    def _c200_scalar(rho_s, rho_ref):
        S = rho_s / rho_ref
        def f(c):
            return (200.0 / 3.0) * c**3 / (np.log1p(c) - c / (1.0 + c)) - S
        try:
            return brentq(f, 1e-6, 1e5)
        except (ValueError, RuntimeError):
            return np.nan

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
            free_samp = f["free_params"][igal, :, :]  # (Nsamp, 4): rs, rt, rho0, mstar

            valid_mask = np.isfinite(free_samp[:, 0])
            samp = free_samp[valid_mask]
            if len(samp) < 10:
                _warn(f"Lens {lens_id}: only {len(samp)} valid MCMC samples — skipping")
                continue

            z_lens = float(best_row["z_lens"])
            rho_crit = cosmo.critical_density(z_lens).to_value("Msun/kpc3")

            _plot_one_galaxy_posterior(
                best_row, lens_id, z_lens, samp, rho_crit,
                _c200_scalar, outdir, plt,
            )
            _plot_single_galaxy_profile(best_row, lens_id, z_lens, outdir, plt)


def _plot_one_galaxy_posterior(row, lens_id, z_lens, samp, rho_crit,
                               c200_func, outdir, plt):
    """Posterior histograms of M*, c200, rt for one galaxy."""
    rs_samp = samp[:, 0]
    rt_samp = samp[:, 1]
    rho0_samp = samp[:, 2]
    mstar_samp = samp[:, 3]

    c200_samp = np.array([c200_func(rho, rho_crit) for rho in rho0_samp])

    # Best-fit values from the CSV
    bf_mstar = row.get("Mstar_free_Msun", np.nan)
    bf_c200 = row.get("c200_free", np.nan)
    bf_rt = row.get("rt_free_kpc", np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)

    panels = [
        (mstar_samp, bf_mstar, r"$M_\star$ [M$_\odot$]", "Stellar mass", True),
        (c200_samp, bf_c200, r"$c_{200}$", "NFW concentration", False),
        (rt_samp, bf_rt, r"$r_t$ [kpc]", "Truncation radius", False),
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



def _plot_single_galaxy_profile(row, lens_id, z_lens, outdir, plt):
    """Density profile plot for one galaxy: PIEMD vs tNFW+Jaffe (fixed & free).

    Upper panel: density profiles on log-log scale.
    Lower panel: fractional residuals (model - PIEMD) / PIEMD.
    """
    r = np.logspace(-2, 3, 500)

    # --- PIEMD total density ---
    r_core = row["core_radius_kpc"]
    r_cut = row["cut_radius_kpc"]
    sigma0 = row["v_disp_sigma0_km_s"]
    rho_piemd = piemd_rho(r, r_core, r_cut, sigma0)

    # --- Fixed-Jaffe mode ---
    rs_fix = row["rs_fix_kpc"]
    rt_fix = row["rt_fix_kpc"]
    rho0_fix = row["rho0_fix_msun_kpc3"]
    mstar_phot = row["Mstar_from_F160W_Msun"]
    re_kpc = row["ReF160W_kpc"]

    ok_fix = np.isfinite(rs_fix) and np.isfinite(rt_fix) and np.isfinite(rho0_fix)
    rho_tnfw_fix = rho_jaffe_fix_clamped = rho_total_fix = None
    if ok_fix:
        rho_tnfw_fix = trunc_nfw_rho(r, rs_fix, rt_fix, rho0_fix)
        rho_jaffe_fix = jaffe_rho(r, mstar_phot, re_kpc)
        # Jaffe is clamped to not exceed PIEMD in the fitting code
        rho_jaffe_fix_clamped = np.minimum(rho_jaffe_fix, rho_piemd)
        rho_total_fix = rho_tnfw_fix + rho_jaffe_fix_clamped

    # --- Free-Jaffe mode ---
    rs_free = row["rs_free_kpc"]
    rt_free = row["rt_free_kpc"]
    rho0_free = row["rho0_free_msun_kpc3"]
    mstar_free = row["Mstar_free_Msun"]

    ok_free = (np.isfinite(rs_free) and np.isfinite(rt_free)
               and np.isfinite(rho0_free) and np.isfinite(mstar_free))
    rho_tnfw_free = rho_jaffe_free = rho_total_free = None
    if ok_free:
        rho_tnfw_free = trunc_nfw_rho(r, rs_free, rt_free, rho0_free)
        rho_jaffe_free = jaffe_rho(r, mstar_free, re_kpc)
        rho_total_free = rho_tnfw_free + rho_jaffe_free

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

    if ok_fix:
        ax_top.loglog(r, rho_tnfw_fix, color="tab:orange", lw=1.0, ls=":",
                      label=f"tNFW (fixed), rs={rs_fix:.1f} kpc")
        ax_top.loglog(r, rho_jaffe_fix_clamped, color="tab:cyan", lw=1.0, ls=":",
                      label=f"Jaffe (phot), M*={mstar_phot:.2e}")
        ax_top.loglog(r, rho_total_fix, color="tab:green", lw=1.8, ls="--",
                      label="tNFW + Jaffe (fixed)")

    if ok_free:
        ax_top.loglog(r, rho_tnfw_free, color="tab:purple", lw=1.0, ls=":",
                      label=f"tNFW (free), rs={rs_free:.1f} kpc")
        ax_top.loglog(r, rho_jaffe_free, color="tab:pink", lw=1.0, ls=":",
                      label=f"Jaffe (free), M*={mstar_free:.2e}")
        ax_top.loglog(r, rho_total_free, color="tab:red", lw=1.8, ls="-.",
                      label="tNFW + Jaffe (free)")

    ax_top.axvline(re_kpc, color="gray", lw=0.8, ls=":", alpha=0.7,
                   label=f"Re = {re_kpc:.2f} kpc")

    ax_top.set_ylabel(r"$\rho(r)$ [M$_\odot$ / kpc$^3$]")
    ax_top.set_title(
        f"Lens {lens_id}  (z={z_lens:.3f}, F160W={row.get('F160W', np.nan):.2f}, "
        f"$\\sigma_0$={sigma0:.0f} km/s)",
        fontsize=10,
    )
    ax_top.legend(fontsize=7, loc="upper right", ncol=2)
    ax_top.set_xlim(r[0], r[-1])

    # ---- Lower panel: fractional residuals ----
    mask = rho_piemd > 0
    r_m = r[mask]
    rho_p = rho_piemd[mask]

    if ok_fix:
        res_fix = (rho_total_fix[mask] - rho_p) / rho_p
        ax_bot.semilogx(r_m, res_fix, color="tab:green", lw=1.5, ls="--",
                        label="fixed")

    if ok_free:
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

    out_prof = outdir / f"validation_profile_lens{lens_id:03d}.png"
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

    ok = check_csv_structure(df)
    if not ok:
        print("\nCSV structure check failed — aborting.")
        sys.exit(1)

    check_physical_plausibility(df)
    check_constraints(df)
    check_fix_vs_free_consistency(df)
    check_mstar_mismatch(df)

    if args.h5:
        check_h5(args.h5, df)
        check_mcmc_percentiles(df, args.h5)
    else:
        # Still check internal MCMC consistency if columns exist
        check_mcmc_percentiles(df, None)

    check_profile_reconstruction(df)

    if args.plot:
        make_summary_plots(df, args.h5, args.plot_dir, n_examples=args.n_examples)

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

