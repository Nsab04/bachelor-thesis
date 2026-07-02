"""
Fit LensTool PIEMD galaxy density profiles with two models:

1) Fixed-stellar model:
      rho_tot(r) = rho_DM(r) + rho_Jaffe(r; M*=M*(F160W), Re=ReF160W)

2) Free-stellar model:
      rho_tot(r) = rho_DM(r) + rho_Jaffe(r; M*=M*_free, rJ=rJ_free)
   where both free stellar mass and free Jaffe scale radius are fitted.

The DM profile is either a truncated NFW (default), a BMO smoothly truncated
NFW with n=0.5 (activated with ``--use-bmo``), a standard (untruncated)
NFW (activated with ``--use-nfw``), an Einasto profile (activated with
``--use-einasto``), or a generalized NFW profile with free inner slope
(activated with ``--use-gnfw``).

Outputs a single CSV with both fit modes per matched galaxy.
"""

from __future__ import annotations

import argparse
import io
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares
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
JAFFE_Re_OVER_rJ = 1.0  # keep same convention used in fitGalaxies_mcmc_h5.py

DEFAULT_PAR = "M0416.par"#"/Users/maxmen3/stiva/pietro_models/M0416_B22.par"
DEFAULT_CATALOG = "/Users/maxmen3/projects/Pierpaoli/m0416.dat"
DEFAULT_README = "/Users/maxmen3/projects/Pierpaoli/ReadMe.txt"

# --- Scaling relations for cluster member galaxies (Bergamini et al.) ---
DEFAULT_MREF_F160W = 17.02
DEFAULT_ALPHA = 0.30
DEFAULT_BETA_CUT = 0.60


def load_bayes_scaling_params(bayes_path):
    """Load LensTool MCMC sample file (bayes_*.dat) and extract sigma_ref, rcut_ref chains.

    Returns
    -------
    sigma_ref_LT : ndarray, shape (nsamples,)
    rcut_ref_arcsec : ndarray, shape (nsamples,)
    """
    with open(bayes_path, "r", encoding="utf-8", errors="replace") as f:
        header_lines, data_lines = [], []
        for line in f:
            if line.strip().startswith("#"):
                header_lines.append(line.strip())
            elif line.strip():
                data_lines.append(line)
    if not header_lines:
        raise ValueError(f"No header lines found in {bayes_path}.")
    colnames = [h.lstrip("#").strip() for h in header_lines]
    try:
        i_sig = colnames.index("Pot0 sigma (km/s)")
        i_rcut = colnames.index("Pot0 rcut (arcsec)")
    except ValueError as e:
        raise ValueError(
            "Could not find 'Pot0 sigma (km/s)' and 'Pot0 rcut (arcsec)' in bayes file headers.\n"
            f"Available: {colnames}"
        ) from e
    data = np.loadtxt(io.StringIO("".join(data_lines)))
    if data.ndim == 1:
        data = data[None, :]
    return data[:, i_sig], data[:, i_rcut]


def luminosity_ratio_from_mag(m, mref=DEFAULT_MREF_F160W):
    """L/Lref = 10^{-0.4(m - mref)}."""
    return 10.0 ** (-0.4 * (m - mref))


def corrcoef_from_samples(samples_2d):
    """Return correlation matrix and number of valid rows for a (Nsamp, Npar) array.

    Rows containing non-finite values are discarded. If fewer than 2 valid rows remain,
    the returned matrix is filled with NaNs.
    """
    arr = np.asarray(samples_2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    npar = arr.shape[1]
    valid = np.all(np.isfinite(arr), axis=1)
    arr_valid = arr[valid]
    n_valid = int(arr_valid.shape[0])
    corr = np.full((npar, npar), np.nan, dtype=float)
    if n_valid < 2:
        return corr, n_valid
    with np.errstate(invalid="ignore", divide="ignore"):
        corr[:, :] = np.corrcoef(arr_valid, rowvar=False)
    return corr, n_valid


def galaxy_dpie_from_scaling(m_f160w, sigma_ref_lt, rcut_ref_arcsec, cosmo, zl,
                             alpha=DEFAULT_ALPHA, beta_cut=DEFAULT_BETA_CUT, mref=DEFAULT_MREF_F160W):
    """Compute PIEMD sigma0 (km/s) and rcut (kpc) from scaling relations + one MCMC sample."""
    lr = luminosity_ratio_from_mag(m_f160w, mref=mref)
    sigma_lt = sigma_ref_lt * (lr ** alpha)
    sigma0 = float(sigma_lt) * np.sqrt(3.0 / 2.0)
    kpc_per_arcsec = cosmo.angular_diameter_distance(zl).to_value("kpc") * (np.pi / 648000.0)
    rcut_kpc = float(rcut_ref_arcsec) * (lr ** beta_cut) * kpc_per_arcsec
    return sigma0, rcut_kpc


def _parse_vizier_bytes_table(readme_text: str, data_basename: str):
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


def stellar_mass_to_light(mag_f160w):
    """
    Grillo+2015 relation used in existing pipeline.
    """
    logm = 18.541 - 0.416 * float(mag_f160w)
    return 10.0 ** logm


def jaffe_scale_radius(re):
    return float(re) / JAFFE_Re_OVER_rJ


def jaffe_rho(r, mstar, re):
    r = np.asarray(r, dtype=float)
    rj = jaffe_scale_radius(re)
    rr = np.maximum(r, np.finfo(float).eps)
    return (float(mstar) * rj) / (4.0 * np.pi * rr**2 * (rr + rj) ** 2)


def jaffe_rho_rj(r, mstar, rj):
    """Jaffe density with explicit scale radius rJ (not derived from Re)."""
    r = np.asarray(r, dtype=float)
    rr = np.maximum(r, np.finfo(float).eps)
    rj = float(rj)
    return (float(mstar) * rj) / (4.0 * np.pi * rr**2 * (rr + rj) ** 2)


def piemd_rho(r, r_core, r_cut, sigma_v):
    r = np.asarray(r, dtype=float)
    return (
        (sigma_v**2)
        / (2.0 * np.pi * G_KPC)
        * (r_cut + r_core)
        / (r_core**2 * r_cut)
        / (1.0 + r**2 / r_core**2)
        / (1.0 + r**2 / r_cut**2)
    )


def trunc_nfw_rho(r, rs, rt, rho0):
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    tau = rt / rs
    trunc = tau**2 / (x**2 + tau**2)
    return trunc * (rho0 / (x * (1.0 + x) ** 2))


def M_tNFW_enclosed(r, rs, rt, rho0):
    """Enclosed mass for the multiplicatively truncated NFW (closed-form).

    Args:
        r: Radius (kpc).
        rs: NFW scale radius (kpc).
        rt: Truncation radius (kpc).
        rho0: NFW scale density (Msun/kpc^3).

    Returns:
        M(<r) in Msun.
    """
    r = np.asanyarray(r, dtype=float)
    rs = np.asanyarray(rs, dtype=float)
    rt = np.asanyarray(rt, dtype=float)
    rho0 = np.asanyarray(rho0, dtype=float)
    x = r / rs
    tau = rt / rs
    tau2 = tau**2
    denom = (tau2 + 1.0) ** 2
    A = tau2 * (tau2 - 1.0) / denom
    B = -tau2 / (tau2 + 1.0)
    D = 2.0 * tau2**2 / denom
    term1 = A * (np.log1p(x) - 0.5 * np.log1p((x / tau) ** 2))
    term2 = B * (x / (1.0 + x))
    term3 = (D / tau) * np.arctan(x / tau)
    F = term1 + term2 + term3
    return 4.0 * np.pi * rho0 * rs**3 * F


def M_tNFW_total(rs, rt, rho0):
    """Total mass (r→∞) of the truncated NFW in closed form."""
    tau = np.asanyarray(rt, float) / np.asanyarray(rs, float)
    tau2 = tau**2
    denom = (tau2 + 1.0) ** 2
    A = tau2 * (tau2 - 1.0) / denom
    B = -tau2 / (tau2 + 1.0)
    D = 2.0 * tau2**2 / denom
    F_inf = A * np.log(tau) + B + (D / tau) * (np.pi / 2.0)
    return 4.0 * np.pi * rho0 * rs**3 * F_inf


# ---------------------------------------------------------------------------
# BMO (Baltz–Marshall–Oguri) smoothly truncated NFW with n=0.5
# ---------------------------------------------------------------------------

def bmo_rho(r, rs, rt, rho0):
    """BMO smoothly truncated NFW density profile (n=0.5).

    rho(r) = rho_0 / [(r/rs)(1 + r/rs)^2] * (rt^2 / (r^2 + rt^2))^(1/2)

    Reference: Baltz, Marshall & Oguri (2009), JCAP, 01, 015.

    Args:
        r: Radius (kpc).
        rs: NFW scale radius (kpc).
        rt: Truncation radius (kpc).
        rho0: NFW characteristic density (Msun/kpc^3).

    Returns:
        rho(r) in Msun/kpc^3.
    """
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    trunc = np.sqrt(rt**2 / (r**2 + rt**2))
    return trunc * (rho0 / (x * (1.0 + x) ** 2))


def M_bmo_enclosed(r, rs, rt, rho0, npts=5000):
    """Enclosed mass for the BMO n=0.5 profile (numerical integration).

    Args:
        r: Radius or array of radii (kpc).
        rs: Scale radius (kpc).
        rt: Truncation radius (kpc).
        rho0: Scale density (Msun/kpc^3).
        npts: Number of integration points.

    Returns:
        M(<r) in Msun (scalar or array matching r).
    """
    r = np.asanyarray(r, dtype=float)
    scalar = r.ndim == 0
    r = np.atleast_1d(r)
    rmax = float(np.max(r))
    rg = np.linspace(0, rmax, npts)
    rhog = bmo_rho(rg, rs, rt, rho0)
    integ = 4.0 * np.pi * rg**2 * rhog
    Mg = np.zeros_like(rg)
    Mg[1:] = np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(rg))
    result = np.interp(r, rg, Mg)
    return float(result) if scalar else result


def compute_bmo_derived(rs, rt, rho0, rho_crit, npts=10000):
    """Compute c200, R200, and M200 from BMO n=0.5 fit parameters.

    Uses numerical integration to find R200 where the mean enclosed
    density equals 200 * rho_crit.

    Args:
        rs: Scale radius (kpc).
        rt: Truncation radius (kpc).
        rho0: Scale density (Msun/kpc^3).
        rho_crit: Critical density at lens redshift (Msun/kpc^3).
        npts: Number of radial grid points for integration.

    Returns:
        dict with keys c200, R200_kpc, M200_Msun.
    """
    result = {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
    if not (np.isfinite(rs) and np.isfinite(rt) and np.isfinite(rho0)
            and rs > 0 and rt > 0 and rho0 > 0):
        return result
    try:
        rmax = max(10.0 * rt, 50.0 * rs, 5000.0)
        rg = np.linspace(1e-4, rmax, npts)
        rhog = bmo_rho(rg, rs, rt, rho0)
        integ = 4.0 * np.pi * rg**2 * rhog
        Mg = np.zeros_like(rg)
        Mg[1:] = np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(rg))
        mean_rho = Mg / ((4.0 / 3.0) * np.pi * rg**3)
        above = mean_rho >= 200.0 * rho_crit
        if np.any(above):
            idx = np.where(above)[0][-1]
            if idx < len(rg) - 1:
                r1, r2 = rg[idx], rg[idx + 1]
                d1 = mean_rho[idx] - 200.0 * rho_crit
                d2 = mean_rho[idx + 1] - 200.0 * rho_crit
                R200 = r1 - d1 * (r2 - r1) / (d2 - d1)
                M200 = float(np.interp(R200, rg, Mg))
                result["c200"] = R200 / rs
                result["R200_kpc"] = float(R200)
                result["M200_Msun"] = M200
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Standard (untruncated) NFW profile
# ---------------------------------------------------------------------------

def nfw_rho(r, rs, rho0):
    """Standard NFW density profile (no truncation).

    rho(r) = rho0 / [(r/rs) * (1 + r/rs)^2]

    Args:
        r: Radius (kpc).
        rs: Scale radius (kpc).
        rho0: Scale density (Msun/kpc^3).

    Returns:
        rho(r) in Msun/kpc^3.
    """
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    return rho0 / (x * (1.0 + x) ** 2)


def M_nfw_enclosed(r, rs, rho0):
    """Enclosed mass for the standard NFW profile (closed-form).

    M(<r) = 4 pi rho0 rs^3 [ln(1 + r/rs) - (r/rs)/(1 + r/rs)]

    Args:
        r: Radius (kpc).
        rs: NFW scale radius (kpc).
        rho0: NFW scale density (Msun/kpc^3).

    Returns:
        M(<r) in Msun.
    """
    r = np.asanyarray(r, dtype=float)
    x = r / rs
    return 4.0 * np.pi * rho0 * rs**3 * (np.log1p(x) - x / (1.0 + x))


def compute_std_nfw_derived(rs, rho0, rho_crit):
    """Compute c200, R200, and M200 from standard NFW fit parameters.

    Args:
        rs: Scale radius (kpc).
        rho0: Scale density (Msun/kpc^3).
        rho_crit: Critical density at lens redshift (Msun/kpc^3).

    Returns:
        dict with keys c200, R200_kpc, M200_Msun.
    """
    result = {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
    if not (np.isfinite(rs) and np.isfinite(rho0) and rs > 0 and rho0 > 0):
        return result
    try:
        c200 = float(c200_from_rs_rhos(rs, rho0, rho_crit))
        result["c200"] = c200
        R200 = c200 * rs
        result["R200_kpc"] = R200
        result["M200_Msun"] = float(M_nfw_enclosed(R200, rs, rho0))
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Einasto profile
# ---------------------------------------------------------------------------

def einasto_rho(r, rs, rho_s, n):
    """Einasto density profile.

    Args:
        r: Radius (kpc).
        rs: Scale radius (kpc) – the radius at which d ln rho / d ln r = -2.
        rho_s: Density at r = rs (Msun/kpc^3).
        n: Einasto index (shape parameter; larger n → shallower inner slope).

    Returns:
        rho(r) in Msun/kpc^3.
    """
    r = np.asarray(r, dtype=float)
    s = np.maximum(r / rs, np.finfo(float).eps)
    alpha = 1.0 / n
    dn = 2.0 * n  # approximate; exact d_n ≈ 2n − 1/3 + ... but 2n is standard
    return rho_s * np.exp(-dn * (s**alpha - 1.0))


def M_einasto_enclosed(r, rs, rho_s, n, npts=5000):
    """Enclosed mass for the Einasto profile (numerical integration).

    Args:
        r: Radius or array of radii (kpc).
        rs: Einasto scale radius (kpc).
        rho_s: Density at rs (Msun/kpc^3).
        n: Einasto index.
        npts: Grid points for the numerical quadrature.

    Returns:
        M(<r) in Msun.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    rmax = r.max()
    r_grid = np.linspace(0.0, rmax, npts)
    rho_grid = einasto_rho(r_grid, rs, rho_s, n)
    integrand = 4.0 * np.pi * r_grid**2 * rho_grid
    M_grid = np.zeros_like(r_grid)
    M_grid[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_grid))
    return np.interp(r, r_grid, M_grid)


def compute_einasto_derived(rs, rho_s, n, rho_crit, Delta=200.0, rmax_search=5000.0, npts=10000):
    """Compute c200, R200 and M200 for an Einasto profile.

    Args:
        rs: Scale radius (kpc).
        rho_s: Density at rs (Msun/kpc^3).
        n: Einasto index.
        rho_crit: Critical density at lens redshift (Msun/kpc^3).
        Delta: Overdensity threshold (default 200).
        rmax_search: Maximum radius to search for R200.
        npts: Grid resolution.

    Returns:
        dict with keys c200, R200_kpc, M200_Msun.
    """
    result = {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
    if not (np.isfinite(rs) and np.isfinite(rho_s) and np.isfinite(n)
            and rs > 0 and rho_s > 0 and n > 0):
        return result
    try:
        r_grid = np.linspace(1e-4, rmax_search, npts)
        M_grid = M_einasto_enclosed(r_grid, rs, rho_s, n, npts=npts)
        mean_rho = M_grid / ((4.0 / 3.0) * np.pi * r_grid**3)
        target = Delta * rho_crit
        # Find where mean density drops below target
        above = mean_rho >= target
        if not np.any(above):
            return result
        idx = np.where(above)[0][-1]
        if idx >= len(r_grid) - 1:
            return result
        # Linear interpolation
        r1, r2 = r_grid[idx], r_grid[idx + 1]
        d1, d2 = mean_rho[idx] - target, mean_rho[idx + 1] - target
        R200 = r1 - d1 * (r2 - r1) / (d2 - d1)
        result["R200_kpc"] = float(R200)
        result["M200_Msun"] = float(np.interp(R200, r_grid, M_grid))
        result["c200"] = float(R200 / rs)
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Generalized NFW (gNFW) profile
# ---------------------------------------------------------------------------

def gnfw_rho(r, rs, rho_s, gamma):
    """Generalized NFW density profile.

    rho(r) = rho_s / [(r/rs)^gamma * (1 + r/rs)^(3 - gamma)]

    Standard NFW is recovered for gamma = 1.

    Args:
        r: Radius (kpc).
        rs: Scale radius (kpc).
        rho_s: Scale density (Msun/kpc^3).
        gamma: Inner slope (0 < gamma < 3).

    Returns:
        rho(r) in Msun/kpc^3.
    """
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    return rho_s / (x**gamma * (1.0 + x) ** (3.0 - gamma))


def M_gnfw_enclosed(r, rs, rho_s, gamma, npts=5000):
    """Enclosed mass for the gNFW profile (numerical integration).

    Args:
        r: Radius or array of radii (kpc).
        rs: gNFW scale radius (kpc).
        rho_s: Scale density (Msun/kpc^3).
        gamma: Inner slope.
        npts: Grid points for the numerical quadrature.

    Returns:
        M(<r) in Msun.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    rmax = r.max()
    r_grid = np.linspace(0.0, rmax, npts)
    rho_grid = gnfw_rho(r_grid, rs, rho_s, gamma)
    integrand = 4.0 * np.pi * r_grid**2 * rho_grid
    M_grid = np.zeros_like(r_grid)
    M_grid[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_grid))
    return np.interp(r, r_grid, M_grid)


def compute_gnfw_derived(rs, rho_s, gamma, rho_crit, Delta=200.0,
                          rmax_search=5000.0, npts=10000):
    """Compute c200, R200 and M200 for a gNFW profile.

    Args:
        rs: Scale radius (kpc).
        rho_s: Scale density (Msun/kpc^3).
        gamma: Inner slope.
        rho_crit: Critical density at lens redshift (Msun/kpc^3).
        Delta: Overdensity threshold (default 200).
        rmax_search: Maximum radius to search for R200.
        npts: Grid resolution.

    Returns:
        dict with keys c200, R200_kpc, M200_Msun.
    """
    result = {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
    if not (np.isfinite(rs) and np.isfinite(rho_s) and np.isfinite(gamma)
            and rs > 0 and rho_s > 0 and 0 < gamma < 3):
        return result
    try:
        r_grid = np.linspace(1e-4, rmax_search, npts)
        M_grid = M_gnfw_enclosed(r_grid, rs, rho_s, gamma, npts=npts)
        mean_rho = M_grid / ((4.0 / 3.0) * np.pi * r_grid**3)
        target = Delta * rho_crit
        above = mean_rho >= target
        if not np.any(above):
            return result
        idx = np.where(above)[0][-1]
        if idx >= len(r_grid) - 1:
            return result
        r1, r2 = r_grid[idx], r_grid[idx + 1]
        d1, d2 = mean_rho[idx] - target, mean_rho[idx + 1] - target
        R200 = r1 - d1 * (r2 - r1) / (d2 - d1)
        result["R200_kpc"] = float(R200)
        result["M200_Msun"] = float(np.interp(R200, r_grid, M_grid))
        result["c200"] = float(R200 / rs)
    except Exception:
        pass
    return result


def rdelta_trunc_nfw(rs, rt, rho0, rho_ref, Delta=200.0, grid_size=20000):
    """R_Delta for a truncated NFW profile (default Delta=200).

    Args:
        rs: NFW scale radius.
        rt: Truncation radius.
        rho0: NFW scale density.
        rho_ref: Reference density (e.g. critical density at z), same units as rho0.
        Delta: Overdensity (default 200).
        grid_size: Resolution for the precomputed integral.

    Returns:
        R_Delta in same units as rs.
    """
    tau = rt / rs
    x_min = max(min(1e-7, 0.1 * tau), 1e-12)
    if tau <= x_min:
        x_min = tau * 1e-6
    x = np.logspace(np.log10(x_min), np.log10(tau), grid_size)
    f = (1.0 / (x * (1.0 + x) ** 2)) * (tau**2 / (x**2 + tau**2))
    integrand = x**2 * f
    F = np.zeros_like(x)
    dx = np.diff(x)
    F[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dx)
    mass_norm = 4.0 * np.pi * rho0 * rs**3

    def M_of_R(R):
        xR = np.clip(np.atleast_1d(R / rs), x[0], x[-1])
        return float((mass_norm * np.interp(xR, x, F)).item())

    target_coeff = (4.0 / 3.0) * np.pi * Delta * rho_ref

    def g(R):
        return M_of_R(R) - target_coeff * R**3

    R_low = 1e-8 * rs
    while g(R_low) <= 0:
        R_low *= 0.1
        if R_low < 1e-30 * rs:
            break
    M_t = M_of_R(rt)
    R_est = (M_t / target_coeff) ** (1.0 / 3.0)
    R_high = max(rt, 2.0 * R_est)
    while g(R_high) > 0:
        R_high *= 2.0
        if R_high > 1e6 * max(rs, rt):
            return np.nan
    for _ in range(80):
        R_mid = 0.5 * (R_low + R_high)
        if g(R_mid) > 0:
            R_low = R_mid
        else:
            R_high = R_mid
    return 0.5 * (R_low + R_high)


def c200_from_rs_rhos(rs, rho0, rho_ref):
    """NFW concentration c200 from scale density rho0 and reference density rho_ref.

    Solves: (200/3) * c^3 / [ln(1+c) - c/(1+c)] = rho0 / rho_ref

    Args:
        rs: Scale radius (only used for broadcasting shape).
        rho0: NFW scale density.
        rho_ref: Reference (critical) density, same units as rho0.

    Returns:
        c200 (scalar or array matching input shapes).
    """
    rs = np.asarray(rs, dtype=float)
    rho0 = np.asarray(rho0, dtype=float)
    S = np.broadcast_to(rho0 / float(rho_ref), np.broadcast(rs, rho0).shape)

    def f(c, Sval):
        return (200.0 / 3.0) * c**3 / (np.log1p(c) - c / (1.0 + c)) - Sval

    c_out = np.empty_like(S, dtype=float)
    it = np.nditer(S, flags=["multi_index"])
    while not it.finished:
        try:
            c_out[it.multi_index] = brentq(f, 1e-6, 1e5, args=(float(it[0]),))
        except (ValueError, RuntimeError):
            c_out[it.multi_index] = np.nan
        it.iternext()
    return c_out


def compute_nfw_derived(rs, rt, rho0, rho_crit):
    """Compute c200, R200, and M200 from tNFW fit parameters.

    Args:
        rs: Scale radius (kpc).
        rt: Truncation radius (kpc).
        rho0: Scale density (Msun/kpc^3).
        rho_crit: Critical density at lens redshift (Msun/kpc^3).

    Returns:
        dict with keys c200, R200_kpc, M200_Msun, Mtot_tNFW_Msun.
        Values are NaN if computation fails.
    """
    result = {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan, "Mtot_tNFW_Msun": np.nan}
    if not (np.isfinite(rs) and np.isfinite(rt) and np.isfinite(rho0)
            and rs > 0 and rt > 0 and rho0 > 0):
        return result
    try:
        c200 = float(c200_from_rs_rhos(rs, rho0, rho_crit))
        result["c200"] = c200
    except Exception:
        pass
    try:
        R200 = rdelta_trunc_nfw(rs, rt, rho0, rho_crit, Delta=200.0)
        result["R200_kpc"] = R200
        if np.isfinite(R200):
            result["M200_Msun"] = float(M_tNFW_enclosed(R200, rs, rt, rho0))
    except Exception:
        pass
    try:
        result["Mtot_tNFW_Msun"] = float(M_tNFW_total(rs, rt, rho0))
    except Exception:
        pass
    return result


def _fit_logspace_model(r, y_target, model_fun, theta0, lower, upper, f_scale=0.1, max_nfev=20000):
    valid = np.isfinite(r) & np.isfinite(y_target) & (r > 0.0) & (y_target > 0.0)
    rr = r[valid]
    yy = y_target[valid]
    if rr.size < 20:
        raise ValueError("Not enough valid points to fit.")

    theta0 = np.clip(np.asarray(theta0, dtype=float), np.asarray(lower) + 1.0e-8, np.asarray(upper) - 1.0e-8)

    def residuals(theta):
        ymod = model_fun(rr, theta)
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        return res[np.isfinite(res)]

    sol = least_squares(
        residuals,
        theta0,
        method="trf",
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=f_scale,
        max_nfev=max_nfev,
    )

    ymod = model_fun(rr, sol.x)
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum((np.log(np.maximum(ymod, np.finfo(float).tiny)) - ylog) ** 2) / sst) if sst > 0 else np.nan

    return sol, {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size), "success": bool(sol.success), "message": str(sol.message)}


def fit_tnfw_plus_fixed_jaffe(r, rho_target, rho_jaffe_fixed, rs_guess, rt_guess, rho0_guess, rmin_fit=0.05, rmax_fit=None, re_kpc=None, penalty_weight=100.0):
    r = np.asarray(r, dtype=float)
    y = np.asarray(rho_target, dtype=float)
    rhoj_orig = np.asarray(rho_jaffe_fixed, dtype=float)

    # Clamp the Jaffe profile so it never exceeds the PIEMD total density.
    # This ensures the DM (tNFW) component remains non-negative.
    rhoj = np.minimum(rhoj_orig, y)

    mask = np.isfinite(r) & np.isfinite(y) & np.isfinite(rhoj) & (r >= rmin_fit)
    if rmax_fit is not None:
        mask &= (r <= rmax_fit)
    rr, yy, jj = r[mask], y[mask], rhoj[mask]

    # Identify points beyond Re for the dominance penalty
    if re_kpc is not None and re_kpc > 0:
        outer_mask = rr > re_kpc
    else:
        outer_mask = np.zeros(rr.size, dtype=bool)

    def model_fun(rad, theta):
        rs = np.exp(theta[0])
        rt = rs * (1.0 + np.exp(theta[1]))
        rho0 = np.exp(theta[2])
        return trunc_nfw_rho(rad, rs, rt, rho0) + jj

    theta0 = np.array(
        [
            np.log(max(float(rs_guess), 1.0e-6)),
            np.log(max(float(rt_guess) / max(float(rs_guess), 1.0e-6) - 1.0, 1.0e-6)),
            np.log(max(float(rho0_guess), 1.0e-20)),
        ],
        dtype=float,
    )
    lower = np.array([np.log(max(rmin_fit, 1.0e-4)), np.log(1.0e-6), np.log(1.0e-30)], dtype=float)
    upper = np.array([np.log(max(5.0 * np.nanmax(rr), 1.0)), np.log(1.0e5), np.log(1.0e20)], dtype=float)

    theta0 = np.clip(theta0, np.asarray(lower) + 1.0e-8, np.asarray(upper) - 1.0e-8)

    def residuals(theta):
        rs = np.exp(theta[0])
        rt = rs * (1.0 + np.exp(theta[1]))
        rho0 = np.exp(theta[2])
        rho_nfw = trunc_nfw_rho(rr, rs, rt, rho0)
        ymod = rho_nfw + jj
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        res = res[np.isfinite(res)]
        # Penalty: at r > Re, tNFW should exceed Jaffe
        if np.any(outer_mask):
            violation = np.maximum(jj[outer_mask] - rho_nfw[outer_mask], 0.0)
            pen = penalty_weight * violation / (np.maximum(jj[outer_mask], np.finfo(float).tiny))
            res = np.concatenate([res, pen])
        return res

    sol = least_squares(
        residuals, theta0, method="trf", bounds=(lower, upper),
        loss="soft_l1", f_scale=0.1, max_nfev=20000,
    )

    rs = float(np.exp(sol.x[0]))
    rt = float(rs * (1.0 + np.exp(sol.x[1])))
    rho0 = float(np.exp(sol.x[2]))

    ymod = model_fun(rr, sol.x)
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan

    diag = {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size),
            "success": bool(sol.success), "message": str(sol.message)}
    diag["jaffe_clamped"] = bool(np.any(np.minimum(rhoj_orig, y)[mask] < rhoj_orig[mask]))
    return rs, rt, rho0, diag


def fit_tnfw_plus_free_jaffe(r, rho_target, rho_jaffe_shape, rs_guess, rt_guess, rho0_guess, mstar_guess, rmin_fit=0.05, rmax_fit=None, re_kpc=None, penalty_weight=100.0):
    """Fit rho_target = tNFW(rs, rt, rho0) + Jaffe(mstar, rJ).

    Both mstar and the Jaffe scale radius rJ are free parameters.
    ``rho_jaffe_shape`` is accepted for API compatibility but ignored;
    the Jaffe profile is computed from (mstar, rJ) inside the residuals.

    Returns:
        rs, rt, rho0, mstar_free, rJ_free, diag
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(rho_target, dtype=float)

    mask = np.isfinite(r) & np.isfinite(y) & (r >= rmin_fit)
    if rmax_fit is not None:
        mask &= (r <= rmax_fit)
    rr, yy = r[mask], y[mask]

    # Initial rJ guess from Re
    rj_guess = jaffe_scale_radius(re_kpc) if (re_kpc is not None and re_kpc > 0) else 1.0

    # Upper bound on mstar: Jaffe(mstar, rj_guess) <= rho_piemd everywhere
    js_init = jaffe_rho_rj(rr, 1.0, rj_guess)
    safe_js = np.where(js_init > 0, js_init, np.inf)
    mstar_max = float(np.min(yy / safe_js))
    mstar_max = max(mstar_max, 1.0e-12)
    # Allow generous upper bound since rJ will change during fit
    mstar_upper = mstar_max * 10.0

    mstar_guess_clamped = min(float(mstar_guess), mstar_max)

    # Identify points beyond Re for the dominance penalty
    if re_kpc is not None and re_kpc > 0:
        outer_mask = rr > re_kpc
    else:
        outer_mask = np.zeros(rr.size, dtype=bool)

    # theta = [log(rs), log(rt/rs - 1), log(rho0), log(mstar), log(rJ)]
    theta0 = np.array(
        [
            np.log(max(float(rs_guess), 1.0e-6)),
            np.log(max(float(rt_guess) / max(float(rs_guess), 1.0e-6) - 1.0, 1.0e-6)),
            np.log(max(float(rho0_guess), 1.0e-20)),
            np.log(max(mstar_guess_clamped, 1.0e-20)),
            np.log(max(rj_guess, 1.0e-4)),
        ],
        dtype=float,
    )
    lower = np.array([np.log(max(rmin_fit, 1.0e-4)), np.log(1.0e-6), np.log(1.0e-30),
                       np.log(1.0e-12), np.log(1.0e-4)], dtype=float)
    upper = np.array([np.log(max(5.0 * np.nanmax(rr), 1.0)), np.log(1.0e5), np.log(1.0e20),
                       np.log(mstar_upper), np.log(max(5.0 * np.nanmax(rr), 10.0))], dtype=float)

    theta0 = np.clip(theta0, np.asarray(lower) + 1.0e-8, np.asarray(upper) - 1.0e-8)

    def residuals(theta):
        rs = np.exp(theta[0])
        rt = rs * (1.0 + np.exp(theta[1]))
        rho0 = np.exp(theta[2])
        mstar = np.exp(theta[3])
        rj = np.exp(theta[4])
        rho_nfw = trunc_nfw_rho(rr, rs, rt, rho0)
        rho_jaffe = jaffe_rho_rj(rr, mstar, rj)
        ymod = rho_nfw + rho_jaffe
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        res = res[np.isfinite(res)]
        # Penalty: at r > Re, tNFW should exceed Jaffe
        if np.any(outer_mask):
            violation = np.maximum(rho_jaffe[outer_mask] - rho_nfw[outer_mask], 0.0)
            pen = penalty_weight * violation / (np.maximum(rho_jaffe[outer_mask], np.finfo(float).tiny))
            res = np.concatenate([res, pen])
        return res

    sol = least_squares(
        residuals, theta0, method="trf", bounds=(lower, upper),
        loss="soft_l1", f_scale=0.1, max_nfev=20000,
    )

    rs = float(np.exp(sol.x[0]))
    rt = float(rs * (1.0 + np.exp(sol.x[1])))
    rho0 = float(np.exp(sol.x[2]))
    mstar_free = float(np.exp(sol.x[3]))
    rJ_free = float(np.exp(sol.x[4]))

    rho_nfw_final = trunc_nfw_rho(rr, rs, rt, rho0)
    ymod = rho_nfw_final + jaffe_rho_rj(rr, mstar_free, rJ_free)
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan

    diag = {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size),
            "success": bool(sol.success), "message": str(sol.message)}
    diag["mstar_max_constraint"] = mstar_max
    return rs, rt, rho0, mstar_free, rJ_free, diag


# ---------------------------------------------------------------------------
# BMO (n=0.5) + Jaffe fitting routines
# ---------------------------------------------------------------------------

def fit_bmo_plus_fixed_jaffe(r, rho_target, rho_jaffe_fixed, rs_guess, rt_guess,
                              rho0_guess, rmin_fit=0.05, rmax_fit=None, re_kpc=None,
                              penalty_weight=100.0):
    """Fit rho_target = bmo(rs, rt, rho0) + jaffe_fixed.

    Returns:
        rs, rt, rho0, diag
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(rho_target, dtype=float)
    rhoj_orig = np.asarray(rho_jaffe_fixed, dtype=float)
    rhoj = np.minimum(rhoj_orig, y)

    mask = np.isfinite(r) & np.isfinite(y) & np.isfinite(rhoj) & (r >= rmin_fit)
    if rmax_fit is not None:
        mask &= (r <= rmax_fit)
    rr, yy, jj = r[mask], y[mask], rhoj[mask]

    if re_kpc is not None and re_kpc > 0:
        outer_mask = rr > re_kpc
    else:
        outer_mask = np.zeros(rr.size, dtype=bool)

    def model_fun(rad, theta):
        rs = np.exp(theta[0])
        rt = rs * (1.0 + np.exp(theta[1]))
        rho0 = np.exp(theta[2])
        return bmo_rho(rad, rs, rt, rho0) + jj

    theta0 = np.array([
        np.log(max(float(rs_guess), 1e-6)),
        np.log(max(float(rt_guess) / max(float(rs_guess), 1e-6) - 1.0, 1e-6)),
        np.log(max(float(rho0_guess), 1e-20)),
    ], dtype=float)
    lower = np.array([np.log(max(rmin_fit, 1e-4)), np.log(1e-6), np.log(1e-30)], dtype=float)
    upper = np.array([np.log(max(5.0 * np.nanmax(rr), 1.0)), np.log(1e5), np.log(1e20)], dtype=float)
    theta0 = np.clip(theta0, np.asarray(lower) + 1e-8, np.asarray(upper) - 1e-8)

    def residuals(theta):
        rs = np.exp(theta[0])
        rt = rs * (1.0 + np.exp(theta[1]))
        rho0 = np.exp(theta[2])
        rho_dm = bmo_rho(rr, rs, rt, rho0)
        ymod = rho_dm + jj
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        res = res[np.isfinite(res)]
        if np.any(outer_mask):
            violation = np.maximum(jj[outer_mask] - rho_dm[outer_mask], 0.0)
            pen = penalty_weight * violation / np.maximum(jj[outer_mask], np.finfo(float).tiny)
            res = np.concatenate([res, pen])
        return res

    sol = least_squares(residuals, theta0, method="trf", bounds=(lower, upper),
                        loss="soft_l1", f_scale=0.1, max_nfev=20000)

    rs = float(np.exp(sol.x[0]))
    rt = float(rs * (1.0 + np.exp(sol.x[1])))
    rho0 = float(np.exp(sol.x[2]))

    ymod = model_fun(rr, sol.x)
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan

    diag = {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size),
            "success": bool(sol.success), "message": str(sol.message)}
    diag["jaffe_clamped"] = bool(np.any(np.minimum(rhoj_orig, y)[mask] < rhoj_orig[mask]))
    return rs, rt, rho0, diag


def fit_bmo_plus_free_jaffe(r, rho_target, rho_jaffe_shape, rs_guess, rt_guess,
                             rho0_guess, mstar_guess, rmin_fit=0.05, rmax_fit=None,
                             re_kpc=None, penalty_weight=100.0):
    """Fit rho_target = bmo(rs, rt, rho0) + Jaffe(mstar, rJ).

    Both mstar and rJ are free parameters.

    Returns:
        rs, rt, rho0, mstar_free, rJ_free, diag
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(rho_target, dtype=float)

    mask = np.isfinite(r) & np.isfinite(y) & (r >= rmin_fit)
    if rmax_fit is not None:
        mask &= (r <= rmax_fit)
    rr, yy = r[mask], y[mask]

    rj_guess = jaffe_scale_radius(re_kpc) if (re_kpc is not None and re_kpc > 0) else 1.0

    js_init = jaffe_rho_rj(rr, 1.0, rj_guess)
    safe_js = np.where(js_init > 0, js_init, np.inf)
    mstar_max = float(np.min(yy / safe_js))
    mstar_max = max(mstar_max, 1e-12)
    mstar_upper = mstar_max * 10.0
    mstar_guess_clamped = min(float(mstar_guess), mstar_max)

    if re_kpc is not None and re_kpc > 0:
        outer_mask = rr > re_kpc
    else:
        outer_mask = np.zeros(rr.size, dtype=bool)

    theta0 = np.array([
        np.log(max(float(rs_guess), 1e-6)),
        np.log(max(float(rt_guess) / max(float(rs_guess), 1e-6) - 1.0, 1e-6)),
        np.log(max(float(rho0_guess), 1e-20)),
        np.log(max(mstar_guess_clamped, 1e-20)),
        np.log(max(rj_guess, 1e-4)),
    ], dtype=float)
    lower = np.array([np.log(max(rmin_fit, 1e-4)), np.log(1e-6), np.log(1e-30),
                       np.log(1e-12), np.log(1e-4)], dtype=float)
    upper = np.array([np.log(max(5.0 * np.nanmax(rr), 1.0)), np.log(1e5), np.log(1e20),
                       np.log(mstar_upper), np.log(max(5.0 * np.nanmax(rr), 10.0))], dtype=float)
    theta0 = np.clip(theta0, np.asarray(lower) + 1e-8, np.asarray(upper) - 1e-8)

    def residuals(theta):
        rs = np.exp(theta[0])
        rt = rs * (1.0 + np.exp(theta[1]))
        rho0 = np.exp(theta[2])
        mstar = np.exp(theta[3])
        rj = np.exp(theta[4])
        rho_dm = bmo_rho(rr, rs, rt, rho0)
        rho_jaffe = jaffe_rho_rj(rr, mstar, rj)
        ymod = rho_dm + rho_jaffe
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        res = res[np.isfinite(res)]
        if np.any(outer_mask):
            violation = np.maximum(rho_jaffe[outer_mask] - rho_dm[outer_mask], 0.0)
            pen = penalty_weight * violation / np.maximum(rho_jaffe[outer_mask], np.finfo(float).tiny)
            res = np.concatenate([res, pen])
        return res

    sol = least_squares(residuals, theta0, method="trf", bounds=(lower, upper),
                        loss="soft_l1", f_scale=0.1, max_nfev=20000)

    rs = float(np.exp(sol.x[0]))
    rt = float(rs * (1.0 + np.exp(sol.x[1])))
    rho0 = float(np.exp(sol.x[2]))
    mstar_free = float(np.exp(sol.x[3]))
    rJ_free = float(np.exp(sol.x[4]))

    rho_dm_final = bmo_rho(rr, rs, rt, rho0)
    ymod = rho_dm_final + jaffe_rho_rj(rr, mstar_free, rJ_free)
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan

    diag = {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size),
            "success": bool(sol.success), "message": str(sol.message)}
    diag["mstar_max_constraint"] = mstar_max
    return rs, rt, rho0, mstar_free, rJ_free, diag


# ---------------------------------------------------------------------------
# Einasto + Jaffe fitting routines
# ---------------------------------------------------------------------------

def fit_einasto_plus_fixed_jaffe(r, rho_target, rho_jaffe_fixed, rs_guess, n_guess,
                                  rho_s_guess, rmin_fit=0.05, rmax_fit=None, re_kpc=None, penalty_weight=100.0):
    """Fit rho_target = einasto(rs, rho_s, n) + jaffe_fixed.

    Returns:
        rs, rho_s, n, diag
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(rho_target, dtype=float)
    rhoj_orig = np.asarray(rho_jaffe_fixed, dtype=float)

    # Clamp Jaffe so DM stays non-negative
    rhoj = np.minimum(rhoj_orig, y)

    mask = np.isfinite(r) & np.isfinite(y) & np.isfinite(rhoj) & (r >= rmin_fit)
    if rmax_fit is not None:
        mask &= (r <= rmax_fit)
    rr, yy, jj = r[mask], y[mask], rhoj[mask]

    if re_kpc is not None and re_kpc > 0:
        outer_mask = rr > re_kpc
    else:
        outer_mask = np.zeros(rr.size, dtype=bool)

    # theta = [log(rs), log(rho_s), log(n)]
    theta0 = np.array([
        np.log(max(float(rs_guess), 1e-6)),
        np.log(max(float(rho_s_guess), 1e-20)),
        np.log(max(float(n_guess), 0.1)),
    ], dtype=float)
    lower = np.array([np.log(max(rmin_fit, 1e-4)), np.log(1e-30), np.log(0.1)])
    upper = np.array([np.log(max(5.0 * np.nanmax(rr), 1.0)), np.log(1e20), np.log(20.0)])

    theta0 = np.clip(theta0, lower + 1e-8, upper - 1e-8)

    def residuals(theta):
        rs = np.exp(theta[0])
        rho_s = np.exp(theta[1])
        n = np.exp(theta[2])
        rho_ein = einasto_rho(rr, rs, rho_s, n)
        ymod = rho_ein + jj
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        res = res[np.isfinite(res)]
        if np.any(outer_mask):
            violation = np.maximum(jj[outer_mask] - rho_ein[outer_mask], 0.0)
            pen = penalty_weight * violation / np.maximum(jj[outer_mask], np.finfo(float).tiny)
            res = np.concatenate([res, pen])
        return res

    sol = least_squares(
        residuals, theta0, method="trf", bounds=(lower, upper),
        loss="soft_l1", f_scale=0.1, max_nfev=20000,
    )

    rs = float(np.exp(sol.x[0]))
    rho_s = float(np.exp(sol.x[1]))
    n = float(np.exp(sol.x[2]))

    ymod = einasto_rho(rr, rs, rho_s, n) + jj
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan

    diag = {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size),
            "success": bool(sol.success), "message": str(sol.message)}
    diag["jaffe_clamped"] = bool(np.any(np.minimum(rhoj_orig, y)[mask] < rhoj_orig[mask]))
    return rs, rho_s, n, diag


def fit_einasto_plus_free_jaffe(r, rho_target, rho_jaffe_shape, rs_guess, n_guess,
                                 rho_s_guess, mstar_guess, rmin_fit=0.05, rmax_fit=None,
                                 re_kpc=None, penalty_weight=100.0):
    """Fit rho_target = einasto(rs, rho_s, n) + Jaffe(mstar, rJ).

    Both mstar and rJ are free parameters.

    Returns:
        rs, rho_s, n, mstar_free, rJ_free, diag
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(rho_target, dtype=float)

    mask = np.isfinite(r) & np.isfinite(y) & (r >= rmin_fit)
    if rmax_fit is not None:
        mask &= (r <= rmax_fit)
    rr, yy = r[mask], y[mask]

    rj_guess = jaffe_scale_radius(re_kpc) if (re_kpc is not None and re_kpc > 0) else 1.0

    js_init = jaffe_rho_rj(rr, 1.0, rj_guess)
    safe_js = np.where(js_init > 0, js_init, np.inf)
    mstar_max = float(np.min(yy / safe_js))
    mstar_max = max(mstar_max, 1e-12)
    mstar_upper = mstar_max * 10.0

    mstar_guess_clamped = min(float(mstar_guess), mstar_max)

    if re_kpc is not None and re_kpc > 0:
        outer_mask = rr > re_kpc
    else:
        outer_mask = np.zeros(rr.size, dtype=bool)

    # theta = [log(rs), log(rho_s), log(n), log(mstar), log(rJ)]
    theta0 = np.array([
        np.log(max(float(rs_guess), 1e-6)),
        np.log(max(float(rho_s_guess), 1e-20)),
        np.log(max(float(n_guess), 0.1)),
        np.log(max(mstar_guess_clamped, 1e-20)),
        np.log(max(rj_guess, 1e-4)),
    ], dtype=float)
    lower = np.array([np.log(max(rmin_fit, 1e-4)), np.log(1e-30), np.log(0.1),
                       np.log(1e-12), np.log(1e-4)])
    upper = np.array([np.log(max(5.0 * np.nanmax(rr), 1.0)), np.log(1e20), np.log(20.0),
                       np.log(mstar_upper), np.log(max(5.0 * np.nanmax(rr), 10.0))])

    theta0 = np.clip(theta0, lower + 1e-8, upper - 1e-8)

    def residuals(theta):
        rs = np.exp(theta[0])
        rho_s = np.exp(theta[1])
        n = np.exp(theta[2])
        mstar = np.exp(theta[3])
        rj = np.exp(theta[4])
        rho_ein = einasto_rho(rr, rs, rho_s, n)
        rho_jaffe = jaffe_rho_rj(rr, mstar, rj)
        ymod = rho_ein + rho_jaffe
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        res = res[np.isfinite(res)]
        if np.any(outer_mask):
            violation = np.maximum(rho_jaffe[outer_mask] - rho_ein[outer_mask], 0.0)
            pen = penalty_weight * violation / np.maximum(rho_jaffe[outer_mask], np.finfo(float).tiny)
            res = np.concatenate([res, pen])
        return res

    sol = least_squares(
        residuals, theta0, method="trf", bounds=(lower, upper),
        loss="soft_l1", f_scale=0.1, max_nfev=20000,
    )

    rs = float(np.exp(sol.x[0]))
    rho_s = float(np.exp(sol.x[1]))
    n = float(np.exp(sol.x[2]))
    mstar_free = float(np.exp(sol.x[3]))
    rJ_free = float(np.exp(sol.x[4]))

    rho_ein_final = einasto_rho(rr, rs, rho_s, n)
    ymod = rho_ein_final + jaffe_rho_rj(rr, mstar_free, rJ_free)
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan

    diag = {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size),
            "success": bool(sol.success), "message": str(sol.message)}
    diag["mstar_max_constraint"] = mstar_max
    return rs, rho_s, n, mstar_free, rJ_free, diag


# ---------------------------------------------------------------------------
# gNFW + Jaffe fitting routines
# ---------------------------------------------------------------------------

def fit_gnfw_plus_fixed_jaffe(r, rho_target, rho_jaffe_fixed, rs_guess,
                               gamma_guess, rho_s_guess, rmin_fit=0.05, rmax_fit=None,
                               re_kpc=None, penalty_weight=100.0):
    """Fit rho_target = gnfw(rs, rho_s, gamma) + jaffe_fixed.

    Returns:
        rs, rho_s, gamma, diag
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(rho_target, dtype=float)
    rhoj_orig = np.asarray(rho_jaffe_fixed, dtype=float)

    rhoj = np.minimum(rhoj_orig, y)

    mask = np.isfinite(r) & np.isfinite(y) & np.isfinite(rhoj) & (r >= rmin_fit)
    if rmax_fit is not None:
        mask &= (r <= rmax_fit)
    rr, yy, jj = r[mask], y[mask], rhoj[mask]

    if re_kpc is not None and re_kpc > 0:
        outer_mask = rr > re_kpc
    else:
        outer_mask = np.zeros(rr.size, dtype=bool)

    # theta = [log(rs), log(rho_s), log(gamma)]
    theta0 = np.array([
        np.log(max(float(rs_guess), 1e-6)),
        np.log(max(float(rho_s_guess), 1e-20)),
        np.log(max(float(gamma_guess), 0.01)),
    ], dtype=float)
    lower = np.array([np.log(max(rmin_fit, 1e-4)), np.log(1e-30), np.log(0.01)])
    upper = np.array([np.log(max(5.0 * np.nanmax(rr), 1.0)), np.log(1e20), np.log(2.99)])

    theta0 = np.clip(theta0, lower + 1e-8, upper - 1e-8)

    def residuals(theta):
        rs = np.exp(theta[0])
        rho_s = np.exp(theta[1])
        gamma = np.exp(theta[2])
        rho_dm = gnfw_rho(rr, rs, rho_s, gamma)
        ymod = rho_dm + jj
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        res = res[np.isfinite(res)]
        if np.any(outer_mask):
            violation = np.maximum(jj[outer_mask] - rho_dm[outer_mask], 0.0)
            pen = penalty_weight * violation / np.maximum(jj[outer_mask], np.finfo(float).tiny)
            res = np.concatenate([res, pen])
        return res

    sol = least_squares(
        residuals, theta0, method="trf", bounds=(lower, upper),
        loss="soft_l1", f_scale=0.1, max_nfev=20000,
    )

    rs = float(np.exp(sol.x[0]))
    rho_s = float(np.exp(sol.x[1]))
    gamma = float(np.exp(sol.x[2]))

    ymod = gnfw_rho(rr, rs, rho_s, gamma) + jj
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan

    diag = {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size),
            "success": bool(sol.success), "message": str(sol.message)}
    diag["jaffe_clamped"] = bool(np.any(np.minimum(rhoj_orig, y)[mask] < rhoj_orig[mask]))
    return rs, rho_s, gamma, diag


def fit_gnfw_plus_free_jaffe(r, rho_target, rho_jaffe_shape, rs_guess,
                              gamma_guess, rho_s_guess, mstar_guess,
                              rmin_fit=0.05, rmax_fit=None, re_kpc=None, penalty_weight=100.0):
    """Fit rho_target = gnfw(rs, rho_s, gamma) + Jaffe(mstar, rJ).

    Both mstar and rJ are free parameters.

    Returns:
        rs, rho_s, gamma, mstar_free, rJ_free, diag
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(rho_target, dtype=float)

    mask = np.isfinite(r) & np.isfinite(y) & (r >= rmin_fit)
    if rmax_fit is not None:
        mask &= (r <= rmax_fit)
    rr, yy = r[mask], y[mask]

    rj_guess = jaffe_scale_radius(re_kpc) if (re_kpc is not None and re_kpc > 0) else 1.0

    js_init = jaffe_rho_rj(rr, 1.0, rj_guess)
    safe_js = np.where(js_init > 0, js_init, np.inf)
    mstar_max = float(np.min(yy / safe_js))
    mstar_max = max(mstar_max, 1e-12)
    mstar_upper = mstar_max * 10.0

    mstar_guess_clamped = min(float(mstar_guess), mstar_max)

    if re_kpc is not None and re_kpc > 0:
        outer_mask = rr > re_kpc
    else:
        outer_mask = np.zeros(rr.size, dtype=bool)

    # theta = [log(rs), log(rho_s), log(gamma), log(mstar), log(rJ)]
    theta0 = np.array([
        np.log(max(float(rs_guess), 1e-6)),
        np.log(max(float(rho_s_guess), 1e-20)),
        np.log(max(float(gamma_guess), 0.01)),
        np.log(max(mstar_guess_clamped, 1e-20)),
        np.log(max(rj_guess, 1e-4)),
    ], dtype=float)
    lower = np.array([np.log(max(rmin_fit, 1e-4)), np.log(1e-30), np.log(0.01),
                       np.log(1e-12), np.log(1e-4)])
    upper = np.array([np.log(max(5.0 * np.nanmax(rr), 1.0)), np.log(1e20), np.log(2.99),
                       np.log(mstar_upper), np.log(max(5.0 * np.nanmax(rr), 10.0))])

    theta0 = np.clip(theta0, lower + 1e-8, upper - 1e-8)

    def residuals(theta):
        rs = np.exp(theta[0])
        rho_s = np.exp(theta[1])
        gamma = np.exp(theta[2])
        mstar = np.exp(theta[3])
        rj = np.exp(theta[4])
        rho_dm = gnfw_rho(rr, rs, rho_s, gamma)
        rho_jaffe = jaffe_rho_rj(rr, mstar, rj)
        ymod = rho_dm + rho_jaffe
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        res = res[np.isfinite(res)]
        if np.any(outer_mask):
            violation = np.maximum(rho_jaffe[outer_mask] - rho_dm[outer_mask], 0.0)
            pen = penalty_weight * violation / np.maximum(rho_jaffe[outer_mask], np.finfo(float).tiny)
            res = np.concatenate([res, pen])
        return res

    sol = least_squares(
        residuals, theta0, method="trf", bounds=(lower, upper),
        loss="soft_l1", f_scale=0.1, max_nfev=20000,
    )

    rs = float(np.exp(sol.x[0]))
    rho_s = float(np.exp(sol.x[1]))
    gamma = float(np.exp(sol.x[2]))
    mstar_free = float(np.exp(sol.x[3]))
    rJ_free = float(np.exp(sol.x[4]))

    rho_dm_final = gnfw_rho(rr, rs, rho_s, gamma)
    ymod = rho_dm_final + jaffe_rho_rj(rr, mstar_free, rJ_free)
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan

    diag = {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size),
            "success": bool(sol.success), "message": str(sol.message)}
    diag["mstar_max_constraint"] = mstar_max
    return rs, rho_s, gamma, mstar_free, rJ_free, diag


# ---------------------------------------------------------------------------
# Standard NFW + Jaffe fitting routines
# ---------------------------------------------------------------------------

def fit_nfw_plus_fixed_jaffe(r, rho_target, rho_jaffe_fixed, rs_guess,
                              rho0_guess, rmin_fit=0.05, rmax_fit=None,
                              re_kpc=None, penalty_weight=100.0):
    """Fit rho_target = nfw(rs, rho0) + jaffe_fixed.

    Returns:
        rs, rho0, diag
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(rho_target, dtype=float)
    rhoj_orig = np.asarray(rho_jaffe_fixed, dtype=float)

    rhoj = np.minimum(rhoj_orig, y)

    mask = np.isfinite(r) & np.isfinite(y) & np.isfinite(rhoj) & (r >= rmin_fit)
    if rmax_fit is not None:
        mask &= (r <= rmax_fit)
    rr, yy, jj = r[mask], y[mask], rhoj[mask]

    if re_kpc is not None and re_kpc > 0:
        outer_mask = rr > re_kpc
    else:
        outer_mask = np.zeros(rr.size, dtype=bool)

    # theta = [log(rs), log(rho0)]
    theta0 = np.array([
        np.log(max(float(rs_guess), 1e-6)),
        np.log(max(float(rho0_guess), 1e-20)),
    ], dtype=float)
    lower = np.array([np.log(max(rmin_fit, 1e-4)), np.log(1e-30)])
    upper = np.array([np.log(max(5.0 * np.nanmax(rr), 1.0)), np.log(1e20)])

    theta0 = np.clip(theta0, lower + 1e-8, upper - 1e-8)

    def residuals(theta):
        rs = np.exp(theta[0])
        rho0 = np.exp(theta[1])
        rho_dm = nfw_rho(rr, rs, rho0)
        ymod = rho_dm + jj
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        res = res[np.isfinite(res)]
        if np.any(outer_mask):
            violation = np.maximum(jj[outer_mask] - rho_dm[outer_mask], 0.0)
            pen = penalty_weight * violation / np.maximum(jj[outer_mask], np.finfo(float).tiny)
            res = np.concatenate([res, pen])
        return res

    sol = least_squares(
        residuals, theta0, method="trf", bounds=(lower, upper),
        loss="soft_l1", f_scale=0.1, max_nfev=20000,
    )

    rs = float(np.exp(sol.x[0]))
    rho0 = float(np.exp(sol.x[1]))

    ymod = nfw_rho(rr, rs, rho0) + jj
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan

    diag = {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size),
            "success": bool(sol.success), "message": str(sol.message)}
    diag["jaffe_clamped"] = bool(np.any(np.minimum(rhoj_orig, y)[mask] < rhoj_orig[mask]))
    return rs, rho0, diag


def fit_nfw_plus_free_jaffe(r, rho_target, rho_jaffe_shape, rs_guess,
                             rho0_guess, mstar_guess, rmin_fit=0.05,
                             rmax_fit=None, re_kpc=None, penalty_weight=100.0):
    """Fit rho_target = nfw(rs, rho0) + Jaffe(mstar, rJ).

    Both mstar and rJ are free parameters.

    Returns:
        rs, rho0, mstar_free, rJ_free, diag
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(rho_target, dtype=float)

    mask = np.isfinite(r) & np.isfinite(y) & (r >= rmin_fit)
    if rmax_fit is not None:
        mask &= (r <= rmax_fit)
    rr, yy = r[mask], y[mask]

    rj_guess = jaffe_scale_radius(re_kpc) if (re_kpc is not None and re_kpc > 0) else 1.0

    js_init = jaffe_rho_rj(rr, 1.0, rj_guess)
    safe_js = np.where(js_init > 0, js_init, np.inf)
    mstar_max = float(np.min(yy / safe_js))
    mstar_max = max(mstar_max, 1e-12)
    mstar_upper = mstar_max * 10.0

    mstar_guess_clamped = min(float(mstar_guess), mstar_max)

    if re_kpc is not None and re_kpc > 0:
        outer_mask = rr > re_kpc
    else:
        outer_mask = np.zeros(rr.size, dtype=bool)

    # theta = [log(rs), log(rho0), log(mstar), log(rJ)]
    theta0 = np.array([
        np.log(max(float(rs_guess), 1e-6)),
        np.log(max(float(rho0_guess), 1e-20)),
        np.log(max(mstar_guess_clamped, 1e-20)),
        np.log(max(rj_guess, 1e-4)),
    ], dtype=float)
    lower = np.array([np.log(max(rmin_fit, 1e-4)), np.log(1e-30),
                       np.log(1e-12), np.log(1e-4)])
    upper = np.array([np.log(max(5.0 * np.nanmax(rr), 1.0)), np.log(1e20),
                       np.log(mstar_upper), np.log(max(5.0 * np.nanmax(rr), 10.0))])

    theta0 = np.clip(theta0, lower + 1e-8, upper - 1e-8)

    def residuals(theta):
        rs = np.exp(theta[0])
        rho0 = np.exp(theta[1])
        mstar = np.exp(theta[2])
        rj = np.exp(theta[3])
        rho_dm = nfw_rho(rr, rs, rho0)
        rho_jaffe = jaffe_rho_rj(rr, mstar, rj)
        ymod = rho_dm + rho_jaffe
        res = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
        res = res[np.isfinite(res)]
        if np.any(outer_mask):
            violation = np.maximum(rho_jaffe[outer_mask] - rho_dm[outer_mask], 0.0)
            pen = penalty_weight * violation / np.maximum(rho_jaffe[outer_mask], np.finfo(float).tiny)
            res = np.concatenate([res, pen])
        return res

    sol = least_squares(
        residuals, theta0, method="trf", bounds=(lower, upper),
        loss="soft_l1", f_scale=0.1, max_nfev=20000,
    )

    rs = float(np.exp(sol.x[0]))
    rho0 = float(np.exp(sol.x[1]))
    mstar_free = float(np.exp(sol.x[2]))
    rJ_free = float(np.exp(sol.x[3]))

    rho_dm_final = nfw_rho(rr, rs, rho0)
    ymod = rho_dm_final + jaffe_rho_rj(rr, mstar_free, rJ_free)
    reslog = np.log(np.maximum(ymod, np.finfo(float).tiny)) - np.log(yy)
    rmse_log = float(np.sqrt(np.mean(reslog**2)))
    ylog = np.log(yy)
    sst = np.sum((ylog - ylog.mean()) ** 2)
    r2_log = float(1.0 - np.sum(reslog**2) / sst) if sst > 0 else np.nan

    diag = {"rmse_log": rmse_log, "r2_log": r2_log, "npts": int(rr.size),
            "success": bool(sol.success), "message": str(sol.message)}
    diag["mstar_max_constraint"] = mstar_max
    return rs, rho0, mstar_free, rJ_free, diag


def main():
    parser = argparse.ArgumentParser(description="Dual PIEMD decomposition: tNFW+Jaffe(fixed M*) and tNFW+Jaffe(free M*).")
    parser.add_argument("--parfile", default=DEFAULT_PAR, help="LensTool parameter file.")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG, help="Photometric catalog file (fixed-width).")
    parser.add_argument("--explain", default=DEFAULT_README, help="ReadMe file for fixed-width parsing.")
    parser.add_argument("--tolerance-arcsec", type=float, default=0.1, help="Sky-match tolerance in arcsec.")
    parser.add_argument("--rmin-fit-kpc", type=float, default=0.05, help="Minimum radius used in fit [kpc].")
    parser.add_argument("--rmax-re-factor", type=float, default=None,
                        help="If set, fit only up to rmax = factor * Re_kpc for each galaxy (e.g. 5.0 for 5*Re).")
    parser.add_argument("--max-galaxies", type=int, default=None, help="If set, stop after this many fitted galaxies.")
    parser.add_argument("--outcsv", default=None, help="Output CSV filename (default depends on DM profile type).")
    parser.add_argument("--plot-dir", default=None, help="Optional per-galaxy plot directory (default depends on DM profile type).")
    parser.add_argument("--max-plots", type=int, default=0, help="Number of diagnostic plots to save (0 disables).")
    parser.add_argument("--bayesfile", type=str, default=None,
                        help="Path to LensTool bayes_*.dat MCMC sample file (to propagate uncertainties).")
    parser.add_argument("--nchains", type=int, default=None,
                        help="Number of MCMC samples to use from bayesfile (default: all).")
    parser.add_argument("--outh5", type=str, default=None,
                        help="Output HDF5 filename for per-sample parameters (default: derived from outcsv).")
    parser.add_argument("--corr-dir", type=str, default=None,
                        help="Optional output directory for per-galaxy correlation matrices (CSV) from MCMC samples.")
    parser.add_argument("--mref", type=float, default=DEFAULT_MREF_F160W,
                        help="Reference F160W magnitude for scaling relations.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Velocity-dispersion scaling exponent alpha.")
    parser.add_argument("--beta-cut", type=float, default=DEFAULT_BETA_CUT,
                        help="Cut-radius scaling exponent beta_cut.")
    dm_group = parser.add_mutually_exclusive_group()
    dm_group.add_argument("--use-bmo", action="store_true", default=False,
                          help="Use BMO (Baltz-Marshall-Oguri, n=0.5) smoothly truncated NFW for the DM component.")
    dm_group.add_argument("--use-nfw", action="store_true", default=False,
                          help="Use standard (untruncated) NFW profile for the DM component.")
    dm_group.add_argument("--use-einasto", action="store_true", default=False,
                          help="Use Einasto profile instead of truncated NFW for the DM component.")
    dm_group.add_argument("--use-gnfw", action="store_true", default=False,
                          help="Use generalized NFW (free inner slope) instead of truncated NFW for the DM component.")
    args = parser.parse_args()

    # --- Resolve default filenames based on the DM profile type ---
    if args.use_bmo:
        dm_tag = "bmo"
    elif args.use_nfw:
        dm_tag = "nfw"
    elif args.use_einasto:
        dm_tag = "einasto"
    elif args.use_gnfw:
        dm_tag = "gnfw"
    else:
        dm_tag = "tnfw"
    if args.outcsv is None:
        args.outcsv = f"lens_galaxy_fits_{dm_tag}_jaffe_dual.csv"
    if args.plot_dir is None:
        args.plot_dir = f"fit_{dm_tag}_jaffe_dual_profiles"

    # --- Load cosmology from the .par file ---
    cosmologie = readLenstoolBlock(best_par=args.parfile, block_name="cosmologie")
    cosmo = FlatLambdaCDM(H0=cosmologie["H0"], Om0=cosmologie["omegaM"])

    # --- Optionally load MCMC chains for uncertainty propagation ---
    sigma_chain = None
    rcut_chain = None
    if args.bayesfile is not None:
        sigma_chain, rcut_chain = load_bayes_scaling_params(args.bayesfile)
        n_total = len(sigma_chain)
        if args.nchains is not None and args.nchains < n_total:
            # Randomly subsample (without replacement) following sampleBayes pattern
            idx = np.random.choice(n_total, size=args.nchains, replace=False)
            sigma_chain = sigma_chain[idx]
            rcut_chain = rcut_chain[idx]
            print(f"Randomly selected {args.nchains} / {n_total} MCMC samples from {args.bayesfile}")
        else:
            print(f"Loaded {n_total} MCMC samples from {args.bayesfile}")

    # --- HDF5 containers for per-sample parameters ---
    do_mcmc = sigma_chain is not None
    if args.outh5 is None and do_mcmc:
        args.outh5 = Path(args.outcsv).with_suffix(".h5").as_posix()

    _H5_PARAM_NAMES_FIX = ["rs_fix_kpc", "rt_fix_kpc", "rho0_fix_Msun_kpc3"]
    _H5_PARAM_NAMES_FREE = ["rs_free_kpc", "rt_free_kpc", "rho0_free_Msun_kpc3", "Mstar_free_Msun", "rJ_free_kpc"]
    _H5_PIEMD_NAMES = ["sigma0_km_s", "rcut_kpc"]
    if args.use_bmo:
        _H5_PARAM_NAMES_FIX = ["rs_bmo_fix_kpc", "rt_bmo_fix_kpc", "rho0_bmo_fix_Msun_kpc3"]
        _H5_PARAM_NAMES_FREE = ["rs_bmo_free_kpc", "rt_bmo_free_kpc", "rho0_bmo_free_Msun_kpc3", "Mstar_free_Msun", "rJ_free_kpc"]
    elif args.use_nfw:
        _H5_PARAM_NAMES_FIX = ["rs_nfw_fix_kpc", "rho0_nfw_fix_Msun_kpc3"]
        _H5_PARAM_NAMES_FREE = ["rs_nfw_free_kpc", "rho0_nfw_free_Msun_kpc3", "Mstar_free_Msun", "rJ_free_kpc"]
    elif args.use_einasto:
        _H5_PARAM_NAMES_FIX = ["rs_ein_fix_kpc", "rhos_ein_fix_Msun_kpc3", "n_ein_fix"]
        _H5_PARAM_NAMES_FREE = ["rs_ein_free_kpc", "rhos_ein_free_Msun_kpc3", "n_ein_free", "Mstar_free_Msun", "rJ_free_kpc"]
    elif args.use_gnfw:
        _H5_PARAM_NAMES_FIX = ["rs_gnfw_fix_kpc", "rhos_gnfw_fix_Msun_kpc3", "gamma_gnfw_fix"]
        _H5_PARAM_NAMES_FREE = ["rs_gnfw_free_kpc", "rhos_gnfw_free_Msun_kpc3", "gamma_gnfw_free", "Mstar_free_Msun", "rJ_free_kpc"]
    _h5_lens_idx: list[int] = []
    _h5_cat_idx: list[int] = []
    _h5_ra: list[float] = []
    _h5_dec: list[float] = []
    _h5_fix_cube: list[np.ndarray] = []   # per-galaxy: (Nsamp, 3)
    _h5_free_cube: list[np.ndarray] = []  # per-galaxy: (Nsamp, 4)
    _h5_piemd_cube: list[np.ndarray] = [] # per-galaxy: (Nsamp, 2)
    _h5_fix_corr: list[np.ndarray] = []   # per-galaxy: (Npar_fix, Npar_fix)
    _h5_free_corr: list[np.ndarray] = []  # per-galaxy: (Npar_free, Npar_free)
    _h5_piemd_corr: list[np.ndarray] = [] # per-galaxy: (2, 2)
    _h5_fix_corr_with_derived: list[np.ndarray] = []   # per-galaxy: (Npar_fix+2, Npar_fix+2)
    _h5_free_corr_with_derived: list[np.ndarray] = []  # per-galaxy: (Npar_free+2, Npar_free+2)
    _h5_corr_nvalid_fix: list[int] = []
    _h5_corr_nvalid_free: list[int] = []
    _h5_corr_nvalid_piemd: list[int] = []
    _h5_corr_nvalid_fix_with_derived: list[int] = []
    _h5_corr_nvalid_free_with_derived: list[int] = []

    if do_mcmc and args.corr_dir is not None:
        Path(args.corr_dir).mkdir(parents=True, exist_ok=True)

    potentiel_all = readLenstoolBlock(best_par=args.parfile, block_name="potentiel")
    potentiel = selectPotentielByType(potentiel_all, ptype="gal")
    df_cat = load_cluster(readme_path=args.explain, data_path=args.catalog)

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

    r = np.logspace(-2, np.log10(1000), 1000)  # kpc, 0.01 to 1000
    rows = []
    nplot = 0
    if args.max_plots > 0:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    for j in range(len(potentiel)):
        if j not in match_map:
            continue
        j_cat = int(match_map[j])
        crow = df_cat.iloc[j_cat]

        try:
            mag_f160w = float(crow["F160W"])
            re_f160w = float(crow["ReF160W"])
        except Exception:
            continue
        if not np.isfinite(mag_f160w) or not np.isfinite(re_f160w) or re_f160w <= 0:
            continue

        # Per-galaxy maximum fitting radius
        rmax_fit = args.rmax_re_factor * re_f160w if args.rmax_re_factor is not None else None

        sigma0 = float(findInBlock(potentiel[j], "v_disp")) * np.sqrt(3.0 / 2.0)  # LensTool sigma_LT -> sigma0
        z_lens = float(findInBlock(potentiel[j], "z_lens"))
        r_cut = float(findInBlock(potentiel[j], "cut_radius_kpc"))
        r_core = float(findInBlock(potentiel[j], "core_radius_kpc"))
        if sigma0 <= 0 or r_cut <= 0 or r_core <= 0:
            continue

        rho_piemd = piemd_rho(r, r_core, r_cut, sigma0)

        mstar_mag = stellar_mass_to_light(mag_f160w)
        rho_j_fixed = jaffe_rho(r, mstar_mag, re_f160w)
        rho_j_shape = jaffe_rho(r, 1.0, re_f160w)

        rs_guess = max(1.0, 0.2 * r_cut)
        rt_guess = max(r_cut, 2.0 * rs_guess)
        rho0_guess = max(float(np.nanmax(rho_piemd) * 1.0e-3), 1.0e-20)
        n_guess = 4.0    # Einasto index initial guess
        gamma_guess = 1.0  # gNFW inner slope initial guess (NFW value)

        if args.use_bmo:
            # --- BMO (n=0.5) + Jaffe fits ---
            try:
                rs_fix, rt_fix, rho0_fix, diag_fix = fit_bmo_plus_fixed_jaffe(
                    r, rho_piemd, rho_j_fixed, rs_guess, rt_guess, rho0_guess,
                    rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w
                )
                ok_fix = True
            except Exception as exc:
                rs_fix, rt_fix, rho0_fix = np.nan, np.nan, np.nan
                diag_fix = {"rmse_log": np.nan, "r2_log": np.nan, "npts": 0, "success": False, "message": str(exc)}
                ok_fix = False

            try:
                rs_free, rt_free, rho0_free, mstar_free, rJ_free, diag_free = fit_bmo_plus_free_jaffe(
                    r, rho_piemd, rho_j_shape, rs_guess, rt_guess, rho0_guess,
                    mstar_guess=mstar_mag, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w
                )
                ok_free = True
            except Exception as exc:
                rs_free, rt_free, rho0_free, mstar_free, rJ_free = np.nan, np.nan, np.nan, np.nan, np.nan
                diag_free = {"rmse_log": np.nan, "r2_log": np.nan, "npts": 0, "success": False, "message": str(exc)}
                ok_free = False

        elif args.use_nfw:
            # --- Standard NFW + Jaffe fits ---
            # Fixed-stellar fit
            try:
                rs_fix, rho0_fix, diag_fix = fit_nfw_plus_fixed_jaffe(
                    r, rho_piemd, rho_j_fixed, rs_guess, rho0_guess,
                    rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w
                )
                ok_fix = True
            except Exception as exc:
                rs_fix, rho0_fix = np.nan, np.nan
                diag_fix = {"rmse_log": np.nan, "r2_log": np.nan, "npts": 0, "success": False, "message": str(exc)}
                ok_fix = False

            # Free-stellar fit
            try:
                rs_free, rho0_free, mstar_free, rJ_free, diag_free = fit_nfw_plus_free_jaffe(
                    r, rho_piemd, rho_j_shape, rs_guess, rho0_guess,
                    mstar_guess=mstar_mag, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w
                )
                ok_free = True
            except Exception as exc:
                rs_free, rho0_free, mstar_free, rJ_free = np.nan, np.nan, np.nan, np.nan
                diag_free = {"rmse_log": np.nan, "r2_log": np.nan, "npts": 0, "success": False, "message": str(exc)}
                ok_free = False

        elif args.use_einasto:
            # --- Einasto + Jaffe fits ---
            # Fixed-stellar fit
            try:
                rs_fix, rho_s_fix, n_fix, diag_fix = fit_einasto_plus_fixed_jaffe(
                    r, rho_piemd, rho_j_fixed, rs_guess, n_guess, rho0_guess,
                    rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w
                )
                ok_fix = True
            except Exception as exc:
                rs_fix, rho_s_fix, n_fix = np.nan, np.nan, np.nan
                diag_fix = {"rmse_log": np.nan, "r2_log": np.nan, "npts": 0, "success": False, "message": str(exc)}
                ok_fix = False

            # Free-stellar fit
            try:
                rs_free, rho_s_free, n_free, mstar_free, rJ_free, diag_free = fit_einasto_plus_free_jaffe(
                    r, rho_piemd, rho_j_shape, rs_guess, n_guess, rho0_guess,
                    mstar_guess=mstar_mag, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w
                )
                ok_free = True
            except Exception as exc:
                rs_free, rho_s_free, n_free, mstar_free, rJ_free = np.nan, np.nan, np.nan, np.nan, np.nan
                diag_free = {"rmse_log": np.nan, "r2_log": np.nan, "npts": 0, "success": False, "message": str(exc)}
                ok_free = False

        elif args.use_gnfw:
            # --- gNFW + Jaffe fits ---
            # Fixed-stellar fit
            try:
                rs_fix, rho_s_fix, gamma_fix, diag_fix = fit_gnfw_plus_fixed_jaffe(
                    r, rho_piemd, rho_j_fixed, rs_guess, gamma_guess, rho0_guess,
                    rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w
                )
                ok_fix = True
            except Exception as exc:
                rs_fix, rho_s_fix, gamma_fix = np.nan, np.nan, np.nan
                diag_fix = {"rmse_log": np.nan, "r2_log": np.nan, "npts": 0, "success": False, "message": str(exc)}
                ok_fix = False

            # Free-stellar fit
            try:
                rs_free, rho_s_free, gamma_free, mstar_free, rJ_free, diag_free = fit_gnfw_plus_free_jaffe(
                    r, rho_piemd, rho_j_shape, rs_guess, gamma_guess, rho0_guess,
                    mstar_guess=mstar_mag, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w
                )
                ok_free = True
            except Exception as exc:
                rs_free, rho_s_free, gamma_free, mstar_free, rJ_free = np.nan, np.nan, np.nan, np.nan, np.nan
                diag_free = {"rmse_log": np.nan, "r2_log": np.nan, "npts": 0, "success": False, "message": str(exc)}
                ok_free = False

        else:
            # --- tNFW + Jaffe fits ---
            # Fixed-stellar fit
            try:
                rs_fix, rt_fix, rho0_fix, diag_fix = fit_tnfw_plus_fixed_jaffe(
                    r, rho_piemd, rho_j_fixed, rs_guess, rt_guess, rho0_guess, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w
                )
                ok_fix = True
            except Exception as exc:
                rs_fix, rt_fix, rho0_fix = np.nan, np.nan, np.nan
                diag_fix = {"rmse_log": np.nan, "r2_log": np.nan, "npts": 0, "success": False, "message": str(exc)}
                ok_fix = False

            # Free-stellar fit
            try:
                rs_free, rt_free, rho0_free, mstar_free, rJ_free, diag_free = fit_tnfw_plus_free_jaffe(
                    r, rho_piemd, rho_j_shape, rs_guess, rt_guess, rho0_guess, mstar_guess=mstar_mag, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w
                )
                ok_free = True
            except Exception as exc:
                rs_free, rt_free, rho0_free, mstar_free, rJ_free = np.nan, np.nan, np.nan, np.nan, np.nan
                diag_free = {"rmse_log": np.nan, "r2_log": np.nan, "npts": 0, "success": False, "message": str(exc)}
                ok_free = False

        rho_dm_fixed = rho_piemd - rho_j_fixed
        negative_center_fixed = bool(np.any(rho_dm_fixed[r <= 1.0] <= 0.0))

        # --- MCMC uncertainty propagation ---
        mcmc_summary = {}
        fix_samp = None
        free_samp = None
        piemd_samp = None
        corr_fix = np.full((len(_H5_PARAM_NAMES_FIX), len(_H5_PARAM_NAMES_FIX)), np.nan, dtype=float)
        corr_free = np.full((len(_H5_PARAM_NAMES_FREE), len(_H5_PARAM_NAMES_FREE)), np.nan, dtype=float)
        corr_piemd = np.full((len(_H5_PIEMD_NAMES), len(_H5_PIEMD_NAMES)), np.nan, dtype=float)
        corr_fix_with_derived = np.full((len(_H5_PARAM_NAMES_FIX) + 2, len(_H5_PARAM_NAMES_FIX) + 2), np.nan, dtype=float)
        corr_free_with_derived = np.full((len(_H5_PARAM_NAMES_FREE) + 2, len(_H5_PARAM_NAMES_FREE) + 2), np.nan, dtype=float)
        corr_nvalid_fix = 0
        corr_nvalid_free = 0
        corr_nvalid_piemd = 0
        corr_nvalid_fix_with_derived = 0
        corr_nvalid_free_with_derived = 0
        if do_mcmc:
            nsamp = len(sigma_chain)
            n_fix_params = 2 if args.use_nfw else 3  # NFW: rs, rho0; tNFW/BMO/Einasto/gNFW: 3 params
            n_free_params = 4 if args.use_nfw else 5  # NFW: rs, rho0, mstar, rJ; others: 5 params
            fix_samp = np.full((nsamp, n_fix_params), np.nan, dtype=float)
            free_samp = np.full((nsamp, n_free_params), np.nan, dtype=float)
            piemd_samp = np.full((nsamp, 2), np.nan, dtype=float)

            mag_par = float(findInBlock(potentiel[j], "mag"))

            for isamp in range(nsamp):
                try:
                    sig0_s, rcut_s = galaxy_dpie_from_scaling(
                        mag_par, sigma_chain[isamp], rcut_chain[isamp], cosmo, z_lens,
                        alpha=args.alpha, beta_cut=args.beta_cut, mref=args.mref,
                    )
                    if sig0_s <= 0 or rcut_s <= 0:
                        continue
                    piemd_samp[isamp, :] = [sig0_s, rcut_s]

                    rho_piemd_s = piemd_rho(r, r_core, rcut_s, sig0_s)
                    rho_j_fixed_s = jaffe_rho(r, mstar_mag, re_f160w)
                    rho_j_shape_s = jaffe_rho(r, 1.0, re_f160w)

                    rs_g = max(1.0, 0.2 * rcut_s)
                    rt_g = max(rcut_s, 2.0 * rs_g)
                    rho0_g = max(float(np.nanmax(rho_piemd_s) * 1.0e-3), 1.0e-20)

                    if args.use_bmo:
                        # Fixed-Jaffe BMO fit
                        try:
                            rs_f, rt_f, rho0_f, _ = fit_bmo_plus_fixed_jaffe(
                                r, rho_piemd_s, rho_j_fixed_s, rs_g, rt_g, rho0_g,
                                rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w,
                            )
                            if np.isfinite(rs_f) and np.isfinite(rt_f) and np.isfinite(rho0_f):
                                fix_samp[isamp, :] = [rs_f, rt_f, rho0_f]
                        except Exception:
                            pass
                        # Free-Jaffe BMO fit
                        try:
                            rs_r, rt_r, rho0_r, mstar_r, rj_r, _ = fit_bmo_plus_free_jaffe(
                                r, rho_piemd_s, rho_j_shape_s, rs_g, rt_g, rho0_g,
                                mstar_guess=mstar_mag, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w,
                            )
                            if np.isfinite(rs_r) and np.isfinite(rt_r) and np.isfinite(rho0_r) and np.isfinite(mstar_r) and np.isfinite(rj_r):
                                free_samp[isamp, :] = [rs_r, rt_r, rho0_r, mstar_r, rj_r]
                        except Exception:
                            pass
                    elif args.use_nfw:
                        # Fixed-Jaffe NFW fit
                        try:
                            rs_f, rho0_f, _ = fit_nfw_plus_fixed_jaffe(
                                r, rho_piemd_s, rho_j_fixed_s, rs_g, rho0_g,
                                rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w,
                            )
                            if np.isfinite(rs_f) and np.isfinite(rho0_f):
                                fix_samp[isamp, :] = [rs_f, rho0_f]
                        except Exception:
                            pass
                        # Free-Jaffe NFW fit
                        try:
                            rs_r, rho0_r, mstar_r, rj_r, _ = fit_nfw_plus_free_jaffe(
                                r, rho_piemd_s, rho_j_shape_s, rs_g, rho0_g,
                                mstar_guess=mstar_mag, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w,
                            )
                            if np.isfinite(rs_r) and np.isfinite(rho0_r) and np.isfinite(mstar_r) and np.isfinite(rj_r):
                                free_samp[isamp, :] = [rs_r, rho0_r, mstar_r, rj_r]
                        except Exception:
                            pass
                    elif args.use_einasto:
                        # Fixed-Jaffe Einasto fit
                        try:
                            rs_f, rho_s_f, n_f, _ = fit_einasto_plus_fixed_jaffe(
                                r, rho_piemd_s, rho_j_fixed_s, rs_g, n_guess, rho0_g,
                                rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w,
                            )
                            if np.isfinite(rs_f) and np.isfinite(rho_s_f) and np.isfinite(n_f):
                                fix_samp[isamp, :] = [rs_f, rho_s_f, n_f]
                        except Exception:
                            pass
                        # Free-Jaffe Einasto fit
                        try:
                            rs_r, rho_s_r, n_r, mstar_r, rj_r, _ = fit_einasto_plus_free_jaffe(
                                r, rho_piemd_s, rho_j_shape_s, rs_g, n_guess, rho0_g,
                                mstar_guess=mstar_mag, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w,
                            )
                            if np.isfinite(rs_r) and np.isfinite(rho_s_r) and np.isfinite(n_r) and np.isfinite(mstar_r) and np.isfinite(rj_r):
                                free_samp[isamp, :] = [rs_r, rho_s_r, n_r, mstar_r, rj_r]
                        except Exception:
                            pass
                    elif args.use_gnfw:
                        # Fixed-Jaffe gNFW fit
                        try:
                            rs_f, rho_s_f, gamma_f, _ = fit_gnfw_plus_fixed_jaffe(
                                r, rho_piemd_s, rho_j_fixed_s, rs_g, gamma_guess, rho0_g,
                                rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w,
                            )
                            if np.isfinite(rs_f) and np.isfinite(rho_s_f) and np.isfinite(gamma_f):
                                fix_samp[isamp, :] = [rs_f, rho_s_f, gamma_f]
                        except Exception:
                            pass
                        # Free-Jaffe gNFW fit
                        try:
                            rs_r, rho_s_r, gamma_r, mstar_r, rj_r, _ = fit_gnfw_plus_free_jaffe(
                                r, rho_piemd_s, rho_j_shape_s, rs_g, gamma_guess, rho0_g,
                                mstar_guess=mstar_mag, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w,
                            )
                            if np.isfinite(rs_r) and np.isfinite(rho_s_r) and np.isfinite(gamma_r) and np.isfinite(mstar_r) and np.isfinite(rj_r):
                                free_samp[isamp, :] = [rs_r, rho_s_r, gamma_r, mstar_r, rj_r]
                        except Exception:
                            pass
                    else:
                        # Fixed-Jaffe tNFW fit
                        try:
                            rs_f, rt_f, rho0_f, _ = fit_tnfw_plus_fixed_jaffe(
                                r, rho_piemd_s, rho_j_fixed_s, rs_g, rt_g, rho0_g,
                                rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w,
                            )
                            if np.isfinite(rs_f) and np.isfinite(rt_f) and np.isfinite(rho0_f):
                                fix_samp[isamp, :] = [rs_f, rt_f, rho0_f]
                        except Exception:
                            pass
                        # Free-Jaffe tNFW fit
                        try:
                            rs_r, rt_r, rho0_r, mstar_r, rj_r, _ = fit_tnfw_plus_free_jaffe(
                                r, rho_piemd_s, rho_j_shape_s, rs_g, rt_g, rho0_g,
                                mstar_guess=mstar_mag, rmin_fit=args.rmin_fit_kpc, rmax_fit=rmax_fit, re_kpc=re_f160w,
                            )
                            if np.isfinite(rs_r) and np.isfinite(rt_r) and np.isfinite(rho0_r) and np.isfinite(mstar_r) and np.isfinite(rj_r):
                                free_samp[isamp, :] = [rs_r, rt_r, rho0_r, mstar_r, rj_r]
                        except Exception:
                            pass
                except Exception:
                    continue

            # Summarise percentiles for CSV
            def _pct(arr, col):
                vals = arr[:, col]
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    return np.nan, np.nan, np.nan
                return tuple(np.nanpercentile(vals, [16, 50, 84]))

            n_ok_fix = int(np.sum(np.isfinite(fix_samp[:, 0])))
            n_ok_free = int(np.sum(np.isfinite(free_samp[:, 0])))
            mcmc_summary["mcmc_nsamp_total"] = nsamp
            mcmc_summary["mcmc_nsamp_ok_fix"] = n_ok_fix
            mcmc_summary["mcmc_nsamp_ok_free"] = n_ok_free

            if args.use_bmo:
                for ic, pname in enumerate(["rs_bmo_fix", "rt_bmo_fix", "rho0_bmo_fix"]):
                    p16, p50, p84 = _pct(fix_samp, ic)
                    mcmc_summary[f"{pname}_p16"] = p16
                    mcmc_summary[f"{pname}_p50"] = p50
                    mcmc_summary[f"{pname}_p84"] = p84
                for ic, pname in enumerate(["rs_bmo_free", "rt_bmo_free", "rho0_bmo_free", "Mstar_free", "rJ_free"]):
                    p16, p50, p84 = _pct(free_samp, ic)
                    mcmc_summary[f"{pname}_p16"] = p16
                    mcmc_summary[f"{pname}_p50"] = p50
                    mcmc_summary[f"{pname}_p84"] = p84
            elif args.use_nfw:
                for ic, pname in enumerate(["rs_nfw_fix", "rho0_nfw_fix"]):
                    p16, p50, p84 = _pct(fix_samp, ic)
                    mcmc_summary[f"{pname}_p16"] = p16
                    mcmc_summary[f"{pname}_p50"] = p50
                    mcmc_summary[f"{pname}_p84"] = p84
                for ic, pname in enumerate(["rs_nfw_free", "rho0_nfw_free", "Mstar_free", "rJ_free"]):
                    p16, p50, p84 = _pct(free_samp, ic)
                    mcmc_summary[f"{pname}_p16"] = p16
                    mcmc_summary[f"{pname}_p50"] = p50
                    mcmc_summary[f"{pname}_p84"] = p84
            elif args.use_einasto:
                for ic, pname in enumerate(["rs_ein_fix", "rhos_ein_fix", "n_ein_fix"]):
                    p16, p50, p84 = _pct(fix_samp, ic)
                    mcmc_summary[f"{pname}_p16"] = p16
                    mcmc_summary[f"{pname}_p50"] = p50
                    mcmc_summary[f"{pname}_p84"] = p84
                for ic, pname in enumerate(["rs_ein_free", "rhos_ein_free", "n_ein_free", "Mstar_free", "rJ_free"]):
                    p16, p50, p84 = _pct(free_samp, ic)
                    mcmc_summary[f"{pname}_p16"] = p16
                    mcmc_summary[f"{pname}_p50"] = p50
                    mcmc_summary[f"{pname}_p84"] = p84
            elif args.use_gnfw:
                for ic, pname in enumerate(["rs_gnfw_fix", "rhos_gnfw_fix", "gamma_gnfw_fix"]):
                    p16, p50, p84 = _pct(fix_samp, ic)
                    mcmc_summary[f"{pname}_p16"] = p16
                    mcmc_summary[f"{pname}_p50"] = p50
                    mcmc_summary[f"{pname}_p84"] = p84
                for ic, pname in enumerate(["rs_gnfw_free", "rhos_gnfw_free", "gamma_gnfw_free", "Mstar_free", "rJ_free"]):
                    p16, p50, p84 = _pct(free_samp, ic)
                    mcmc_summary[f"{pname}_p16"] = p16
                    mcmc_summary[f"{pname}_p50"] = p50
                    mcmc_summary[f"{pname}_p84"] = p84
            else:
                for ic, pname in enumerate(["rs_fix", "rt_fix", "rho0_fix"]):
                    p16, p50, p84 = _pct(fix_samp, ic)
                    mcmc_summary[f"{pname}_p16"] = p16
                    mcmc_summary[f"{pname}_p50"] = p50
                    mcmc_summary[f"{pname}_p84"] = p84
                for ic, pname in enumerate(["rs_free", "rt_free", "rho0_free", "Mstar_free", "rJ_free"]):
                    p16, p50, p84 = _pct(free_samp, ic)
                    mcmc_summary[f"{pname}_p16"] = p16
                    mcmc_summary[f"{pname}_p50"] = p50
                    mcmc_summary[f"{pname}_p84"] = p84

            # Derive c200, R200, M200 for each MCMC sample and compute percentiles
            rho_crit_z = cosmo.critical_density(z_lens).to_value("Msun/kpc3")
            c200_fix_arr = np.full(nsamp, np.nan, dtype=float)
            c200_free_arr = np.full(nsamp, np.nan, dtype=float)
            m200_fix_arr = np.full(nsamp, np.nan, dtype=float)
            m200_free_arr = np.full(nsamp, np.nan, dtype=float)
            r200_fix_arr = np.full(nsamp, np.nan, dtype=float)
            r200_free_arr = np.full(nsamp, np.nan, dtype=float)

            if args.use_bmo:
                for isamp in range(nsamp):
                    if np.isfinite(fix_samp[isamp, 0]):
                        d = compute_bmo_derived(
                            fix_samp[isamp, 0], fix_samp[isamp, 1], fix_samp[isamp, 2], rho_crit_z
                        )
                        c200_fix_arr[isamp] = d["c200"]
                        r200_fix_arr[isamp] = d["R200_kpc"]
                        m200_fix_arr[isamp] = d["M200_Msun"]
                    if np.isfinite(free_samp[isamp, 0]):
                        d = compute_bmo_derived(
                            free_samp[isamp, 0], free_samp[isamp, 1], free_samp[isamp, 2], rho_crit_z
                        )
                        c200_free_arr[isamp] = d["c200"]
                        r200_free_arr[isamp] = d["R200_kpc"]
                        m200_free_arr[isamp] = d["M200_Msun"]
            elif args.use_nfw:
                for isamp in range(nsamp):
                    if np.isfinite(fix_samp[isamp, 0]):
                        d = compute_std_nfw_derived(fix_samp[isamp, 0], fix_samp[isamp, 1], rho_crit_z)
                        c200_fix_arr[isamp] = d["c200"]
                        r200_fix_arr[isamp] = d["R200_kpc"]
                        m200_fix_arr[isamp] = d["M200_Msun"]
                    if np.isfinite(free_samp[isamp, 0]):
                        d = compute_std_nfw_derived(free_samp[isamp, 0], free_samp[isamp, 1], rho_crit_z)
                        c200_free_arr[isamp] = d["c200"]
                        r200_free_arr[isamp] = d["R200_kpc"]
                        m200_free_arr[isamp] = d["M200_Msun"]
            elif args.use_einasto:
                for isamp in range(nsamp):
                    if np.isfinite(fix_samp[isamp, 0]):
                        d = compute_einasto_derived(
                            fix_samp[isamp, 0], fix_samp[isamp, 1], fix_samp[isamp, 2], rho_crit_z
                        )
                        c200_fix_arr[isamp] = d["c200"]
                        r200_fix_arr[isamp] = d["R200_kpc"]
                        m200_fix_arr[isamp] = d["M200_Msun"]
                    if np.isfinite(free_samp[isamp, 0]):
                        d = compute_einasto_derived(
                            free_samp[isamp, 0], free_samp[isamp, 1], free_samp[isamp, 2], rho_crit_z
                        )
                        c200_free_arr[isamp] = d["c200"]
                        r200_free_arr[isamp] = d["R200_kpc"]
                        m200_free_arr[isamp] = d["M200_Msun"]
            elif args.use_gnfw:
                for isamp in range(nsamp):
                    if np.isfinite(fix_samp[isamp, 0]):
                        d = compute_gnfw_derived(
                            fix_samp[isamp, 0], fix_samp[isamp, 1], fix_samp[isamp, 2], rho_crit_z
                        )
                        c200_fix_arr[isamp] = d["c200"]
                        r200_fix_arr[isamp] = d["R200_kpc"]
                        m200_fix_arr[isamp] = d["M200_Msun"]
                    if np.isfinite(free_samp[isamp, 0]):
                        d = compute_gnfw_derived(
                            free_samp[isamp, 0], free_samp[isamp, 1], free_samp[isamp, 2], rho_crit_z
                        )
                        c200_free_arr[isamp] = d["c200"]
                        r200_free_arr[isamp] = d["R200_kpc"]
                        m200_free_arr[isamp] = d["M200_Msun"]
            else:
                for isamp in range(nsamp):
                    if np.isfinite(fix_samp[isamp, 0]):
                        d = compute_nfw_derived(
                            fix_samp[isamp, 0], fix_samp[isamp, 1], fix_samp[isamp, 2], rho_crit_z
                        )
                        c200_fix_arr[isamp] = d["c200"]
                        m200_fix_arr[isamp] = d["M200_Msun"]
                    if np.isfinite(free_samp[isamp, 0]):
                        d = compute_nfw_derived(
                            free_samp[isamp, 0], free_samp[isamp, 1], free_samp[isamp, 2], rho_crit_z
                        )
                        c200_free_arr[isamp] = d["c200"]
                        m200_free_arr[isamp] = d["M200_Msun"]

            def _pct_list(vals):
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    return np.nan, np.nan, np.nan
                return tuple(np.nanpercentile(vals, [16, 50, 84]))

            # c200 and M200 percentiles are common to all profiles
            for vals, prefix in [
                (c200_fix_arr, "c200_fix"),
                (m200_fix_arr, "M200_fix"),
                (c200_free_arr, "c200_free"),
                (m200_free_arr, "M200_free"),
            ]:
                p16, p50, p84 = _pct_list(vals)
                mcmc_summary[f"{prefix}_p16"] = p16
                mcmc_summary[f"{prefix}_p50"] = p50
                mcmc_summary[f"{prefix}_p84"] = p84

            # R200 percentiles for NFW/BMO/Einasto/gNFW (for tNFW, R200 is derived differently)
            if args.use_bmo:
                for vals, prefix in [(r200_fix_arr, "R200_bmo_fix"), (r200_free_arr, "R200_bmo_free")]:
                    p16, p50, p84 = _pct_list(vals)
                    mcmc_summary[f"{prefix}_p16"] = p16
                    mcmc_summary[f"{prefix}_p50"] = p50
                    mcmc_summary[f"{prefix}_p84"] = p84
            elif args.use_nfw:
                for vals, prefix in [(r200_fix_arr, "R200_nfw_fix"), (r200_free_arr, "R200_nfw_free")]:
                    p16, p50, p84 = _pct_list(vals)
                    mcmc_summary[f"{prefix}_p16"] = p16
                    mcmc_summary[f"{prefix}_p50"] = p50
                    mcmc_summary[f"{prefix}_p84"] = p84
            elif args.use_einasto:
                for vals, prefix in [(r200_fix_arr, "R200_ein_fix"), (r200_free_arr, "R200_ein_free")]:
                    p16, p50, p84 = _pct_list(vals)
                    mcmc_summary[f"{prefix}_p16"] = p16
                    mcmc_summary[f"{prefix}_p50"] = p50
                    mcmc_summary[f"{prefix}_p84"] = p84
            elif args.use_gnfw:
                for vals, prefix in [(r200_fix_arr, "R200_gnfw_fix"), (r200_free_arr, "R200_gnfw_free")]:
                    p16, p50, p84 = _pct_list(vals)
                    mcmc_summary[f"{prefix}_p16"] = p16
                    mcmc_summary[f"{prefix}_p50"] = p50
                    mcmc_summary[f"{prefix}_p84"] = p84

            # Correlation matrices from per-sample MCMC propagated fits.
            corr_fix, corr_nvalid_fix = corrcoef_from_samples(fix_samp)
            corr_free, corr_nvalid_free = corrcoef_from_samples(free_samp)
            corr_piemd, corr_nvalid_piemd = corrcoef_from_samples(piemd_samp)

            fix_with_derived = np.column_stack([fix_samp, c200_fix_arr, m200_fix_arr])
            free_with_derived = np.column_stack([free_samp, c200_free_arr, m200_free_arr])
            corr_fix_with_derived, corr_nvalid_fix_with_derived = corrcoef_from_samples(fix_with_derived)
            corr_free_with_derived, corr_nvalid_free_with_derived = corrcoef_from_samples(free_with_derived)

            mcmc_summary["corr_nvalid_fix"] = corr_nvalid_fix
            mcmc_summary["corr_nvalid_free"] = corr_nvalid_free
            mcmc_summary["corr_nvalid_piemd"] = corr_nvalid_piemd
            mcmc_summary["corr_nvalid_fix_with_derived"] = corr_nvalid_fix_with_derived
            mcmc_summary["corr_nvalid_free_with_derived"] = corr_nvalid_free_with_derived

            # Optional per-galaxy CSV export.
            if args.corr_dir is not None:
                lens_prefix = f"lens{int(j):03d}_cat{int(j_cat):03d}"
                fix_names = list(_H5_PARAM_NAMES_FIX)
                free_names = list(_H5_PARAM_NAMES_FREE)
                piemd_names = list(_H5_PIEMD_NAMES)
                fix_ext_names = fix_names + ["c200_fix", "M200_fix_Msun"]
                free_ext_names = free_names + ["c200_free", "M200_free_Msun"]

                pd.DataFrame(corr_fix, index=fix_names, columns=fix_names).to_csv(
                    Path(args.corr_dir) / f"{lens_prefix}_fix_corr.csv"
                )
                pd.DataFrame(corr_free, index=free_names, columns=free_names).to_csv(
                    Path(args.corr_dir) / f"{lens_prefix}_free_corr.csv"
                )
                pd.DataFrame(corr_piemd, index=piemd_names, columns=piemd_names).to_csv(
                    Path(args.corr_dir) / f"{lens_prefix}_piemd_corr.csv"
                )
                pd.DataFrame(corr_fix_with_derived, index=fix_ext_names, columns=fix_ext_names).to_csv(
                    Path(args.corr_dir) / f"{lens_prefix}_fix_corr_with_derived.csv"
                )
                pd.DataFrame(corr_free_with_derived, index=free_ext_names, columns=free_ext_names).to_csv(
                    Path(args.corr_dir) / f"{lens_prefix}_free_corr_with_derived.csv"
                )

        # --- Build output row ---
        rho_crit_z = cosmo.critical_density(z_lens).to_value("Msun/kpc3")

        row = {
            "lens_index": int(j),
            "cat_index": int(j_cat),
            "ra_lens": float(ra_lens[j]),
            "dec_lens": float(dec_lens[j]),
            "z_lens": z_lens,
            "F160W": mag_f160w,
            "ReF160W_kpc": re_f160w,
            "rmax_fit_kpc": rmax_fit if rmax_fit is not None else np.nan,
            "v_disp_sigma0_km_s": sigma0,
            "core_radius_kpc": r_core,
            "cut_radius_kpc": r_cut,
            "Mstar_from_F160W_Msun": mstar_mag,
            "negative_center_fixedJaffe": negative_center_fixed,
        }

        if args.use_bmo:
            # BMO derived quantities
            bmo_fix = compute_bmo_derived(rs_fix, rt_fix, rho0_fix, rho_crit_z) if ok_fix else {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
            bmo_free = compute_bmo_derived(rs_free, rt_free, rho0_free, rho_crit_z) if ok_free else {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
            row.update({
                "dm_profile": "BMO",
                "rs_bmo_fix_kpc": rs_fix,
                "rt_bmo_fix_kpc": rt_fix if ok_fix else np.nan,
                "rho0_bmo_fix_msun_kpc3": rho0_fix if ok_fix else np.nan,
                "fit_fix_rmse_log": diag_fix["rmse_log"],
                "fit_fix_r2_log": diag_fix["r2_log"],
                "fit_fix_npts": diag_fix["npts"],
                "fit_fix_success": diag_fix["success"],
                "fit_fix_message": diag_fix["message"],
                "fit_fix_jaffe_clamped": diag_fix.get("jaffe_clamped", False),
                "c200_fix": bmo_fix["c200"],
                "R200_bmo_fix_kpc": bmo_fix["R200_kpc"],
                "M200_bmo_fix_Msun": bmo_fix["M200_Msun"],
                "rs_bmo_free_kpc": rs_free,
                "rt_bmo_free_kpc": rt_free if ok_free else np.nan,
                "rho0_bmo_free_msun_kpc3": rho0_free if ok_free else np.nan,
                "Mstar_free_Msun": mstar_free,
                "rJ_free_kpc": rJ_free if ok_free else np.nan,
                "fit_free_rmse_log": diag_free["rmse_log"],
                "fit_free_r2_log": diag_free["r2_log"],
                "fit_free_npts": diag_free["npts"],
                "fit_free_success": diag_free["success"],
                "fit_free_message": diag_free["message"],
                "c200_free": bmo_free["c200"],
                "R200_bmo_free_kpc": bmo_free["R200_kpc"],
                "M200_bmo_free_Msun": bmo_free["M200_Msun"],
                "Mstar_max_constraint_Msun": diag_free.get("mstar_max_constraint", np.nan),
                "Mstar_ratio_free_over_phot": mstar_free / mstar_mag if np.isfinite(mstar_free) and mstar_mag > 0 else np.nan,
                "Mstar_log_ratio_free_over_phot": np.log10(mstar_free / mstar_mag) if np.isfinite(mstar_free) and mstar_free > 0 and mstar_mag > 0 else np.nan,
                "Mstar_ratio_phot_over_max": mstar_mag / diag_free.get("mstar_max_constraint", np.nan) if mstar_mag > 0 else np.nan,
            })
        elif args.use_nfw:
            # Standard NFW derived quantities
            nfw_s_fix = compute_std_nfw_derived(rs_fix, rho0_fix, rho_crit_z) if ok_fix else {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
            nfw_s_free = compute_std_nfw_derived(rs_free, rho0_free, rho_crit_z) if ok_free else {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
            row.update({
                "dm_profile": "NFW",
                "rs_nfw_fix_kpc": rs_fix,
                "rho0_nfw_fix_msun_kpc3": rho0_fix if ok_fix else np.nan,
                "fit_fix_rmse_log": diag_fix["rmse_log"],
                "fit_fix_r2_log": diag_fix["r2_log"],
                "fit_fix_npts": diag_fix["npts"],
                "fit_fix_success": diag_fix["success"],
                "fit_fix_message": diag_fix["message"],
                "fit_fix_jaffe_clamped": diag_fix.get("jaffe_clamped", False),
                "c200_fix": nfw_s_fix["c200"],
                "R200_nfw_fix_kpc": nfw_s_fix["R200_kpc"],
                "M200_nfw_fix_Msun": nfw_s_fix["M200_Msun"],
                "rs_nfw_free_kpc": rs_free,
                "rho0_nfw_free_msun_kpc3": rho0_free if ok_free else np.nan,
                "Mstar_free_Msun": mstar_free,
                "rJ_free_kpc": rJ_free if ok_free else np.nan,
                "fit_free_rmse_log": diag_free["rmse_log"],
                "fit_free_r2_log": diag_free["r2_log"],
                "fit_free_npts": diag_free["npts"],
                "fit_free_success": diag_free["success"],
                "fit_free_message": diag_free["message"],
                "c200_free": nfw_s_free["c200"],
                "R200_nfw_free_kpc": nfw_s_free["R200_kpc"],
                "M200_nfw_free_Msun": nfw_s_free["M200_Msun"],
                "Mstar_max_constraint_Msun": diag_free.get("mstar_max_constraint", np.nan),
                "Mstar_ratio_free_over_phot": mstar_free / mstar_mag if np.isfinite(mstar_free) and mstar_mag > 0 else np.nan,
                "Mstar_log_ratio_free_over_phot": np.log10(mstar_free / mstar_mag) if np.isfinite(mstar_free) and mstar_free > 0 and mstar_mag > 0 else np.nan,
                "Mstar_ratio_phot_over_max": mstar_mag / diag_free.get("mstar_max_constraint", np.nan) if mstar_mag > 0 else np.nan,
            })
        elif args.use_einasto:
            # Einasto derived quantities
            ein_fix = compute_einasto_derived(rs_fix, rho_s_fix, n_fix, rho_crit_z) if ok_fix else {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
            ein_free = compute_einasto_derived(rs_free, rho_s_free, n_free, rho_crit_z) if ok_free else {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
            row.update({
                "dm_profile": "einasto",
                "rs_ein_fix_kpc": rs_fix,
                "rhos_ein_fix_msun_kpc3": rho_s_fix if ok_fix else np.nan,
                "n_ein_fix": n_fix if ok_fix else np.nan,
                "fit_fix_rmse_log": diag_fix["rmse_log"],
                "fit_fix_r2_log": diag_fix["r2_log"],
                "fit_fix_npts": diag_fix["npts"],
                "fit_fix_success": diag_fix["success"],
                "fit_fix_message": diag_fix["message"],
                "fit_fix_jaffe_clamped": diag_fix.get("jaffe_clamped", False),
                "c200_fix": ein_fix["c200"],
                "R200_ein_fix_kpc": ein_fix["R200_kpc"],
                "M200_ein_fix_Msun": ein_fix["M200_Msun"],
                "rs_ein_free_kpc": rs_free,
                "rhos_ein_free_msun_kpc3": rho_s_free if ok_free else np.nan,
                "n_ein_free": n_free if ok_free else np.nan,
                "Mstar_free_Msun": mstar_free,
                "rJ_free_kpc": rJ_free if ok_free else np.nan,
                "fit_free_rmse_log": diag_free["rmse_log"],
                "fit_free_r2_log": diag_free["r2_log"],
                "fit_free_npts": diag_free["npts"],
                "fit_free_success": diag_free["success"],
                "fit_free_message": diag_free["message"],
                "c200_free": ein_free["c200"],
                "R200_ein_free_kpc": ein_free["R200_kpc"],
                "M200_ein_free_Msun": ein_free["M200_Msun"],
                "Mstar_max_constraint_Msun": diag_free.get("mstar_max_constraint", np.nan),
                "Mstar_ratio_free_over_phot": mstar_free / mstar_mag if np.isfinite(mstar_free) and mstar_mag > 0 else np.nan,
                "Mstar_log_ratio_free_over_phot": np.log10(mstar_free / mstar_mag) if np.isfinite(mstar_free) and mstar_free > 0 and mstar_mag > 0 else np.nan,
                "Mstar_ratio_phot_over_max": mstar_mag / diag_free.get("mstar_max_constraint", np.nan) if mstar_mag > 0 else np.nan,
            })
        elif args.use_gnfw:
            # gNFW derived quantities
            gnfw_fix = compute_gnfw_derived(rs_fix, rho_s_fix, gamma_fix, rho_crit_z) if ok_fix else {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
            gnfw_free = compute_gnfw_derived(rs_free, rho_s_free, gamma_free, rho_crit_z) if ok_free else {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
            row.update({
                "dm_profile": "gNFW",
                "rs_gnfw_fix_kpc": rs_fix,
                "rhos_gnfw_fix_msun_kpc3": rho_s_fix if ok_fix else np.nan,
                "gamma_gnfw_fix": gamma_fix if ok_fix else np.nan,
                "fit_fix_rmse_log": diag_fix["rmse_log"],
                "fit_fix_r2_log": diag_fix["r2_log"],
                "fit_fix_npts": diag_fix["npts"],
                "fit_fix_success": diag_fix["success"],
                "fit_fix_message": diag_fix["message"],
                "fit_fix_jaffe_clamped": diag_fix.get("jaffe_clamped", False),
                "c200_fix": gnfw_fix["c200"],
                "R200_gnfw_fix_kpc": gnfw_fix["R200_kpc"],
                "M200_gnfw_fix_Msun": gnfw_fix["M200_Msun"],
                "rs_gnfw_free_kpc": rs_free,
                "rhos_gnfw_free_msun_kpc3": rho_s_free if ok_free else np.nan,
                "gamma_gnfw_free": gamma_free if ok_free else np.nan,
                "Mstar_free_Msun": mstar_free,
                "rJ_free_kpc": rJ_free if ok_free else np.nan,
                "fit_free_rmse_log": diag_free["rmse_log"],
                "fit_free_r2_log": diag_free["r2_log"],
                "fit_free_npts": diag_free["npts"],
                "fit_free_success": diag_free["success"],
                "fit_free_message": diag_free["message"],
                "c200_free": gnfw_free["c200"],
                "R200_gnfw_free_kpc": gnfw_free["R200_kpc"],
                "M200_gnfw_free_Msun": gnfw_free["M200_Msun"],
                "Mstar_max_constraint_Msun": diag_free.get("mstar_max_constraint", np.nan),
                "Mstar_ratio_free_over_phot": mstar_free / mstar_mag if np.isfinite(mstar_free) and mstar_mag > 0 else np.nan,
                "Mstar_log_ratio_free_over_phot": np.log10(mstar_free / mstar_mag) if np.isfinite(mstar_free) and mstar_free > 0 and mstar_mag > 0 else np.nan,
                "Mstar_ratio_phot_over_max": mstar_mag / diag_free.get("mstar_max_constraint", np.nan) if mstar_mag > 0 else np.nan,
            })
        else:
            nfw_fix = compute_nfw_derived(rs_fix, rt_fix, rho0_fix, rho_crit_z)
            nfw_free = compute_nfw_derived(rs_free, rt_free, rho0_free, rho_crit_z)
            row.update({
                "dm_profile": "tNFW",
                "rs_fix_kpc": rs_fix,
                "rt_fix_kpc": rt_fix,
                "rho0_fix_msun_kpc3": rho0_fix,
                "fit_fix_rmse_log": diag_fix["rmse_log"],
                "fit_fix_r2_log": diag_fix["r2_log"],
                "fit_fix_npts": diag_fix["npts"],
                "fit_fix_success": diag_fix["success"],
                "fit_fix_message": diag_fix["message"],
                "fit_fix_jaffe_clamped": diag_fix.get("jaffe_clamped", False),
                "c200_fix": nfw_fix["c200"],
                "R200_fix_kpc": nfw_fix["R200_kpc"],
                "M200_fix_Msun": nfw_fix["M200_Msun"],
                "Mtot_tNFW_fix_Msun": nfw_fix["Mtot_tNFW_Msun"],
                "rs_free_kpc": rs_free,
                "rt_free_kpc": rt_free,
                "rho0_free_msun_kpc3": rho0_free,
                "Mstar_free_Msun": mstar_free,
                "rJ_free_kpc": rJ_free if ok_free else np.nan,
                "fit_free_rmse_log": diag_free["rmse_log"],
                "fit_free_r2_log": diag_free["r2_log"],
                "fit_free_npts": diag_free["npts"],
                "fit_free_success": diag_free["success"],
                "fit_free_message": diag_free["message"],
                "c200_free": nfw_free["c200"],
                "R200_free_kpc": nfw_free["R200_kpc"],
                "M200_free_Msun": nfw_free["M200_Msun"],
                "Mtot_tNFW_free_Msun": nfw_free["Mtot_tNFW_Msun"],
                "Mstar_max_constraint_Msun": diag_free.get("mstar_max_constraint", np.nan),
                "Mstar_ratio_free_over_phot": mstar_free / mstar_mag if np.isfinite(mstar_free) and mstar_mag > 0 else np.nan,
                "Mstar_log_ratio_free_over_phot": np.log10(mstar_free / mstar_mag) if np.isfinite(mstar_free) and mstar_free > 0 and mstar_mag > 0 else np.nan,
                "Mstar_ratio_phot_over_max": mstar_mag / diag_free.get("mstar_max_constraint", np.nan) if mstar_mag > 0 else np.nan,
            })

        row.update(mcmc_summary)
        rows.append(row)

        # Collect per-sample cubes for HDF5
        if do_mcmc and fix_samp is not None:
            _h5_lens_idx.append(int(j))
            _h5_cat_idx.append(int(j_cat))
            _h5_ra.append(float(ra_lens[j]))
            _h5_dec.append(float(dec_lens[j]))
            _h5_fix_cube.append(fix_samp)
            _h5_free_cube.append(free_samp)
            _h5_piemd_cube.append(piemd_samp)
            _h5_fix_corr.append(corr_fix)
            _h5_free_corr.append(corr_free)
            _h5_piemd_corr.append(corr_piemd)
            _h5_fix_corr_with_derived.append(corr_fix_with_derived)
            _h5_free_corr_with_derived.append(corr_free_with_derived)
            _h5_corr_nvalid_fix.append(int(corr_nvalid_fix))
            _h5_corr_nvalid_free.append(int(corr_nvalid_free))
            _h5_corr_nvalid_piemd.append(int(corr_nvalid_piemd))
            _h5_corr_nvalid_fix_with_derived.append(int(corr_nvalid_fix_with_derived))
            _h5_corr_nvalid_free_with_derived.append(int(corr_nvalid_free_with_derived))

        if args.max_plots > 0 and nplot < args.max_plots:
            if args.use_bmo:
                dm_label = "BMO"
            elif args.use_nfw:
                dm_label = "NFW"
            elif args.use_einasto:
                dm_label = "Einasto"
            elif args.use_gnfw:
                dm_label = "gNFW"
            else:
                dm_label = "tNFW"
            fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.4), constrained_layout=True)
            ax.loglog(r, rho_piemd, color="black", lw=2.0, label="PIEMD total")
            ax.loglog(r, rho_j_fixed, color="tab:blue", lw=1.4, label="Jaffe fixed (from F160W)")

            if ok_fix:
                if args.use_bmo:
                    rho_dm_fix = bmo_rho(r, rs_fix, rt_fix, rho0_fix)
                elif args.use_nfw:
                    rho_dm_fix = nfw_rho(r, rs_fix, rho0_fix)
                elif args.use_einasto:
                    rho_dm_fix = einasto_rho(r, rs_fix, rho_s_fix, n_fix)
                elif args.use_gnfw:
                    rho_dm_fix = gnfw_rho(r, rs_fix, rho_s_fix, gamma_fix)
                else:
                    rho_dm_fix = trunc_nfw_rho(r, rs_fix, rt_fix, rho0_fix)
                ax.loglog(r, rho_dm_fix, color="tab:orange", lw=1.3, label=f"{dm_label} fixed mode")
                ax.loglog(r, rho_dm_fix + rho_j_fixed, color="tab:green", lw=1.6, ls="--", label=f"{dm_label}+Jaffe fixed")

            if ok_free:
                rho_j_free = jaffe_rho_rj(r, mstar_free, rJ_free)
                if args.use_bmo:
                    rho_dm_free = bmo_rho(r, rs_free, rt_free, rho0_free)
                elif args.use_nfw:
                    rho_dm_free = nfw_rho(r, rs_free, rho0_free)
                elif args.use_einasto:
                    rho_dm_free = einasto_rho(r, rs_free, rho_s_free, n_free)
                elif args.use_gnfw:
                    rho_dm_free = gnfw_rho(r, rs_free, rho_s_free, gamma_free)
                else:
                    rho_dm_free = trunc_nfw_rho(r, rs_free, rt_free, rho0_free)
                ax.loglog(r, rho_j_free, color="tab:purple", lw=1.2, label=f"Jaffe free-fit (rJ={rJ_free:.2f} kpc)")
                ax.loglog(r, rho_dm_free, color="tab:orange", lw=1.2, ls=":", label=f"{dm_label} free-fit")
                ax.loglog(r, rho_dm_free + rho_j_free, color="tab:red", lw=1.6, ls=":", label=f"{dm_label}+Jaffe free")

            ax.axvline(re_f160w, color="tab:blue", lw=1.0, ls=":", label=f"Re={re_f160w:.2f} kpc")
            ax.set_xlabel("r [kpc]")
            ax.set_ylabel(r"$\rho(r)$ [Msun/kpc$^3$]")
            ax.set_title(
                f"Lens {j} / Cat {j_cat} | F160W={mag_f160w:.2f} | DM={dm_label}\n"
                f"fixed rmse={diag_fix['rmse_log']:.3f}, free rmse={diag_free['rmse_log']:.3f}"
            )
            ax.legend(fontsize=8)
            out_png = Path(args.plot_dir) / f"fit_{dm_tag}_jaffe_dual_lens{j:03d}.png"
            fig.savefig(out_png, dpi=160)
            plt.close(fig)

            # --- Diagnostic histograms: PIEMD params from bayes chains vs .par ---
            if do_mcmc and piemd_samp is not None:
                valid_mask = np.all(np.isfinite(piemd_samp), axis=1)
                if valid_mask.sum() > 10:
                    sig0_chain = piemd_samp[valid_mask, 0]  # sigma0 (km/s) from chains
                    rcut_chain_vals = piemd_samp[valid_mask, 1]  # rcut (kpc) from chains

                    fig_h, axes_h = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

                    # --- sigma0 histogram ---
                    ax_s = axes_h[0]
                    ax_s.hist(sig0_chain, bins=40, density=True, alpha=0.65,
                              color="steelblue", edgecolor="white", lw=0.4, label="MCMC chains")
                    ax_s.axvline(sigma0, color="red", lw=2.0, ls="--",
                                 label=f".par best-fit = {sigma0:.2f}")
                    p16, p50, p84 = np.percentile(sig0_chain, [16, 50, 84])
                    ax_s.axvline(p50, color="orange", lw=1.5, ls="-",
                                 label=f"median = {p50:.2f}")
                    ax_s.axvspan(p16, p84, alpha=0.12, color="orange",
                                 label=f"68% CI [{p16:.2f}, {p84:.2f}]")
                    ax_s.set_xlabel(r"$\sigma_0$ [km/s]")
                    ax_s.set_ylabel("Density")
                    ax_s.set_title(f"Lens {j} — $\\sigma_0$ distribution")
                    ax_s.legend(fontsize=7)

                    # --- rcut histogram ---
                    ax_r = axes_h[1]
                    ax_r.hist(rcut_chain_vals, bins=40, density=True, alpha=0.65,
                              color="seagreen", edgecolor="white", lw=0.4, label="MCMC chains")
                    ax_r.axvline(r_cut, color="red", lw=2.0, ls="--",
                                 label=f".par best-fit = {r_cut:.2f}")
                    p16, p50, p84 = np.percentile(rcut_chain_vals, [16, 50, 84])
                    ax_r.axvline(p50, color="orange", lw=1.5, ls="-",
                                 label=f"median = {p50:.2f}")
                    ax_r.axvspan(p16, p84, alpha=0.12, color="orange",
                                 label=f"68% CI [{p16:.2f}, {p84:.2f}]")
                    ax_r.set_xlabel(r"$r_{\rm cut}$ [kpc]")
                    ax_r.set_ylabel("Density")
                    ax_r.set_title(f"Lens {j} — $r_{{\\rm cut}}$ distribution")
                    ax_r.legend(fontsize=7)

                    fig_h.suptitle(
                        f"Lens {j} / Cat {j_cat} | F160W={mag_f160w:.2f} | "
                        f"N_valid={valid_mask.sum()}/{len(piemd_samp)}",
                        fontsize=10)
                    out_hist = Path(args.plot_dir) / f"hist_piemd_params_lens{j:03d}.png"
                    fig_h.savefig(out_hist, dpi=160)
                    plt.close(fig_h)

            nplot += 1

        print(
            f"[fit] lens={j:3d} cat={j_cat:3d} "
            f"fix_rmse={diag_fix['rmse_log']:.3f} free_rmse={diag_free['rmse_log']:.3f} "
            f"M*mag={mstar_mag:.3e} M*free={mstar_free:.3e}"
        )

        if args.max_galaxies is not None and len(rows) >= args.max_galaxies:
            break

    if len(rows) == 0:
        raise RuntimeError("No successful galaxy processing was produced.")

    df_out = pd.DataFrame(rows).sort_values("lens_index").reset_index(drop=True)
    df_out.to_csv(args.outcsv, index=False)
    print(f"Saved {len(df_out)} rows to {args.outcsv}")
    print(df_out.head(10).to_string(index=False))

    # --- Save per-sample MCMC cubes to HDF5 ---
    if do_mcmc and len(_h5_fix_cube) > 0:
        fix_cube = np.stack(_h5_fix_cube, axis=0)    # (Ngal, Nsamp, Npar_fix)
        free_cube = np.stack(_h5_free_cube, axis=0)   # (Ngal, Nsamp, Npar_free)
        piemd_cube = np.stack(_h5_piemd_cube, axis=0) # (Ngal, Nsamp, 2)
        fix_corr_cube = np.stack(_h5_fix_corr, axis=0)  # (Ngal, Npar_fix, Npar_fix)
        free_corr_cube = np.stack(_h5_free_corr, axis=0)  # (Ngal, Npar_free, Npar_free)
        piemd_corr_cube = np.stack(_h5_piemd_corr, axis=0)  # (Ngal, 2, 2)
        fix_corr_ext_cube = np.stack(_h5_fix_corr_with_derived, axis=0)  # (Ngal, Npar_fix+2, Npar_fix+2)
        free_corr_ext_cube = np.stack(_h5_free_corr_with_derived, axis=0)  # (Ngal, Npar_free+2, Npar_free+2)
        corr_nvalid_fix = np.asarray(_h5_corr_nvalid_fix, dtype=int)
        corr_nvalid_free = np.asarray(_h5_corr_nvalid_free, dtype=int)
        corr_nvalid_piemd = np.asarray(_h5_corr_nvalid_piemd, dtype=int)
        corr_nvalid_fix_ext = np.asarray(_h5_corr_nvalid_fix_with_derived, dtype=int)
        corr_nvalid_free_ext = np.asarray(_h5_corr_nvalid_free_with_derived, dtype=int)
        fix_corr_ext_names = np.asarray(list(_H5_PARAM_NAMES_FIX) + ["c200_fix", "M200_fix_Msun"], dtype="S")
        free_corr_ext_names = np.asarray(list(_H5_PARAM_NAMES_FREE) + ["c200_free", "M200_free_Msun"], dtype="S")
        nsamp = fix_cube.shape[1]

        with h5py.File(args.outh5, "w") as f:
            f.create_dataset("lens_index", data=np.asarray(_h5_lens_idx, dtype=int))
            f.create_dataset("cat_index", data=np.asarray(_h5_cat_idx, dtype=int))
            f.create_dataset("ra_lens_deg", data=np.asarray(_h5_ra, dtype=float))
            f.create_dataset("dec_lens_deg", data=np.asarray(_h5_dec, dtype=float))
            f.create_dataset("sample_id", data=np.arange(nsamp, dtype=int))

            # Fixed-Jaffe fit cubes
            f.create_dataset("fix_param_names", data=np.asarray(_H5_PARAM_NAMES_FIX, dtype="S"))
            f.create_dataset("fix_params", data=fix_cube, compression="gzip", shuffle=True)
            if args.use_bmo:
                f["fix_params"].attrs["description"] = (
                    "Per-sample fixed-Jaffe BMO (n=0.5) fit parameters: shape (N_gal, N_samp, 3). "
                    "Parameter order: [rs_bmo_kpc, rt_bmo_kpc, rho0_bmo_Msun_kpc3]."
                )
            elif args.use_nfw:
                f["fix_params"].attrs["description"] = (
                    "Per-sample fixed-Jaffe NFW fit parameters: shape (N_gal, N_samp, 2). "
                    "Parameter order: [rs_nfw_kpc, rho0_nfw_Msun_kpc3]."
                )
            elif args.use_einasto:
                f["fix_params"].attrs["description"] = (
                    "Per-sample fixed-Jaffe Einasto fit parameters: shape (N_gal, N_samp, 3). "
                    "Parameter order: [rs_ein_kpc, rhos_ein_Msun_kpc3, n_ein]."
                )
            elif args.use_gnfw:
                f["fix_params"].attrs["description"] = (
                    "Per-sample fixed-Jaffe gNFW fit parameters: shape (N_gal, N_samp, 3). "
                    "Parameter order: [rs_gnfw_kpc, rhos_gnfw_Msun_kpc3, gamma_gnfw]."
                )
            else:
                f["fix_params"].attrs["description"] = (
                    "Per-sample fixed-Jaffe tNFW fit parameters: shape (N_gal, N_samp, 3). "
                    "Parameter order: [rs_kpc, rt_kpc, rho0_Msun_kpc3]."
                )

            # Free-Jaffe fit cubes
            f.create_dataset("free_param_names", data=np.asarray(_H5_PARAM_NAMES_FREE, dtype="S"))
            f.create_dataset("free_params", data=free_cube, compression="gzip", shuffle=True)
            if args.use_bmo:
                f["free_params"].attrs["description"] = (
                    "Per-sample free-Jaffe BMO+Jaffe fit parameters: shape (N_gal, N_samp, 5). "
                    "Parameter order: [rs_bmo_kpc, rt_bmo_kpc, rho0_bmo_Msun_kpc3, Mstar_free_Msun, rJ_free_kpc]."
                )
            elif args.use_nfw:
                f["free_params"].attrs["description"] = (
                    "Per-sample free-Jaffe NFW+Jaffe fit parameters: shape (N_gal, N_samp, 4). "
                    "Parameter order: [rs_nfw_kpc, rho0_nfw_Msun_kpc3, Mstar_free_Msun, rJ_free_kpc]."
                )
            elif args.use_einasto:
                f["free_params"].attrs["description"] = (
                    "Per-sample free-Jaffe Einasto+Jaffe fit parameters: shape (N_gal, N_samp, 5). "
                    "Parameter order: [rs_ein_kpc, rhos_ein_Msun_kpc3, n_ein, Mstar_free_Msun, rJ_free_kpc]."
                )
            elif args.use_gnfw:
                f["free_params"].attrs["description"] = (
                    "Per-sample free-Jaffe gNFW+Jaffe fit parameters: shape (N_gal, N_samp, 5). "
                    "Parameter order: [rs_gnfw_kpc, rhos_gnfw_Msun_kpc3, gamma_gnfw, Mstar_free_Msun, rJ_free_kpc]."
                )
            else:
                f["free_params"].attrs["description"] = (
                    "Per-sample free-Jaffe tNFW+Jaffe fit parameters: shape (N_gal, N_samp, 5). "
                    "Parameter order: [rs_kpc, rt_kpc, rho0_Msun_kpc3, Mstar_free_Msun, rJ_free_kpc]."
                )

            # PIEMD parameters per sample
            f.create_dataset("piemd_param_names", data=np.asarray(_H5_PIEMD_NAMES, dtype="S"))
            f.create_dataset("piemd_params", data=piemd_cube, compression="gzip", shuffle=True)
            f["piemd_params"].attrs["description"] = (
                "Per-sample PIEMD parameters from scaling relations: shape (N_gal, N_samp, 2). "
                "Parameter order: [sigma0_km_s, rcut_kpc]."
            )

            # Correlation matrices per galaxy from MCMC propagated samples.
            f.create_dataset("fix_corrcoef", data=fix_corr_cube, compression="gzip", shuffle=True)
            f.create_dataset("free_corrcoef", data=free_corr_cube, compression="gzip", shuffle=True)
            f.create_dataset("piemd_corrcoef", data=piemd_corr_cube, compression="gzip", shuffle=True)
            f["fix_corrcoef"].attrs["description"] = (
                "Per-galaxy correlation matrices from valid rows of fix_params. "
                "Shape: (N_gal, Npar_fix, Npar_fix)."
            )
            f["free_corrcoef"].attrs["description"] = (
                "Per-galaxy correlation matrices from valid rows of free_params. "
                "Shape: (N_gal, Npar_free, Npar_free)."
            )
            f["piemd_corrcoef"].attrs["description"] = (
                "Per-galaxy correlation matrices from valid rows of piemd_params. "
                "Shape: (N_gal, 2, 2)."
            )

            f.create_dataset("fix_corrcoef_with_derived", data=fix_corr_ext_cube, compression="gzip", shuffle=True)
            f.create_dataset("free_corrcoef_with_derived", data=free_corr_ext_cube, compression="gzip", shuffle=True)
            f.create_dataset("fix_corrcoef_with_derived_names", data=fix_corr_ext_names)
            f.create_dataset("free_corrcoef_with_derived_names", data=free_corr_ext_names)
            f["fix_corrcoef_with_derived"].attrs["description"] = (
                "Per-galaxy correlation matrices from [fix_params, c200_fix, M200_fix_Msun]. "
                "Shape: (N_gal, Npar_fix+2, Npar_fix+2)."
            )
            f["free_corrcoef_with_derived"].attrs["description"] = (
                "Per-galaxy correlation matrices from [free_params, c200_free, M200_free_Msun]. "
                "Shape: (N_gal, Npar_free+2, Npar_free+2)."
            )

            f.create_dataset("corr_nvalid_fix", data=corr_nvalid_fix)
            f.create_dataset("corr_nvalid_free", data=corr_nvalid_free)
            f.create_dataset("corr_nvalid_piemd", data=corr_nvalid_piemd)
            f.create_dataset("corr_nvalid_fix_with_derived", data=corr_nvalid_fix_ext)
            f.create_dataset("corr_nvalid_free_with_derived", data=corr_nvalid_free_ext)

            # Metadata
            f.attrs["dm_profile"] = dm_tag
            f.attrs["bayesfile"] = str(args.bayesfile)
            f.attrs["parfile"] = str(args.parfile)
            f.attrs["catalog"] = str(args.catalog)
            f.attrs["mref_F160W"] = float(args.mref)
            f.attrs["alpha"] = float(args.alpha)
            f.attrs["beta_cut"] = float(args.beta_cut)
            f.attrs["rmin_fit_kpc"] = float(args.rmin_fit_kpc)
            f.attrs["rmax_re_factor"] = float(args.rmax_re_factor) if args.rmax_re_factor is not None else -1.0

        print(f"Saved per-sample MCMC parameters to HDF5: {args.outh5}")
        print(f"  fix_params:  {fix_cube.shape}  (NaN frac: {np.isnan(fix_cube).mean():.1%})")
        print(f"  free_params: {free_cube.shape}  (NaN frac: {np.isnan(free_cube).mean():.1%})")
        print(f"  fix_corrcoef:  {fix_corr_cube.shape}")
        print(f"  free_corrcoef: {free_corr_cube.shape}")
        if args.corr_dir is not None:
            print(f"Saved per-galaxy correlation CSVs to: {args.corr_dir}")

    # --- Scatter plot: M* photometry vs M* free Jaffe fit ---
    plot_mstar_comparison(df_out, args.outcsv)


def plot_mstar_comparison(df_out, outcsv):
    """Scatter plot comparing stellar masses from photometry and from the free Jaffe fit."""
    mask = (
        np.isfinite(df_out["Mstar_from_F160W_Msun"])
        & np.isfinite(df_out["Mstar_free_Msun"])
        & (df_out["Mstar_from_F160W_Msun"] > 0)
        & (df_out["Mstar_free_Msun"] > 0)
    )
    df_plot = df_out[mask].copy()
    if len(df_plot) == 0:
        print("No valid data for M* comparison plot.")
        return

    mstar_phot = df_plot["Mstar_from_F160W_Msun"].values
    mstar_free = df_plot["Mstar_free_Msun"].values
    mstar_max = df_plot["Mstar_max_constraint_Msun"].values
    clamped = df_plot["fit_fix_jaffe_clamped"].values.astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # --- Left panel: M*_phot vs M*_free ---
    ax = axes[0]
    lo = min(np.nanmin(mstar_phot), np.nanmin(mstar_free)) * 0.3
    hi = max(np.nanmax(mstar_phot), np.nanmax(mstar_free)) * 3.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1")

    ax.scatter(
        mstar_phot[~clamped], mstar_free[~clamped],
        c="tab:blue", s=20, alpha=0.7, edgecolors="none",
        label=f"unclamped ({(~clamped).sum()})",
    )
    ax.scatter(
        mstar_phot[clamped], mstar_free[clamped],
        c="tab:red", s=30, alpha=0.8, marker="^", edgecolors="none",
        label=f"clamped ({clamped.sum()})",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$M_\star^{\rm phot}$ (F160W) [M$_\odot$]")
    ax.set_ylabel(r"$M_\star^{\rm free}$ (Jaffe fit) [M$_\odot$]")
    ax.set_title("Photometric vs Free-fit Stellar Mass")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    # --- Right panel: M*_phot vs M*_max ---
    ax2 = axes[1]
    valid_max = np.isfinite(mstar_max) & (mstar_max > 0)
    if np.any(valid_max):
        lo2 = min(np.nanmin(mstar_phot[valid_max]), np.nanmin(mstar_max[valid_max])) * 0.3
        hi2 = max(np.nanmax(mstar_phot[valid_max]), np.nanmax(mstar_max[valid_max])) * 3.0
        ax2.plot([lo2, hi2], [lo2, hi2], "k--", lw=1.0, label="1:1")

        ax2.scatter(
            mstar_phot[valid_max & ~clamped], mstar_max[valid_max & ~clamped],
            c="tab:blue", s=20, alpha=0.7, edgecolors="none",
            label=f"unclamped ({(valid_max & ~clamped).sum()})",
        )
        ax2.scatter(
            mstar_phot[valid_max & clamped], mstar_max[valid_max & clamped],
            c="tab:red", s=30, alpha=0.8, marker="^", edgecolors="none",
            label=f"clamped ({(valid_max & clamped).sum()})",
        )

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlim(lo2, hi2)
        ax2.set_ylim(lo2, hi2)
        ax2.set_aspect("equal", adjustable="box")

    ax2.set_xlabel(r"$M_\star^{\rm phot}$ (F160W) [M$_\odot$]")
    ax2.set_ylabel(r"$M_\star^{\rm max}$ (PIEMD constraint) [M$_\odot$]")
    ax2.set_title(r"Photometric $M_\star$ vs Max Allowed $M_\star$")
    ax2.legend(fontsize=9, loc="upper left")

    out_png = Path(outcsv).with_suffix(".mstar_comparison.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved M* comparison plot to {out_png}")


if __name__ == "__main__":
    main()
