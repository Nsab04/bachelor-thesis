import pyLensLib.lenstool as lst
from pyLensLib.lenstool import *
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import io
import argparse
import astropy.io.fits as fits
# import wcs from astropy
from astropy.wcs import WCS
from scipy.optimize import least_squares, brentq
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy import units as u

import re, os

prefix_models = '/Users/maxmen3/stiva/pietro_models/'
prefix_catalogs = '/Users/maxmen3/projects/Pierpaoli/'

JAFFE_Re_OVER_rJ = 1.0# 0.7447  # Re = 0.7447 * rJ


# --- Scaling relations for cluster member galaxies (Bergamini et al.) ---
# In LensTool, member-galaxy dPIE parameters are scaled with luminosity:
#   sigma_LT = sigma_ref_LT * (L/L_ref)^alpha
#   r_cut    = r_cut_ref    * (L/L_ref)^beta_cut
# where L/L_ref = 10^{-0.4 (m - m_ref)} using F160W magnitudes.
#
# Defaults below match the M0416 model setup (m_ref=17.02, alpha=0.3, beta_cut=0.6).
DEFAULT_MREF_F160W = 17.02
DEFAULT_ALPHA = 0.30
DEFAULT_BETA_CUT = 0.60

def load_bayes_scaling_params(bayes_path):
    """Load LensTool MCMC sample file (bayes_*.dat) and extract the scaling-relation parameters.

    The function searches header lines for:
      - '#Pot0 sigma (km/s)'  -> sigma_ref_LT (km/s)
      - '#Pot0 rcut (arcsec)' -> r_cut_ref (arcsec)

    Returns
    -------
    sigma_ref_LT : ndarray, shape (nsamples,)
    rcut_ref_arcsec : ndarray, shape (nsamples,)
    """
    # Read header lines to infer column indices
    with open(bayes_path, 'r', encoding='utf-8', errors='replace') as f:
        header_lines = []
        data_lines = []
        for line in f:
            if line.strip().startswith('#'):
                header_lines.append(line.strip())
            elif line.strip():
                data_lines.append(line)

    if len(header_lines) == 0:
        raise ValueError(f"No header lines found in {bayes_path}.")

    # Column order is the order of header lines in this file format
    colnames = [h.lstrip('#').strip() for h in header_lines]
    try:
        i_sig = colnames.index('Pot0 sigma (km/s)')
        i_rcut = colnames.index('Pot0 rcut (arcsec)')
    except ValueError as e:
        raise ValueError(
            "Could not find 'Pot0 sigma (km/s)' and 'Pot0 rcut (arcsec)' in bayes file headers.\n"
            f"Available header columns: {colnames}"
        ) from e

    data = np.loadtxt(io.StringIO(''.join(data_lines)))
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] != len(colnames):
        raise ValueError(
            f"Mismatch between number of header columns ({len(colnames)}) and data columns ({data.shape[1]}).")

    sigma_ref_LT = data[:, i_sig]
    rcut_ref_arcsec = data[:, i_rcut]

    return sigma_ref_LT, rcut_ref_arcsec

def luminosity_ratio_from_mag(m, mref=DEFAULT_MREF_F160W):
    """Return L/Lref from magnitudes (AB): L/Lref = 10^{-0.4(m - mref)}."""
    return 10.0 ** (-0.4 * (m - mref))

def galaxy_dpie_from_scaling(m_F160W, sigma_ref_LT, rcut_ref_arcsec, cosmo, zl,
                            alpha=DEFAULT_ALPHA, beta_cut=DEFAULT_BETA_CUT, mref=DEFAULT_MREF_F160W):
    """Compute member-galaxy dPIE (PIEMD) parameters from scaling relations.

    Returns
    -------
    sigma0_kms : float
        Central velocity dispersion (sigma0) in km/s (converted from LensTool sigma_LT).
    rcut_kpc : float
        Cut radius in kpc.
    """
    Lratio = luminosity_ratio_from_mag(m_F160W, mref=mref)
    sigma_LT = sigma_ref_LT * (Lratio ** alpha)
    # Convert LensTool sigma_LT to sigma0 used by PIEMD_rho (see comment in Bergamini et al.)
    sigma0 = float(sigma_LT) * np.sqrt(3.0 / 2.0)

    # Convert rcut from arcsec to kpc
    # kpc per arcsec = D_A(z) * (1 arcsec in rad) in kpc
    kpc_per_arcsec = (cosmo.angular_diameter_distance(zl).to_value('kpc') * (np.pi / 648000.0))
    rcut_kpc = float(rcut_ref_arcsec) * (Lratio ** beta_cut) * kpc_per_arcsec
    return sigma0, rcut_kpc

def jaffe_scale_radius(Re):
    """Return r_J from projected effective radius Re (same length units)."""
    return Re / JAFFE_Re_OVER_rJ

def jaffe_rho(r, M, Re):
    """3D density ρ(r) for a Jaffe model with total mass M and effective radius Re."""
    rJ = jaffe_scale_radius(Re)
    return (M * rJ) / (4.0 * np.pi * r**2 * (r + rJ)**2)

def jaffe_Menc(r, M, Re):
    """Enclosed mass M(<r) for the same model."""
    rJ = jaffe_scale_radius(Re)
    return M * r / (r + rJ)

def PIEMD_rho(r, r_core, r_cut, sigma_v):
    """
    PIEMD density profile
    :param r: radius
    :param r_core: core radius
    :param r_cut: cut radius
    :param sigma_v: velocity dispersion
    :return: density at radius r
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    rho = (sigma_v ** 2) / (2 * np.pi * G) * (r_cut+r_core)/(r_core**2*r_cut)/ (1 + r ** 2/r_core ** 2) / (1 + r ** 2/r_cut ** 2)
    return rho

def PIEMD_Menc(r, r_core, r_cut, sigma_v):
    """
    Enclosed mass of a PIEMD profile
    :param r: radius
    :param r_core: core radius
    :param r_cut: cut radius
    :param sigma_v: velocity dispersion
    :return: mass within radius r
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    #M = np.pi* (sigma_v ** 2) / G * (r_cut - r_core) * (1 - np.sqrt(r_core**2 + r**2)/ (r_cut - r_core) * np.arctan(r_cut/np.sqrt(r_core**2 + r**2)) + np.sqrt(r_cut**2 + r**2)/(r_cut - r_core) * np.arctan(r_cut/np.sqrt(r_cut**2 + r**2)))
    M = 2.0 * sigma_v**2 /G * r_cut/(r_cut - r_core) * (r_cut * np.arctan(r / r_cut) - r_core * np.arctan(r / r_core))
    return M

def PIEMD_mtotal(r_core, r_cut, sigma_v):
    """
    Total mass of a PIEMD profile
    :param r_core: core radius
    :param r_cut: cut radius
    :param sigma_v: velocity dispersion
    :return: total mass within radius r
    """
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    #M = np.pi* (sigma_v ** 2) / G * (r_cut - r_core)
    M = np.pi* (sigma_v ** 2) / G * r_cut 
    return M

def trunc_NFW_rho(r, Rs, r_trunc, rho0):
    """
    Multiplicatively truncated NFW density (valid for any r).

    rho(r) = (r_trunc^2 / (r_trunc^2 + r^2)) * rho0 / [ (r/Rs) * (1 + r/Rs)^2 ]

    Parameters
    ----------
    r : float or array_like
        Radius.
    Rs : float or array_like
        NFW scale radius (>0).
    r_trunc : float or array_like
        Truncation radius (>0).
    rho0 : float or array_like
        NFW scale density.

    Returns
    -------
    rho : ndarray
        Density with broadcasted shape of (r, Rs, r_trunc, rho0).
    """
    r = np.asanyarray(r, dtype=float)
    Rs = np.asanyarray(Rs, dtype=float)
    r_trunc = np.asanyarray(r_trunc, dtype=float)
    rho0 = np.asanyarray(rho0, dtype=float)
    if np.any(Rs <= 0) or np.any(r_trunc <= 0):
        raise ValueError("Rs and r_trunc must be > 0")

    with np.errstate(divide='ignore', invalid='ignore'):
        x = r / Rs
        trunc = (r_trunc**2) / (r_trunc**2 + r**2)
        rho = trunc * (rho0 / (x * (1.0 + x)**2))

    return rho

def rdelta_trunc_nfw(r_s, r_t, rho_s, rho_ref, Delta=200.0, grid_size=20000):
    """
    R_Delta (default Delta=200) for a truncated NFW:
        rho(r) = rho_s / [x (1+x)^2] * [tau^2 / (x^2 + tau^2)] for r<r_t, else 0
    where x=r/r_s and tau=r_t/r_s.

    Parameters
    ----------
    r_s : float
        NFW scale radius.
    r_t : float
        Truncation radius.
    rho_s : float
        NFW scale density.
    rho_ref : float
        Reference density (e.g. critical density at z): same units as rho_s.
    Delta : float
        Overdensity (200 for R200).
    grid_size : int
        Resolution for the precomputed integral.

    Returns
    -------
    R_Delta : float
        Radius where mean enclosed density equals Delta * rho_ref.
    """
    tau = r_t / r_s
    if tau <= 0:
        raise ValueError("r_t must be > 0")
    # Build a log grid in x from tiny to tau (handles inner cusp cleanly)
    x_min = min(1e-7, 0.1 * tau)
    x_min = max(x_min, 1e-12)
    if tau <= x_min:  # pathological case: extremely small r_t
        x_min = tau * 1e-6
    x = np.logspace(np.log10(x_min), np.log10(tau), grid_size)

    # Dimensionless density factor f(x) and mass integrand x^2 f(x)
    f = (1.0 / (x * (1.0 + x)**2)) * (tau**2 / (x**2 + tau**2))
    integrand = x**2 * f

    # Cumulative integral F(x) = ∫_0^x x'^2 f(x') dx' via trapezoid rule
    F = np.zeros_like(x)
    dx = np.diff(x)
    F[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dx)

    # Mass normalisation: M(<x) = 4π ρ_s r_s^3 F(x)
    mass_norm = 4.0 * np.pi * rho_s * r_s**3

    def M_of_R(R):
        # Interpolate F at x=R/r_s, clip to tau so mass is constant for R>=r_t
        xR = np.atleast_1d(R / r_s)
        xR_clip = np.clip(xR, x[0], x[-1])
        F_interp = np.interp(xR_clip, x, F)
        M = mass_norm * F_interp
        return M if np.ndim(R) else float(M)

    # Target mean-density condition: M - (4/3)π Δ ρ_ref R^3 = 0
    target_coeff = (4.0 / 3.0) * np.pi * Delta * rho_ref
    def g(R):
        return M_of_R(R) - target_coeff * R**3

    # Bracket the root
    R_low = 1e-8 * r_s
    # At very small R, mean density -> ∞, so g(R_low) should be > 0; safeguard:
    while g(R_low) <= 0:
        R_low *= 0.1
        if R_low < 1e-30 * r_s:
            break

    M_t = M_of_R(r_t)
    R_est = (M_t / target_coeff)**(1.0 / 3.0)  # if solution falls beyond r_t
    R_high = max(r_t, 2.0 * R_est)
    while g(R_high) > 0:  # increase until mean density drops below target
        R_high *= 2.0
        if R_high > 1e6 * max(r_s, r_t):
            raise RuntimeError("Failed to bracket R200; check parameters/units.")

    # Bisection
    for _ in range(80):
        R_mid = 0.5 * (R_low + R_high)
        if g(R_mid) > 0:
            R_low = R_mid
        else:
            R_high = R_mid
    return 0.5 * (R_low + R_high)

def c200_from_rs_rhos(r_s, rho_s, rho_ref):
    """
    Solve for c200 given NFW r_s and rho_s, with respect to rho_ref (e.g. rho_crit or rho_mean).

    Parameters
    ----------
    r_s : float or array_like
        NFW scale radius (>0). Only used for shape; c200 itself does not depend on r_s.
    rho_s : float or array_like
        NFW scale density (>0).
    rho_ref : float
        Reference density (>0): choose rho_crit(z) for c200c, or rho_mean(z) for c200m.
        Must be in the same units as rho_s.

    Returns
    -------
    c200 : ndarray
        Concentration(s) c200, broadcast to the shape of rho_s and r_s.
    """
    r_s = np.asarray(r_s, dtype=float)
    rho_s = np.asarray(rho_s, dtype=float)
    if np.any(r_s <= 0) or np.any(rho_s <= 0) or not np.isfinite(rho_ref) or rho_ref <= 0:
        raise ValueError("r_s, rho_s, and rho_ref must be positive.")

    S = np.broadcast_to(rho_s / float(rho_ref), np.broadcast(r_s, rho_s).shape)

    def f(c, Sval):
        # f(c) = delta_c(c) - S  (root at f=0)
        return (200.0/3.0) * c**3 / (np.log1p(c) - c/(1.0 + c)) - Sval

    c_out = np.empty_like(S, dtype=float)
    c_lo, c_hi = 1e-6, 1e5  # generous bracket: f(c_lo)<0, f(c_hi)>0 for any reasonable S

    it = np.nditer(S, flags=['multi_index'])
    while not it.finished:
        Sval = float(it[0])
        # Solve f(c)=0
        c_out[it.multi_index] = brentq(f, c_lo, c_hi, args=(Sval,))
        it.iternext()

    return c_out

def M_nfw_enclosed(r, Rs, rho0):
    """
    Enclosed mass for standard NFW:
      rho(r) = rho0 / [ (r/Rs) * (1 + r/Rs)^2 ]
      M(<r)  = 4π rho0 Rs^3 [ ln(1+x) - x/(1+x) ],  x=r/Rs
    """
    r   = np.asanyarray(r, dtype=float)
    Rs  = np.asanyarray(Rs, dtype=float)
    rho0= np.asanyarray(rho0, dtype=float)
    if np.any(Rs <= 0) or np.any(rho0 <= 0) or np.any(r < 0):
        raise ValueError("Rs>0, rho0>0 and r>=0 required.")
    x = r / Rs
    return 4.0*np.pi * rho0 * Rs**3 * (np.log1p(x) - x/(1.0 + x))


def M_tNFW_enclosed(r, Rs, r_trunc, rho0):
    """
    Enclosed mass for the *multiplicatively truncated* NFW:
      rho(r) = rho0 / [x (1+x)^2] * [tau^2 / (x^2 + tau^2)],  x=r/Rs, tau=r_trunc/Rs

    Closed-form primitive:
      Let A = tau^2 (tau^2 - 1) / (tau^2 + 1)^2
          B = - tau^2 / (tau^2 + 1)
          D =  2 tau^4 / (tau^2 + 1)^2
      Then F(x) = A[ ln(1+x) - 0.5 ln(1 + (x/tau)^2) ] + B * x/(1+x) + (D/tau) * arctan(x/tau)
      and   M(<r) = 4π rho0 Rs^3 F(x).

    Notes:
    - Vectorized and numerically stable (uses log1p, arctan).
    - As r→∞, M(<r) → 4π rho0 Rs^3 [ A ln tau + B + (D/tau)*π/2 ] (finite total mass).
    """
    r        = np.asanyarray(r, dtype=float)
    Rs       = np.asanyarray(Rs, dtype=float)
    r_trunc  = np.asanyarray(r_trunc, dtype=float)
    rho0     = np.asanyarray(rho0, dtype=float)
    if np.any(Rs <= 0) or np.any(r_trunc <= 0) or np.any(rho0 <= 0) or np.any(r < 0):
        raise ValueError("Rs>0, r_trunc>0, rho0>0 and r>=0 required.")

    x   = r / Rs
    tau = r_trunc / Rs

    # coefficients
    tau2 = tau**2
    denom = (tau2 + 1.0)**2
    A = tau2 * (tau2 - 1.0) / denom
    B = - tau2 / (tau2 + 1.0)
    D = 2.0 * tau2**2 / denom  # = 2 * tau^4 / (tau^2 + 1)^2

    # primitive F(x)
    term1 = A * (np.log1p(x) - 0.5*np.log1p((x/tau)**2))
    term2 = B * (x / (1.0 + x))
    term3 = (D / tau) * np.arctan(x / tau)
    F = term1 + term2 + term3

    return 4.0*np.pi * rho0 * Rs**3 * F


# (optional) total mass of the truncated NFW (r->∞) in closed form
def M_tNFW_total(Rs, r_trunc, rho0):
    tau = np.asanyarray(r_trunc, float) / np.asanyarray(Rs, float)
    tau2 = tau**2
    denom = (tau2 + 1.0)**2
    A = tau2 * (tau2 - 1.0) / denom
    B = - tau2 / (tau2 + 1.0)
    D = 2.0 * tau2**2 / denom
    F_inf = A * np.log(tau) + B + (D / tau) * (np.pi/2.0)
    return 4.0*np.pi * rho0 * (Rs**3) * F_inf

def fit_with_tNFW(r, rho, r_s_guess, r_t_guess, rho_s_guess,
                  loss="soft_l1", f_scale=0.1, max_nfev=20000,
                  fit_quantity="rho_over_r2", y_err=None):
    """
    Fit either rho(r) or rho(r)/r^2 for a multiplicatively truncated NFW profile.
    Returns (r_s, r_t, rho_s, diagnostics_dict).

    Parameters
    ----------
    r, rho : arrays
        Radii and 3D density values.
    fit_quantity : {"rho", "rho_over_r2"}
        What to fit. If "rho_over_r2", the target is y = rho / r^2.
    y_err : None or array
        1-sigma uncertainty on y in *linear* units (rho or rho/r^2).
        If provided, used to compute chi2 and reduced chi2 in log space via
        sigma_log ~ y_err / y.

    Diagnostics returned include:
      - npts, dof, rss_log, r2_log, rmse_log
      - chi2_log, chi2_red_log (if y_err given)
      - aic, bic (based on log-residual RSS)
      - bias_dex (median residual in dex), scatter68_dex
      - param_errors (1-sigma from covariance), success flag and message
    """
    r   = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)

    # Choose y to fit
    if fit_quantity == "rho_over_r2":
        y = rho * (r**2)
    else:
        y = rho

    mask = (r > 0) & np.isfinite(y) & (y > 0)
    if y_err is not None:
        y_err = np.asarray(y_err, dtype=float)
        mask &= np.isfinite(y_err) & (y_err > 0)

    r_fit = r[mask]
    y_fit = y[mask]
    yerr_fit = None
    if y_err is not None:
        yerr_fit = y_err[mask]
    n = r_fit.size
    if n < 3:
        raise ValueError("Not enough valid points to fit.")

    # Parametrization: r_s = exp(t0) > 0
    #                  r_t = r_s * (1 + exp(t1)) > r_s
    #                  rho_s = exp(t2) > 0
    def unpack(theta):
        r_s   = np.exp(theta[0])
        r_t   = r_s * (1.0 + np.exp(theta[1]))
        rho_s = np.exp(theta[2])
        return r_s, r_t, rho_s

    eps = 1e-8
    theta0 = np.array([
        np.log(max(r_s_guess, eps)),
        np.log(max(r_t_guess / max(r_s_guess, eps) - 1.0, eps)),
        np.log(max(rho_s_guess, eps))
    ])

    # Model in the same space as y_fit
    def model_linear(r, r_s, r_t, rho_s):
        m = trunc_NFW_rho(r, r_s, r_t, rho_s)
        return m * (r**2) if fit_quantity == "rho_over_r2" else m

    def residuals(theta):
        r_s, r_t, rho_s = unpack(theta)
        y_model = model_linear(r_fit, r_s, r_t, rho_s)
        res = np.log(y_model) - np.log(y_fit)   # log-space residuals
        # drop non-finite in case of numerical glitches
        return res[np.isfinite(res)]

    sol = least_squares(residuals, theta0, method="trf",
                        loss=loss, f_scale=f_scale, max_nfev=max_nfev)

    r_s, r_t, rho_s = unpack(sol.x)

    # ----- Diagnostics -----
    y_model = model_linear(r_fit, r_s, r_t, rho_s)
    res_log = np.log(y_model) - np.log(y_fit)
    res_log = res_log[np.isfinite(res_log)]
    n_eff = res_log.size
    k = 3  # parameters
    dof = max(n_eff - k, 1)
    rss_log = np.sum(res_log**2)
    rmse_log = np.sqrt(rss_log / n_eff)

    # R^2 in log space
    ylog = np.log(y_fit)
    ylog = ylog[np.isfinite(ylog)]
    ylog_model = np.log(y_model[np.isfinite(np.log(y_model))])
    # align lengths if any infs were dropped
    mmin = min(ylog.size, ylog_model.size, res_log.size)
    ylog = ylog[:mmin]; ylog_model = ylog_model[:mmin]
    sst = np.sum((ylog - ylog.mean())**2)
    r2_log = 1.0 - (np.sum((ylog_model - ylog)**2) / sst) if sst > 0 else np.nan

    # If uncertainties are given, compute chi^2 in log space
    chi2_log = chi2_red = np.nan
    if y_err is not None and yerr_fit is not None:
        # sigma_log ≈ sigma_y / y (for small errors)
        sigma_log = (yerr_fit / y_fit)[np.isfinite(res_log)]
        chi2_log = np.sum((res_log / sigma_log)**2)
        chi2_red = chi2_log / dof

    # AIC/BIC using log-space RSS
    # (Note: absolute scale depends on using log residuals, but useful for comparison)
    aic = n_eff * np.log(rss_log / n_eff) + 2 * k
    bic = n_eff * np.log(rss_log / n_eff) + k * np.log(n_eff)

    # Bias and scatter in dex
    ln10 = np.log(10.0)
    res_dex = res_log / ln10
    bias_dex = np.median(res_dex)
    p16, p84 = np.percentile(res_dex, [16, 84])
    scatter68_dex = 0.5 * (p84 - p16)

    # Parameter covariance & 1-sigma errors from Jacobian
    param_errors = np.full(3, np.nan)
    if sol.jac is not None and sol.jac.size == n_eff * k:
        J = sol.jac  # shape (n_eff, k) for residuals (log space)
        # scale of residuals: sigma^2 = RSS/dof
        s2 = rss_log / dof
        JTJ = J.T @ J
        try:
            cov_theta = s2 * np.linalg.inv(JTJ)
            # map to (r_s, r_t, rho_s) via Jacobian of unpack at solution
            t = sol.x
            r_s_, r_t_, rho_s_ = unpack(t)
            # derivatives of (r_s, r_t, rho_s) wrt theta
            drs_dt0 = r_s_
            drt_dt0 = (1 + np.exp(t[1])) * r_s_
            drt_dt1 = r_s_ * np.exp(t[1])
            drhos_dt2 = rho_s_
            A = np.array([[drs_dt0,        0.0,        0.0],
                          [drt_dt0,     drt_dt1,        0.0],
                          [0.0,            0.0,    drhos_dt2]])
            cov_phys = A @ cov_theta @ A.T
            param_errors = np.sqrt(np.diag(cov_phys))
        except np.linalg.LinAlgError:
            pass  # ill-conditioned; skip uncertainties

    diag = dict(
        npts=n_eff, dof=dof, rss_log=rss_log, rmse_log=rmse_log,
        r2_log=r2_log, chi2_log=chi2_log, chi2_red_log=chi2_red,
        aic=aic, bic=bic, bias_dex=bias_dex, scatter68_dex=scatter68_dex,
        param_errors=dict(r_s_err=param_errors[0],
                          r_t_err=param_errors[1],
                          rho_s_err=param_errors[2]),
        success=sol.success, message=sol.message
    )

    return r_s, r_t, rho_s, diag

def stellar_mass_to_light(mag):
    """
    convert stellar mass to light using the relation from Grillo et al. 2015
    :param mag: magnitude in F160W
    :return: stellar mass in solar masses
    """
    logM = 18.541 - 0.416 * mag
    M = 10 ** logM
    return M


def validate_matching(ra_lens,dec_lens, ra_cat, dec_cat, idx, rgbimage='M0416_B22_rgb.fits'):
    """
    Validate the matching between a photometric catalog and a lens model by overlaying the positions onto an RGB image.
    :param ra_lens: array of RA from the lens model
    :param dec_lens: array of Dec from the lens model
    :param ra_cat: array of RA from the photometric catalog
    :param dec_cat: array of Dec from the photometric catalog
    :param rgbimage: rgb image file in fits format
    :return:
    """

    if os.path.exists(prefix_models + rgbimage):
        with fits.open(prefix_models + rgbimage) as hdul:
            rgb_fits_data = hdul[0].data  # Access the primary HDU's data
            # get wcs from the fits header
            wcs = WCS(hdul[0].header, naxis=2)

        rgb_image = np.transpose(rgb_fits_data, (1, 2, 0))
        rgb_image = rgb_image / np.max(rgb_image)
        #wcs = wcs.slice((slice(None), slice(None), 0))
        #print (wcs)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': wcs})
        ax.imshow(rgb_image, origin='lower')
        ax.scatter(ra_lens, dec_lens, s=50, edgecolor='blue', facecolor='none', label='Lens Model', alpha=0.7,transform=ax.get_transform('world'))
        ax.scatter(ra_cat[idx], dec_cat[idx], s=30, edgecolor='red', facecolor='none', label='Catalog Matches', alpha=0.7,transform=ax.get_transform('world'))
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        plt.show()


def _parse_vizier_bytes_table(readme_text: str, data_basename: str):
    """Return (colspecs, names) from a VizieR ReadMe for a given data file."""
    # Find the byte-by-byte section whose header includes our data file name
    sec_pat = re.compile(r"Byte-by-byte Description of file:\s*(.*)")
    lines = readme_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        m = sec_pat.search(line)
        if m and data_basename in m.group(1):
            start_idx = i
            break
    if start_idx is None:  # fallback: first byte-by-byte section
        for i, line in enumerate(lines):
            if sec_pat.search(line):
                start_idx = i
                break
    if start_idx is None:
        raise ValueError("No 'Byte-by-byte' section found in ReadMe.")

    # Parse rows like: "  1-  3  I3  ---  ID  ..."
    row_pat = re.compile(
        r"^\s*(\d+)\s*-\s*(\d+)\s+[A-Za-z0-9.]+\s+\S+\s+([A-Za-z0-9_]+)"
    )
    colspecs, names = [], []
    for line in lines[start_idx+1:]:
        if line.strip().startswith("Byte-by-byte Description of file:"):
            break  # next section
        m = row_pat.match(line)
        if m:
            a, b, label = m.groups()
            colspecs.append((int(a) - 1, int(b)))  # pandas uses 0-based, end-exclusive
            names.append(label)
        elif colspecs and not line.strip():
            break  # end of table
    if not colspecs:
        raise ValueError("Could not parse column specs from ReadMe.")
    return colspecs, names

def load_cluster(readme_path="ReadMe.txt", data_path="m0416.dat"):
    """
    Parse VizieR-style ReadMe to get colspecs & names, then read the fixed-width data file.
    Returns a pandas.DataFrame.
    """
    with open(readme_path, encoding="utf-8", errors="replace") as f:
        readme_text = f.read()
    colspecs, names = _parse_vizier_bytes_table(readme_text, os.path.basename(data_path))
    return pd.read_fwf(data_path, colspecs=colspecs, names=names)

def validate_density_profiles():
    """Test analytical density profile functions against known values."""
    print("\n=== Validating Density Profiles ===")
    
    # Test PIEMD enclosed mass consistency
    r_test = np.array([1.0, 5.0, 10.0])
    r_core, r_cut, sigma = 0.15, 100.0, 200.0
    M_enc = PIEMD_Menc(r_test, r_core, r_cut, sigma)
    print (M_enc)
    M_tot = PIEMD_mtotal(r_core, r_cut, sigma)
    
    # M_enc should be monotonic and < M_tot
    assert np.all(np.diff(M_enc) > 0), "PIEMD mass not monotonic"
    assert np.all(M_enc < M_tot), f"Enclosed mass {M_enc[-1]:.2e} exceeds total {M_tot:.2e}"
    print(f"✓ PIEMD: M(<{r_test[-1]:.1f} kpc) = {M_enc[-1]:.2e} Msun < M_tot = {M_tot:.2e}")
    
    # Test Jaffe profile normalization
    M_star, Re = 1e10, 1.0
    M_at_Re = jaffe_Menc(Re, M_star, Re)
    expected_ratio = Re / (Re + jaffe_scale_radius(Re))
    assert np.isclose(M_at_Re / M_star, expected_ratio, rtol=1e-3), "Jaffe M(<Re) mismatch"
    print(f"✓ Jaffe: M(<Re)/M_star = {M_at_Re/M_star:.4f} (expected {expected_ratio:.4f})")
    
    # Test truncated NFW total mass convergence
    r_s, r_t, rho_s = 10.0, 100.0, 1e7
    M_tot_tNFW = M_tNFW_total(r_s, r_t, rho_s)
    M_at_10rt = M_tNFW_enclosed(10*r_t, r_s, r_t, rho_s)
    assert M_at_10rt / M_tot_tNFW > 0.99, "tNFW not converged at 10*r_t"
    print(f"✓ tNFW: M(<10r_t)/M_tot = {M_at_10rt/M_tot_tNFW:.4f} (>0.99 expected)")

    # show plot of profiles
    r_plot = np.logspace(-1, 3, 200)
    rho_nfw = trunc_NFW_rho(r_plot, r_s, r_t, rho_s)
    rho_piemd = PIEMD_rho(r_plot, r_core, r_cut, sigma)
    plt.figure(figsize=(8,5))
    plt.loglog(r_plot, rho_nfw, label='tNFW Density')
    plt.loglog(r_plot, rho_piemd, label='PIEMD Density')
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Density (Msun/kpc^3)')
    plt.title('Truncated NFW Density Profile and PIEMD Density Profile')
    plt.legend()
    plt.grid()
    plt.show()

def validate_concentration_calculation():
    """Test c200 calculation against known NFW c-M relation."""
    print("\n=== Validating Concentration ===")
    
    # Use typical cluster-scale halo parameters
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    z = 0.4
    rho_crit = cosmo.critical_density(z).to_value('Msun/kpc3')
    
    # For c200=5, delta_c = 200*c^3/(3[ln(1+c)-c/(1+c)]) ~ 5208
    c200_input = 5.0
    delta_c = (200.0/3.0) * c200_input**3 / (np.log1p(c200_input) - c200_input/(1.0+c200_input))
    rho_s = delta_c * rho_crit
    r_s = 50.0  # arbitrary scale
    
    c200_recovered = c200_from_rs_rhos(r_s, rho_s, rho_crit)
    assert np.isclose(c200_recovered, c200_input, rtol=0.01), \
        f"c200 mismatch: input={c200_input}, recovered={c200_recovered}"
    print(f"✓ c200 roundtrip: {c200_input:.2f} → {c200_recovered:.2f}")

def validate_fitting_convergence():
    """Test tNFW fitting on synthetic data with known parameters."""
    print("\n=== Validating tNFW Fitting ===")
    
    # Generate synthetic tNFW density
    r_s_true, r_t_true, rho_s_true = 8.0, 80.0, 1e7
    r_grid = np.logspace(-1, 2.5, 200)
    rho_true = trunc_NFW_rho(r_grid, r_s_true, r_t_true, rho_s_true)
    
    # Add 10% noise
    np.random.seed(42)
    rho_noisy = rho_true * (1.0 + 0.1*np.random.randn(len(r_grid)))
    
    # Fit with slightly perturbed initial guess
    r_s_fit, r_t_fit, rho_s_fit, diag = fit_with_tNFW(
        r_grid, rho_noisy, 
        r_s_guess=r_s_true*1.2, r_t_guess=r_t_true*0.9, rho_s_guess=rho_s_true*1.1
    )
    
    # Check recovery within 5% (accounting for noise)
    assert np.abs(r_s_fit/r_s_true - 1.0) < 0.05, f"r_s off by {100*(r_s_fit/r_s_true-1):.1f}%"
    assert np.abs(r_t_fit/r_t_true - 1.0) < 0.05, f"r_t off by {100*(r_t_fit/r_t_true-1):.1f}%"
    assert np.abs(rho_s_fit/rho_s_true - 1.0) < 0.10, f"rho_s off by {100*(rho_s_fit/rho_s_true-1):.1f}%"
    assert diag['success'], f"Fit failed: {diag['message']}"
    print(f"✓ Fit recovery: r_s={r_s_fit/r_s_true:.3f}, r_t={r_t_fit/r_t_true:.3f}, rho_s={rho_s_fit/rho_s_true:.3f} (x true)")
    print(f"  Diagnostics: RMSE_log={diag['rmse_log']:.3f}, npts={diag['npts']}, R²={diag['r2_log']:.3f}")

def validate_mcmc_propagation(bayesfile, catalog, parfile, ntest=3):
    """Test MCMC uncertainty propagation on a few galaxies."""
    if bayesfile is None or not os.path.exists(bayesfile):
        print("\n⊘ Skipping MCMC validation (no bayesfile provided)")
        return
    
    print(f"\n=== Validating MCMC Propagation (first {ntest} galaxies) ===")
    
    # Load small subset
    sigma_chain, rcut_chain = load_bayes_scaling_params(bayesfile)
    sigma_chain = sigma_chain[:100]  # use only 100 samples for speed
    rcut_chain = rcut_chain[:100]
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    df_cat = load_cluster(catalog_path=catalog)
    
    for i in range(min(ntest, len(df_cat))):
        m_F160W = df_cat.iloc[i]['F160W']
        zl = 0.4  # typical cluster redshift
        
        r200_samples = []
        for sig_ref, rcut_ref in zip(sigma_chain, rcut_chain):
            sigma0, rcut_kpc = galaxy_dpie_from_scaling(
                m_F160W, sig_ref, rcut_ref, cosmo, zl
            )
            # Quick estimate: assume R200 ~ rcut for validation
            r200_samples.append(rcut_kpc)
        
        r200_median = np.nanmedian(r200_samples)
        r200_std = np.nanstd(r200_samples)
        fractional_unc = r200_std / r200_median if r200_median > 0 else np.nan
        
        print(f"  Galaxy {i}: R200 ~ {r200_median:.1f} ± {r200_std:.1f} kpc (δR/R = {fractional_unc:.2%})")
        assert fractional_unc < 0.5, f"Uncertainty {fractional_unc:.1%} too large (>50%)"
    
    print("✓ MCMC uncertainties are reasonable (<50% scatter)")

def validate_hdf5_output(h5file):
    """Verify HDF5 structure and data integrity."""
    if h5file is None or not os.path.exists(h5file):
        print("\n⊘ Skipping HDF5 validation (file not found)")
        return
    
    print(f"\n=== Validating HDF5 Output: {h5file} ===")
    
    with h5py.File(h5file, 'r') as f:
        # Check required datasets exist
        required = ['lens_index', 'cat_index', 'ra_lens_deg', 'dec_lens_deg', 
                   'sample_id', 'param_names', 'dm_params']
        for key in required:
            assert key in f, f"Missing dataset: {key}"
        
        # Check shapes are consistent
        n_gal = f['lens_index'].shape[0]
        n_samp = f['sample_id'].shape[0]
        n_par = f['param_names'].shape[0]
        dm_shape = f['dm_params'].shape
        
        assert dm_shape == (n_gal, n_samp, n_par), \
            f"dm_params shape {dm_shape} != ({n_gal}, {n_samp}, {n_par})"
        
        # Check for NaN contamination (should be <50% for valid samples)
        nan_frac = np.isnan(f['dm_params'][:]).mean()
        assert nan_frac < 0.5, f"Too many NaNs in dm_params: {nan_frac:.1%}"
        
        print(f"✓ HDF5 structure valid: {n_gal} galaxies × {n_samp} samples × {n_par} params")
        print(f"  NaN fraction: {nan_frac:.1%} (expected <50% for failed fits)")
        
        # Spot-check parameter ranges
        params = f['dm_params'][:]
        param_names = [name.decode() for name in f['param_names'][:]]
        for i, name in enumerate(param_names):
            valid = params[:, :, i][np.isfinite(params[:, :, i])]
            if len(valid) > 0:
                print(f"  {name}: [{valid.min():.2e}, {valid.max():.2e}]")

def validate_mcmc_propagation(bayesfile, readme_path, data_path, ntest=3):
    """Test MCMC uncertainty propagation on a few galaxies."""
    if bayesfile is None or not os.path.exists(bayesfile):
        print("\n⊘ Skipping MCMC validation (no bayesfile provided)")
        return
    
    print(f"\n=== Validating MCMC Propagation (first {ntest} galaxies) ===")
    
    # Load small subset
    sigma_chain, rcut_chain = load_bayes_scaling_params(bayesfile)
    sigma_chain = sigma_chain[:100]  # use only 100 samples for speed
    rcut_chain = rcut_chain[:100]
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    df_cat = load_cluster(readme_path=readme_path, data_path=data_path)
    
    for i in range(min(ntest, len(df_cat))):
        m_F160W = df_cat.iloc[i]['F160W']
        zl = 0.4  # typical cluster redshift
        
        r200_samples = []
        for sig_ref, rcut_ref in zip(sigma_chain, rcut_chain):
            sigma0, rcut_kpc = galaxy_dpie_from_scaling(
                m_F160W, sig_ref, rcut_ref, cosmo, zl
            )
            # Quick estimate: assume R200 ~ rcut for validation
            r200_samples.append(rcut_kpc)
        
        r200_median = np.nanmedian(r200_samples)
        r200_std = np.nanstd(r200_samples)
        fractional_unc = r200_std / r200_median if r200_median > 0 else np.nan
        
        print(f"  Galaxy {i}: R200 ~ {r200_median:.1f} ± {r200_std:.1f} kpc (δR/R = {fractional_unc:.2%})")
        assert fractional_unc < 0.5, f"Uncertainty {fractional_unc:.1%} too large (>50%)"
    
    print("✓ MCMC uncertainties are reasonable (<50% scatter)")

def validate_mcmc_posteriors(h5file, df_results, ngal=3):
    """
    Plot MCMC posteriors for r_s and r_t for the first few galaxies.
    Shows histograms of posterior samples with fiducial best-fit overlaid.
    
    Parameters
    ----------
    h5file : str
        Path to HDF5 file containing per-sample DM parameters.
    df_results : pd.DataFrame
        Results table with fiducial fit values.
    ngal : int
        Number of galaxies to plot (default: 3).
    """
    if h5file is None or not os.path.exists(h5file):
        print("\n⊘ Skipping MCMC posterior validation (no HDF5 file)")
        return
    
    print(f"\n=== Validating MCMC Posteriors (first {ngal} galaxies) ===")
    
    with h5py.File(h5file, 'r') as f:
        dm_params = f['dm_params'][:]  # shape (N_gal, N_samp, N_par)
        param_names = [name.decode() for name in f['param_names'][:]]
        lens_indices = f['lens_index'][:]
    
    # Find parameter indices for r_s and r_t
    i_rs = param_names.index('r_s_kpc')
    i_rt = param_names.index('r_t_kpc')
    
    ngal_actual = min(ngal, dm_params.shape[0], len(df_results))
    
    fig, axes = plt.subplots(ngal_actual, 2, figsize=(12, 4*ngal_actual))
    if ngal_actual == 1:
        axes = axes[np.newaxis, :]  # ensure 2D array
    
    for igal in range(ngal_actual):
        # Extract MCMC samples for this galaxy
        rs_samples = dm_params[igal, :, i_rs]
        rt_samples = dm_params[igal, :, i_rt]
        
        # Remove NaNs (failed fits)
        rs_valid = rs_samples[np.isfinite(rs_samples)]
        rt_valid = rt_samples[np.isfinite(rt_samples)]
        
        # Get fiducial (single best-fit) values from df_results
        row = df_results.iloc[igal]
        rs_fiducial = row['r_s_tNFW']
        rt_fiducial = row['r_t_tNFW']
        
        # Get MCMC percentiles if available
        rs_p16 = row.get('r_s_tNFW_p16', np.nan)
        rs_p50 = row.get('r_s_tNFW_p50', np.nan)
        rs_p84 = row.get('r_s_tNFW_p84', np.nan)
        rt_p16 = row.get('r_t_tNFW_p16', np.nan)
        rt_p50 = row.get('r_t_tNFW_p50', np.nan)
        rt_p84 = row.get('r_t_tNFW_p84', np.nan)
        
        # --- Plot r_s posterior ---
        ax_rs = axes[igal, 0]
        if len(rs_valid) > 0:
            ax_rs.hist(rs_valid, bins=30, alpha=0.7, color='steelblue', 
                      edgecolor='black', label='MCMC samples')
            ax_rs.axvline(rs_fiducial, color='red', linestyle='--', linewidth=2, 
                         label=f'Fiducial: {rs_fiducial:.2f} kpc')
            if np.isfinite(rs_p50):
                ax_rs.axvline(rs_p50, color='orange', linestyle='-', linewidth=2, 
                             label=f'MCMC median: {rs_p50:.2f} kpc')
                ax_rs.axvspan(rs_p16, rs_p84, alpha=0.2, color='orange', 
                             label=f'68% CI: [{rs_p16:.2f}, {rs_p84:.2f}]')
        else:
            ax_rs.text(0.5, 0.5, 'No valid samples', ha='center', va='center', 
                      transform=ax_rs.transAxes, fontsize=12)
        
        ax_rs.set_xlabel(r'$r_s$ [kpc]', fontsize=12)
        ax_rs.set_ylabel('N samples', fontsize=12)
        ax_rs.set_title(f'Galaxy {igal} (lens #{int(lens_indices[igal])}): $r_s$ posterior', 
                       fontsize=13)
        ax_rs.legend(fontsize=9)
        ax_rs.grid(alpha=0.3)
        
        # --- Plot r_t posterior ---
        ax_rt = axes[igal, 1]
        if len(rt_valid) > 0:
            ax_rt.hist(rt_valid, bins=30, alpha=0.7, color='forestgreen', 
                      edgecolor='black', label='MCMC samples')
            ax_rt.axvline(rt_fiducial, color='red', linestyle='--', linewidth=2, 
                         label=f'Fiducial: {rt_fiducial:.2f} kpc')
            if np.isfinite(rt_p50):
                ax_rt.axvline(rt_p50, color='orange', linestyle='-', linewidth=2, 
                             label=f'MCMC median: {rt_p50:.2f} kpc')
                ax_rt.axvspan(rt_p16, rt_p84, alpha=0.2, color='orange', 
                             label=f'68% CI: [{rt_p16:.2f}, {rt_p84:.2f}]')
        else:
            ax_rt.text(0.5, 0.5, 'No valid samples', ha='center', va='center', 
                      transform=ax_rt.transAxes, fontsize=12)
        
        ax_rt.set_xlabel(r'$r_t$ [kpc]', fontsize=12)
        ax_rt.set_ylabel('N samples', fontsize=12)
        ax_rt.set_title(f'Galaxy {igal} (lens #{int(lens_indices[igal])}): $r_t$ posterior', 
                       fontsize=13)
        ax_rt.legend(fontsize=9)
        ax_rt.grid(alpha=0.3)
        
        # Diagnostic: warn if fiducial is outside MCMC range
        if len(rs_valid) > 0 and (rs_fiducial < rs_valid.min() or rs_fiducial > rs_valid.max()):
            print(f"  ⚠ Galaxy {igal}: Fiducial r_s={rs_fiducial:.2f} outside MCMC range "
                  f"[{rs_valid.min():.2f}, {rs_valid.max():.2f}] kpc")
        if len(rt_valid) > 0 and (rt_fiducial < rt_valid.min() or rt_fiducial > rt_valid.max()):
            print(f"  ⚠ Galaxy {igal}: Fiducial r_t={rt_fiducial:.2f} outside MCMC range "
                  f"[{rt_valid.min():.2f}, {rt_valid.max():.2f}] kpc")
    
    plt.tight_layout()
    plt.savefig('mcmc_posteriors_validation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved MCMC posterior validation plot: mcmc_posteriors_validation.png")
    plt.show()


def validate_scaling_relation_consistency(bayesfile, parfile, catalog, readme_path, 
                                          alpha=DEFAULT_ALPHA, beta_cut=DEFAULT_BETA_CUT, 
                                          mref=DEFAULT_MREF_F160W):
    """
    Compare MCMC-sampled reference parameters (sigma_ref, rcut_ref) to those implied by 
    the .par file's first galaxy. Plots histograms of MCMC samples with .par-derived 
    fiducial values overlaid as vertical lines.
    
    This validates whether the .par file's galaxy parameters are consistent with the 
    MCMC posterior's scaling relation.
    
    Parameters
    ----------
    bayesfile : str
        Path to bayes_*.dat MCMC sample file
    parfile : str
        Path to LensTool .par file
    catalog : str
        Path to photometric catalog
    readme_path : str
        Path to ReadMe for catalog parsing
    alpha, beta_cut, mref : float
        Scaling relation parameters
    """
    if bayesfile is None or not os.path.exists(bayesfile):
        print("\n⊘ Skipping scaling relation validation (no bayesfile)")
        return
    
    print("\n=== Validating Scaling Relation Consistency ===")
    
    # Load MCMC samples
    sigma_ref_chain, rcut_ref_arcsec_chain = load_bayes_scaling_params(bayesfile)
    
    # Load .par file and catalog
    potentiel_ = readLenstoolBlock(best_par=parfile, block_name='potentiel')
    potentiel = selectPotentielByType(potentiel_, ptype='gal')
    cosmologie = readLenstoolBlock(best_par=parfile, block_name='cosmologie')
    cosmo = FlatLambdaCDM(H0=cosmologie['H0'], Om0=cosmologie['omegaM'])
    df_cat = load_cluster(readme_path=readme_path, data_path=catalog)
    
    # Match first galaxy in .par to catalog (to get its F160W magnitude)
    x, y = getXYfromPotentiel(potentiel)
    ra_ref, dec_ref = getRef_RA_DEC(parfile)
    ra, dec = getRADECfromXY(x, y, ra_ref, dec_ref)
    
    coords_lens = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    coords_cat = SkyCoord(ra=df_cat['RAdeg'].values * u.deg, dec=df_cat['DEdeg'].values * u.deg)
    idx_cat, d2d, _ = coords_lens.match_to_catalog_sky(coords_cat)
    
    # Use first matched galaxy (lens index 0)
    tolerance = 0.1 * u.arcsec
    mask = d2d < tolerance
    lens_idx = np.where(mask)[0]
    
    if len(lens_idx) == 0:
        print("  ⚠ No galaxies matched - cannot validate")
        return
    
    j = lens_idx[0]  # first galaxy in .par
    j_cat = idx_cat[j]
    #m_F160Wcat = df_cat.iloc[j_cat]['F160W']
    zl = findInBlock(potentiel[j], 'z_lens')
    
    # Extract .par file's galaxy parameters (already scaled for this galaxy's luminosity)
    v_disp_par = float(findInBlock(potentiel[j], 'v_disp'))  # LensTool sigma
    rcut_kpc_par = float(findInBlock(potentiel[j], 'cut_radius_kpc'))  # Already in kpc!
    rcut_arcsec_par_ = float(findInBlock(potentiel[j], 'cut_radius'))  # in arcsec (not used here)
    m_F160W = float(findInBlock(potentiel[j], 'mag'))

    
    # Back-calculate reference parameters from .par using inverse scaling relation
    # Scaling: sigma_LT = sigma_ref_LT * (L/L_ref)^alpha
    #          rcut_kpc = rcut_ref_arcsec * kpc_per_arcsec * (L/L_ref)^beta_cut
    # Inverse: sigma_ref_LT = sigma_LT / (L/L_ref)^alpha
    #          rcut_ref_arcsec = (rcut_kpc / kpc_per_arcsec) / (L/L_ref)^beta_cut
    
    Lratio = luminosity_ratio_from_mag(m_F160W, mref=mref)
    sigma_ref_par = v_disp_par / (Lratio ** alpha)  # LensTool sigma (not σ₀)
    
    kpc_per_arcsec = cosmo.angular_diameter_distance(zl).to_value('kpc') * (np.pi / 648000.0)
    print (f"  Galaxy 0: F160W={m_F160W:.2f}, L/L_ref={Lratio:.3f}, z={zl:.3f}, kpc/arcsec={kpc_per_arcsec:.3f}")
    # CORRECTED: rcut_kpc_par is already in kpc, so convert to arcsec first, then inverse-scale
    rcut_arcsec_par = rcut_kpc_par / kpc_per_arcsec
    rcut_ref_arcsec_par = rcut_arcsec_par / (Lratio ** beta_cut)
    print (f"    .par galaxy 0: v_disp={v_disp_par:.1f} km/s, r_cut={rcut_kpc_par:.2f} kpc "
           f"-> r_cut={rcut_arcsec_par:.3f} arcsec")
    print (f"for comparison: r_cut from .par in arcsec = {rcut_arcsec_par_:.3f} arcsec")
    
    print (f"    Inferred .par reference params: σ_ref={sigma_ref_par:.1f} km/s, "
           f"r_cut_ref={rcut_ref_arcsec_par:.3f} arcsec")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- sigma_ref histogram ---
    ax = axes[0]
    ax.hist(sigma_ref_chain, bins=40, alpha=0.7, color='steelblue', 
            edgecolor='black', label='MCMC posterior')
    ax.axvline(sigma_ref_par, color='red', linestyle='--', linewidth=2.5,
               label=f'.par fiducial: {sigma_ref_par:.1f} km/s')
    
    # Show MCMC percentiles
    p16, p50, p84 = np.percentile(sigma_ref_chain, [16, 50, 84])
    ax.axvline(p50, color='orange', linestyle='-', linewidth=2, alpha=0.8,
               label=f'MCMC median: {p50:.1f} km/s')
    ax.axvspan(p16, p84, alpha=0.2, color='orange', 
               label=f'68% CI: [{p16:.1f}, {p84:.1f}]')
    
    ax.set_xlabel(r'$\sigma_{\rm ref,LT}$ [km/s]', fontsize=13)
    ax.set_ylabel('N samples', fontsize=13)
    ax.set_title('Reference Velocity Dispersion\n(LensTool convention)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Diagnostic message
    if sigma_ref_par < p16 or sigma_ref_par > p84:
        print(f"  ⚠ .par σ_ref={sigma_ref_par:.1f} km/s outside MCMC 68% CI [{p16:.1f}, {p84:.1f}]")
        print(f"    Galaxy 0: F160W={m_F160W:.2f}, L/L_ref={Lratio:.3f}, z={zl:.3f}")
    else:
        print(f"  ✓ .par σ_ref={sigma_ref_par:.1f} km/s within MCMC range [{p16:.1f}, {p84:.1f}]")
    
    # --- r_cut_ref histogram ---
    ax = axes[1]
    ax.hist(rcut_ref_arcsec_chain, bins=40, alpha=0.7, color='forestgreen',
            edgecolor='black', label='MCMC posterior')
    ax.axvline(rcut_ref_arcsec_par, color='red', linestyle='--', linewidth=2.5,
               label=f'.par fiducial: {rcut_ref_arcsec_par:.3f} arcsec')
    
    p16, p50, p84 = np.percentile(rcut_ref_arcsec_chain, [16, 50, 84])
    ax.axvline(p50, color='orange', linestyle='-', linewidth=2, alpha=0.8,
               label=f'MCMC median: {p50:.3f} arcsec')
    ax.axvspan(p16, p84, alpha=0.2, color='orange',
               label=f'68% CI: [{p16:.3f}, {p84:.3f}]')
    
    ax.set_xlabel(r'$r_{\rm cut,ref}$ [arcsec]', fontsize=13)
    ax.set_ylabel('N samples', fontsize=13)
    ax.set_title('Reference Cut Radius', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    if rcut_ref_arcsec_par < np.percentile(rcut_ref_arcsec_chain, 16) or \
       rcut_ref_arcsec_par > np.percentile(rcut_ref_arcsec_chain, 84):
        print(f"  ⚠ .par r_cut_ref={rcut_ref_arcsec_par:.3f} arcsec outside MCMC 68% CI [{p16:.3f}, {p84:.3f}]")
    else:
        print(f"  ✓ .par r_cut_ref={rcut_ref_arcsec_par:.3f} arcsec within MCMC range [{p16:.3f}, {p84:.3f}]")
    
    plt.tight_layout()
    plt.savefig('scaling_relation_validation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved scaling relation validation plot: scaling_relation_validation.png")
    plt.show()

if __name__ == "__main__":

    # parse the name of the photometric catalog from command line arguments
    parser = argparse.ArgumentParser(description='Read a photo-morpho catalog and match it with the best.par.')
    parser.add_argument('--parfile', type=str, required=False, help='Path to the lenstool parameter file', default='/Users/maxmen3/stiva/pietro_models/M0416_B22.par')
    parser.add_argument('--catalog', type=str, required=False, help='Path to the photometric catalog file', default='/Users/maxmen3/projects/Pierpaoli/m0416.dat')
    parser.add_argument('--explain', type=str, required=False, help='Path to the ReadMe file',
                        default='/Users/maxmen3/projects/Pierpaoli/ReadMe.txt')

    parser.add_argument('--bayesfile', type=str, required=False,
                        help='Path to LensTool bayes_*.dat MCMC sample file (to propagate uncertainties).',
                        default=None)
    parser.add_argument('--nchains', type=int, required=False,
                        help='Number of MCMC samples (chains) to use from bayesfile (default: use all).',
                        default=None)
    parser.add_argument('--mref', type=float, required=False, default=DEFAULT_MREF_F160W,
                        help='Reference F160W magnitude used in the scaling relations (default from Bergamini et al.).')
    parser.add_argument('--alpha', type=float, required=False, default=DEFAULT_ALPHA,
                        help='Velocity-dispersion scaling exponent alpha (default from Bergamini et al.).')
    parser.add_argument('--beta_cut', type=float, required=False, default=DEFAULT_BETA_CUT,
                        help='Cut-radius scaling exponent beta_cut (default from Bergamini et al.).')
    parser.add_argument('--outcsv', type=str, required=False, default='lens_galaxy_fits.csv',
                        help='Output CSV filename.')
    parser.add_argument('--outh5', type=str, required=False, default=None,
                        help='Output HDF5 filename for per-sample DM parameters (default: derived from outcsv).')
    parser.add_argument('--run-validation', action='store_true', default=False,
                        help='Run built-in validation tests and exit.')
    args = parser.parse_args()

    # Quick validation suite (unit-level) -------------------------------------
    if args.run_validation:
        validate_density_profiles()
        validate_concentration_calculation()
        validate_fitting_convergence()
        # MCMC validation only if a bayesfile is provided
        if args.bayesfile:
            validate_mcmc_propagation(args.bayesfile, args.explain, args.catalog, ntest=3)
            # NEW: Validate scaling relation consistency
            validate_scaling_relation_consistency(
                args.bayesfile, args.parfile, args.catalog, args.explain,
                alpha=args.alpha, beta_cut=args.beta_cut, mref=args.mref
            )
        else:
            print("⊘ Skipping MCMC validation (no --bayesfile provided)")
        # No need to proceed further
        exit(0)

    # read a lenstool parameter file
    runmode = readLenstoolBlock(best_par=args.parfile, block_name='runmode')
    grille = readLenstoolBlock(best_par=args.parfile, block_name='grille')
    potentiel_ = readLenstoolBlock(best_par=args.parfile, block_name='potentiel')
    potentiel = selectPotentielByType(potentiel_, ptype='gal')
    cline = readLenstoolBlock(best_par=args.parfile, block_name='cline')
    grande = readLenstoolBlock(best_par=args.parfile, block_name='grande')
    cosmologie = readLenstoolBlock(best_par=args.parfile, block_name='cosmologie')
    champ = readLenstoolBlock(best_par=args.parfile, block_name='champ')
    print(cosmologie)
    cosmo = FlatLambdaCDM(H0=cosmologie['H0'], Om0=cosmologie['omegaM'])


    # read the photometric catalog as .vot file
    df_cat = load_cluster(readme_path=args.explain, data_path=args.catalog)
    print(df_cat.head(10))

    # Optionally load LensTool MCMC samples (bayes_*.dat) to propagate errors on the DM fit parameters
    sigma_ref_LT_chain = None
    rcut_ref_arcsec_chain = None
    if args.bayesfile is not None:
        sigma_ref_LT_chain, rcut_ref_arcsec_chain = load_bayes_scaling_params(args.bayesfile)
        n_total = len(sigma_ref_LT_chain)
        if args.nchains is not None and args.nchains < n_total:
            # Randomly subsample (without replacement) following sampleBayes pattern
            idx = np.random.choice(n_total, size=args.nchains, replace=False)
            sigma_ref_LT_chain = sigma_ref_LT_chain[idx]
            rcut_ref_arcsec_chain = rcut_ref_arcsec_chain[idx]
            print(f"Randomly selected {args.nchains} / {n_total} MCMC samples from {args.bayesfile}")
        else:
            print(f"Loaded {n_total} MCMC samples from {args.bayesfile}")

    # --- Containers for per-sample DM-fit parameters (for HDF5 output) ---
    save_h5_samples = (args.bayesfile is not None)
    if args.outh5 is None and save_h5_samples:
        # default: replace .csv with .h5, else append .h5
        if args.outcsv.lower().endswith('.csv'):
            args.outh5 = args.outcsv[:-4] + '.h5'
        else:
            args.outh5 = args.outcsv + '.h5'
    # We'll store a cube: (N_gal_saved, N_samples, N_params)
    _h5_lens_index = []
    _h5_cat_index = []
    _h5_ra = []
    _h5_dec = []
    _h5_dm_params = []
    _h5_piemd_params = []  # Store PIEMD params (v_disp, cut_kpc) per sample
    _h5_jaffe_params = []  # Store Jaffe (stellar) params per galaxy (not per sample)
    _h5_sample_ids = None
    _h5_param_names = ['r_s_kpc', 'r_t_kpc', 'rho_s_Msun_kpc3', 'R200_kpc', 'c200', 'M200_Msun']
    _h5_piemd_param_names = ['v_disp_km_s', 'cut_radius_kpc']
    _h5_jaffe_param_names = ['M_star_Msun', 'Re_kpc', 'r_J_kpc']

    # get the reference RA and Dec from the parameter file


    x,y = getXYfromPotentiel(potentiel)
    ra_ref, dec_ref = getRef_RA_DEC(args.parfile)

    print ('Reference RA, Dec:', ra_ref, dec_ref)

    ra,dec = getRADECfromXY(x,y, ra_ref, dec_ref)

    # get ra and dec from the catalog
    ra_cat = df_cat['RAdeg'].values
    dec_cat = df_cat['DEdeg'].values

    # match ra, dec to ra_cat, dec_cat
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    # Build coords
    coords_lens = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    coords_cat = SkyCoord(ra=ra_cat * u.deg, dec=dec_cat * u.deg)

    # For each lens point, the nearest catalog index + separation
    idx_cat, d2d, _ = coords_lens.match_to_catalog_sky(coords_cat)

    # Keep only good matches
    tolerance = 0.1 * u.arcsec
    mask = d2d < tolerance

    # Lens indices that matched, and their corresponding catalog indices
    lens_idx = np.where(mask)[0]  # indices into lens arrays (ra, dec, potentiel, ...)
    cat_idx = idx_cat[mask]  # corresponding indices into df_cat

    #validate_matching(ra, dec, ra_cat, dec_cat, cat_idx)

    # (Optional but handy) a DataFrame of the matches
    matches = pd.DataFrame({
        "lens_index": lens_idx,
        "cat_index": cat_idx,
        "ra_lens": ra[lens_idx],
        "dec_lens": dec[lens_idx],
        "RAdeg": ra_cat[cat_idx],
        "DEdeg": dec_cat[cat_idx],
        "sep_arcsec": d2d[mask].arcsec
    })

    # Attach magnitudes (e.g. F160W) from the catalog
    matches["F160W"] = df_cat["F160W"].values[matches["cat_index"]]

    print(matches.head())
    print("Number of matches:", len(matches))

    # Build a map lens_index -> cat_index
    match_map = dict(zip(matches["lens_index"].to_numpy(), matches["cat_index"].to_numpy()))

    imatch = 0
    for j in range(len(potentiel)):  # j is a LENS index
        if j in match_map:

            j_cat = match_map[j]  # corresponding CATALOG row
            #m_F160W = df_cat.iloc[j_cat]["F160W"]
            Re_F160W = df_cat.iloc[j_cat]["ReF814W"]

            v_disp_ = findInBlock(potentiel[j], 'v_disp')
            zl = findInBlock(potentiel[j], 'z_lens')
            v_disp_ = float(v_disp_)*np.sqrt(3.0/2.0)
            cut_radius_ = findInBlock(potentiel[j], 'cut_radius_kpc')
            core_radius_ = findInBlock(potentiel[j], 'core_radius_kpc')
            m_F160W = float(findInBlock(potentiel[j], 'mag'))
            M_star = stellar_mass_to_light(m_F160W)
            rj = jaffe_scale_radius(Re=Re_F160W)

            # Enforce physical bound: stellar mass within Re cannot exceed total mass within Re
            Mj_Re_orig = jaffe_Menc(Re_F160W, M_star, Re_F160W)
            Mp_Re = PIEMD_Menc(Re_F160W, float(core_radius_), float(cut_radius_), v_disp_)
            stellar_mass_capped = False
            if Mj_Re_orig > Mp_Re:
                scale = Mp_Re / Mj_Re_orig
                M_star *= scale
                stellar_mass_capped = True
                print(f"  ⚠ Galaxy #{j}: stellar mass capped (M*_orig={Mj_Re_orig:.2e} > M_PIEMD={Mp_Re:.2e}), "
                      f"scale={scale:.3f}")
            # recompute Jaffe enclosed mass after scaling for consistency
            Mj_Re = jaffe_Menc(Re_F160W, M_star, Re_F160W)

            print(f"Lens #{j}: matched cat #{j_cat}, F160W={m_F160W:.3f}, "
                  f"RA/Dec(lens)=({ra[j]:.6f},{dec[j]:.6f}), "
                  f"RA/Dec(cat)=({df_cat.iloc[j_cat]['RAdeg']:.6f},{df_cat.iloc[j_cat]['DEdeg']:.6f})",
                  f"v_disp={v_disp_}, cut_radius={cut_radius_}, core_radius={core_radius_}, M*={M_star:.2e} Msun, r_J={rj:.2f}")

            r = np.logspace(-2, 3, 1000)  # kpc
            rho_j = jaffe_rho(r, M_star, Re_F160W)
            rho_piemd = PIEMD_rho(r, float(core_radius_), float(cut_radius_), v_disp_)

            mask = rho_piemd > 2*rho_j

            print (f"  M_Jaffe={jaffe_Menc(Re_F160W, M_star, Re_F160W):.2e} Msun, M_PIEMD={PIEMD_Menc(Re_F160W,float(core_radius_), float(cut_radius_), v_disp_):.2e} Msun")
            f = jaffe_Menc(Re_F160W, M_star, Re_F160W)/PIEMD_Menc(Re_F160W,float(core_radius_), float(cut_radius_), v_disp_)

            # Dark-matter density must be non-negative. Clip residuals to >= 0.
            # Use the same positive-DM mask for radii used in the fit.
            rho_dm = np.clip(rho_piemd - rho_j, 0.0, None)
            dm_mask = mask & np.isfinite(rho_dm) & (rho_dm > 0)
            rho2fit = rho_dm[dm_mask]
            rfit = r[dm_mask]

            # fit a truncated NFW to rho2fit
            r_s_guess = 3.0#cut_radius_
            r_t_guess = cut_radius_
            rho_s_guess = 1e7
            r_s, r_t, rho_s, diag = fit_with_tNFW(rfit, rho2fit, r_s_guess, r_t_guess, rho_s_guess)

            rhonfw = trunc_NFW_rho(rfit, r_s, r_t, rho_s)
            rho_crit = cosmo.critical_density(zl).to_value('Msun/kpc3')
            r200 = rdelta_trunc_nfw(r_s, r_t, rho_s, rho_ref=rho_crit, Delta=200.0)
            c200nfw = c200_from_rs_rhos(r_s, rho_s, rho_ref=rho_crit)
            m200 = M_tNFW_enclosed(r200, r_s, r_t, rho_s)
            mtot = PIEMD_mtotal(float(core_radius_), float(cut_radius_), v_disp_)
            print(f'  tNFW fit: r_s={r_s:.2f}, r_t={r_t:.2f}, rho_s={rho_s:.2e}, c200={r200/r_s:.2f}, R200={r200:.2f} kpc, c200nfw={c200nfw:.2f}')

            # --- If MCMC samples are provided, propagate scaling-relation uncertainties into DM-fit uncertainties ---
            mcmc_summary = {}
            if sigma_ref_LT_chain is not None and rcut_ref_arcsec_chain is not None:
                nsamp_chain = len(sigma_ref_LT_chain)
                # Per-sample storage (shape: Nsamp x Npar). Filled with NaN when a sample fit fails.
                dm_params_samp = np.full((nsamp_chain, 6), np.nan, dtype=float)
                piemd_params_samp = np.full((nsamp_chain, 2), np.nan, dtype=float)  # v_disp, cut_kpc

                r_s_samp = []
                r_t_samp = []
                rho_s_samp = []
                r200_samp = []
                c200_samp = []
                m200_samp = []

                # Precompute stellar density once (it does not depend on the lens model chain)
                r_grid = r  # already defined
                rho_star_grid = rho_j

                for isamp, (sig_ref, rcut_ref_as) in enumerate(zip(sigma_ref_LT_chain, rcut_ref_arcsec_chain)):
                    try:
                        v_disp_chain, cut_kpc_chain = galaxy_dpie_from_scaling(
                            m_F160W, sig_ref, rcut_ref_as, cosmo, zl,
                            alpha=args.alpha, beta_cut=args.beta_cut, mref=args.mref
                        )
                        # Store PIEMD parameters for this sample
                        piemd_params_samp[isamp, :] = [v_disp_chain, cut_kpc_chain]
                        
                        rho_tot_chain = PIEMD_rho(r_grid, float(core_radius_), float(cut_kpc_chain), v_disp_chain)

                        dm_mask_chain = rho_tot_chain > 2.0 * rho_star_grid
                        if np.count_nonzero(dm_mask_chain) < 20:
                            continue

                        rho_dm_chain = rho_tot_chain[dm_mask_chain] - rho_star_grid[dm_mask_chain]
                        rfit_chain = r_grid[dm_mask_chain]

                        # Fit truncated NFW
                        r_s_c, r_t_c, rho_s_c, diag_c = fit_with_tNFW(
                            rfit_chain, rho_dm_chain,
                            r_s_guess, float(cut_kpc_chain), rho_s_guess
                        )
                        if not np.isfinite(r_s_c) or not np.isfinite(r_t_c) or not np.isfinite(rho_s_c):
                            continue

                        rho_crit_c = rho_crit  # same zl for all
                        r200_c = rdelta_trunc_nfw(r_s_c, r_t_c, rho_s_c, rho_ref=rho_crit_c, Delta=200.0)
                        m200_c = M_tNFW_enclosed(r200_c, r_s_c, r_t_c, rho_s_c)

                        # Fill the per-sample array (even if we later summarize)
                        dm_params_samp[isamp, :] = [r_s_c, r_t_c, rho_s_c, r200_c, r200_c / r_s_c, m200_c]

                        # Also collect for quick percentiles in the CSV
                        r_s_samp.append(r_s_c)
                        r_t_samp.append(r_t_c)
                        rho_s_samp.append(rho_s_c)
                        r200_samp.append(r200_c)
                        c200_samp.append(r200_c / r_s_c)
                        m200_samp.append(m200_c)
                    except Exception:
                        continue

                def _pct(x):
                    if len(x) == 0:
                        return (np.nan, np.nan, np.nan)
                    x = np.asarray(x, dtype=float)
                    return tuple(np.nanpercentile(x, [16, 50, 84]))

                (rs16, rs50, rs84) = _pct(r_s_samp)
                (rt16, rt50, rt84) = _pct(r_t_samp)
                (rhos16, rhos50, rhos84) = _pct(rho_s_samp)
                (r200_16, r200_50, r200_84) = _pct(r200_samp)
                (c200_16, c200_50, c200_84) = _pct(c200_samp)
                (m200_16, m200_50, m200_84) = _pct(m200_samp)

                mcmc_summary = {
                    'nsamp_mcmc_used': len(r_s_samp),
                    'r_s_tNFW_p16': rs16, 'r_s_tNFW_p50': rs50, 'r_s_tNFW_p84': rs84,
                    'r_t_tNFW_p16': rt16, 'r_t_tNFW_p50': rt50, 'r_t_tNFW_p84': rt84,
                    'rho_s_tNFW_p16': rhos16, 'rho_s_tNFW_p50': rhos50, 'rho_s_tNFW_p84': rhos84,
                    'R200_tNFW_kpc_p16': r200_16, 'R200_tNFW_kpc_p50': r200_50, 'R200_tNFW_kpc_p84': r200_84,
                    'c200_tNFW_p16': c200_16, 'c200_tNFW_p50': c200_50, 'c200_tNFW_p84': c200_84,
                    'M200_tNFW_Msun_p16': m200_16, 'M200_tNFW_Msun_p50': m200_50, 'M200_tNFW_Msun_p84': m200_84,
                }
            # create a dictionary with the results, including ra and dec of the each lens
            results = {
                'lens_index': j,
                'cat_index': j_cat,
                'ra_lens': ra[j],
                'dec_lens': dec[j],
                'F160W': m_F160W,
                'Re_F160W': Re_F160W,
                'M_star': M_star,
                'r_J': rj,
                'v_disp': v_disp_,
                'core_radius_kpc': float(core_radius_),
                'cut_radius_kpc': float(cut_radius_),
                'r_s_tNFW': r_s,
                'r_t_tNFW': r_t,
                'rho_s_tNFW': rho_s,
                'c200_tNFW': r200/r_s,
                'R200_tNFW_kpc': r200,
                'c200_NFW': c200nfw,
                'M200_tNFW_Msun': m200,
                'Mtot_PIEMD_Msun': mtot,
                'fstar_Re': f,
                'stellar_mass_capped': stellar_mass_capped,  # True if M* was reduced to match PIEMD
                'npts': diag['npts'],
                'diag': diag['rmse_log']
            }
            # attach MCMC-derived percentiles (if computed)
            if mcmc_summary:
                results.update(mcmc_summary)

            # Warn if stellar-to-total mass ratio is suspiciously high (fstar > 0.9)
            if f > 0.9:
                print(f"  ⚠ Galaxy #{j}: fstar_Re={f:.3f} > 0.9 (stellar mass dominates, DM fit unreliable)")

            if diag['rmse_log'] > 110.6:
                print('  Warning: high rmse_log =', diag['rmse_log'], diag['npts'], diag['message'])
                fig, ax = plt.subplots(1,1)
                ax.loglog(r[mask], rho_j[mask]*r[mask]**2, label='Jaffe (stars)', color='blue')
                ax.loglog(r[mask], rho_piemd[mask]*r[mask]**2, label='PIEMD (total)', color='red')
                ax.loglog(rfit, rho2fit*rfit**2, label='PIEMD - Jaffe', color='green')
                ax.loglog(rfit, rhonfw*rfit**2, label='tNFW fit to PIEMD - Jaffe', color='orange')
                ax.axvline(r_s, color='orange', linestyle='--', label='r_s')
                ax.axvline(r_t, color='orange', linestyle=':', label='r_t')
                ax.set_xlabel('r [kpc]')
                ax.set_ylabel(r'$\rho(r)$ [$M_\odot$ kpc$^{-3}$]')
                ax.set_title(f'Lens #{j}, cat #{j_cat}, F160W={m_F160W:.2f}, Re={Re_F160W:.2f} arcsec')
                ax.legend()
                plt.show()
                #exit()


            # append results to a dataframe
            if diag['rmse_log'] < 0.6:
                if imatch == 0:
                    df_results = pd.DataFrame(results, index=[0])
                else:
                    df_results = pd.concat([df_results, pd.DataFrame(results, index=[0])], ignore_index=True)
                imatch += 1
                # Store per-sample DM-fit parameters for HDF5 output (aligned with df_results rows)
                if save_h5_samples and (sigma_ref_LT_chain is not None) and ('dm_params_samp' in locals()):
                    _h5_lens_index.append(j)
                    _h5_cat_index.append(j_cat)
                    _h5_ra.append(ra[j])
                    _h5_dec.append(dec[j])
                    _h5_dm_params.append(dm_params_samp)
                    _h5_piemd_params.append(piemd_params_samp)
                    # Jaffe (stellar) params are fixed per galaxy (from photometry, not MCMC)
                    _h5_jaffe_params.append([M_star, Re_F160W, rj])
                    if _h5_sample_ids is None:
                        _h5_sample_ids = np.arange(dm_params_samp.shape[0], dtype=int)

    # save the dataframe to a csv file
    df_results.to_csv(args.outcsv, index=False)

    # Optionally save full posterior samples (per galaxy x per MCMC sample) to HDF5
    if save_h5_samples and len(_h5_dm_params) > 0:
        dm_cube = np.stack(_h5_dm_params, axis=0)  # (Ngal, Nsamp, Npar)
        piemd_cube = np.stack(_h5_piemd_params, axis=0)  # (Ngal, Nsamp, 2)
        jaffe_arr = np.asarray(_h5_jaffe_params, dtype=float)  # (Ngal, 3)
        with h5py.File(args.outh5, 'w') as f:
            f.create_dataset('lens_index', data=np.asarray(_h5_lens_index, dtype=int))
            f.create_dataset('cat_index', data=np.asarray(_h5_cat_index, dtype=int))
            f.create_dataset('ra_lens_deg', data=np.asarray(_h5_ra, dtype=float))
            f.create_dataset('dec_lens_deg', data=np.asarray(_h5_dec, dtype=float))
            f.create_dataset('sample_id', data=np.asarray(_h5_sample_ids, dtype=int))
            f.create_dataset('param_names', data=np.asarray(_h5_param_names, dtype='S'))
            f.create_dataset('dm_params', data=dm_cube, compression='gzip', shuffle=True)
            # Save PIEMD parameters (v_disp, cut_kpc) per sample
            f.create_dataset('piemd_param_names', data=np.asarray(_h5_piemd_param_names, dtype='S'))
            f.create_dataset('piemd_params', data=piemd_cube, compression='gzip', shuffle=True)
            # Save Jaffe (stellar) parameters per galaxy (not per sample - derived from photometry)
            f.create_dataset('jaffe_param_names', data=np.asarray(_h5_jaffe_param_names, dtype='S'))
            f.create_dataset('jaffe_params', data=jaffe_arr, compression='gzip', shuffle=True)
            # Helpful attributes
            f['dm_params'].attrs['description'] = (
                'Per-sample DM-fit parameters for each galaxy: shape (N_gal, N_samp, N_par). '
                'Parameter order given by param_names.'
            )
            f['dm_params'].attrs['units'] = 'r_s[kpc], r_t[kpc], rho_s[Msun/kpc^3], R200[kpc], c200[dimensionless], M200[Msun]'
            f['piemd_params'].attrs['description'] = (
                'Per-sample PIEMD parameters for each galaxy: shape (N_gal, N_samp, 2). '
                'Parameter order: [v_disp (km/s, sigma_0 convention), cut_radius (kpc)].'
            )
            f['piemd_params'].attrs['units'] = 'v_disp[km/s], cut_radius[kpc]'
            f['jaffe_params'].attrs['description'] = (
                'Per-galaxy Jaffe (stellar) parameters: shape (N_gal, 3). '
                'Parameter order: [M_star (Msun), Re (kpc), r_J (kpc)]. '
                'These are derived from photometry and are the same for all MCMC samples.'
            )
            f['jaffe_params'].attrs['units'] = 'M_star[Msun], Re[kpc], r_J[kpc]'
            f.attrs['bayesfile'] = str(args.bayesfile)
            f.attrs['parfile'] = str(args.parfile)
            f.attrs['catalog'] = str(args.catalog)
            f.attrs['mref_F160W'] = float(args.mref)
            f.attrs['alpha'] = float(args.alpha)
            f.attrs['beta_cut'] = float(args.beta_cut)
        print(f"Saved per-sample DM-fit parameters to HDF5: {args.outh5}")
        
        # Validate HDF5 structure
        validate_hdf5_output(args.outh5)
        
        # Validate MCMC posteriors visually (first 3 galaxies)
        validate_mcmc_posteriors(args.outh5, df_results, ngal=3)

    print(df_results.head(10))

    #plot concentration vs Mstar/Mtotal
    fig, ax = plt.subplots(1,1)
    ax.scatter(df_results['fstar_Re'], df_results['c200_NFW'], label='Lens galaxies', color='green')
    ax.set_xlabel(r'$M_{*}/M_{total}$')
    ax.set_ylabel(r'$c_{200}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Concentration vs Stellar-to-Total Mass Ratio')
    plt.show()

    # plot concentration vs diagnostics
    fig, ax = plt.subplots(1,1)
    ax.scatter(df_results['diag'], df_results['c200_NFW'], label='Lens galaxies', color='red')
    ax.set_xlabel(r'Fit Diagnostics (rmse_log)')
    ax.set_ylabel(r'$c_{200}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Concentration vs Fit Diagnostics')
    plt.show()

















