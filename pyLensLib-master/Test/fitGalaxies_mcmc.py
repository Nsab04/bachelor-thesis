import pyLensLib.lenstool as lst
from pyLensLib.lenstool import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import argparse
import astropy.io.fits as fits
# import wcs from astropy
from astropy.wcs import WCS
from scipy.optimize import least_squares, brentq
from astropy.cosmology import FlatLambdaCDM

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
    M = np.pi* (sigma_v ** 2) / G * (r_cut - r_core) * (1 - np.sqrt(r_core**2 + r**2)/ (r_cut - r_core) * np.arctan(r_cut/np.sqrt(r_core**2 + r**2)) + np.sqrt(r_cut**2 + r**2)/(r_cut - r_core) * np.arctan(r_cut/np.sqrt(r_cut**2 + r**2)))
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
    M = np.pi* (sigma_v ** 2) / G * (r_cut - r_core)
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
    if y_err is not None:
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
        with fits.open(prefix_models + rgbimage) as hudl:
            rgb_fits_data = hudl[0].data

        rgb_image = np.transpose(rgb_fits_data, (1, 2, 0))
        rgb_image = rgb_image / np.max(rgb_image)
        # get wcs from the fits header
        wcs = WCS(hudl[0].header,naxis=2)
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

    args = parser.parse_args()

    # read a lenstool parameter file
    runmode = readLenstoolBlock(best_par=args.parfile, block_name='runmode')
    grille = readLenstoolBlock(best_par=args.parfile, block_name='grille')
    potentiel = readLenstoolBlock(best_par=args.parfile, block_name='potentiel')
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
        if args.nchains is not None:
            sigma_ref_LT_chain = sigma_ref_LT_chain[:args.nchains]
            rcut_ref_arcsec_chain = rcut_ref_arcsec_chain[:args.nchains]
        print(f"Loaded {len(sigma_ref_LT_chain)} MCMC samples from {args.bayesfile}")

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
            m_F160W = df_cat.iloc[j_cat]["F160W"]
            Re_F160W = df_cat.iloc[j_cat]["ReF814W"]

            v_disp_ = findInBlock(potentiel[j], 'v_disp')
            zl = findInBlock(potentiel[j], 'z_lens')
            v_disp_ = float(v_disp_)*np.sqrt(3.0/2.0)
            cut_radius_ = findInBlock(potentiel[j], 'cut_radius_kpc')
            core_radius_ = findInBlock(potentiel[j], 'core_radius_kpc')
            M_star = stellar_mass_to_light(m_F160W)
            rj = jaffe_scale_radius(Re=Re_F160W)

            # Enforce physical bound: stellar mass within Re cannot exceed total mass within Re
            Mj_Re = jaffe_Menc(Re_F160W, M_star, Re_F160W)
            Mp_Re = PIEMD_Menc(Re_F160W, float(core_radius_), float(cut_radius_), v_disp_)
            if Mj_Re > Mp_Re:
                scale = Mp_Re / Mj_Re
                M_star *= scale
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
                r_s_samp = []
                r_t_samp = []
                rho_s_samp = []
                r200_samp = []
                c200_samp = []
                m200_samp = []

                # Precompute stellar density once (it does not depend on the lens model chain)
                r_grid = r  # already defined
                rho_star_grid = rho_j

                for sig_ref, rcut_ref_as in zip(sigma_ref_LT_chain, rcut_ref_arcsec_chain):
                    try:
                        v_disp_chain, cut_kpc_chain = galaxy_dpie_from_scaling(
                            m_F160W, sig_ref, rcut_ref_as, cosmo, zl,
                            alpha=args.alpha, beta_cut=args.beta_cut, mref=args.mref
                        )
                        rho_tot_chain = PIEMD_rho(r_grid, float(core_radius_), float(cut_kpc_chain), v_disp_chain)

                        dm_mask_chain = rho_tot_chain > 2.0 * rho_star_grid
                        if np.count_nonzero(dm_mask_chain) < 20:
                            continue

                        rho_dm_chain = rho_tot_chain[dm_mask_chain] - rho_star_grid[dm_mask_chain]
                        rfit_chain = r_grid[dm_mask_chain]

                        # Fit truncated NFW
                        r_s_c, r_t_c, rho_s_c, diag_c = fit_with_tNFW(rfit_chain, rho_dm_chain,
                                                                     r_s_guess, float(cut_kpc_chain), rho_s_guess)
                        if not np.isfinite(r_s_c) or not np.isfinite(r_t_c) or not np.isfinite(rho_s_c):
                            continue

                        rho_crit_c = rho_crit  # same zl for all
                        r200_c = rdelta_trunc_nfw(r_s_c, r_t_c, rho_s_c, rho_ref=rho_crit_c, Delta=200.0)
                        m200_c = M_tNFW_enclosed(r200_c, r_s_c, r_t_c, rho_s_c)

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
                'npts': diag['npts'],
                'diag': diag['rmse_log']
            }
            # attach MCMC-derived percentiles (if computed)
            if mcmc_summary:
                results.update(mcmc_summary)

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

    # save the dataframe to a csv file
    df_results.to_csv(args.outcsv, index=False)
    print(df_results.head(10))


    # plot the concentration-mass relation
    fig, ax = plt.subplots(1,1)
    ax.scatter(df_results['Mtot_PIEMD_Msun'], df_results['c200_NFW'], label='Lens galaxies', color='blue')
    ax.set_xlabel(r'$M_{200}$ [$M_\odot$]')
    ax.set_ylabel(r'$c_{200}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Concentration-Mass relation for lens galaxies')
    plt.show()

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

















