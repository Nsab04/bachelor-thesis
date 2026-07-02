from pyLensLib.lenstool import *
import numpy as np
import pandas as pd
import argparse
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from scipy.optimize import least_squares, brentq
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
import re, os

# -------------------- CONFIG / CONSTANTS --------------------
prefix_models   = '/Users/maxmen3/stiva/pietro_models/'
prefix_catalogs = '/Users/maxmen3/projects/Pierpaoli/'

# Re (projected half-mass/light) and Jaffe scale: Re = 0.7447 * rJ  => rJ ≈ 1.342 * Re
JAFFE_Re_OVER_rJ = 0.7447
G_KPC = 4.30091e-6  # kpc (km/s)^2 / Msun

# --- knobs you can tweak ---
USE_STAR_SLACK     = True   # weak priors on M_* and rJ (recommended)
PRIOR_LOGM_DEx     = 0.005   # ~0.25 dex scatter on M*/L (Grillo+15-like)
PRIOR_LOGRJ_DEx    = 0.2   # ~0.2 dex scatter on rJ/Re mapping
W_SLOPE            = 0.10   # small weight on log-slope residuals (0..0.3 typical)
ROBUST_LOSS        = "soft_l1"
F_SCALE            = 0.1
MAX_NFEV           = 30000
N_STARTS           = 5      # multi-starts for stability

# -------------------- PROFILE DEFINITIONS (3D) --------------------
def jaffe_scale_radius(Re):
    """Return r_J from projected effective radius Re (same units)."""
    return Re / JAFFE_Re_OVER_rJ

def rho_jaffe(r, Mstar, rJ):
    r = np.asarray(r, float)
    return (Mstar * rJ) / (4.0*np.pi * r**2 * (r + rJ)**2)

def rho_tnfw(r, Rs, rtr, rho0):
    """Multiplicatively truncated NFW (valid for ANY r)."""
    r  = np.asarray(r, float)
    x  = r / Rs
    trunc = (rtr**2) / (rtr**2 + r**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        return rho0 / (x * (1.0 + x)**2) * trunc

def rho_piemd_3d(r, r_core, r_cut, sigma_v):
    """
    Spherical dPIE/PIEMD 3D density (your normalization).
    Keep your original choice to preserve backward compatibility.
    """
    r = np.asarray(r, float)
    G = G_KPC
    return (sigma_v ** 2) / (2 * np.pi * G) * (r_cut + r_core) / (r_core**2 * r_cut) \
           / (1 + (r / r_core) ** 2) / (1 + (r / r_cut) ** 2)

def ln_slope(r, rho):
    r = np.asarray(r, float); rho = np.asarray(rho, float)
    lr, lρ = np.log(r), np.log(rho)
    return np.gradient(lρ, lr)

# --------- tNFW enclosed mass (analytic) & RΔ for multiplicative truncation ---------
def M_tNFW_enclosed(r, Rs, r_trunc, rho0):
    r   = np.asanyarray(r, dtype=float)
    Rs  = float(Rs)
    tau = float(r_trunc / Rs)
    rho0 = float(rho0)
    x = r / Rs
    tau2 = tau*tau
    denom = (tau2 + 1.0)**2
    A = tau2 * (tau2 - 1.0) / denom
    B = - tau2 / (tau2 + 1.0)
    D = 2.0 * (tau2**2) / denom
    term1 = A * (np.log1p(x) - 0.5*np.log1p((x/tau)**2))
    term2 = B * (x / (1.0 + x))
    term3 = (D / tau) * np.arctan(x / tau)
    F = term1 + term2 + term3
    return 4.0*np.pi * rho0 * (Rs**3) * F

def Rdelta_tNFW(Rs, rtr, rho0, rho_ref, Delta=200.0):
    """
    Solve for R_Delta where <rho>(<R) = 3M/(4πR^3) = Delta * rho_ref
    for the multiplicatively truncated NFW using the analytic M(<r).
    """
    K = (4.0/3.0) * np.pi * Delta * float(rho_ref)
    def g(R):
        return M_tNFW_enclosed(R, Rs, rtr, rho0) - K * R**3
    # bracket: small R => g>0, large R => g<0
    R_lo = 1e-9 * Rs
    while g(R_lo) <= 0:  # ensure positive (mean density huge at tiny R)
        R_lo *= 0.1
        if R_lo < 1e-30*Rs: break
    R_hi = max(3*Rs, rtr, 10.0*Rs)
    while g(R_hi) > 0:
        R_hi *= 2.0
        if R_hi > 1e8*max(Rs, rtr):
            raise RuntimeError("Failed to bracket RΔ")
    return brentq(lambda R: g(R), R_lo, R_hi)

def c200_from_rs_rhos(r_s, rho_s, rho_ref):
    """Solve δ_c(c)=rho_s/rho_ref for NFW c200 (no truncation)."""
    r_s = float(r_s)
    rho_s = float(rho_s)
    if r_s <= 0 or rho_s <= 0 or rho_ref <= 0:
        raise ValueError("r_s, rho_s, rho_ref must be >0.")

    S = rho_s/float(rho_ref)
    def f(c):
        return (200.0/3.0) * c**3 / (np.log1p(c) - c/(1.0+c)) - S
    return brentq(f, 1e-6, 1e5)

# -------------------- Fitting: Jaffe + tNFW → PIEMD(3D) --------------------
def fit_piemd_3d_with_jaffe_tnfw(
    r,                       # radii where we compare 3D densities
    r_core, r_cut, sigma_v,  # PIEMD parameters (TOTAL 3D)
    Mstar_phot, Re_kpc,      # stellar proxies (M* from Grillo15; Re in kpc)
    Rs_guess, rtr_guess, rho0_guess,
    use_star_slack=USE_STAR_SLACK,
    prior_logM_dex=PRIOR_LOGM_DEx,
    prior_logrJ_dex=PRIOR_LOGRJ_DEx,
    w_slope=W_SLOPE,
    loss=ROBUST_LOSS, f_scale=F_SCALE,
    max_nfev=MAX_NFEV, n_starts=N_STARTS, seed=42
):
    rng = np.random.default_rng(seed)

    r = np.asarray(r, float)
    rho_tot = rho_piemd_3d(r, r_core, r_cut, sigma_v)
    good = np.isfinite(rho_tot) & (rho_tot > 0) & (r > 0)
    r_fit = r[good]; rho_fit = rho_tot[good]
    if r_fit.size < 8:
        raise ValueError("Too few radii in fit window.")

    # stellar priors (weak)
    rJ_nom = jaffe_scale_radius(Re_kpc)
    lnM0   = np.log(Mstar_phot)
    lnrJ0  = np.log(rJ_nom)
    sM  = prior_logM_dex * np.log(10.0)
    sRJ = prior_logrJ_dex * np.log(10.0)

    # Parameters θ = (a,b,c,d,e)
    # stars: M* = exp(a), rJ = exp(b)
    # DM:    Rs = exp(c), rtr = Rs*(1+exp(d)), rho0 = exp(e)
    def unpack(theta):
        Mstar = np.exp(theta[0])
        rJ    = np.exp(theta[1])
        Rs    = np.exp(theta[2])
        rtr   = Rs * (1.0 + np.exp(theta[3]))
        rho0  = np.exp(theta[4])
        return Mstar, rJ, Rs, rtr, rho0

    seeds = []
    for _ in range(n_starts):
        f = 10.0**rng.uniform(-0.2, 0.2)  # ±0.2 dex
        g = 10.0**rng.uniform(-0.2, 0.2)
        h = 10.0**rng.uniform(-0.3, 0.3)
        Rs0   = max(Rs_guess * f, 1e-6)
        rtr0  = max(rtr_guess * g, Rs0*1.2)
        rho00 = max(rho0_guess * h, 1e-12)
        theta0 = np.array([
            lnM0, lnrJ0,
            np.log(Rs0),
            np.log(max(rtr0/Rs0 - 1.0, 1e-6)),
            np.log(rho00)
        ])
        seeds.append(theta0)

    best = None

    def residuals(theta):
        Mstar, rJ, Rs, rtr, rho0 = unpack(theta)

        # lock stars if requested
        if not use_star_slack:
            Mstar, rJ = np.exp(lnM0), np.exp(lnrJ0)

        rho_model = rho_jaffe(r_fit, Mstar, rJ) + rho_tnfw(r_fit, Rs, rtr, rho0)

        # fit residuals (log space) -> 1-D
        res_fit = (np.log(rho_model) - np.log(rho_fit)).ravel()

        # optional slope residuals -> 1-D (or empty)
        if w_slope > 0:
            s_data = ln_slope(r_fit, rho_fit)
            s_model = ln_slope(r_fit, rho_model)
            res_slope = (w_slope * (s_model - s_data)).ravel()
        else:
            res_slope = np.array([], dtype=float)

        # priors as a small 1-D vector (size 0 if disabled)
        if use_star_slack:
            prior_vec = np.array([
                (np.log(Mstar) - lnM0) / sM,
                (np.log(rJ) - lnrJ0) / sRJ
            ], dtype=float)
        else:
            prior_vec = np.array([], dtype=float)

        # concatenate ONLY 1-D arrays
        return np.concatenate([res_fit, res_slope, prior_vec])

    for theta0 in seeds:
        sol = least_squares(residuals, theta0, method="trf",
                            loss=loss, f_scale=f_scale, max_nfev=max_nfev)
        if (best is None) or (sol.cost < best["cost"]):
            best = dict(cost=sol.cost, sol=sol)

    sol = best["sol"]
    Mstar, rJ, Rs, rtr, rho0 = unpack(sol.x)

    # Diagnostics in log space
    rho_model = rho_jaffe(r_fit, Mstar, rJ) + rho_tnfw(r_fit, Rs, rtr, rho0)
    res_log = np.log(rho_model) - np.log(rho_fit)
    rmse_log = float(np.sqrt(np.mean(res_log**2)))
    y = np.log(rho_fit); yhat = np.log(rho_model)
    sst = np.sum((y - y.mean())**2)
    r2_log = float(1.0 - np.sum((yhat - y)**2)/sst) if sst > 0 else np.nan

    return dict(Mstar=Mstar, rJ=rJ, Rs=Rs, rtr=rtr, rho0=rho0,
                rmse_log=rmse_log, r2_log=r2_log, nfev=sol.nfev, success=sol.success, message=sol.message)

# -------------------- PHOTOMETRY M* RELATION (Grillo+15) --------------------
def stellar_mass_to_light(mag_F160W):
    """log10 M* = 18.541 - 0.416 * m_F160W  (Msun)"""
    return 10.0**(18.541 - 0.416 * mag_F160W)

# -------------------- VizieR ReadMe parser (unchanged) --------------------
def _parse_vizier_bytes_table(readme_text: str, data_basename: str):
    sec_pat = re.compile(r"Byte-by-byte Description of file:\s*(.*)")
    lines = readme_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        m = sec_pat.search(line)
        if m and data_basename in m.group(1):
            start_idx = i; break
    if start_idx is None:
        for i, line in enumerate(lines):
            if sec_pat.search(line):
                start_idx = i; break
    if start_idx is None:
        raise ValueError("No 'Byte-by-byte' section found in ReadMe.")

    row_pat = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s+[A-Za-z0-9.]+\s+\S+\s+([A-Za-z0-9_]+)")
    colspecs, names = [], []
    for line in lines[start_idx+1:]:
        if line.strip().startswith("Byte-by-byte Description of file:"):
            break
        m = row_pat.match(line)
        if m:
            a, b, label = m.groups()
            colspecs.append((int(a)-1, int(b))); names.append(label)
        elif colspecs and not line.strip():
            break
    if not colspecs:
        raise ValueError("Could not parse column specs from ReadMe.")
    return colspecs, names

def load_cluster(readme_path="ReadMe.txt", data_path="m0416.dat"):
    with open(readme_path, encoding="utf-8", errors="replace") as f:
        readme_text = f.read()
    colspecs, names = _parse_vizier_bytes_table(readme_text, os.path.basename(data_path))
    return pd.read_fwf(data_path, colspecs=colspecs, names=names)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit PIEMD (3D) with Jaffe + truncated NFW for many galaxies.')
    parser.add_argument('--parfile', type=str, default=f'{prefix_models}/M0416_B22.par')
    parser.add_argument('--catalog', type=str, default=f'{prefix_catalogs}/m0416.dat')
    parser.add_argument('--explain', type=str, default=f'{prefix_catalogs}/ReadMe.txt')
    args = parser.parse_args()

    # Read lenstool parameter blocks
    runmode   = readLenstoolBlock(best_par=args.parfile, block_name='runmode')
    grille    = readLenstoolBlock(best_par=args.parfile, block_name='grille')
    potentiel = readLenstoolBlock(best_par=args.parfile, block_name='potentiel')
    cline     = readLenstoolBlock(best_par=args.parfile, block_name='cline')
    grande    = readLenstoolBlock(best_par=args.parfile, block_name='grande')
    cosmologie= readLenstoolBlock(best_par=args.parfile, block_name='cosmologie')
    champ     = readLenstoolBlock(best_par=args.parfile, block_name='champ')
    print("Cosmology:", cosmologie)
    cosmo = FlatLambdaCDM(H0=cosmologie['H0'], Om0=cosmologie['omegaM'])

    # Load catalog
    df_cat = load_cluster(readme_path=args.explain, data_path=args.catalog)
    print(df_cat.head(3))

    # Lens ref coords
    x,y = getXYfromPotentiel(potentiel)
    ra_ref, dec_ref = getRef_RA_DEC(args.parfile)
    ra,dec = getRADECfromXY(x,y, ra_ref, dec_ref)

    # Cross-match to catalog (≤0.1")
    coords_lens = SkyCoord(ra=ra*u.deg,     dec=dec*u.deg)
    coords_cat  = SkyCoord(ra=df_cat['RAdeg'].values*u.deg, dec=df_cat['DEdeg'].values*u.deg)
    idx_cat, d2d, _ = coords_lens.match_to_catalog_sky(coords_cat)
    mask = d2d < 0.1*u.arcsec
    lens_idx = np.where(mask)[0]
    cat_idx  = idx_cat[mask]

    matches = pd.DataFrame({
        "lens_index": lens_idx,
        "cat_index":  cat_idx,
        "ra_lens":    ra[lens_idx],
        "dec_lens":   dec[lens_idx],
        "RAdeg":      df_cat['RAdeg'].values[cat_idx],
        "DEdeg":      df_cat['DEdeg'].values[cat_idx],
        "sep_arcsec": d2d[mask].arcsec,
        "F160W":      df_cat["F160W"].values[cat_idx],
        "ReF814W":    df_cat["ReF814W"].values[cat_idx],  # assumed arcsec
    })
    print(matches.head(), "\n# matches:", len(matches))

    results_rows = []

    for j in range(14):#range(len(potentiel)):
        if j not in set(matches["lens_index"].values):
            continue
        j_cat = matches.loc[matches["lens_index"]==j, "cat_index"].values[0]
        m_F160W = df_cat.iloc[j_cat]["F160W"]
        Re_kpc = float(df_cat.iloc[j_cat]["ReF814W"])  # already in kpc
        rJ = jaffe_scale_radius(Re_kpc)

        # PIEMD parameters (use your fields; keep units in kpc, km/s)
        v_disp = float(findInBlock(potentiel[j], 'v_disp')) * np.sqrt(3.0/2.0)
        r_cut  = float(findInBlock(potentiel[j], 'cut_radius_kpc'))
        r_core = float(findInBlock(potentiel[j], 'core_radius_kpc'))
        zl     = float(findInBlock(potentiel[j], 'z_lens'))
        mtot = np.pi * v_disp**2 / G_KPC * (r_cut-r_core)

        # M* and rJ (Re arcsec -> kpc)
        M_star = stellar_mass_to_light(m_F160W)

        print(f"Lens #{j}: cat #{j_cat}, F160W={m_F160W:.3f}, Re={Re_kpc:.3f} kpc | "
              f"v_disp={v_disp:.2f}, r_core={r_core:.3f} kpc, r_cut={r_cut:.3f} kpc | "
              f"M*={M_star:.2e} Msun, rJ={rJ:.3f} kpc")

        # ----- Fit window & grid -----
        rmin = 3*rJ #max(0.5*r_core, 1e-3*r_cut)
        rmax = 1e3#10*r_cut
        if rmax <= rmin:
            print("  Skipping: invalid fit window.")
            continue
        r = np.logspace(np.log10(rmin), np.log10(rmax), 1000)

        # DM initial guesses (reasonable defaults)
        Rs_guess   = 0.2 * r_cut
        rtr_guess  = max(0.5 * r_cut, 10 * Rs_guess)
        # use mtot as a guide for rho0

        rho0_guess = 1e8

        # ----- Fit SUM in 3D (robust, log-space) -----
        try:
            fit = fit_piemd_3d_with_jaffe_tnfw(
                r=r, r_core=r_core, r_cut=r_cut, sigma_v=v_disp,
                Mstar_phot=M_star, Re_kpc=Re_kpc,
                Rs_guess=Rs_guess, rtr_guess=rtr_guess, rho0_guess=rho0_guess
            )
        except Exception as e:
            print(f"  Fit failed: {e}")
            continue

        # Diagnostics & derived quantities
        rho_crit = cosmo.critical_density(zl).to_value('Msun/kpc3')
        try:
            R200 = Rdelta_tNFW(fit["Rs"], fit["rtr"], fit["rho0"], rho_crit, Delta=200.0)
            c200 = R200 / fit["Rs"]
        except Exception:
            R200, c200 = np.nan, np.nan
        try:
            c200_nfw = c200_from_rs_rhos(fit["Rs"], fit["rho0"], rho_crit)
        except Exception:
            c200_nfw = np.nan

        # Store row
        row = dict(
            lens_index=j, cat_index=j_cat,
            ra_lens=ra[j], dec_lens=dec[j],
            F160W=m_F160W, Re_kpc=Re_kpc,
            M_star=fit["Mstar"], rJ_kpc=fit["rJ"],
            v_disp=v_disp, core_radius_kpc=r_core, cut_radius_kpc=r_cut,
            r_s_tNFW=fit["Rs"], r_t_tNFW=fit["rtr"], rho_s_tNFW=fit["rho0"],
            R200_tNFW_kpc=R200, c200_tNFW=c200, c200_NFW=c200_nfw,
            rmse_log=fit["rmse_log"], r2_log=fit["r2_log"],
            success=fit["success"]
        )
        results_rows.append(row)

        # Plot per galaxy (optional; comment out to speed up)
        rho_piemd = rho_piemd_3d(r, r_core, r_cut, v_disp)
        rho_star  = rho_jaffe(r, fit["Mstar"], fit["rJ"])
        rho_dm    = rho_tnfw(r, fit["Rs"], fit["rtr"], fit["rho0"])
        rho_sum   = rho_star + rho_dm

        fig, ax = plt.subplots(1,1, figsize=(7,5.2))
        ax.loglog(r, rho_piemd, label="PIEMD (total 3D)", lw=2, alpha=0.9)
        ax.loglog(r, rho_star,  label="Jaffe (stars)", ls="--")
        ax.loglog(r, rho_dm,    label="tNFW (DM)", ls="-.")
        ax.loglog(r, rho_sum,   label="Sum (model)", lw=2)
        ax.axvline(fit["Rs"],  ls=":", alpha=0.6)
        ax.axvline(fit["rtr"], ls=":", alpha=0.6)
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel(r"$\rho(r)\ \,[M_\odot\,\mathrm{kpc}^{-3}]$")
        ttl = f"Lens #{j} | rmse_log={fit['rmse_log']:.3f}, R200={R200:.1f} kpc, c200={c200:.2f}"
        ax.set_title(ttl)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend()
        plt.show()

    # -------------------- Save & summary plots --------------------
    if len(results_rows) == 0:
        print("No successful fits.")
        raise SystemExit

    df_results = pd.DataFrame(results_rows)
    df_results.to_csv('lens_galaxy_fits.csv', index=False)
    print(df_results.head(10))

    # C-M relation (using PIEMD total mass proxy: π σ^2/G (r_cut - r_core))
    Mtot_PIEMD = np.pi * (df_results['v_disp']**2) / G_KPC * (df_results['cut_radius_kpc'] - df_results['core_radius_kpc'])
    fig, ax = plt.subplots(1,1)
    ax.scatter(Mtot_PIEMD, df_results['c200_NFW'], color='blue')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$M_{\mathrm{tot,PIEMD}}$ [M$_\odot$]')
    ax.set_ylabel(r'$c_{200}$ (NFW; from $\rho_s/\rho_\mathrm{crit}$)')
    ax.set_title('Concentration–Mass (lens galaxies)')
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.show()

    # c200 vs fit quality
    fig, ax = plt.subplots(1,1)
    ax.scatter(df_results['rmse_log'], df_results['c200_NFW'], color='purple')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('rmse_log (log-space)')
    ax.set_ylabel(r'$c_{200}$')
    ax.set_title('Concentration vs fit diagnostics')
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.show()