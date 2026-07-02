#!/usr/bin/env python3
"""Generate the tutorial notebook for density profile fits."""
import json

cells = []

def md(text):
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(text):
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type": "code", "metadata": {}, "source": src,
                  "outputs": [], "execution_count": None})

# =====================================================================
# Cell 0: Title
# =====================================================================
md("# Galaxy Density Profile Fits \u2014 Data Products Tutorial\n"
   "\n"
   "This notebook explains how to read and use the CSV and HDF5 output files\n"
   "produced by `fitGalaxies_tnfw_jaffe_dual.py`. It covers:\n"
   "\n"
   "1. **Background & physical model**\n"
   "2. **CSV file structure** \u2014 best-fit parameters and derived quantities\n"
   "3. **HDF5 file structure** \u2014 per-sample MCMC fit parameters\n"
   "4. **Reproducing density profiles** \u2014 DM + Jaffe and each component separately\n"
   "5. **Error estimation from MCMC samples** \u2014 posterior envelopes\n"
   "6. **Derived quantities** \u2014 $c_{200}$, $R_{200}$, $M_{200}$\n"
   "\n"
   "---")

# =====================================================================
# Cell 1: Physical model
# =====================================================================
md("## 1. Background & Physical Model\n"
   "\n"
   "### Overview\n"
   "\n"
   "Each galaxy in a galaxy cluster has its total mass distribution modelled by\n"
   "the LensTool code as a **PIEMD** (Pseudo-Isothermal Elliptical Mass\n"
   "Distribution). The fitting code *decomposes* this total PIEMD profile into a\n"
   "**dark-matter (DM) component** plus a **stellar component**, assuming\n"
   "spherical symmetry for the 3-D density profile:\n"
   "\n"
   "$$\\rho_{\\mathrm{total}}(r) = \\rho_{\\mathrm{DM}}(r) + \\rho_{\\star}(r)$$\n"
   "\n"
   "### 1.1 PIEMD density profile (target)\n"
   "\n"
   "$$\\rho_{\\mathrm{PIEMD}}(r) = \\frac{\\sigma_0^2}{2\\pi G}"
   "\\frac{r_{\\mathrm{cut}} + r_{\\mathrm{core}}}{r_{\\mathrm{core}}^2 \\, r_{\\mathrm{cut}}}"
   "\\frac{1}{(1 + r^2/r_{\\mathrm{core}}^2)(1 + r^2/r_{\\mathrm{cut}}^2)}$$\n"
   "\n"
   "where $\\sigma_0$ is the central velocity dispersion (km/s), $r_{\\mathrm{core}}$ is the\n"
   "core radius (kpc), $r_{\\mathrm{cut}}$ is the cut radius (kpc), and\n"
   "$G = 4.302 \\times 10^{-6}$ kpc (km/s)$^2$ / M$_\\odot$.\n"
   "\n"
   "### 1.2 Jaffe stellar profile\n"
   "\n"
   "$$\\rho_{\\mathrm{Jaffe}}(r) = \\frac{M_\\star \\, r_J}{4\\pi \\, r^2 \\, (r + r_J)^2}$$\n"
   "\n"
   "where $M_\\star$ is the total stellar mass (M$_\\odot$) and $r_J = R_e$ is the\n"
   "Jaffe scale radius, set equal to the effective (half-light) radius $R_e$ (F160W).\n"
   "\n"
   "### 1.3 Truncated NFW (tNFW) DM profile\n"
   "\n"
   "$$\\rho_{\\mathrm{tNFW}}(r) = \\frac{\\rho_0}{(r/r_s)(1 + r/r_s)^2}"
   "\\cdot \\frac{\\tau^2}{(r/r_s)^2 + \\tau^2}$$\n"
   "\n"
   "where $r_s$ is the scale radius, $r_t$ is the truncation radius,\n"
   "$\\tau \\equiv r_t / r_s$, and $\\rho_0$ is the NFW characteristic density.\n"
   "\n"
   "### 1.4 Alternative DM profiles\n"
   "\n"
   "**Einasto:**\n"
   "$$\\rho_{\\mathrm{Ein}}(r) = \\rho_s \\exp\\!\\left[-2n\\left(\\left(\\frac{r}{r_s}\\right)^{1/n} - 1\\right)\\right]$$\n"
   "\n"
   "**Generalized NFW (gNFW):**\n"
   "$$\\rho_{\\mathrm{gNFW}}(r) = \\frac{\\rho_s}{(r/r_s)^\\gamma (1 + r/r_s)^{3-\\gamma}}$$\n"
   "\n"
   "### 1.5 Two fitting modes\n"
   "\n"
   "| Mode | Stellar mass | Fit parameters |\n"
   "|------|-------------|----------------|\n"
   "| **Fixed-Jaffe** | $M_\\star$ fixed from F160W photometry | DM params only |\n"
   "| **Free-Jaffe** | $M_\\star$ is a free parameter | DM params + $M_\\star$ |\n"
   "\n"
   "### 1.6 Derived quantities\n"
   "\n"
   "- $c_{200} = R_{200} / r_s$ \u2014 concentration parameter\n"
   "- $R_{200}$ \u2014 radius enclosing mean density $200\\,\\rho_{\\mathrm{crit}}(z)$\n"
   "- $M_{200} = M(<R_{200})$ \u2014 mass within $R_{200}$")

# =====================================================================
# Cell 2: Setup
# =====================================================================
md("---\n## 2. Setup & Imports")

code("import numpy as np\n"
     "import pandas as pd\n"
     "import h5py\n"
     "import matplotlib.pyplot as plt\n"
     "\n"
     "%matplotlib inline\n"
     "plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})")

md("### File paths\n\nSet these to point to your actual files.")

code('CSV_FILE = "lens_galaxy_fits_tnfw_jaffe_dual.csv"   # <-- change to your CSV path\n'
     'H5_FILE  = "lens_galaxy_fits_tnfw_jaffe_dual.h5"    # <-- change to your HDF5 path')

# =====================================================================
# Cell 3: CSV
# =====================================================================
md("---\n## 3. Reading the CSV File\n\n"
   "The CSV contains one row per galaxy with the best-fit parameters from both\n"
   "fitting modes, plus derived quantities and MCMC percentile summaries.")

code('df = pd.read_csv(CSV_FILE)\n'
     'print(f"Number of galaxies: {len(df)}")\n'
     'print(f"DM profile type: {df[\'dm_profile\'].iloc[0]}")\n'
     'print(f"\\nAll columns ({len(df.columns)}):")\n'
     'for i, col in enumerate(df.columns):\n'
     '    print(f"  {i:3d}. {col}")')

md("### 3.1 CSV Column Reference\n"
   "\n"
   "#### Identification & position\n"
   "| Column | Description |\n"
   "|--------|-------------|\n"
   "| `lens_index` | Galaxy index in the LensTool model |\n"
   "| `cat_index` | Matched index in the photometric catalog |\n"
   "| `ra_lens`, `dec_lens` | Sky position (degrees) |\n"
   "| `z_lens` | Lens redshift |\n"
   "\n"
   "#### Observed properties\n"
   "| Column | Description |\n"
   "|--------|-------------|\n"
   "| `F160W` | Apparent magnitude in the F160W filter |\n"
   "| `ReF160W_kpc` | Effective (half-light) radius $R_e$ in kpc |\n"
   "| `Mstar_from_F160W_Msun` | Stellar mass from photometry (M$_\\odot$) |\n"
   "\n"
   "#### PIEMD input parameters\n"
   "| Column | Description |\n"
   "|--------|-------------|\n"
   "| `v_disp_sigma0_km_s` | Central velocity dispersion $\\sigma_0$ (km/s) |\n"
   "| `core_radius_kpc` | PIEMD core radius (kpc) |\n"
   "| `cut_radius_kpc` | PIEMD cut radius (kpc) |\n"
   "\n"
   "#### Fixed-Jaffe fit results (tNFW example)\n"
   "| Column | Description |\n"
   "|--------|-------------|\n"
   "| `rs_fix_kpc` | Best-fit scale radius $r_s$ (kpc) |\n"
   "| `rt_fix_kpc` | Best-fit truncation radius $r_t$ (kpc) |\n"
   "| `rho0_fix_msun_kpc3` | Best-fit $\\rho_0$ (M$_\\odot$/kpc$^3$) |\n"
   "| `fit_fix_success` | Optimizer converged (True/False) |\n"
   "| `fit_fix_rmse_log` | RMS residual in log-density |\n"
   "| `fit_fix_jaffe_clamped` | Whether Jaffe was clamped to $\\leq$ PIEMD |\n"
   "| `c200_fix` | Concentration $c_{200}$ |\n"
   "| `R200_fix_kpc`, `M200_fix_Msun` | $R_{200}$, $M_{200}$ |\n"
   "\n"
   "#### Free-Jaffe fit results\n"
   "Same columns with `_free` suffix, plus `Mstar_free_Msun`.\n"
   "\n"
   "#### Stellar mass diagnostics\n"
   "| Column | Description |\n"
   "|--------|-------------|\n"
   "| `Mstar_max_constraint_Msun` | Maximum allowed $M_\\star$ |\n"
   "| `Mstar_ratio_free_over_phot` | $M_\\star^{\\mathrm{free}}/M_\\star^{\\mathrm{phot}}$ |\n"
   "\n"
   "#### MCMC percentile columns\n"
   "For each parameter `X`: `X_p16`, `X_p50`, `X_p84` (16th, 50th, 84th percentiles).\n"
   "\n"
   "> **Einasto columns:** `rs_ein_*`, `rhos_ein_*`, `n_ein_*`\n"
   "> **gNFW columns:** `rs_gnfw_*`, `rhos_gnfw_*`, `gamma_gnfw_*`")

# =====================================================================
# Cell 4: HDF5
# =====================================================================
md("---\n## 4. Reading the HDF5 File\n\n"
   "The HDF5 file stores the full MCMC sample-by-sample fit results as 3-D\n"
   "arrays, enabling posterior analysis beyond simple percentiles.")

code('with h5py.File(H5_FILE, "r") as f:\n'
     '    print("HDF5 datasets:")\n'
     '    for key in f.keys():\n'
     '        ds = f[key]\n'
     '        print(f"  {key:25s}  shape={ds.shape}  dtype={ds.dtype}")\n'
     '    print("\\nAttributes (metadata):")\n'
     '    for key, val in f.attrs.items():\n'
     '        print(f"  {key}: {val}")')

md("### 4.1 HDF5 Dataset Reference\n"
   "\n"
   "| Dataset | Shape | Description |\n"
   "|---------|-------|-------------|\n"
   "| `lens_index` | `(N_gal,)` | Galaxy index matching CSV |\n"
   "| `fix_params` | `(N_gal, N_samp, 3)` | Fixed-Jaffe DM params per sample |\n"
   "| `free_params` | `(N_gal, N_samp, 4)` | Free-Jaffe DM+M* params per sample |\n"
   "| `piemd_params` | `(N_gal, N_samp, 2)` | PIEMD params per sample |\n"
   "\n"
   "**tNFW parameter order:** `[rs, rt, rho0]` (fix), `[rs, rt, rho0, M*]` (free)\n\n"
   "**Einasto:** `[rs, rho_s, n]` (fix), `[rs, rho_s, n, M*]` (free)\n\n"
   "**gNFW:** `[rs, rho_s, gamma]` (fix), `[rs, rho_s, gamma, M*]` (free)\n\n"
   "NaN entries indicate samples where the fit did not converge.")

code('with h5py.File(H5_FILE, "r") as f:\n'
     '    h5_lens_idx  = f["lens_index"][:]\n'
     '    fix_params   = f["fix_params"][:]      # (N_gal, N_samp, 3)\n'
     '    free_params  = f["free_params"][:]     # (N_gal, N_samp, 4)\n'
     '    piemd_params = f["piemd_params"][:]    # (N_gal, N_samp, 2)\n'
     '    dm_profile   = f.attrs.get("dm_profile", "tNFW")\n'
     '    fix_names    = [s.decode() for s in f["fix_param_names"][:]]\n'
     '    free_names   = [s.decode() for s in f["free_param_names"][:]]\n'
     '\n'
     'N_gal, N_samp, _ = fix_params.shape\n'
     'print(f"DM profile: {dm_profile}")\n'
     'print(f"N_gal = {N_gal}, N_samp = {N_samp}")\n'
     'print(f"Fix param names:  {fix_names}")\n'
     'print(f"Free param names: {free_names}")')

# =====================================================================
# Cell 5: Density functions
# =====================================================================
md("---\n## 5. Density Profile Functions\n\n"
   "Self-contained Python implementations of all density profiles.")

code('''G_KPC = 4.302e-6   # kpc (km/s)^2 / Msun


def piemd_rho(r, r_core, r_cut, sigma_v):
    """PIEMD 3-D density (Msun/kpc^3)."""
    r = np.asarray(r, dtype=float)
    return (
        (sigma_v**2) / (2.0 * np.pi * G_KPC)
        * (r_cut + r_core) / (r_core**2 * r_cut)
        / (1.0 + r**2 / r_core**2)
        / (1.0 + r**2 / r_cut**2)
    )


def jaffe_rho(r, mstar, re):
    """Jaffe stellar density (Msun/kpc^3). r_J = Re."""
    r = np.asarray(r, dtype=float)
    rj = float(re)
    rr = np.maximum(r, np.finfo(float).eps)
    return (float(mstar) * rj) / (4.0 * np.pi * rr**2 * (rr + rj)**2)


def trunc_nfw_rho(r, rs, rt, rho0):
    """Truncated NFW density (Msun/kpc^3)."""
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    tau = rt / rs
    trunc = tau**2 / (x**2 + tau**2)
    return trunc * (rho0 / (x * (1.0 + x)**2))


def einasto_rho(r, rs, rho_s, n):
    """Einasto density profile (Msun/kpc^3)."""
    r = np.asarray(r, dtype=float)
    s = np.maximum(r / rs, np.finfo(float).eps)
    alpha = 1.0 / n
    dn = 2.0 * n
    return rho_s * np.exp(-dn * (s**alpha - 1.0))


def gnfw_rho(r, rs, rho_s, gamma):
    """Generalized NFW density profile (Msun/kpc^3). gamma=1 is standard NFW."""
    r = np.asarray(r, dtype=float)
    x = np.maximum(r / rs, np.finfo(float).eps)
    return rho_s / (x**gamma * (1.0 + x)**(3.0 - gamma))''')

# =====================================================================
# Cell 6: Single galaxy profile
# =====================================================================
md("---\n## 6. Reproducing Density Profiles for a Single Galaxy\n\n"
   "### 6.1 Select a galaxy and extract its parameters")

code('# Pick a galaxy (first with both fits converged)\n'
     'mask = df["fit_fix_success"].astype(bool) & df["fit_free_success"].astype(bool)\n'
     'gal = df.loc[mask].iloc[0]\n'
     '\n'
     'print(f"Lens index: {int(gal[\'lens_index\'])}")\n'
     'print(f"z_lens = {gal[\'z_lens\']:.4f}")\n'
     'print(f"F160W  = {gal[\'F160W\']:.2f}")\n'
     'print(f"Re     = {gal[\'ReF160W_kpc\']:.3f} kpc")\n'
     'print(f"sigma0 = {gal[\'v_disp_sigma0_km_s\']:.1f} km/s")\n'
     'print(f"r_core = {gal[\'core_radius_kpc\']:.4f} kpc")\n'
     'print(f"r_cut  = {gal[\'cut_radius_kpc\']:.2f} kpc")\n'
     'print(f"M* (phot) = {gal[\'Mstar_from_F160W_Msun\']:.3e} Msun")')

md("### 6.2 Compute and plot the density profiles\n\n"
   "Handles all three DM profile types automatically via the `dm_profile` column.")

code('''r = np.logspace(-2, 3, 500)  # radii from 0.01 to 1000 kpc

# --- PIEMD total ---
rho_piemd = piemd_rho(r, gal["core_radius_kpc"], gal["cut_radius_kpc"],
                      gal["v_disp_sigma0_km_s"])

# --- Jaffe stellar profiles ---
re = gal["ReF160W_kpc"]
mstar_phot = gal["Mstar_from_F160W_Msun"]
mstar_free = gal["Mstar_free_Msun"]

rho_jaffe_phot = jaffe_rho(r, mstar_phot, re)
rho_jaffe_free = jaffe_rho(r, mstar_free, re)
rho_jaffe_clamped = np.minimum(rho_jaffe_phot, rho_piemd)

# --- DM profiles (auto-detect type) ---
profile = str(gal["dm_profile"]).strip().lower()

if profile == "tnfw":
    rho_dm_fix  = trunc_nfw_rho(r, gal["rs_fix_kpc"],  gal["rt_fix_kpc"],  gal["rho0_fix_msun_kpc3"])
    rho_dm_free = trunc_nfw_rho(r, gal["rs_free_kpc"], gal["rt_free_kpc"], gal["rho0_free_msun_kpc3"])
    dm_label = "tNFW"
elif profile == "einasto":
    rho_dm_fix  = einasto_rho(r, gal["rs_ein_fix_kpc"],  gal["rhos_ein_fix_msun_kpc3"],  gal["n_ein_fix"])
    rho_dm_free = einasto_rho(r, gal["rs_ein_free_kpc"], gal["rhos_ein_free_msun_kpc3"], gal["n_ein_free"])
    dm_label = "Einasto"
elif profile == "gnfw":
    rho_dm_fix  = gnfw_rho(r, gal["rs_gnfw_fix_kpc"],  gal["rhos_gnfw_fix_msun_kpc3"],  gal["gamma_gnfw_fix"])
    rho_dm_free = gnfw_rho(r, gal["rs_gnfw_free_kpc"], gal["rhos_gnfw_free_msun_kpc3"], gal["gamma_gnfw_free"])
    dm_label = "gNFW"

rho_total_fix  = rho_dm_fix  + rho_jaffe_clamped
rho_total_free = rho_dm_free + rho_jaffe_free''')

code('''fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

ax1.loglog(r, rho_piemd, "k-", lw=2.5, label="PIEMD (total)", zorder=5)
ax1.loglog(r, rho_jaffe_clamped, "C0:", lw=1.2, label=f"Jaffe (phot, M*={mstar_phot:.2e})")
ax1.loglog(r, rho_jaffe_free, "C3:", lw=1.2, label=f"Jaffe (free, M*={mstar_free:.2e})")
ax1.loglog(r, rho_dm_fix, "C1--", lw=1.0, label=f"{dm_label} (fixed)")
ax1.loglog(r, rho_dm_free, "C2--", lw=1.0, label=f"{dm_label} (free)")
ax1.loglog(r, rho_total_fix, "C1-", lw=1.8, label=f"{dm_label}+Jaffe (fixed)")
ax1.loglog(r, rho_total_free, "C2-", lw=1.8, label=f"{dm_label}+Jaffe (free)")
ax1.axvline(re, color="gray", ls=":", lw=0.8, alpha=0.7, label=f"$R_e$ = {re:.2f} kpc")
ax1.set_ylabel(r"$\\rho(r)$ [M$_\\odot$ / kpc$^3$]")
ax1.set_title(f"Lens {int(gal['lens_index'])} (z={gal['z_lens']:.3f}, F160W={gal['F160W']:.2f})")
ax1.legend(fontsize=7, ncol=2, loc="upper right")

pos = rho_piemd > 0
ax2.semilogx(r[pos], (rho_total_fix[pos]-rho_piemd[pos])/rho_piemd[pos], "C1-", lw=1.5, label="fixed")
ax2.semilogx(r[pos], (rho_total_free[pos]-rho_piemd[pos])/rho_piemd[pos], "C2-", lw=1.5, label="free")
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xlabel("r [kpc]"); ax2.set_ylabel("Fractional residual"); ax2.set_ylim(-1, 1)
ax2.legend(fontsize=9)
plt.tight_layout(); plt.show()''')

# =====================================================================
# Cell 7: MCMC error estimation
# =====================================================================
md("---\n## 7. Error Estimation from MCMC Samples\n\n"
   "### 7.1 Find the galaxy in the HDF5 file")

code('lens_id = int(gal["lens_index"])\n'
     'igal = int(np.where(h5_lens_idx == lens_id)[0][0])\n'
     'print(f"Galaxy with lens_index={lens_id} is at HDF5 row {igal}")')

md("### 7.2 Extract MCMC samples and filter valid ones")

code('free_samp = free_params[igal]   # (N_samp, 4)\n'
     'fix_samp  = fix_params[igal]    # (N_samp, 3)\n'
     '\n'
     'valid_free = np.all(np.isfinite(free_samp), axis=1)\n'
     'valid_fix  = np.all(np.isfinite(fix_samp), axis=1)\n'
     '\n'
     'print(f"Valid free-fit samples: {valid_free.sum()} / {N_samp}")\n'
     'print(f"Valid fixed-fit samples: {valid_fix.sum()} / {N_samp}")')

md("### 7.3 Compute density profile envelopes")

code('''def compute_profiles_for_samples(r, samples, profile_type, re_kpc, mstar_phot, mode="free"):
    """Compute DM, Jaffe, and total density for each MCMC sample.

    Supports legacy free-Jaffe format (no explicit rJ) and new format with free rJ.
    """
    N, Nr = len(samples), len(r)
    rho_dm    = np.full((N, Nr), np.nan, dtype=float)
    rho_jaffe = np.full((N, Nr), np.nan, dtype=float)
    profile_type = str(profile_type).strip().lower()
    for i in range(N):
        p = samples[i]
        if profile_type == "tnfw":
            rho_dm[i] = trunc_nfw_rho(r, p[0], p[1], p[2])
        elif profile_type == "nfw":
            x = np.maximum(r / p[0], np.finfo(float).eps)
            rho_dm[i] = p[1] / (x * (1.0 + x)**2)
        elif profile_type == "bmo":
            x = np.maximum(r / p[0], np.finfo(float).eps)
            trunc = np.sqrt(p[1]**2 / (r**2 + p[1]**2))
            rho_dm[i] = trunc * (p[2] / (x * (1.0 + x)**2))
        elif profile_type == "einasto":
            rho_dm[i] = einasto_rho(r, p[0], p[1], p[2])
        elif profile_type == "gnfw":
            rho_dm[i] = gnfw_rho(r, p[0], p[1], p[2])
        else:
            raise ValueError(f"Unsupported dm profile '{profile_type}' in section 7.3")

        has_rj = (profile_type == "nfw" and len(p) >= 4) or (profile_type != "nfw" and len(p) >= 5)
        if mode == "free":
            ms_idx = -2 if has_rj else -1
            ms = p[ms_idx]
            rj = p[-1] if has_rj else re_kpc
        else:
            ms = mstar_phot
            rj = re_kpc

        rr = np.maximum(r, np.finfo(float).eps)
        rho_jaffe[i] = (float(ms) * float(rj)) / (4.0 * np.pi * rr**2 * (rr + float(rj))**2)
    return rho_dm, rho_jaffe, rho_dm + rho_jaffe''')

code('''r_plot = np.logspace(-2, 3, 300)

dm_free, jaffe_free_arr, total_free = compute_profiles_for_samples(
    r_plot, free_samp[valid_free], profile, re, mstar_phot, mode="free")
dm_fix_arr, jaffe_fix_arr, total_fix_arr = compute_profiles_for_samples(
    r_plot, fix_samp[valid_fix], profile, re, mstar_phot, mode="fix")

def pct(arr, q=[16, 50, 84]):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected 2-D array of shape (N_samples, N_radii)")
    if arr.shape[0] == 0:
        return np.full((len(q), arr.shape[1]), np.nan, dtype=float)
    return np.nanpercentile(arr, q, axis=0)

free_total_pct = pct(total_free)
free_dm_pct    = pct(dm_free)
free_jaffe_pct = pct(jaffe_free_arr)
fix_total_pct  = pct(total_fix_arr)

print(f"Samples used in envelopes: free={total_free.shape[0]}, fix={total_fix_arr.shape[0]}")''')

code('''fig, ax = plt.subplots(figsize=(9, 6))

ax.loglog(r_plot, piemd_rho(r_plot, gal["core_radius_kpc"],
          gal["cut_radius_kpc"], gal["v_disp_sigma0_km_s"]),
          "k-", lw=2.5, label="PIEMD (best fit)", zorder=5)

ax.fill_between(r_plot, free_total_pct[0], free_total_pct[2],
                alpha=0.25, color="C2", label=f"{dm_label}+Jaffe free (68% CI)")
ax.loglog(r_plot, free_total_pct[1], "C2-", lw=1.5, label=f"{dm_label}+Jaffe free (median)")

ax.fill_between(r_plot, fix_total_pct[0], fix_total_pct[2],
                alpha=0.2, color="C1", label=f"{dm_label}+Jaffe fixed (68% CI)")
ax.loglog(r_plot, fix_total_pct[1], "C1--", lw=1.5, label=f"{dm_label}+Jaffe fixed (median)")

ax.fill_between(r_plot, free_dm_pct[0], free_dm_pct[2], alpha=0.15, color="C4")
ax.loglog(r_plot, free_dm_pct[1], "C4:", lw=1.0, label=f"{dm_label} only (free)")

ax.fill_between(r_plot, free_jaffe_pct[0], free_jaffe_pct[2], alpha=0.15, color="C3")
ax.loglog(r_plot, free_jaffe_pct[1], "C3:", lw=1.0, label="Jaffe (free)")

ax.axvline(re, color="gray", ls=":", lw=0.8)
ax.set_xlabel("r [kpc]"); ax.set_ylabel(r"$\\rho(r)$ [M$_\\odot$ / kpc$^3$]")
ax.set_title(f"Lens {lens_id} — MCMC posterior envelopes ({valid_free.sum()} samples)")
ax.set_xlim(r_plot[0], r_plot[-1])
ax.legend(fontsize=7, ncol=2, loc="upper right")
plt.tight_layout(); plt.show()''')

md("### 7.4 Parameter posterior distributions")

code('''mstar_samples = free_samp[valid_free, 3]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.log10(mstar_samples), bins=30, density=True, alpha=0.7,
        color="steelblue", edgecolor="white", lw=0.3)
p16, p50, p84 = np.percentile(np.log10(mstar_samples), [16, 50, 84])
ax.axvline(p50, color="red", lw=1.5, label=f"median = {10**p50:.2e}")
ax.axvspan(p16, p84, alpha=0.15, color="red", label=f"68% CI")
ax.axvline(np.log10(mstar_free), color="k", ls="--", lw=1.5, label=f"best fit = {mstar_free:.2e}")
ax.set_xlabel(r"$\\log_{10}(M_\\star / M_\\odot)$"); ax.set_ylabel("Density")
ax.set_title(f"Lens {lens_id} — Stellar mass posterior")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()''')

# =====================================================================
# Cell 8: Derived quantities
# =====================================================================
md("---\n## 8. Derived Quantities: $c_{200}$, $R_{200}$, $M_{200}$\n\n"
   "### 8.1 From CSV")

code('''if profile == "tnfw":
    for mode in ["fix", "free"]:
        print(f"--- {mode} fit ---")
        print(f"  c200 = {gal.get(f'c200_{mode}', 'N/A'):.1f}")
        if mode == "fix":
            print(f"  R200 = {gal.get('R200_fix_kpc', 'N/A'):.1f} kpc")
            print(f"  M200 = {gal.get('M200_fix_Msun', 'N/A'):.3e} Msun")
        else:
            print(f"  R200 = {gal.get('R200_free_kpc', 'N/A'):.1f} kpc")
            print(f"  M200 = {gal.get('M200_free_Msun', 'N/A'):.3e} Msun")
elif profile == "einasto":
    for mode in ["fix", "free"]:
        print(f"--- {mode} fit ---")
        print(f"  c200 = {gal.get(f'c200_{mode}', 'N/A')}")
        print(f"  R200 = {gal.get(f'R200_ein_{mode}_kpc', 'N/A')} kpc")
        print(f"  M200 = {gal.get(f'M200_ein_{mode}_Msun', 'N/A')} Msun")
elif profile == "gnfw":
    for mode in ["fix", "free"]:
        print(f"--- {mode} fit ---")
        print(f"  c200 = {gal.get(f'c200_{mode}', 'N/A')}")
        print(f"  R200 = {gal.get(f'R200_gnfw_{mode}_kpc', 'N/A')} kpc")
        print(f"  M200 = {gal.get(f'M200_gnfw_{mode}_Msun', 'N/A')} Msun")''')

md("### 8.2 Computing $c_{200}$, $M_{200}$ from MCMC samples")

code('''from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import brentq

cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
z_lens = gal["z_lens"]
rho_crit = cosmo.critical_density(z_lens).to_value("Msun/kpc3")
print(f"rho_crit(z={z_lens:.3f}) = {rho_crit:.4e} Msun/kpc^3")''')

code('''def c200_from_tnfw(rs, rho0, rho_crit):
    """Solve (200/3)*c^3/[ln(1+c)-c/(1+c)] = rho0/rho_crit."""
    S = rho0 / rho_crit
    try:
        return brentq(lambda c: (200/3)*c**3/(np.log1p(c)-c/(1+c)) - S, 1e-6, 1e5)
    except:
        return np.nan

def M_tNFW_enclosed(r, rs, rt, rho0):
    """Enclosed mass of the truncated NFW (closed-form)."""
    x, tau = r/rs, rt/rs
    tau2 = tau**2; denom = (tau2+1)**2
    A = tau2*(tau2-1)/denom; B = -tau2/(tau2+1); D = 2*tau2**2/denom
    return 4*np.pi*rho0*rs**3*(A*(np.log1p(x)-0.5*np.log1p((x/tau)**2)) + B*x/(1+x) + (D/tau)*np.arctan(x/tau))

def derived_from_sample(sample, profile_type, rho_crit):
    """Compute c200, R200, M200 from one MCMC sample (first 3 params = DM)."""
    result = {"c200": np.nan, "R200_kpc": np.nan, "M200_Msun": np.nan}
    rs = sample[0]
    if profile_type == "tnfw":
        rt, rho0 = sample[1], sample[2]
        c200 = c200_from_tnfw(rs, rho0, rho_crit)
        if np.isfinite(c200):
            R200 = c200 * rs
            result.update(c200=c200, R200_kpc=R200, M200_Msun=float(M_tNFW_enclosed(R200, rs, rt, rho0)))
    else:
        rho_s, sp = sample[1], sample[2]
        rg = np.linspace(1e-4, 5000, 10000)
        rhog = einasto_rho(rg, rs, rho_s, sp) if profile_type == "einasto" else gnfw_rho(rg, rs, rho_s, sp)
        integ = 4*np.pi*rg**2*rhog
        Mg = np.zeros_like(rg); Mg[1:] = np.cumsum(0.5*(integ[1:]+integ[:-1])*np.diff(rg))
        mean_rho = Mg / ((4/3)*np.pi*rg**3)
        above = mean_rho >= 200*rho_crit
        if np.any(above):
            idx = np.where(above)[0][-1]
            if idx < len(rg)-1:
                r1, r2 = rg[idx], rg[idx+1]
                d1, d2 = mean_rho[idx]-200*rho_crit, mean_rho[idx+1]-200*rho_crit
                R200 = r1 - d1*(r2-r1)/(d2-d1)
                result.update(c200=R200/rs, R200_kpc=float(R200), M200_Msun=float(np.interp(R200, rg, Mg)))
    return result''')

code('''valid_samples = free_samp[valid_free]
c200_arr = np.array([derived_from_sample(s, profile, rho_crit)["c200"] for s in valid_samples])
m200_arr = np.array([derived_from_sample(s, profile, rho_crit)["M200_Msun"] for s in valid_samples])

ok = np.isfinite(c200_arr) & np.isfinite(m200_arr)
print(f"Valid c200/M200 for {ok.sum()}/{len(valid_samples)} samples")
if ok.sum() > 0:
    p16, p50, p84 = np.percentile(c200_arr[ok], [16, 50, 84])
    print(f"c200: median={p50:.1f}, 68% CI=[{p16:.1f}, {p84:.1f}]")
    p16, p50, p84 = np.percentile(np.log10(m200_arr[ok]), [16, 50, 84])
    print(f"log10(M200/Msun): median={p50:.2f}, 68% CI=[{p16:.2f}, {p84:.2f}]")''')

# =====================================================================
# Cell 9: Batch processing
# =====================================================================
md("---\n## 9. Batch Processing: Looping Over All Galaxies")

code('''r_eval = np.logspace(-1, 2.5, 200)
converged = df["fit_free_success"].astype(bool)
print(f"Processing {converged.sum()} converged galaxies...\\n")

for _, row in df.loc[converged].head(5).iterrows():
    lid = int(row["lens_index"])
    prof = str(row["dm_profile"]).strip().lower()
    re_i = row["ReF160W_kpc"]
    rho_p = piemd_rho(r_eval, row["core_radius_kpc"], row["cut_radius_kpc"], row["v_disp_sigma0_km_s"])
    if prof == "tnfw":
        rho_d = trunc_nfw_rho(r_eval, row["rs_free_kpc"], row["rt_free_kpc"], row["rho0_free_msun_kpc3"])
    elif prof == "einasto":
        rho_d = einasto_rho(r_eval, row["rs_ein_free_kpc"], row["rhos_ein_free_msun_kpc3"], row["n_ein_free"])
    elif prof == "gnfw":
        rho_d = gnfw_rho(r_eval, row["rs_gnfw_free_kpc"], row["rhos_gnfw_free_msun_kpc3"], row["gamma_gnfw_free"])
    rho_j = jaffe_rho(r_eval, row["Mstar_free_Msun"], re_i)
    m = rho_p > 0
    med_res = np.median(np.abs((rho_d+rho_j)[m] - rho_p[m]) / rho_p[m])
    print(f"Lens {lid:4d} | z={row['z_lens']:.3f} | F160W={row['F160W']:5.2f} | median |drho/rho| = {med_res:.4f}")''')

# =====================================================================
# Cell 10: Formula summary
# =====================================================================
md("---\n## 10. Formula Summary\n"
   "\n"
   "| Profile | Formula | Parameters |\n"
   "|---------|---------|------------|\n"
   "| PIEMD | $\\rho = \\frac{\\sigma_0^2}{2\\pi G}\\frac{r_\\mathrm{cut}+r_\\mathrm{core}}{r_\\mathrm{core}^2 r_\\mathrm{cut}}\\frac{1}{(1+r^2/r_\\mathrm{core}^2)(1+r^2/r_\\mathrm{cut}^2)}$ | $\\sigma_0$, $r_\\mathrm{core}$, $r_\\mathrm{cut}$ |\n"
   "| Jaffe | $\\rho = \\frac{M_\\star r_J}{4\\pi r^2(r+r_J)^2}$ | $M_\\star$, $r_J = R_e$ |\n"
   "| tNFW | $\\rho = \\frac{\\rho_0}{(r/r_s)(1+r/r_s)^2}\\cdot\\frac{\\tau^2}{(r/r_s)^2+\\tau^2}$ | $r_s$, $r_t$, $\\rho_0$; $\\tau=r_t/r_s$ |\n"
   "| Einasto | $\\rho = \\rho_s\\exp[-2n((r/r_s)^{1/n}-1)]$ | $r_s$, $\\rho_s$, $n$ |\n"
   "| gNFW | $\\rho = \\frac{\\rho_s}{(r/r_s)^\\gamma(1+r/r_s)^{3-\\gamma}}$ | $r_s$, $\\rho_s$, $\\gamma$ |\n"
   "\n"
   "**Derived:**\n"
   "- $c_{200} = R_{200}/r_s$\n"
   "- $R_{200}$: where $\\bar{\\rho}(<R_{200}) = 200\\,\\rho_\\mathrm{crit}(z)$\n"
   "- $M_{200} = M(<R_{200})$\n"
   "- tNFW $c_{200}$: $(200/3)\\,c^3/[\\ln(1+c) - c/(1+c)] = \\rho_0/\\rho_\\mathrm{crit}$\n"
   "\n"
   "**Stellar mass from photometry:** $\\log_{10}(M_\\star/M_\\odot) = 18.541 - 0.416\\,m_\\mathrm{F160W}$\n"
   "\n"
   "**Constants:** $G = 4.302 \\times 10^{-6}$ kpc (km/s)$^2$ / M$_\\odot$\n"
   "\n"
   "---\n"
   "*Tutorial for pyLensLib `fitGalaxies_tnfw_jaffe_dual.py` outputs.*")

# =====================================================================
# Write the notebook
# =====================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

outpath = "/Users/maxmen3/projects/pyLensLib/Test/tutorial_density_profile_fits.ipynb"
with open(outpath, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook written with {len(cells)} cells to {outpath}")
