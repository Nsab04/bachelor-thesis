import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

def read_galaxy_fits(csv_file):
    """
    Read the output CSV file from fitGalaxies_mcmc_h5.py containing galaxy DM-fit results.
    
    Returns a pandas DataFrame with columns:
      - Lens geometry: lens_index, cat_index, ra_lens, dec_lens
      - Photometry: F160W, Re_F160W, M_star, r_J
      - PIEMD (total mass): v_disp, core_radius_kpc, cut_radius_kpc, Mtot_PIEMD_Msun, fstar_Re
      - tNFW (dark matter): r_s_tNFW, r_t_tNFW, rho_s_tNFW, c200_tNFW, R200_tNFW_kpc, c200_NFW, M200_tNFW_Msun
      - Fit quality: npts, diag (rmse_log)
      - MCMC percentiles (if available): r_s_tNFW_p16/p50/p84, r_t_tNFW_p16/p50/p84, etc.
    
    Parameters
    ----------
    csv_file : str
        Path to the output CSV file.
    
    Returns
    -------
    df : pd.DataFrame
        Parsed results with appropriate dtypes.
    
    Examples
    --------
    >>> df = read_galaxy_fits('lens_galaxy_fits.csv')
    >>> print(df[['F160W', 'r_s_tNFW', 'c200_NFW']].head())
    >>> # Plot concentration vs stellar fraction
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(df['fstar_Re'], df['c200_NFW'], 'o', alpha=0.6)
    >>> plt.xlabel('$M_*/M_{total}$')
    >>> plt.ylabel('$c_{200}$')
    >>> plt.show()
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Validate expected columns exist
    required_cols = ['lens_index', 'cat_index', 'ra_lens', 'dec_lens', 'F160W', 
                     'r_s_tNFW', 'r_t_tNFW', 'rho_s_tNFW', 'c200_NFW', 'M200_tNFW_Msun', 
                     'diag', 'npts']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert column dtypes for clarity
    int_cols = ['lens_index', 'cat_index', 'npts']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    float_cols = [col for col in df.columns if col not in int_cols + ['lens_index', 'cat_index']]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"✓ Loaded {len(df)} galaxy fits from {csv_file}")
    print(f"  Columns: {list(df.columns)}")
    
    return df

def read_galaxy_fits_h5(h5_file):
    """
    Read the HDF5 output file from fitGalaxies_mcmc_h5.py containing per-sample DM-fit parameters.
    
    Returns a dictionary with datasets and metadata. The main data cube `dm_params` has shape
    (N_gal, N_samp, N_par) where each galaxy has N_samp posterior samples across N_par parameters.
    
    Args:
        h5_file: Path to the output HDF5 file.
    
    Returns:
        dict: Contains the following keys:
            - `dm_params`: (N_gal, N_samp, N_par) array of per-sample DM parameters
            - `param_names`: List of N_par parameter names (e.g., ['r_s_kpc', 'r_t_kpc', ...])
            - `lens_index`: (N_gal,) array of lens indices (from .par file)
            - `cat_index`: (N_gal,) array of catalog indices (from photometric catalog)
            - `ra_lens_deg`: (N_gal,) array of lens RA positions [deg]
            - `dec_lens_deg`: (N_gal,) array of lens Dec positions [deg]
            - `sample_id`: (N_samp,) array of sample indices (0 to N_samp-1)
            - `metadata`: Dict of file-level attributes (bayesfile, parfile, catalog, mref, alpha, beta_cut)
    
    Raises:
        FileNotFoundError: If h5_file does not exist.
        ValueError: If required datasets are missing.
    
    Examples:
        >>> data = read_galaxy_fits_h5('lens_galaxy_fits.h5')
        >>> dm_params = data['dm_params']  # shape (N_gal, N_samp, N_par)
        >>> param_names = data['param_names']
        >>> print(f"Loaded {dm_params.shape[0]} galaxies with {dm_params.shape[1]} samples")
        
        >>> # Extract r_s samples for the first galaxy
        >>> i_rs = param_names.index('r_s_kpc')
        >>> rs_samples = dm_params[0, :, i_rs]
        >>> rs_median = np.nanmedian(rs_samples)
        >>> rs_std = np.nanstd(rs_samples)
        >>> print(f"r_s: {rs_median:.2f} ± {rs_std:.2f} kpc")
        
        >>> # Plot posterior for first 3 galaxies
        >>> import matplotlib.pyplot as plt
        >>> for igal in range(min(3, dm_params.shape[0])):
        ...     rs_valid = dm_params[igal, :, i_rs][np.isfinite(dm_params[igal, :, i_rs])]
        ...     plt.figure()
        ...     plt.hist(rs_valid, bins=30, edgecolor='black', alpha=0.7)
        ...     plt.xlabel('$r_s$ [kpc]')
        ...     plt.title(f'Galaxy {igal}')
        ...     plt.show()
    """
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
    
    result = {}
    
    # Required datasets
    required_datasets = ['dm_params', 'param_names', 'lens_index', 'cat_index',
                        'ra_lens_deg', 'dec_lens_deg', 'sample_id']
    
    with h5py.File(h5_file, 'r') as f:
        # Validate required datasets
        missing = [ds for ds in required_datasets if ds not in f]
        if missing:
            raise ValueError(f"Missing required datasets in HDF5 file: {missing}")
        
        # Load datasets
        result['dm_params'] = f['dm_params'][:]  # shape (N_gal, N_samp, N_par)
        result['lens_index'] = f['lens_index'][:]
        result['cat_index'] = f['cat_index'][:]
        result['ra_lens_deg'] = f['ra_lens_deg'][:]
        result['dec_lens_deg'] = f['dec_lens_deg'][:]
        result['sample_id'] = f['sample_id'][:]
        
        # Decode parameter names (stored as bytes)
        param_names_bytes = f['param_names'][:]
        result['param_names'] = [name.decode() if isinstance(name, bytes) else name 
                                for name in param_names_bytes]
        
        # Extract file-level metadata
        result['metadata'] = {
            'bayesfile': f.attrs.get('bayesfile', None),
            'parfile': f.attrs.get('parfile', None),
            'catalog': f.attrs.get('catalog', None),
            'mref_F160W': f.attrs.get('mref_F160W', np.nan),
            'alpha': f.attrs.get('alpha', np.nan),
            'beta_cut': f.attrs.get('beta_cut', np.nan),
        }
        
        # Extract dataset-level descriptions (if available)
        if 'dm_params' in f:
            dm_attrs = dict(f['dm_params'].attrs)
            result['metadata']['dm_params_description'] = dm_attrs.get('description', '')
            result['metadata']['dm_params_units'] = dm_attrs.get('units', '')
        
        # Load PIEMD parameters if available (v_disp, cut_kpc per sample)
        if 'piemd_params' in f:
            result['piemd_params'] = f['piemd_params'][:]  # shape (N_gal, N_samp, 2)
            piemd_names_bytes = f['piemd_param_names'][:]
            result['piemd_param_names'] = [name.decode() if isinstance(name, bytes) else name 
                                           for name in piemd_names_bytes]
            piemd_attrs = dict(f['piemd_params'].attrs)
            result['metadata']['piemd_params_description'] = piemd_attrs.get('description', '')
            result['metadata']['piemd_params_units'] = piemd_attrs.get('units', '')
        else:
            result['piemd_params'] = None
            result['piemd_param_names'] = []
    
    # Print summary
    n_gal, n_samp, n_par = result['dm_params'].shape
    nan_frac = np.isnan(result['dm_params']).mean()
    print(f"✓ Loaded HDF5 file: {h5_file}")
    print(f"  Galaxies: {n_gal}, Samples: {n_samp}, Parameters: {n_par}")
    print(f"  Parameters: {result['param_names']}")
    if result['piemd_params'] is not None:
        print(f"  PIEMD parameters: {result['piemd_param_names']}")
    print(f"  NaN fraction: {nan_frac:.1%}")
    print(f"  Metadata: bayesfile={result['metadata']['bayesfile']}, "
          f"alpha={result['metadata']['alpha']:.2f}, beta_cut={result['metadata']['beta_cut']:.2f}")
    
    return result


def extract_percentiles_from_h5(data, param_name='r_s_kpc', percentiles=(16, 50, 84)):
    """
    Extract percentiles across MCMC samples for a given parameter across all galaxies.
    
    Args:
        data: Dictionary returned by read_galaxy_fits_h5().
        param_name: Name of parameter to extract (e.g., 'r_s_kpc', 'c200').
        percentiles: Tuple of percentile levels to compute (default: 16, 50, 84 for 68% CI).
    
    Returns:
        dict: Contains keys for each percentile (e.g., 'p16', 'p50', 'p84') with arrays
              of shape (N_gal,) containing the percentile values for each galaxy.
    
    Examples:
        >>> data = read_galaxy_fits_h5('lens_galaxy_fits.h5')
        >>> rs_pcts = extract_percentiles_from_h5(data, 'r_s_kpc')
        >>> print(f"r_s: {rs_pcts['p50']} +{rs_pcts['p84'] - rs_pcts['p50']} -{rs_pcts['p50'] - rs_pcts['p16']}")
    """
    if param_name not in data['param_names']:
        raise ValueError(f"Parameter '{param_name}' not found. Available: {data['param_names']}")
    
    i_param = data['param_names'].index(param_name)
    dm_params = data['dm_params']  # shape (N_gal, N_samp, N_par)
    
    result = {}
    for pct in percentiles:
        pct_array = np.full(dm_params.shape[0], np.nan)
        for igal in range(dm_params.shape[0]):
            samples = dm_params[igal, :, i_param]
            valid_samples = samples[np.isfinite(samples)]
            if len(valid_samples) > 0:
                pct_array[igal] = np.percentile(valid_samples, pct)
        result[f'p{pct}'] = pct_array
    
    return result


def summarize_posteriors_to_csv(h5_file, output_csv='posterior_summary.csv'):
    """
    Convert HDF5 posterior samples to a summary CSV with percentiles for all parameters.
    
    Creates a CSV with columns like: lens_index, cat_index, ra_lens_deg, dec_lens_deg,
    r_s_kpc_p16, r_s_kpc_p50, r_s_kpc_p84, r_t_kpc_p16, ..., c200_p16, c200_p50, c200_p84, etc.
    
    Args:
        h5_file: Path to HDF5 file from fitGalaxies_mcmc_h5.py.
        output_csv: Output CSV filename.
    
    Returns:
        pd.DataFrame: Summary table with percentiles.
    
    Examples:
        >>> df = summarize_posteriors_to_csv('lens_galaxy_fits.h5', 'posteriors_summary.csv')
        >>> print(df[['lens_index', 'r_s_kpc_p50', 'c200_p50']])
    """
    data = read_galaxy_fits_h5(h5_file)
    
    # Start with positional information
    summary = pd.DataFrame({
        'lens_index': data['lens_index'],
        'cat_index': data['cat_index'],
        'ra_lens_deg': data['ra_lens_deg'],
        'dec_lens_deg': data['dec_lens_deg'],
    })
    
    # For each parameter, compute and append percentiles
    for param_name in data['param_names']:
        pcts = extract_percentiles_from_h5(data, param_name, percentiles=(16, 50, 84))
        for pct_key, pct_vals in pcts.items():
            summary[f'{param_name}_{pct_key}'] = pct_vals
    
    if output_csv is not None:

        summary.to_csv(output_csv, index=False)
        print(f"✓ Saved posterior summary to: {output_csv}")
        print(f"  Shape: {summary.shape}")
        print(f"  Columns: {list(summary.columns)[:8]}...")  # print first 8

    # make histograms of some parameters as example
    import matplotlib.pyplot as plt
    for param in ['r_s_kpc', 'c200_NFW']:
        p50 = summary[f'{param}_p50']
        plt.figure()
        plt.hist(p50[np.isfinite(p50)], bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel(f'{param} (median)')
        plt.ylabel('Number of galaxies')
        plt.title(f'Posterior median distribution of {param}')
        plt.show()


    
    return summary


def _resolve_param_name_h5(param_names, candidates):
    """Return the first candidate present in HDF5 param_names."""
    for name in candidates:
        if name in param_names:
            return name
    raise ValueError(f"No candidate found among {candidates}. Available: {param_names}")

def plot_concentration_mass_relation(h5_file=None, csv_file=None, figsize=(10, 8)):
    """
    Plot the concentration-mass relation for galaxies with MCMC-derived error bars.
    """
    import matplotlib.pyplot as plt
    
    if h5_file is None and csv_file is None:
        raise ValueError("Must provide either h5_file or csv_file")
    
    # Load data
    if h5_file is not None and os.path.exists(h5_file):
        print(f"Loading posterior samples from HDF5: {h5_file}")
        data = read_galaxy_fits_h5(h5_file)
        
        # Resolve parameter names (supports both tNFW and generic names)
        m200_name = _resolve_param_name_h5(data['param_names'], ['M200_tNFW_Msun', 'M200_Msun'])
        c200_name = _resolve_param_name_h5(data['param_names'], ['c200_tNFW', 'c200'])
        
        # Extract M200 and c200 with percentiles
        m200_pcts = extract_percentiles_from_h5(data, m200_name, percentiles=(16, 50, 84))
        c200_pcts = extract_percentiles_from_h5(data, c200_name, percentiles=(16, 50, 84))
        
        m200_median = m200_pcts['p50']
        m200_err_low = m200_median - m200_pcts['p16']
        m200_err_high = m200_pcts['p84'] - m200_median
        
        c200_median = c200_pcts['p50']
        c200_err_low = c200_median - c200_pcts['p16']
        c200_err_high = c200_pcts['p84'] - c200_median
        
    elif csv_file is not None and os.path.exists(csv_file):
        print(f"Loading summary from CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Resolve column base names
        m200_base = 'M200_tNFW_Msun' if f'M200_tNFW_Msun_p50' in df.columns else 'M200_Msun'
        c200_base = 'c200_tNFW' if f'c200_tNFW_p50' in df.columns else 'c200'
        
        required = [f'{m200_base}_p16', f'{m200_base}_p50', f'{m200_base}_p84',
                    f'{c200_base}_p16', f'{c200_base}_p50', f'{c200_base}_p84']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")
        
        m200_median = df[f'{m200_base}_p50'].values
        m200_err_low = m200_median - df[f'{m200_base}_p16'].values
        m200_err_high = df[f'{m200_base}_p84'].values - m200_median
        
        c200_median = df[f'{c200_base}_p50'].values
        c200_err_low = c200_median - df[f'{c200_base}_p16'].values
        c200_err_high = df[f'{c200_base}_p84'].values - c200_median
    else:
        raise FileNotFoundError(f"Could not find h5_file={h5_file} or csv_file={csv_file}")
    
    # Filter out NaN values
    valid = np.isfinite(m200_median) & np.isfinite(c200_median) & \
            np.isfinite(m200_err_low) & np.isfinite(m200_err_high) & \
            np.isfinite(c200_err_low) & np.isfinite(c200_err_high)
    
    m200_median = m200_median[valid]
    m200_err_low = m200_err_low[valid]
    m200_err_high = m200_err_high[valid]
    c200_median = c200_median[valid]
    c200_err_low = c200_err_low[valid]
    c200_err_high = c200_err_high[valid]
    
    n_valid = np.sum(valid)
    print(f"✓ Loaded {n_valid} galaxies with valid c200 and M200 measurements")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with asymmetric error bars (68% credible interval)
    ax.errorbar(m200_median, c200_median,
               xerr=[m200_err_low, m200_err_high],
               yerr=[c200_err_low, c200_err_high],
               fmt='o', markersize=8, alpha=0.6, elinewidth=1.5, capsize=4,
               color='steelblue', ecolor='steelblue', label='Galaxy fits (68% CI)')
    
    # Formatting
    ax.set_xlabel(r'$M_{200}$ [M$_\odot$]', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$c_{200}$', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')

    # Add axis labels with proper formatting
    ax.tick_params(labelsize=11)

    # Print statistics
    print(f"\nConcentration-Mass Relation Statistics:")
    print(f"  M200 range: [{m200_median.min():.2e}, {m200_median.max():.2e}] M☉")
    print(f"  c200 range: [{c200_median.min():.2f}, {c200_median.max():.2f}]")
    print(f"  Median c200 uncertainty: +{np.median(c200_err_high):.2f} -{np.median(c200_err_low):.2f}")
    print(f"  Median M200 uncertainty: +{np.median(m200_err_high):.2e} -{np.median(m200_err_low):.2e} M☉")
    
    plt.tight_layout()
    
    return fig, ax


def plot_concentration_mass_with_theory(h5_file=None, csv_file=None, 
                                        theory_name='Dutton & Maccio 2014',
                                        figsize=(12, 8)):
    """
    Plot concentration-mass relation with comparison to theoretical predictions.
    
    Overlays empirical c(M) relations (e.g., Dutton & Maccio 2014, Ludlow et al. 2016) 
    for visual comparison with measurements. This follows pyLensLib's philosophy of 
    using plots to validate model consistency against known scaling relations.
    
    Args:
        h5_file: Path to HDF5 file from fitGalaxies_mcmc_h5.py.
        csv_file: Path to CSV summary file.
        theory_name: Which theoretical relation to overlay ('Dutton & Maccio 2014' or 
                    'Ludlow et al. 2016'). Default: 'Dutton & Maccio 2014'.
        figsize: Tuple of (width, height) in inches.
    
    Returns:
        fig, ax: Matplotlib figure and axes objects.
    
    Examples:
        >>> fig, ax = plot_concentration_mass_with_theory(h5_file='lens_galaxy_fits.h5')
        >>> plt.savefig('cM_relation.png', dpi=150, bbox_inches='tight')
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    # Plot the data first
    fig, ax = plot_concentration_mass_relation(h5_file=h5_file, csv_file=csv_file, 
                                               figsize=figsize)
    
    # Define theoretical c(M) relations
    m_theory = np.logspace(10.5, 14.5, 200)  # 10^10.5 to 10^14.5 M☉
    
    if theory_name == 'Dutton & Maccio 2014':
        # Dutton & Maccio 2014: c200 = 10.6 * (M200/10^12)^(-0.1)
        c_theory = 10.6 * (m_theory / 1e12) ** (-0.1)
        label = 'Dutton & Maccio 2014'
        linestyle = '--'
        color = 'red'
    elif theory_name == 'Ludlow et al. 2016':
        # Ludlow et al. 2016: c200 = A * (M200/M_pivot)^B for z~0
        # Example: A=8.5, B=-0.08 at z=0 (from their fitting functions)
        A, B = 8.5, -0.08
        c_theory = A * (m_theory / 1e12) ** B
        label = 'Ludlow et al. 2016'
        linestyle = '-.'
        color = 'darkred'
    else:
        raise ValueError(f"Unknown theory_name: {theory_name}")
    
    ax.plot(m_theory, c_theory, color=color, linestyle=linestyle, linewidth=2.5,
           label=label, zorder=5)
    
    ax.legend(fontsize=11, loc='best')
    ax.set_title(f'Concentration-Mass Relation with {theory_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    
    return fig, ax

def extract_piemd_percentiles(data, param_name='v_disp_km_s', percentiles=(16, 50, 84)):
    """
    Extract percentiles across MCMC samples for PIEMD parameters (v_disp, cut_kpc).
    
    Args:
        data: Dictionary returned by read_galaxy_fits_h5().
        param_name: 'v_disp_km_s' or 'cut_radius_kpc'.
        percentiles: Tuple of percentile levels to compute (default: 16, 50, 84 for 68% CI).
    
    Returns:
        dict: Contains keys for each percentile (e.g., 'p16', 'p50', 'p84') with arrays
              of shape (N_gal,) containing the percentile values for each galaxy.
    
    Raises:
        ValueError: If piemd_params not available or param_name not found.
    
    Examples:
        >>> data = read_galaxy_fits_h5('lens_galaxy_fits.h5')
        >>> vdisp_pcts = extract_piemd_percentiles(data, 'v_disp_km_s')
        >>> print(f"v_disp: {vdisp_pcts['p50'][0]:.1f} +{vdisp_pcts['p84'][0]-vdisp_pcts['p50'][0]:.1f} -{vdisp_pcts['p50'][0]-vdisp_pcts['p16'][0]:.1f} km/s")
    """
    if data['piemd_params'] is None:
        raise ValueError("PIEMD parameters not available in this HDF5 file")
    
    if param_name not in data['piemd_param_names']:
        raise ValueError(f"Parameter '{param_name}' not found. Available: {data['piemd_param_names']}")
    
    i_param = data['piemd_param_names'].index(param_name)
    piemd_params = data['piemd_params']  # shape (N_gal, N_samp, 2)
    
    result = {}
    for pct in percentiles:
        pct_array = np.full(piemd_params.shape[0], np.nan)
        for igal in range(piemd_params.shape[0]):
            samples = piemd_params[igal, :, i_param]
            valid_samples = samples[np.isfinite(samples)]
            if len(valid_samples) > 0:
                pct_array[igal] = np.percentile(valid_samples, pct)
        result[f'p{pct}'] = pct_array
    
    return result

if __name__ == "__main__":
    # Example usage
    h5_file = 'lens_galaxy_fits.h5'
    #output_csv = None
    #summarize_posteriors_to_csv(h5_file, output_csv)

    # Plot c-M relation from HDF5
    fig, ax = plot_concentration_mass_relation(h5_file='lens_galaxy_fits.h5')
    plt.show()

    # Plot with comparison to theory
    fig, ax = plot_concentration_mass_with_theory(h5_file='lens_galaxy_fits.h5',
                                             theory_name='Dutton & Maccio 2014')
    plt.savefig('cM_relation_theory.png', dpi=150, bbox_inches='tight')
    plt.show()

    data = read_galaxy_fits_h5('lens_galaxy_fits.h5')

    # Get v_disp posteriors
    if data['piemd_params'] is not None:
        vdisp_pcts = extract_piemd_percentiles(data, 'v_disp_km_s')
        cut_pcts = extract_piemd_percentiles(data, 'cut_radius_kpc')
    
        # First galaxy
        print(f"Galaxy 0: v_disp = {vdisp_pcts['p50'][0]:.1f} km/s")
        print(f"Galaxy 0: cut_radius = {cut_pcts['p50'][0]:.2f} kpc")

