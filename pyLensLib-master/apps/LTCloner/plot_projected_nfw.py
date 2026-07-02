#!/usr/bin/env python3
"""
Plot the projected NFW profile using _f_proj_nfw function.

This script visualizes:
1. The cumulative mass function f(x) for projected NFW
2. The surface density profile (derivative of f)
3. Comparison for different scale radii

TERMINOLOGY CLARIFICATION:
- f(x) = _f_proj_nfw(x) = CUMULATIVE MASS within radius R (integral from 0 to R)
- Σ(R) = df/dR = SURFACE DENSITY at radius R (mass per unit area)
- dM = Σ(R) × 2πR dR = mass in an ANNULUS between R and R+dR
- For dimensionless units: df/dx × x ∝ dM/d(ln R) = mass per logarithmic bin

When sampling galaxy positions:
- We sample from the CDF of cumulative mass f(x)
- This is equivalent to sampling from the PDF ∝ Σ(R) × 2πR
- The probability of landing in an annulus at R is proportional to the mass in that annulus
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/')
from pyLensLib.lenstool import _f_proj_nfw

def plot_cumulative_mass_function():
    """Plot the dimensionless cumulative mass function f(x)."""

    # Create x values (r/rs)
    x = np.logspace(-2, 2, 1000)  # From 0.01 to 100 in units of rs

    # Compute cumulative mass function
    f_x = _f_proj_nfw(x)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Linear scale
    ax1.plot(x, f_x, 'b-', linewidth=2, label='f(x) = F(R/rs)')
    ax1.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='x = 1 (r = rs)')
    ax1.set_xlabel('x = R / rs', fontsize=12)
    ax1.set_ylabel('f(x) [cumulative mass]', fontsize=12)
    ax1.set_title('Projected NFW Cumulative Mass Function', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 10)

    # Plot 2: Log-log scale
    ax2.loglog(x, f_x, 'b-', linewidth=2, label='f(x)')
    ax2.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='x = 1 (r = rs)')
    ax2.set_xlabel('x = R / rs', fontsize=12)
    ax2.set_ylabel('f(x) [cumulative mass]', fontsize=12)
    ax2.set_title('Projected NFW (log-log)', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()

    plt.tight_layout()
    return fig

def plot_surface_density():
    """Plot the surface density profile Σ(R) ∝ df/dR."""

    # Create radius values in units of rs
    x = np.logspace(-2, 2, 1000)

    # Compute cumulative function (total mass within radius R)
    f_x = _f_proj_nfw(x)

    # Compute surface density via numerical derivative
    # Σ(R) = df/dR = surface mass density at radius R (mass per unit area)
    df_dx = np.gradient(f_x, x)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Surface density (mass per unit area at radius R)
    ax1.loglog(x, df_dx, 'g-', linewidth=2, label='Σ(R) = df/dx')
    ax1.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='x = 1 (r = rs)')
    ax1.set_xlabel('x = R / rs', fontsize=12)
    ax1.set_ylabel('df/dx [surface density: mass per unit area]', fontsize=12)
    ax1.set_title('Projected NFW Surface Density Σ(R)', fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()

    # Plot 2: Mass in annulus (for galaxy sampling)
    # For galaxy sampling, we need Σ(R) × 2πR = mass in annulus at radius R
    # Since we're in dimensionless units, this is proportional to df/dx × x
    mass_in_annulus = df_dx * x  # This is dM/d(ln R) = mass per logarithmic radius bin
    ax2.loglog(x, mass_in_annulus, 'm-', linewidth=2, label='df/dx × x = dM/d(ln R)')
    ax2.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='x = 1 (r = rs)')
    ax2.set_xlabel('x = R / rs', fontsize=12)
    ax2.set_ylabel('df/dx × x [mass in annulus at R]', fontsize=12)
    ax2.set_title('Mass in Annulus (∝ Σ(R) × 2πR)', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()

    plt.tight_layout()
    return fig

def plot_different_scale_radii():
    """Plot profiles for different physical scale radii."""

    # Different scale radii in arcsec
    rs_values = [10, 30, 50, 100]
    colors = plt.cm.viridis(np.linspace(0, 1, len(rs_values)))

    # Physical radius in arcsec
    r_arcsec = np.logspace(0, 3, 1000)  # 1 to 1000 arcsec

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for rs, color in zip(rs_values, colors):
        # Compute x = R/rs
        x = r_arcsec / rs

        # Compute cumulative function
        f_x = _f_proj_nfw(x)

        # Compute surface density
        df_dx = np.gradient(f_x, x)

        # Convert to physical units (arbitrary normalization)
        Sigma = df_dx / rs  # Surface density in units of mass per arcsec²

        ax1.loglog(r_arcsec, f_x, color=color, linewidth=2, label=f'rs = {rs}"')
        ax2.loglog(r_arcsec, Sigma, color=color, linewidth=2, label=f'rs = {rs}"')

    ax1.set_xlabel('Radius [arcsec]', fontsize=12)
    ax1.set_ylabel('Cumulative Mass F(R)', fontsize=12)
    ax1.set_title('Cumulative Mass for Different Scale Radii', fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend()

    ax2.set_xlabel('Radius [arcsec]', fontsize=12)
    ax2.set_ylabel('Surface Density Σ(R) [arbitrary units]', fontsize=12)
    ax2.set_title('Surface Density for Different Scale Radii', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()

    plt.tight_layout()
    return fig

def plot_truncation_effects():
    """Demonstrate the effect of truncating at different rmax."""

    rs = 30.0  # arcsec
    rmax_values = [50, 100, 200, 500]
    colors = plt.cm.plasma(np.linspace(0, 1, len(rmax_values)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Full profile (no truncation)
    r_full = np.logspace(0, 3, 1000)
    x_full = r_full / rs
    f_full = _f_proj_nfw(x_full)
    f_full_norm = f_full / f_full[-1]

    ax1.semilogx(r_full, f_full_norm, 'k-', linewidth=3, label='Full profile', alpha=0.5)

    for rmax, color in zip(rmax_values, colors):
        # Truncated profile
        r = np.logspace(0, np.log10(rmax), 500)
        x = r / rs
        f = _f_proj_nfw(x)

        # Normalize to rmax (truncated distribution)
        f_norm_trunc = f / f[-1]

        # CDF value at rmax in the full distribution
        cdf_at_rmax = np.interp(rmax, r_full, f_full_norm)

        ax1.semilogx(r, f_norm_trunc, color=color, linewidth=2,
                     label=f'Truncated at {rmax}" (CDF={cdf_at_rmax:.3f})')

        # Show fraction of mass within rmax
        ax2.bar(rmax, cdf_at_rmax, width=rmax*0.3, color=color, alpha=0.7,
                label=f'rmax={rmax}"')

    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% mass')
    ax1.set_xlabel('Radius [arcsec]', fontsize=12)
    ax1.set_ylabel('Normalized CDF', fontsize=12)
    ax1.set_title(f'Truncation Effects (rs = {rs}")', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_xlim(1, 1000)

    ax2.set_xlabel('rmax [arcsec]', fontsize=12)
    ax2.set_ylabel('Fraction of Total Mass', fontsize=12)
    ax2.set_title('Mass Fraction within rmax', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    return fig

def main():
    """Generate all plots."""

    print("="*80)
    print("PROJECTED NFW PROFILE VISUALIZATION")
    print("="*80)
    print("\nGenerating plots using _f_proj_nfw function...\n")

    # Generate plots
    print("1. Cumulative mass function...")
    fig1 = plot_cumulative_mass_function()
    fig1.savefig('nfw_cumulative_mass.png', dpi=150, bbox_inches='tight')
    print("   Saved: nfw_cumulative_mass.png")

    print("2. Surface density profile...")
    fig2 = plot_surface_density()
    fig2.savefig('nfw_surface_density.png', dpi=150, bbox_inches='tight')
    print("   Saved: nfw_surface_density.png")

    print("3. Different scale radii...")
    fig3 = plot_different_scale_radii()
    fig3.savefig('nfw_different_rs.png', dpi=150, bbox_inches='tight')
    print("   Saved: nfw_different_rs.png")

    print("4. Truncation effects...")
    fig4 = plot_truncation_effects()
    fig4.savefig('nfw_truncation_effects.png', dpi=150, bbox_inches='tight')
    print("   Saved: nfw_truncation_effects.png")

    print("\n" + "="*80)
    print("All plots generated successfully!")
    print("="*80)

    # Show plots
    plt.show()

if __name__ == "__main__":
    main()

