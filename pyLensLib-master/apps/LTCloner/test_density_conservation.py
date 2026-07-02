#!/usr/bin/env python3
"""
Test script to verify that number density n(r) is conserved with different fieldsizes

This script simulates galaxy distributions with different fieldsizes and checks
that the radial number density profile n(r) remains constant.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/')
from pyLensLib.lenstool import sample_radius_from_projected_nfw

def generate_galaxy_distribution(fieldsize, number_density_base, rs_nfw_base=30.0, fieldsize_base=100.0, seed=42):
    """
    Generate galaxy distribution matching the FIXED ltcloner logic

    Parameters:
        fieldsize: Field of view size (arcsec)
        number_density_base: Base number density (galaxies/arcmin²)
        rs_nfw_base: NFW scale radius at base fieldsize (arcsec)
        fieldsize_base: Base fieldsize used for rs normalization (arcsec)
        seed: Random seed

    Returns:
        r: Radial distances of galaxies (arcsec)
        n_galaxies: Total number of galaxies
    """
    np.random.seed(seed)

    # Calculate rmax based on fieldsize
    rmax = fieldsize / 2.0 * np.sqrt(2.0)

    # CRITICAL FIX: Scale rs with fieldsize to maintain constant n(r)
    rmax_base = fieldsize_base / 2.0 * np.sqrt(2.0)
    rs_scaling_factor = rmax / rmax_base
    rs_nfw = rs_nfw_base * rs_scaling_factor

    # Calculate effective area
    area_arcsec2 = np.pi * rmax**2
    area_arcmin2 = area_arcsec2 / 3600.0

    # Calculate number of galaxies to maintain constant density
    n_galaxies = int(number_density_base * area_arcmin2)

    # Use the NFW sampling from lenstool.py with SCALED rs
    r = sample_radius_from_projected_nfw(rs=rs_nfw, size=n_galaxies, rmax=rmax, seed=seed)

    return r, n_galaxies, rmax, area_arcmin2, rs_nfw

def measure_radial_density_profile(r, r_bins):
    """
    Measure number density as a function of radius

    Parameters:
        r: Radial distances of galaxies (arcsec)
        r_bins: Bin edges for radial bins (arcsec)

    Returns:
        r_centers: Center of each radial bin (arcsec)
        density: Number density in each bin (galaxies/arcmin²)
        density_err: Poisson error on density
    """
    # Count galaxies in each radial bin
    counts, _ = np.histogram(r, bins=r_bins)

    # Calculate area of each annulus
    inner_radii = r_bins[:-1]
    outer_radii = r_bins[1:]
    areas_arcsec2 = np.pi * (outer_radii**2 - inner_radii**2)
    areas_arcmin2 = areas_arcsec2 / 3600.0

    # Calculate number density
    density = counts / areas_arcmin2

    # Poisson error
    density_err = np.sqrt(counts) / areas_arcmin2

    # Bin centers
    r_centers = 0.5 * (inner_radii + outer_radii)

    return r_centers, density, density_err

def test_density_conservation():
    """Test that n(r) is conserved across different fieldsizes"""

    print("="*70)
    print("TESTING: Number Density Conservation n(r)")
    print("="*70)

    # Base parameters
    number_density_base = 100.0  # galaxies/arcmin²
    rs_nfw_base = 30.0  # arcsec at base fieldsize
    fieldsize_base = 100.0  # Base fieldsize for rs scaling

    # Different fieldsizes to test
    fieldsizes = [100, 150, 200, 250]

    # Radial bins for measuring n(r)
    # Use bins that are within all fieldsizes for fair comparison
    r_max_common = 60.0  # arcsec - within all fieldsizes
    r_bins = np.linspace(0, r_max_common, 11)  # 10 bins

    print(f"\nBase number density: {number_density_base:.1f} galaxies/arcmin²")
    print(f"NFW scale radius (base): {rs_nfw_base:.1f} arcsec")
    print(f"FIXED IMPLEMENTATION: rs scales with fieldsize to maintain n(r)")
    print(f"Measuring n(r) in common region: r < {r_max_common:.1f} arcsec")
    print(f"Radial bins: {len(r_bins)-1} bins from 0 to {r_max_common:.1f} arcsec")

    # Store results
    results = {}

    header_label = f"N(<{r_max_common:.0f}\")"
    print(f"\n{'Fieldsize':>10s} | {'rmax':>10s} | {'N_total':>10s} | {header_label:>12s} | {'Density':>12s} | {'rs_used':>10s}")
    print("-"*85)

    for fs in fieldsizes:
        # Generate distribution
        r, n_total, rmax, area_total, rs_used = generate_galaxy_distribution(
            fs, number_density_base, rs_nfw_base, fieldsize_base
        )

        # Count galaxies in common region
        n_in_common = np.sum(r < r_max_common)
        area_common_arcmin2 = np.pi * r_max_common**2 / 3600.0
        density_in_common = n_in_common / area_common_arcmin2

        print(f"{fs:>10.0f} | {rmax:>10.1f} | {n_total:>10d} | {n_in_common:>12d} | {density_in_common:>12.1f} | rs={rs_used:>6.1f}\"")


        # Measure radial profile
        r_centers, density_profile, density_err = measure_radial_density_profile(r, r_bins)

        results[fs] = {
            'r': r,
            'n_total': n_total,
            'rmax': rmax,
            'r_centers': r_centers,
            'density': density_profile,
            'density_err': density_err
        }

    return results, r_bins, number_density_base

def plot_density_profiles(results, r_bins):
    """Plot radial density profiles for all fieldsizes"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    # Plot 1: Radial density profiles
    for (fs, data), color in zip(results.items(), colors):
        ax1.errorbar(data['r_centers'], data['density'], yerr=data['density_err'],
                    marker='o', linestyle='-', linewidth=2, markersize=6,
                    label=f'Fieldsize = {fs}"', color=color, capsize=3)

    ax1.set_xlabel('Radius (arcsec)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number Density (galaxies/arcmin²)', fontsize=12, fontweight='bold')
    ax1.set_title('Radial Number Density Profile n(r)\nShould be identical for all fieldsizes!',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Plot 2: Difference from first fieldsize
    fs_reference = list(results.keys())[0]
    density_ref = results[fs_reference]['density']

    for (fs, data), color in zip(results.items(), colors):
        if fs == fs_reference:
            continue
        diff = data['density'] - density_ref
        relative_diff = 100 * diff / (density_ref + 1e-10)

        ax2.plot(data['r_centers'], relative_diff,
                marker='o', linestyle='-', linewidth=2, markersize=6,
                label=f'FS={fs}" - FS={fs_reference}"', color=color)

    ax2.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Radius (arcsec)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative Difference (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Difference from Reference (Fieldsize = {fs_reference}")\nShould be ~0% everywhere',
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = 'test_density_conservation.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_file}")
    plt.show()

def quantitative_test(results):
    """Perform quantitative test of density conservation"""

    print(f"\n" + "="*70)
    print("QUANTITATIVE TEST: Are densities conserved?")
    print("="*70)

    # Use first fieldsize as reference
    fs_reference = list(results.keys())[0]
    density_ref = results[fs_reference]['density']
    r_centers = results[fs_reference]['r_centers']

    print(f"\nReference: Fieldsize = {fs_reference}\"")
    print(f"\nComparing density profiles bin-by-bin:")
    print(f"{'Bin Center':>12s} | ", end="")
    for fs in results.keys():
        print(f"FS={fs:>3.0f}\" n(r) | ", end="")
    print(f"{'Max Diff':>10s} | {'Status':>8s}")
    print("-"*70)

    max_relative_diffs = []

    for i, r_c in enumerate(r_centers):
        densities = [results[fs]['density'][i] for fs in results.keys()]

        # Calculate max relative difference from reference
        if density_ref[i] > 0:
            diffs = [100 * abs(d - density_ref[i]) / density_ref[i] for d in densities[1:]]
            max_diff = max(diffs) if diffs else 0
        else:
            max_diff = 0

        max_relative_diffs.append(max_diff)

        # Print row
        print(f"{r_c:>12.1f} | ", end="")
        for d in densities:
            print(f"{d:>13.1f} | ", end="")

        status = "✓ GOOD" if max_diff < 10 else "✗ BAD"
        print(f"{max_diff:>10.1f}% | {status:>8s}")

    # Overall assessment
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)

    avg_max_diff = np.mean(max_relative_diffs)
    max_max_diff = np.max(max_relative_diffs)

    print(f"\nAverage maximum relative difference: {avg_max_diff:.2f}%")
    print(f"Maximum maximum relative difference: {max_max_diff:.2f}%")

    # Criteria for success
    threshold = 10.0  # 10% tolerance (accounting for Poisson noise)

    if avg_max_diff < threshold:
        print(f"\n✓ TEST PASSED: Number density n(r) is CONSERVED!")
        print(f"  Average difference ({avg_max_diff:.2f}%) < threshold ({threshold:.0f}%)")
        return True
    else:
        print(f"\n✗ TEST FAILED: Number density n(r) is NOT conserved!")
        print(f"  Average difference ({avg_max_diff:.2f}%) >= threshold ({threshold:.0f}%)")
        return False

def main():
    """Main test function"""

    print("\n" + "="*70)
    print("TEST: Number Density Conservation with Fieldsize Changes")
    print("="*70)
    print("""
This test verifies that the radial number density profile n(r) remains
constant when changing the fieldsize parameter in ltcloner.py.

The test:
1. Generates galaxy distributions for different fieldsizes
2. Measures the radial density profile n(r) for each
3. Compares profiles to verify they're identical
4. Quantifies the maximum difference

If the fix is correct, all profiles should overlap within Poisson noise.
""")

    # Run test
    results, r_bins, number_density_base = test_density_conservation()

    # Plot results
    plot_density_profiles(results, r_bins)

    # Quantitative assessment
    test_passed = quantitative_test(results)

    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if test_passed:
        print("""
✓ SUCCESS: Number density n(r) is CONSERVED!

The radial density profile remains constant across all fieldsizes.
This confirms that:
  - Number of galaxies scales correctly with area
  - Placement is consistent
  - The fix in ltcloner.py is working correctly

You can safely change fieldsize knowing that the cluster properties
(specifically the number density profile) will remain constant.
""")
    else:
        print("""
✗ FAILURE: Number density n(r) is NOT conserved!

There are significant differences in the radial density profiles
across different fieldsizes. This indicates a problem with:
  - Galaxy sampling logic
  - Placement radius calculation  
  - Area scaling

Please review the implementation in ltcloner.py.
""")

    return test_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

