#!/usr/bin/env python3
"""
CORRECT test for NFW density conservation.

The key insight: When fieldsize increases, you sample MORE galaxies (N ∝ area).
The test should verify that when you scale N proportionally with area,
the local density at each radius stays constant.
"""

import numpy as np
import sys
sys.path.insert(0, '/')
from pyLensLib.lenstool import sample_radius_from_projected_nfw

def test_correct_density_conservation():
    """
    CORRECT test: Scale N with area, check if local density is conserved
    """

    rs = 30.0  # Fixed observed scale radius
    density_target = 100.0  # Target density (galaxies per arcmin²)

    print("="*80)
    print("CORRECT TEST: Density Conservation with Scaling N")
    print("="*80)
    print(f"\nObserved NFW parameters:")
    print(f"  rs = {rs} arcsec (fixed, from observations)")
    print(f"  Target density = {density_target} galaxies/arcmin²")

    print(f"\nTest: Sample N ∝ area, check if n(r) is constant")

    # Test region
    r_test = 50.0  # Check density at r < 50"
    area_test_arcmin2 = np.pi * r_test**2 / 3600.0

    header1 = f"N(r<{r_test:.0f}\")"
    header2 = f"n(r<{r_test:.0f}\")"
    print(f"\nMeasuring density in test region r < {r_test}\":")
    print(f"{'rmax':>8s} | {'Area':>10s} | {'N_total':>10s} | {header1:>12s} | {header2:>12s} | {'Status':>10s}")
    print("-"*80)

    results = []
    for rmax in [70, 100, 150, 200]:
        # Calculate total area
        area_total_arcmin2 = np.pi * rmax**2 / 3600.0

        # Sample N proportional to total area (this is what your code does!)
        n_total = int(density_target * area_total_arcmin2)

        # Sample positions
        r = sample_radius_from_projected_nfw(rs=rs, size=n_total, rmax=rmax, seed=42)

        # Count in test region
        n_test = np.sum(r < r_test)

        # Calculate density in test region
        density_test = n_test / area_test_arcmin2

        # Check if close to target
        error = abs(density_test - density_target) / density_target * 100
        status = "✓ GOOD" if error < 10 else "✗ BAD"

        results.append(density_test)

        print(f"{rmax:>8.0f} | {area_total_arcmin2:>10.2f} | {n_total:>10d} | {n_test:>12d} | {density_test:>12.2f} | {status:>10s}")

    # Statistical analysis
    print(f"\n{'Metric':>25s} | {'Value':>12s} | {'Status':>10s}")
    print("-"*55)

    mean_density = np.mean(results)
    std_density = np.std(results)
    rel_std = 100 * std_density / mean_density
    max_deviation = 100 * max([abs(d - density_target)/density_target for d in results])

    print(f"{'Target density':>25s} | {density_target:>12.2f} |")
    print(f"{'Mean measured density':>25s} | {mean_density:>12.2f} |")
    print(f"{'Std deviation':>25s} | {std_density:>12.2f} |")
    print(f"{'Relative std (%)':>25s} | {rel_std:>12.2f} |")
    print(f"{'Max deviation from target':>25s} | {max_deviation:>11.2f}% |")

    # Test criteria
    success = rel_std < 10.0 and max_deviation < 20.0

    print(f"{'TEST RESULT':>25s} | {'PASSED ✓' if success else 'FAILED ✗':>12s} |")

    print("\n" + "="*80)

    return success

def compare_with_analytical():
    """
    Compare sampled distribution with analytical NFW surface density
    """
    print("\nCOMPARISON WITH ANALYTICAL NFW")
    print("="*80)

    rs = 30.0
    rmax = 150.0
    n_sample = 5000

    # Sample
    r_samples = sample_radius_from_projected_nfw(rs=rs, size=n_sample, rmax=rmax, seed=42)

    # Bin and measure density
    r_bins = np.linspace(0, rmax, 20)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    counts, _ = np.histogram(r_samples, bins=r_bins)

    # Area of each annulus
    areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
    densities_sampled = counts / areas

    # Analytical NFW surface density (proportional to df/dr × r)
    from pyLensLib.lenstool import _f_proj_nfw
    x_centers = r_centers / rs
    f_vals = _f_proj_nfw(x_centers)

    # Numerical derivative for surface density
    dr = r_centers[1] - r_centers[0]
    df_dr = np.gradient(f_vals, dr/rs) / rs  # Proper scaling
    surface_density_analytical = df_dr * r_centers

    # Normalize analytical to match sampled
    surface_density_analytical *= np.sum(densities_sampled) / np.sum(surface_density_analytical)

    # Compare
    print(f"\n{'Radius':>8s} | {'Sampled':>12s} | {'Analytical':>12s} | {'Ratio':>10s}")
    print("-"*50)

    for i in range(min(10, len(r_centers))):
        ratio = densities_sampled[i] / surface_density_analytical[i] if surface_density_analytical[i] > 0 else 0
        print(f"{r_centers[i]:>8.1f} | {densities_sampled[i]:>12.4f} | {surface_density_analytical[i]:>12.4f} | {ratio:>10.3f}")

    print("="*80)

if __name__ == "__main__":
    success = test_correct_density_conservation()
    compare_with_analytical()

    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {'ALL TESTS PASSED ✓' if success else 'TESTS FAILED - NEEDS MORE WORK ✗'}")
    print(f"{'='*80}\n")

