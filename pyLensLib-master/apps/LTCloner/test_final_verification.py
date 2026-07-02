#!/usr/bin/env python3
"""
Final verification: Check if the CDF normalization fix works
by comparing density profiles at the SAME radius for different rmax
"""

import numpy as np
import sys
sys.path.insert(0, '/')
from pyLensLib.lenstool import sample_radius_from_projected_nfw

def test_fixed_radius_density():
    """
    Test at a SINGLE fixed radius with different rmax values.
    If normalization is correct, density at r=40" should be constant.
    """

    rs = 30.0
    r_test = 40.0  # Test radius
    n_samples = 10000  # Large number for statistics

    print("="*80)
    print("VERIFICATION: Fixed Normalization Test")
    print("="*80)
    print(f"\nTest: Sample {n_samples} galaxies with different rmax")
    print(f"Measure: Fraction falling at r ≈ {r_test}\" (bin width = 2\")")
    print(f"Expected: Fraction should be constant (independent of rmax)")

    # Bin around r_test
    r_bin = [r_test - 1, r_test + 1]

    print(f"\n{'rmax':>8s} | {'N in bin':>12s} | {'Fraction':>12s} | {'Ratio to rmax=70':>18s}")
    print("-"*65)

    fractions = []
    for rmax in [70, 100, 150, 200, 300]:
        r = sample_radius_from_projected_nfw(rs=rs, size=n_samples, rmax=rmax, seed=42)
        n_in_bin = np.sum((r >= r_bin[0]) & (r < r_bin[1]))
        fraction = n_in_bin / n_samples
        fractions.append(fraction)

        ratio = fraction / fractions[0] if len(fractions) > 0 else 1.0
        print(f"{rmax:>8.0f} | {n_in_bin:>12d} | {fraction:>12.6f} | {ratio:>18.3f}")

    # Check consistency
    std = np.std(fractions)
    mean = np.mean(fractions)
    rel_std = 100 * std / mean

    print(f"\n{'Metric':>25s} | {'Value':>12s}")
    print("-"*40)
    print(f"{'Mean fraction':>25s} | {mean:>12.6f}")
    print(f"{'Std deviation':>25s} | {std:>12.6f}")
    print(f"{'Relative std (%)':>25s} | {rel_std:>12.2f}")

    success = rel_std < 5.0
    print(f"{'TEST RESULT':>25s} | {'PASSED ✓' if success else 'FAILED ✗':>12s}")

    print("="*80)
    return success

def test_density_scaling():
    """
    The REAL test: Does density scale correctly with N?
    """
    print("\nREAL TEST: Density Scaling")
    print("="*80)

    rs = 30.0
    r_test = 40.0

    # For each rmax, calculate expected number from NFW
    from pyLensLib.lenstool import _f_proj_nfw

    print(f"\nTest: For each rmax, calculate expected N from NFW integral")
    header_label = f"n at r={r_test:.0f}\""
    print(f"Then sample that many and check density at r={r_test}\"")

    print(f"\n{'rmax':>8s} | {'CDF(rmax)':>12s} | {'N expected':>12s} | {header_label:>15s} | {'Status':>10s}")
    print("-"*70)

    densities = []
    for rmax in [70, 100, 150, 200]:
        # Calculate what fraction of NFW mass is within rmax
        x_max = rmax / rs
        f_max = _f_proj_nfw(np.array([x_max]))[0]

        # Normalize to large radius
        x_norm = 1000
        f_norm = _f_proj_nfw(np.array([x_norm]))[0]
        cdf_at_rmax = f_max / f_norm

        # If we want density=100 gal/arcmin² in full area:
        area_arcmin2 = np.pi * rmax**2 / 3600.0
        n_expected = int(100 * area_arcmin2 * cdf_at_rmax)  # Scaled by CDF

        # Sample
        r = sample_radius_from_projected_nfw(rs=rs, size=n_expected, rmax=rmax, seed=42)

        # Measure density in annulus at r_test
        dr = 2.0
        mask = (r >= r_test - dr/2) & (r < r_test + dr/2)
        n_annulus = np.sum(mask)
        area_annulus = np.pi * ((r_test + dr/2)**2 - (r_test - dr/2)**2) / 3600.0
        density = n_annulus / area_annulus

        densities.append(density)
        status = "✓" if 80 < density < 120 else "✗"

        print(f"{rmax:>8.0f} | {cdf_at_rmax:>12.4f} | {n_expected:>12d} | {density:>15.2f} | {status:>10s}")

    mean_d = np.mean(densities)
    std_d = np.std(densities)
    rel_std = 100 * std_d / mean_d

    print(f"\nRelative std: {rel_std:.2f}%")
    success = rel_std < 15.0
    print(f"TEST: {'PASSED ✓' if success else 'FAILED ✗'}")

    print("="*80)
    return success

if __name__ == "__main__":
    test1 = test_fixed_radius_density()
    test2 = test_density_scaling()

    print(f"\n{'='*80}")
    if test1 and test2:
        print("✓ ALL TESTS PASSED - Normalization fix is working!")
    else:
        print("✗ TESTS FAILED - More investigation needed")
    print(f"{'='*80}\n")

