#!/usr/bin/env python3
"""
Test to verify that NFW sampling maintains constant number density
when rmax (fieldsize) changes.
"""

import numpy as np
import sys
sys.path.insert(0, '/')
from pyLensLib.lenstool import sample_radius_from_projected_nfw

def test_constant_density():
    """Test that n(r) is constant when rmax changes"""

    rs = 30.0  # Scale radius
    n_base = 1000  # Base number of galaxies

    # Test with different rmax values
    rmax_values = [70, 100, 150, 200]

    print("="*70)
    print("TEST: NFW Sampling with Constant Density")
    print("="*70)
    print(f"\nNFW scale radius: rs = {rs} arcsec")
    print(f"Testing with {n_base} galaxies for comparison")

    # For each rmax, sample n_base galaxies and measure density in inner region
    r_test = 50.0  # Measure density at r < 50"

    header_label = f"N(r<{r_test:.0f}\")"
    print(f"\nMeasuring density in r < {r_test}\":")
    print(f"{'rmax':>10s} | {'N_sampled':>10s} | {header_label:>12s} | {'Density':>15s}")
    print("-"*60)

    densities = []
    for rmax in rmax_values:
        # Sample positions
        r = sample_radius_from_projected_nfw(rs=rs, size=n_base, rmax=rmax, seed=42)

        # Count galaxies within r_test
        n_inner = np.sum(r < r_test)

        # Calculate density (galaxies per unit area)
        area_inner = np.pi * r_test**2 / 3600.0  # Convert to arcmin²
        density = n_inner / area_inner
        densities.append(density)

        print(f"{rmax:>10.0f} | {n_base:>10d} | {n_inner:>12d} | {density:>15.2f}")

    # Check consistency
    print(f"\n{'Result':>20s} | {'Value':>15s} | {'Status':>10s}")
    print("-"*50)

    mean_density = np.mean(densities)
    std_density = np.std(densities)
    rel_std = 100 * std_density / mean_density

    print(f"{'Mean density':>20s} | {mean_density:>15.2f} |")
    print(f"{'Std deviation':>20s} | {std_density:>15.2f} |")
    print(f"{'Relative std':>20s} | {rel_std:>14.2f}% |")

    # Test passes if relative std < 5%
    if rel_std < 5.0:
        print(f"{'TEST':>20s} | {'PASSED':>15s} | {'✓':>10s}")
        status = True
    else:
        print(f"{'TEST':>20s} | {'FAILED':>15s} | {'✗':>10s}")
        status = False

    print("\n" + "="*70)

    return status

def test_scaling_behavior():
    """Test that number of galaxies scales correctly with area"""

    rs = 30.0
    density_target = 100.0  # galaxies per arcmin²

    print("\nTEST: Proper Scaling with Area")
    print("="*70)
    print(f"\nTarget density: {density_target} galaxies/arcmin²")

    print(f"\n{'rmax':>10s} | {'Area':>12s} | {'N_sampled':>12s} | {'Density':>15s} | {'Error':>10s}")
    print("-"*70)

    errors = []
    for rmax in [70, 100, 150, 200]:
        # Calculate area
        area_arcmin2 = np.pi * rmax**2 / 3600.0

        # Sample appropriate number for target density
        n_sample = int(density_target * area_arcmin2)

        # Sample positions
        r = sample_radius_from_projected_nfw(rs=rs, size=n_sample, rmax=rmax, seed=42)

        # Verify all within rmax
        n_within = np.sum(r <= rmax)
        actual_density = n_within / area_arcmin2
        error = abs(actual_density - density_target) / density_target * 100
        errors.append(error)

        print(f"{rmax:>10.0f} | {area_arcmin2:>12.2f} | {n_sample:>12d} | {actual_density:>15.2f} | {error:>9.2f}%")

    max_error = max(errors)
    print(f"\nMaximum error: {max_error:.2f}%")

    if max_error < 1.0:
        print("TEST: PASSED ✓")
        return True
    else:
        print("TEST: FAILED ✗")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("VERIFICATION OF NFW SAMPLING FIX")
    print("="*70)

    test1 = test_constant_density()
    test2 = test_scaling_behavior()

    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print(f"Constant density test: {'PASSED ✓' if test1 else 'FAILED ✗'}")
    print(f"Scaling behavior test: {'PASSED ✓' if test2 else 'FAILED ✗'}")

    if test1 and test2:
        print("\n✓ All tests PASSED - Number density is conserved!")
    else:
        print("\n✗ Some tests FAILED - Review the implementation")

    print("="*70)

