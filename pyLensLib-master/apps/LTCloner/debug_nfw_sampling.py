#!/usr/bin/env python3
"""
Debug: Why is density not conserved?
Let's trace through what's happening step by step.
"""

import numpy as np
import sys
sys.path.insert(0, '/')
from pyLensLib.lenstool import sample_radius_from_projected_nfw, _f_proj_nfw

# Test parameters
rs = 30.0  # Scale radius
r_test = 40.0  # Test at this radius

print("="*80)
print("DEBUG: Understanding the NFW Sampling")
print("="*80)

# Test with two different rmax values
for rmax in [100, 300]:
    print(f"\n{'='*80}")
    print(f"rmax = {rmax}")
    print(f"{'='*80}")

    # Reproduce what the function does
    r_norm = 100.0 * rs  # 3000
    print(f"r_norm = {r_norm}")

    # Build grid
    r_vals = np.logspace(np.log10(0.05), np.log10(r_norm), 5000)
    x_vals = r_vals / rs
    cdf_vals = _f_proj_nfw(x_vals)
    cdf_vals_unnorm = cdf_vals.copy()
    cdf_vals /= cdf_vals[-1]  # Normalize to 1

    print(f"CDF at r_norm ({r_norm}): {cdf_vals[-1]:.6f} (should be 1.0)")

    # Find CDF at specific radii
    cdf_at_rmax = np.interp(rmax, r_vals, cdf_vals)
    cdf_at_rtest = np.interp(r_test, r_vals, cdf_vals)

    print(f"CDF at rmax ({rmax}): {cdf_at_rmax:.6f}")
    print(f"CDF at r_test ({r_test}): {cdf_at_rtest:.6f}")

    # The key ratio
    print(f"\nKey ratio: CDF({r_test}) / CDF({rmax}) = {cdf_at_rtest/cdf_at_rmax:.6f}")
    print(f"This is the fraction of galaxies that land at r < {r_test}")
    print(f"when sampling {1000} galaxies within rmax={rmax}")

    # Sample
    n_sample = 10000
    u = np.random.uniform(0, cdf_at_rmax, n_sample)
    r_samples = np.interp(u, cdf_vals, r_vals)

    # Count at r_test
    n_at_rtest = np.sum(r_samples < r_test)
    frac_at_rtest = n_at_rtest / n_sample

    print(f"\nActual sampling:")
    print(f"N samples: {n_sample}")
    print(f"N at r<{r_test}: {n_at_rtest}")
    print(f"Fraction: {frac_at_rtest:.6f}")
    print(f"Expected fraction: {cdf_at_rtest/cdf_at_rmax:.6f}")
    print(f"Match: {'YES' if abs(frac_at_rtest - cdf_at_rtest/cdf_at_rmax) < 0.01 else 'NO'}")

    # Density calculation
    area_at_rtest = np.pi * r_test**2 / 3600.0  # arcmin²
    density_at_rtest = n_at_rtest / area_at_rtest
    print(f"\nDensity at r<{r_test}: {density_at_rtest:.2f} gal/arcmin²")

print(f"\n{'='*80}")
print("CONCLUSION:")
print("="*80)
print("If the density at r<40\" is the SAME for both rmax values,")
print("then the fix is working. If it's DIFFERENT, there's still an issue.")

