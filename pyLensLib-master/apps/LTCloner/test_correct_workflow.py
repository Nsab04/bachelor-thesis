#!/usr/bin/env python3
"""
CORRECT TEST: Sample N ∝ area, then check if density is conserved
Testing: number density (galaxies per unit area) at a specific radius should be constant
"""

import numpy as np
import sys
sys.path.insert(0, '/')
from pyLensLib.lenstool import sample_radius_from_projected_nfw

rs = 30.0
density_target = 100.0  # gal/arcmin²
r_test = 40.0  # Test at this radius
dr = 5.0  # Annulus width

print("="*80)
print("CORRECT TEST: N ∝ area, check local number density n(r)")
print("="*80)
print(f"Target overall density: {density_target} gal/arcmin²")
print(f"Testing LOCAL density in annulus at r = {r_test}\" ± {dr/2}\"")
print(f"Annulus: {r_test-dr/2:.1f}\" < r < {r_test+dr/2:.1f}\"")
print()

results = []
for rmax in [100, 200, 300, 400]:
    # Calculate total area
    area_total_arcmin2 = np.pi * rmax**2 / 3600.0

    # Sample N proportional to total area
    n_sample = int(density_target * area_total_arcmin2)

    # Sample positions
    r = sample_radius_from_projected_nfw(rs=rs, size=n_sample, rmax=rmax, seed=42)

    # Measure LOCAL density in ANNULUS at r_test
    in_annulus = (r >= r_test - dr/2) & (r < r_test + dr/2)
    n_in_annulus = np.sum(in_annulus)

    # Area of annulus
    area_annulus = np.pi * ((r_test + dr/2)**2 - (r_test - dr/2)**2) / 3600.0  # arcmin²

    # Local density (galaxies per unit area at radius r_test)
    local_density = n_in_annulus / area_annulus

    results.append(local_density)

    print(f"rmax={rmax:>3.0f}: N_total={n_sample:>5d}, N_in_annulus={n_in_annulus:>4d}, "
          f"n(r={r_test}\")={local_density:>7.2f} gal/arcmin²")

print()
print("="*80)
mean_density = np.mean(results)
std_density = np.std(results)
rel_std = 100 * std_density / mean_density

print(f"Mean local density: {mean_density:.2f} gal/arcmin²")
print(f"Std deviation: {std_density:.2f} gal/arcmin²")
print(f"Relative std: {rel_std:.2f}%")
print()

if rel_std < 10:
    print("✓ LOCAL DENSITY IS CONSERVED!")
else:
    print("✗ LOCAL DENSITY IS NOT CONSERVED")
print("="*80)

