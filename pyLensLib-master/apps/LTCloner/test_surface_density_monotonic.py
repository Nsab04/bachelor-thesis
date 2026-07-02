#!/usr/bin/env python3
"""
Test: Check if surface density is monotonic
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/')
from pyLensLib.lenstool import _f_proj_nfw

# Create x values
x = np.logspace(-2, 2, 1000)

# Compute cumulative function
f_x = _f_proj_nfw(x)

# Method 1: np.gradient (what we currently use)
df_dx_gradient = np.gradient(f_x, x)

# Method 2: Analytical - compute directly from f values
# For more accurate derivative with log-spaced data
dx = np.diff(x)
df = np.diff(f_x)
df_dx_diff = df / dx
x_mid = 0.5 * (x[:-1] + x[1:])

# Check monotonicity
is_monotonic_gradient = np.all(df_dx_gradient[1:] <= df_dx_gradient[:-1])
is_monotonic_diff = np.all(df_dx_diff[1:] <= df_dx_diff[:-1])

print("="*80)
print("SURFACE DENSITY MONOTONICITY TEST")
print("="*80)
print(f"\nUsing np.gradient: Monotonic decreasing? {is_monotonic_gradient}")
print(f"Using np.diff: Monotonic decreasing? {is_monotonic_diff}")

# Plot both methods
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Compare methods
ax1.loglog(x, df_dx_gradient, 'g-', linewidth=2, label='np.gradient', alpha=0.7)
ax1.loglog(x_mid, df_dx_diff, 'r--', linewidth=2, label='np.diff', alpha=0.7)
ax1.set_xlabel('x = R / rs')
ax1.set_ylabel('df/dx [surface density]')
ax1.set_title('Comparison of Derivative Methods')
ax1.legend()
ax1.grid(True, alpha=0.3, which='both')

# Right: Check for non-monotonic regions
ratio = df_dx_gradient[1:] / df_dx_gradient[:-1]
ax2.semilogx(x[:-1], ratio, 'b-', linewidth=2)
ax2.axhline(1.0, color='r', linestyle='--', label='Monotonic boundary')
ax2.set_xlabel('x = R / rs')
ax2.set_ylabel('Σ(x[i+1]) / Σ(x[i])')
ax2.set_title('Monotonicity Check (should be ≤ 1)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('surface_density_monotonicity_test.png', dpi=150)
print("\nSaved: surface_density_monotonicity_test.png")

# Find regions where it's not monotonic
if not is_monotonic_gradient:
    bad_indices = np.where(df_dx_gradient[1:] > df_dx_gradient[:-1])[0]
    print(f"\nNon-monotonic at {len(bad_indices)} points:")
    print(f"  x range: {x[bad_indices].min():.4f} to {x[bad_indices].max():.4f}")
    print(f"  Ratio range: {ratio[bad_indices].min():.6f} to {ratio[bad_indices].max():.6f}")

plt.show()

