#!/usr/bin/env python3
"""
Deep dive: Why doesn't rejection sampling work?
"""

import numpy as np
import sys
sys.path.insert(0, '/')
from pyLensLib.lenstool import _f_proj_nfw

rs = 30.0
r_test = 40.0
r_norm = 100.0 * rs  # 3000

print("="*80)
print("Understanding NFW CDF")
print("="*80)

# Build CDF
r_vals = np.logspace(np.log10(0.05), np.log10(r_norm), 5000)
x_vals = r_vals / rs
f_vals = _f_proj_nfw(x_vals)
f_vals_normalized = f_vals / f_vals[-1]

# Check values at key radii
for r in [40, 100, 200, 300, r_norm]:
    f_at_r = np.interp(r, r_vals, f_vals_normalized)
    print(f"CDF({r:>4.0f}\") = {f_at_r:.6f}")

print(f"\nNow let's see what happens when we sample...")
print("="*80)

for rmax in [100, 200, 300]:
    cdf_at_rmax = np.interp(rmax, r_vals, f_vals_normalized)
    cdf_at_rtest = np.interp(r_test, r_vals, f_vals_normalized)

    print(f"\nrmax = {rmax}")
    print(f"  CDF(rmax={rmax}) = {cdf_at_rmax:.6f}")
    print(f"  CDF(r_test={r_test}) = {cdf_at_rtest:.6f}")
    print(f"  Ratio: CDF({r_test})/CDF({rmax}) = {cdf_at_rtest/cdf_at_rmax:.6f}")
    print(f"  → If I sample N galaxies from [0, 1] and keep only r<{rmax},")
    print(f"     {cdf_at_rmax*100:.1f}% will be within rmax")
    print(f"     Of those, {(cdf_at_rtest/cdf_at_rmax)*100:.1f}% will be at r<{r_test}")

    # The problem: when we sample from [0,1] and reject, we're effectively
    # renormalizing the PDF within [0, rmax], which changes the shape!

print("\n" + "="*80)
print("THE PROBLEM:")
print("="*80)
print("When we sample from [0,1] and REJECT samples > rmax, we're implicitly")
print("creating a TRUNCATED NFW distribution, which has different shape!")
print("This is NOT the same as sampling from the untruncated NFW.")

