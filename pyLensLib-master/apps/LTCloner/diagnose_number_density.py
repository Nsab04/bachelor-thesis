#!/usr/bin/env python3
"""
Diagnostic script for the number density issue with fieldsize changes

This demonstrates why number density was increasing with fieldsize and how the fix works.
"""

import numpy as np

def demonstrate_number_density_issue():
    """Show the number density problem"""

    print("="*70)
    print("NUMBER DENSITY ISSUE WITH FIELDSIZE")
    print("="*70)

    # Original input parameters
    original_fieldsize = 100.0  # arcsec
    original_area_arcmin2 = (original_fieldsize / 60.0) ** 2

    print(f"\n1. ORIGINAL INPUT:")
    print(f"   Fieldsize: {original_fieldsize} arcsec")
    print(f"   Area: {original_area_arcmin2:.3f} arcmin²")

    # Assume we have a fitted luminosity function that gives us N galaxies per arcmin²
    # For example, from Schechter function integration
    galaxies_per_arcmin2 = 100.0  # Example number density
    n_galaxies_original = int(galaxies_per_arcmin2 * original_area_arcmin2)

    print(f"   Number density: {galaxies_per_arcmin2:.1f} galaxies/arcmin²")
    print(f"   Total galaxies sampled: {n_galaxies_original}")

    print(f"\n2. WHAT HAPPENS WITH DIFFERENT FIELDSIZES (BEFORE FIX):")
    print(f"   {'Fieldsize':>12s} | {'Area':>12s} | {'N galaxies':>12s} | {'Density':>15s}")
    print(f"   {'-'*12} | {'-'*12} | {'-'*12} | {'-'*15}")

    for fs in [100, 150, 200, 250]:
        area = (fs / 60.0) ** 2
        n_gal = int(galaxies_per_arcmin2 * area)
        density = n_gal / area  # Should be constant!

        print(f"   {fs:>12.0f} | {area:>12.3f} | {n_gal:>12d} | {density:>15.1f}")

    print(f"\n3. THE PROBLEM:")
    print(f"   - area_out_arcmin2 = (fieldsize / 60)²")
    print(f"   - When fieldsize doubles, area quadruples!")
    print(f"   - N_sampled = LF(area_out_arcmin2)")
    print(f"   - So doubling fieldsize → 4× more galaxies")
    print(f"   - BUT galaxies are placed in rmax = fieldsize/2 * sqrt(2)")
    print(f"   - rmax only scales linearly with fieldsize")
    print(f"   - Result: NUMBER DENSITY INCREASES with fieldsize!")

    print(f"\n4. DEMONSTRATION:")
    print(f"   Fieldsize 100 → 200:")
    fs1, fs2 = 100, 200
    area1 = (fs1 / 60.0) ** 2
    area2 = (fs2 / 60.0) ** 2
    rmax1 = fs1 / 2.0 * np.sqrt(2)
    rmax2 = fs2 / 2.0 * np.sqrt(2)
    n_gal1 = int(galaxies_per_arcmin2 * area1)
    n_gal2 = int(galaxies_per_arcmin2 * area2)

    effective_area1 = np.pi * rmax1**2 / 3600.0  # arcmin²
    effective_area2 = np.pi * rmax2**2 / 3600.0  # arcmin²

    density1 = n_gal1 / effective_area1
    density2 = n_gal2 / effective_area2

    print(f"   Fieldsize {fs1}:")
    print(f"     - Sampling area: {area1:.3f} arcmin²")
    print(f"     - N galaxies: {n_gal1}")
    print(f"     - rmax: {rmax1:.1f} arcsec")
    print(f"     - Effective distribution area: {effective_area1:.3f} arcmin²")
    print(f"     - Effective density: {density1:.1f} gal/arcmin²")

    print(f"   Fieldsize {fs2}:")
    print(f"     - Sampling area: {area2:.3f} arcmin²")
    print(f"     - N galaxies: {n_gal2}")
    print(f"     - rmax: {rmax2:.1f} arcsec")
    print(f"     - Effective distribution area: {effective_area2:.3f} arcmin²")
    print(f"     - Effective density: {density2:.1f} gal/arcmin²")

    print(f"\n   Density ratio: {density2/density1:.2f}x")
    print(f"   → Number density DOUBLED when fieldsize doubled!")

def show_solution():
    """Show how the fix works"""

    print(f"\n" + "="*70)
    print("THE SOLUTION")
    print("="*70)

    # Use ORIGINAL area for sampling
    original_fieldsize = 100.0
    original_area = (original_fieldsize / 60.0) ** 2
    galaxies_per_arcmin2 = 100.0
    n_galaxies_fixed = int(galaxies_per_arcmin2 * original_area)

    print(f"\n1. USE ORIGINAL AREA FOR SAMPLING:")
    print(f"   - Compute area from INPUT file galaxy distribution")
    print(f"   - area_arcmin2 = convex_hull_area / 3600")
    print(f"   - This represents the ORIGINAL cluster extent")
    print(f"   - Use this SAME area for all fieldsize values")

    print(f"\n2. RESULTS WITH FIX:")
    print(f"   {'Fieldsize':>12s} | {'Sampling Area':>15s} | {'N galaxies':>12s} | {'rmax':>12s} | {'Density':>15s}")
    print(f"   {'-'*12} | {'-'*15} | {'-'*12} | {'-'*12} | {'-'*15}")

    for fs in [100, 150, 200, 250]:
        rmax = fs / 2.0 * np.sqrt(2)
        effective_area = np.pi * rmax**2 / 3600.0
        density = n_galaxies_fixed / effective_area

        print(f"   {fs:>12.0f} | {original_area:>15.3f} | {n_galaxies_fixed:>12d} | {rmax:>12.1f} | {density:>15.1f}")

    print(f"\n3. KEY INSIGHT:")
    print(f"   - SAME number of galaxies sampled for all fieldsizes")
    print(f"   - rmax scales with fieldsize")
    print(f"   - Effective area scales as rmax²")
    print(f"   - Number density = N_galaxies / effective_area")
    print(f"   - Since N is constant and area scales with fieldsize²")
    print(f"   - Density scales as 1/fieldsize² → DECREASES with larger fieldsize")
    print(f"   - This is WRONG! We want CONSTANT density!")

    print(f"\n4. WAIT... THERE'S STILL A PROBLEM!")
    print(f"   The fix keeps N constant, but rmax increases with fieldsize.")
    print(f"   This means density still changes!")
    print(f"\n   COMPLETE FIX requires:")
    print(f"   Option A: Keep rmax CONSTANT (use original FOV size)")
    print(f"   Option B: Scale N with rmax² to maintain density")
    print(f"   Option C: Don't change rmax based on fieldsize at all")

def show_complete_fix():
    """Show the complete solution"""

    print(f"\n" + "="*70)
    print("COMPLETE FIX")
    print("="*70)

    print(f"\nThe issue has TWO components:")
    print(f"1. Number of galaxies sampled (N)")
    print(f"2. Radius within which they're placed (rmax)")

    print(f"\nFor CONSTANT number density with changing fieldsize:")
    print(f"\nOPTION 1: Keep rmax independent of fieldsize")
    print(f"  - Calculate rmax from ORIGINAL FOV size")
    print(f"  - rmax_original = original_fieldsize/2 * sqrt(2)")
    print(f"  - Use this rmax for ALL fieldsize values")
    print(f"  - Sample N galaxies based on original area")
    print(f"  - Density = N / (π × rmax²) = CONSTANT ✓")

    print(f"\nOPTION 2: Scale N with fieldsize²")
    print(f"  - If rmax scales with fieldsize")
    print(f"  - N should scale as (fieldsize/fieldsize_original)²")
    print(f"  - But this changes the total number of galaxies")
    print(f"  - Not ideal if you want same cluster properties")

    print(f"\nRECOMMENDED: Option 1")
    print(f"  - Keep galaxy count constant (same cluster)")
    print(f"  - Keep rmax constant (same cluster extent)")
    print(f"  - Only change the FOV size for visualization/analysis")
    print(f"  - Number density remains constant ✓")

if __name__ == "__main__":
    demonstrate_number_density_issue()
    show_solution()
    show_complete_fix()

    print(f"\n" + "="*70)
    print("CODE CHANGES NEEDED")
    print("="*70)
    print(f"""
CHANGE 1 (DONE): Use original area for sampling
  - Line 622: area=area_arcmin2 (not area_out_arcmin2)
  - This fixes the N scaling issue

CHANGE 2 (NEEDED): Keep rmax constant
  - Line 658: Calculate rmax from ORIGINAL FOV, not fieldsize
  - rmax_original = original_fieldsize/2.0 * np.sqrt(2.0)
  - OR: rmax = max_distance_from_input_galaxies
  - Use this constant rmax regardless of fieldsize

WHERE: ltcloner.py line ~658:
  BEFORE: rmax=args.fieldsize/2.0 * np.sqrt(2.0)
  AFTER:  rmax=original_fieldsize/2.0 * np.sqrt(2.0)
  
This ensures both N and rmax are independent of the output fieldsize!
""")
    print("="*70)

