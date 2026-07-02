#!/usr/bin/env python3
"""
Diagnostic script to demonstrate the fieldsize inconsistency issue in ltcloner.py

The problem: When fieldsize changes, the galaxy distribution doesn't scale consistently
because the FOV coordinate system and the galaxy placement coordinate system have
different centers.
"""

import numpy as np

def demonstrate_issue():
    """Demonstrate the fieldsize inconsistency"""

    print("="*70)
    print("FIELDSIZE INCONSISTENCY ISSUE")
    print("="*70)

    # Original input .par file has FOV like [0, 100] x [0, 100]
    # Galaxies are positioned relative to this FOV
    original_fov_min = 0.0
    original_fov_max = 100.0
    original_fov_center = (original_fov_max + original_fov_min) / 2.0

    print(f"\n1. ORIGINAL INPUT .PAR FILE:")
    print(f"   FOV: [{original_fov_min}, {original_fov_max}] x [{original_fov_min}, {original_fov_max}]")
    print(f"   FOV center: ({original_fov_center}, {original_fov_center})")

    # Example galaxy positions from input file (absolute coordinates)
    galaxy_positions = np.array([
        [10.0, 20.0],   # Galaxy 1
        [30.0, 40.0],   # Galaxy 2
        [50.0, 50.0],   # Galaxy 3 (at FOV center)
        [70.0, 60.0],   # Galaxy 4
        [90.0, 80.0],   # Galaxy 5
    ])

    print(f"\n2. GALAXY POSITIONS FROM INPUT (absolute coordinates):")
    for i, pos in enumerate(galaxy_positions):
        print(f"   Galaxy {i+1}: ({pos[0]:.1f}, {pos[1]:.1f})")

    # In ltcloner, the champ block is overridden to be centered at (0, 0)
    fieldsize_100 = 100.0
    new_champ_100 = {
        'xmin': -fieldsize_100 / 2.0,  # -50
        'xmax': fieldsize_100 / 2.0,   # +50
        'ymin': -fieldsize_100 / 2.0,  # -50
        'ymax': fieldsize_100 / 2.0,   # +50
    }

    print(f"\n3. LTCLONER SETS CHAMP (fieldsize=100):")
    print(f"   champ['xmin'] = {new_champ_100['xmin']:.1f}")
    print(f"   champ['xmax'] = {new_champ_100['xmax']:.1f}")
    print(f"   champ['ymin'] = {new_champ_100['ymin']:.1f}")
    print(f"   champ['ymax'] = {new_champ_100['ymax']:.1f}")
    print(f"   NEW FOV center: (0.0, 0.0)")

    print(f"\n4. THE PROBLEM:")
    print(f"   - Input galaxies are in [0, 100] coordinate system (center at 50)")
    print(f"   - Output FOV is [-50, 50] coordinate system (center at 0)")
    print(f"   - Galaxy positions are NOT adjusted to match the new coordinate system!")
    print(f"   - They keep their original absolute coordinates")

    print(f"\n5. WHAT HAPPENS:")
    print(f"   Original galaxy at (50, 50) [center of old FOV]")
    print(f"   is still at (50, 50) in new FOV")
    print(f"   but new FOV center is at (0, 0)")
    print(f"   so the galaxy appears SHIFTED by (+50, +50)!")

    print(f"\n6. WHEN FIELDSIZE CHANGES TO 200:")
    fieldsize_200 = 200.0
    new_champ_200 = {
        'xmin': -fieldsize_200 / 2.0,  # -100
        'xmax': fieldsize_200 / 2.0,   # +100
        'ymin': -fieldsize_200 / 2.0,  # -100
        'ymax': fieldsize_200 / 2.0,   # +100
    }
    print(f"   champ['xmin'] = {new_champ_200['xmin']:.1f}")
    print(f"   champ['xmax'] = {new_champ_200['xmax']:.1f}")
    print(f"   FOV is now [-100, 100] but galaxies are still at [0, 100] coordinates")
    print(f"   The distribution looks different because the FOV changed but positions didn't!")

    print(f"\n7. THE INCONSISTENCY:")
    print(f"   - rmax in assignGalaxiesFromNFW is set to: fieldsize/2 * sqrt(2)")
    print(f"   - For fieldsize=100: rmax = 70.7 arcsec")
    print(f"   - For fieldsize=200: rmax = 141.4 arcsec")
    print(f"   - NEW galaxies are distributed within rmax from (x0, y0)")
    print(f"   - But (x0, y0) is from the ORIGINAL input coordinates!")
    print(f"   - If original main halo was at (50, 50), new galaxies are placed")
    print(f"     within 70.7 arcsec of (50, 50) for fieldsize=100")
    print(f"     OR within 141.4 arcsec of (50, 50) for fieldsize=200")
    print(f"   - This is INCONSISTENT because the coordinate system changed!")

    print(f"\n" + "="*70)
    print(f"SOLUTION:")
    print(f"="*70)
    print(f"\nThe code needs to CENTER the input galaxy positions when changing fieldsize.")
    print(f"Option 1: Center input galaxies to match the new FOV center")
    print(f"   - Read original FOV from input .par file")
    print(f"   - Compute original center: (xmin + xmax)/2, (ymin + ymax)/2")
    print(f"   - Subtract this center from all galaxy positions")
    print(f"   - Now galaxies are in centered coordinates matching new champ")
    print(f"\nOption 2: Use relative coordinates from the start")
    print(f"   - Keep galaxies in centered coordinates throughout")
    print(f"   - Don't change the coordinate system between input and output")
    print(f"\nOption 3: Don't override the champ block")
    print(f"   - Keep the original FOV from the input file")
    print(f"   - Scale rmax based on the ORIGINAL FOV size, not the new fieldsize")

def show_code_location():
    """Show where the problem occurs in the code"""
    print(f"\n" + "="*70)
    print(f"CODE LOCATIONS:")
    print(f"="*70)
    print(f"\n1. ltcloner.py lines 526-529: FOV is overridden to be centered")
    print(f"   champ['xmax'] = args.fieldsize / 2.0")
    print(f"   champ['xmin'] = -args.fieldsize / 2.0")
    print(f"   champ['ymax'] = args.fieldsize / 2.0")
    print(f"   champ['ymin'] = -args.fieldsize / 2.0")
    print(f"\n2. ltcloner.py line 622: rmax scales with fieldsize")
    print(f"   rmax=args.fieldsize/2.0 * np.sqrt(2.0)")
    print(f"\n3. lenstool.py assignGalaxiesFromNFW: Galaxies placed around main_center")
    print(f"   - main_center comes from the input file (not centered)")
    print(f"   - New galaxies are placed within rmax of main_center")
    print(f"   - This creates inconsistent distribution when fieldsize changes")
    print(f"\n4. The mismatch:")
    print(f"   - Input coordinates: Use original FOV (e.g., [0, 100])")
    print(f"   - Output coordinates: Use centered FOV (e.g., [-50, 50])")
    print(f"   - No transformation applied to galaxy positions!")

if __name__ == "__main__":
    demonstrate_issue()
    show_code_location()

    print(f"\n" + "="*70)
    print(f"RECOMMENDATIONS:")
    print(f"="*70)
    print(f"\n1. IMMEDIATE FIX: Center the galaxy coordinates")
    print(f"   After reading galaxies from input file, center them:")
    print(f"   ```python")
    print(f"   original_fov = getFoV(args.parfile)")
    print(f"   xcen = 0.5 * (original_fov[1] + original_fov[0])")
    print(f"   ycen = 0.5 * (original_fov[3] + original_fov[2])")
    print(f"   ")
    print(f"   for g in gals:")
    print(f"       g['x_centre'] = float(g['x_centre']) - xcen")
    print(f"       g['y_centre'] = float(g['y_centre']) - ycen")
    print(f"   ```")
    print(f"\n2. LONG-TERM: Use consistent coordinate system")
    print(f"   Always use centered coordinates: [-L/2, L/2] x [-L/2, L/2]")
    print(f"   This matches the physics (lens centered at origin)")
    print(f"\n3. DOCUMENT: Add clear comments about coordinate conventions")
    print(f"   Specify whether coordinates are absolute or centered")
    print(f"="*70)

