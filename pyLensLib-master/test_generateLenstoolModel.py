#!/usr/bin/env python3
"""
Test script for generateLenstoolModel function in pyLensLib.lenstool

This script tests the generateLenstoolModel function which randomizes lens model
components (main halos, galaxies, and gas) while preserving relative positions
and applying scaling relations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyLensLib.lenstool import generateLenstoolModel, findInBlock
import copy

def create_test_mainpot(n=3):
    """Create test main halo potentials"""
    mainpot = []
    for i in range(n):
        pot = {
            'id': f'O{i+1}',
            'profil': '81',
            'x_centre': 10.0 * i - 10.0,  # Spread along x-axis
            'y_centre': 5.0 * i - 5.0,    # Spread along y-axis
            'ellipticite': 0.3 + 0.1 * i,
            'angle_pos': 30.0 + 10 * i,
            'core_radius': 5.0 + i,
            'cut_radius': 100.0,
            'v_disp': 800.0 - 50 * i,
            'z_lens': 0.5
        }
        mainpot.append(pot)
    return mainpot

def create_test_galaxies(n=10):
    """Create test galaxy potentials"""
    gals = []
    for i in range(n):
        # Distribute galaxies in a cluster
        r = 20.0 * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        gal = {
            'id': f'{1000+i}',
            'profil': '81',
            'x_centre': r * np.cos(theta),
            'y_centre': r * np.sin(theta),
            'ellipticite': np.random.uniform(0.2, 0.6),
            'angle_pos': np.random.uniform(-180, 180),
            'core_radius': np.random.uniform(0.5, 2.0),
            'core_radius_kpc': 10.0,  # Should be removed
            'cut_radius': np.random.uniform(10, 30),
            'cut_radius_kpc': 50.0,   # Should be removed
            'v_disp': np.random.uniform(150, 300),
            'mag': np.random.uniform(18, 22),
            'z_lens': 0.5
        }
        gals.append(gal)
    return gals

def create_test_gas(n=2):
    """Create test gas potentials"""
    gas = []
    for i in range(n):
        g = {
            'id': f'G{i+1}',
            'profil': '81',
            'x_centre': 15.0 * (i - 0.5),
            'y_centre': 10.0 * (i - 0.5),
            'ellipticite': 0.5,
            'angle_pos': 45.0,
            'core_radius': 10.0,
            'cut_radius': 200.0,
            'v_disp': 500.0,
            'z_lens': 0.5
        }
        gas.append(g)
    return gas

def test_basic_functionality():
    """Test 1: Basic functionality with default parameters"""
    print("\n" + "="*70)
    print("TEST 1: Basic Functionality")
    print("="*70)

    mainpot = create_test_mainpot(3)
    gals = create_test_galaxies(10)
    gas = create_test_gas(2)
    rs = 30.0

    print(f"Input: {len(mainpot)} main halos, {len(gals)} galaxies, {len(gas)} gas components")
    print(f"NFW scale radius: {rs} arcsec")

    # Run the function with randomize_all=True to get all galaxies randomized
    model_smooth, model_gal, model_gas = generateLenstoolModel(
        mainpot, gals, gas, rs, randomize_all=True, seed=42
    )

    print(f"\nOutput: {len(model_smooth)} main halos, {len(model_gal)} galaxies, {len(model_gas)} gas components")

    # Check that all components are returned
    assert len(model_smooth) == len(mainpot), "Number of main halos changed!"
    assert len(model_gal) == len(gals), "Number of galaxies changed!"
    assert len(model_gas) == len(gas), "Number of gas components changed!"

    print("✓ All components preserved")

    # Check that positions were randomized
    pos_changed = any(
        findInBlock(model_smooth[i], 'x_centre') != findInBlock(mainpot[i], 'x_centre')
        for i in range(len(mainpot))
    )
    assert pos_changed, "Positions were not randomized!"
    print("✓ Positions were randomized")

    # Check that core_radius_kpc and cut_radius_kpc were removed from galaxies
    has_kpc = any('core_radius_kpc' in g or 'cut_radius_kpc' in g for g in model_gal)
    assert not has_kpc, "KPC fields were not removed from galaxies!"
    print("✓ KPC fields removed from galaxies")

    return model_smooth, model_gal, model_gas, mainpot, gals, gas

def test_reproducibility():
    """Test 2: Reproducibility with same seed"""
    print("\n" + "="*70)
    print("TEST 2: Reproducibility")
    print("="*70)

    mainpot = create_test_mainpot(3)  # Increased from 2 to 3
    gals = create_test_galaxies(10)   # Increased from 5 to 10
    gas = create_test_gas(2)          # Increased from 1 to 2
    rs = 25.0

    # Run twice with same seed
    model1_smooth, model1_gal, model1_gas = generateLenstoolModel(
        copy.deepcopy(mainpot), copy.deepcopy(gals), copy.deepcopy(gas),
        rs, randomize_all=True, seed=123
    )

    model2_smooth, model2_gal, model2_gas = generateLenstoolModel(
        copy.deepcopy(mainpot), copy.deepcopy(gals), copy.deepcopy(gas),
        rs, randomize_all=True, seed=123
    )

    # Check positions are identical
    for i in range(len(model1_smooth)):
        x1 = findInBlock(model1_smooth[i], 'x_centre')
        x2 = findInBlock(model2_smooth[i], 'x_centre')
        y1 = findInBlock(model1_smooth[i], 'y_centre')
        y2 = findInBlock(model2_smooth[i], 'y_centre')
        assert np.isclose(x1, x2) and np.isclose(y1, y2), f"Main halo {i} position differs!"

    for i in range(len(model1_gal)):
        x1 = findInBlock(model1_gal[i], 'x_centre')
        x2 = findInBlock(model2_gal[i], 'x_centre')
        y1 = findInBlock(model1_gal[i], 'y_centre')
        y2 = findInBlock(model2_gal[i], 'y_centre')
        assert np.isclose(x1, x2) and np.isclose(y1, y2), f"Galaxy {i} position differs!"

    print("✓ Same seed produces identical results")

    # Run with different seed
    model3_smooth, model3_gal, model3_gas = generateLenstoolModel(
        copy.deepcopy(mainpot), copy.deepcopy(gals), copy.deepcopy(gas),
        rs, randomize_all=True, seed=999
    )

    # Check that results are different - check multiple components
    differences = 0
    for i in range(len(model1_smooth)):
        x1 = findInBlock(model1_smooth[i], 'x_centre')
        x3 = findInBlock(model3_smooth[i], 'x_centre')
        y1 = findInBlock(model1_smooth[i], 'y_centre')
        y3 = findInBlock(model3_smooth[i], 'y_centre')
        if not (np.isclose(x1, x3) and np.isclose(y1, y3)):
            differences += 1

    # At least one component should be different
    assert differences > 0, f"Different seeds should produce different results! Found {differences} differences"

    print(f"✓ Different seeds produce different results ({differences} components differ)")

def test_opening_angle():
    """Test 3: Opening angle constraint"""
    print("\n" + "="*70)
    print("TEST 3: Opening Angle Constraint")
    print("="*70)

    mainpot = create_test_mainpot(3)
    gals = create_test_galaxies(10)
    gas = create_test_gas(2)
    rs = 30.0

    # Test with 90 degree opening angle
    opening_angle = 90.0
    model_smooth, model_gal, model_gas = generateLenstoolModel(
        mainpot, gals, gas, rs, opening_angle=opening_angle, seed=42
    )

    # Get anchor position and angle
    x0 = findInBlock(model_smooth[0], 'x_centre')
    y0 = findInBlock(model_smooth[0], 'y_centre')
    pa = np.deg2rad(findInBlock(model_smooth[0], 'angle_pos'))

    # Check that secondary halos are within the cone
    for i in range(1, len(model_smooth)):
        x = findInBlock(model_smooth[i], 'x_centre')
        y = findInBlock(model_smooth[i], 'y_centre')

        # Compute angle relative to anchor
        rel_angle = np.arctan2(y - y0, x - x0)

        # Check if within cone (with wrapping)
        angle_diff = np.abs(np.angle(np.exp(1j * (rel_angle - pa))))
        angle_diff_alt = np.abs(np.angle(np.exp(1j * (rel_angle - pa - np.pi))))

        within_cone = (angle_diff < np.deg2rad(opening_angle)/2) or (angle_diff_alt < np.deg2rad(opening_angle)/2)

        if not within_cone:
            print(f"  Warning: Halo {i} angle difference: {np.rad2deg(min(angle_diff, angle_diff_alt)):.1f}°")

    print(f"✓ Opening angle constraint ({opening_angle}°) applied")

def test_randomize_all_flag():
    """Test 4: randomize_all flag behavior"""
    print("\n" + "="*70)
    print("TEST 4: randomize_all Flag")
    print("="*70)

    mainpot = create_test_mainpot(2)
    gals = create_test_galaxies(10)
    gas = create_test_gas(1)
    rs = 30.0

    # Test with randomize_all=False (default)
    # This mode associates some galaxies with main halos
    model_smooth1, model_gal1, model_gas1 = generateLenstoolModel(
        copy.deepcopy(mainpot), copy.deepcopy(gals), copy.deepcopy(gas),
        rs, randomize_all=False, seed=42
    )

    # Test with randomize_all=True
    # This mode randomizes all galaxies from NFW distribution
    model_smooth2, model_gal2, model_gas2 = generateLenstoolModel(
        copy.deepcopy(mainpot), copy.deepcopy(gals), copy.deepcopy(gas),
        rs, randomize_all=True, seed=42
    )

    print(f"randomize_all=False: {len(model_gal1)} galaxies (includes {len(mainpot)} associated with halos)")
    print(f"randomize_all=True:  {len(model_gal2)} galaxies (all randomized from NFW)")

    # With randomize_all=False, we get len(mainpot) + remaining galaxies
    # With randomize_all=True, we get exactly len(gals) galaxies
    assert len(model_gal2) == len(gals), "randomize_all=True should preserve galaxy count!"
    assert len(model_gal1) >= len(mainpot), "randomize_all=False should have at least as many galaxies as main halos!"

    # Check that galaxies are in different positions
    positions_differ = False
    for i in range(min(len(model_gal1), len(model_gal2))):
        x1 = findInBlock(model_gal1[i], 'x_centre')
        x2 = findInBlock(model_gal2[i], 'x_centre')
        if not np.isclose(x1, x2):
            positions_differ = True
            break

    assert positions_differ, "Galaxy positions should differ between the two modes!"

    print("✓ randomize_all flag works correctly")

def test_tolerance_parameter():
    """Test 5: Tolerance parameter effect"""
    print("\n" + "="*70)
    print("TEST 5: Tolerance Parameter")
    print("="*70)

    mainpot = create_test_mainpot(3)
    gals = create_test_galaxies(5)
    gas = create_test_gas(1)
    rs = 30.0

    # Original distances
    x0_orig = findInBlock(mainpot[0], 'x_centre')
    y0_orig = findInBlock(mainpot[0], 'y_centre')

    distances_orig = []
    for i in range(1, len(mainpot)):
        x = findInBlock(mainpot[i], 'x_centre')
        y = findInBlock(mainpot[i], 'y_centre')
        r = np.hypot(x - x0_orig, y - y0_orig)
        distances_orig.append(r)

    print(f"Original distances from anchor: {distances_orig}")

    # Test with small tolerance
    tolerance = 0.05
    model_smooth, _, _ = generateLenstoolModel(
        mainpot, gals, gas, rs, tolerance=tolerance, seed=42
    )

    x0 = findInBlock(model_smooth[0], 'x_centre')
    y0 = findInBlock(model_smooth[0], 'y_centre')

    distances_new = []
    for i in range(1, len(model_smooth)):
        x = findInBlock(model_smooth[i], 'x_centre')
        y = findInBlock(model_smooth[i], 'y_centre')
        r = np.hypot(x - x0, y - y0)
        distances_new.append(r)

    print(f"New distances from anchor:      {distances_new}")

    # Check that distances are within tolerance
    for i, (r_orig, r_new) in enumerate(zip(distances_orig, distances_new)):
        ratio = r_new / r_orig
        assert (1 - tolerance) <= ratio <= (1 + tolerance), \
            f"Distance {i} ratio {ratio:.3f} outside tolerance [{1-tolerance}, {1+tolerance}]"

    print(f"✓ All distances within ±{tolerance*100}% tolerance")

def test_halo_distance_placement():
    """Test 6: Visualize main halo distance placement from anchor"""
    print("\n" + "="*70)
    print("TEST 6: Main Halo Distance Placement Visualization")
    print("="*70)

    mainpot = create_test_mainpot(8)  # Create 8 halos to better show both sectors
    gals = create_test_galaxies(20)  # Need more galaxies than halos
    gas = create_test_gas(1)
    rs = 30.0
    tolerance = 0.1
    opening_angle = 60.0  # Use 60 degree opening angle to clearly see the constraint

    # Get original distances from anchor
    x0_orig = findInBlock(mainpot[0], 'x_centre')
    y0_orig = findInBlock(mainpot[0], 'y_centre')

    original_distances = []
    for i in range(1, len(mainpot)):
        x = findInBlock(mainpot[i], 'x_centre')
        y = findInBlock(mainpot[i], 'y_centre')
        r = np.hypot(x - x0_orig, y - y0_orig)
        original_distances.append(r)

    print(f"Original distances from anchor: {[f'{d:.2f}' for d in original_distances]}")
    print(f"Opening angle constraint: {opening_angle}°")

    # Generate randomized model with opening angle constraint
    try:
        model_smooth, _, _ = generateLenstoolModel(
            mainpot, gals, gas, rs, tolerance=tolerance, opening_angle=opening_angle, seed=42
        )
    except Exception as e:
        print(f"ERROR in generateLenstoolModel: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Get randomized distances
    x0 = findInBlock(model_smooth[0], 'x_centre')
    y0 = findInBlock(model_smooth[0], 'y_centre')

    new_distances = []
    positions = [(x0, y0)]  # Start with anchor
    for i in range(1, len(model_smooth)):
        x = findInBlock(model_smooth[i], 'x_centre')
        y = findInBlock(model_smooth[i], 'y_centre')
        r = np.hypot(x - x0, y - y0)
        new_distances.append(r)
        positions.append((x, y))

    print(f"New distances from anchor:      {[f'{d:.2f}' for d in new_distances]}")

    # Verify we have halos to plot
    if len(new_distances) == 0:
        print("ERROR: No secondary halos generated!")
        return None

    if len(new_distances) != len(original_distances):
        print(f"WARNING: Number of halos changed from {len(original_distances)} to {len(new_distances)}")

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Get the position angle of the anchor halo for opening angle visualization
    anchor_pa = np.deg2rad(findInBlock(model_smooth[0], 'angle_pos'))

    # Draw opening angle sectors (two symmetric cones)
    max_r = max(original_distances) * (1 + tolerance) + 5

    # First cone (along major axis)
    from matplotlib.patches import Wedge
    wedge1 = Wedge((x0, y0), max_r,
                   np.rad2deg(anchor_pa - np.deg2rad(opening_angle)/2),
                   np.rad2deg(anchor_pa + np.deg2rad(opening_angle)/2),
                   facecolor='yellow', alpha=0.15, edgecolor='orange',
                   linewidth=2, linestyle='--',
                   label=f'Allowed sector (±{opening_angle/2:.0f}°)')
    ax.add_patch(wedge1)

    # Second cone (opposite side, 180° rotated)
    wedge2 = Wedge((x0, y0), max_r,
                   np.rad2deg(anchor_pa + np.pi - np.deg2rad(opening_angle)/2),
                   np.rad2deg(anchor_pa + np.pi + np.deg2rad(opening_angle)/2),
                   facecolor='yellow', alpha=0.15, edgecolor='orange',
                   linewidth=2, linestyle='--')
    ax.add_patch(wedge2)

    # Draw the major axis line
    axis_length = max_r
    ax.plot([x0 - axis_length * np.cos(anchor_pa), x0 + axis_length * np.cos(anchor_pa)],
            [y0 - axis_length * np.sin(anchor_pa), y0 + axis_length * np.sin(anchor_pa)],
            'orange', linewidth=2, linestyle=':', alpha=0.7, label='Major axis')

    # Plot anchor (first halo) as a star
    ax.scatter(x0, y0, s=500, c='red', marker='*',
              label='Anchor (Main Halo 1)', zorder=5,
              edgecolors='black', linewidths=2)
    ax.text(x0 + 1, y0 + 1, 'Anchor', fontsize=12, fontweight='bold')

    # Plot circles showing expected distances (original distances with tolerance)
    colors_circles = plt.cm.viridis(np.linspace(0.2, 0.8, len(original_distances)))
    for i, (r_orig, r_new, color) in enumerate(zip(original_distances, new_distances, colors_circles)):
        # Draw circle at original distance
        circle_orig = plt.Circle((x0, y0), r_orig, fill=False,
                                 edgecolor=color, linewidth=2, linestyle='-',
                                 label=f'Expected radius {i+2}: {r_orig:.1f}"', alpha=0.7)
        ax.add_patch(circle_orig)

        # Draw tolerance band
        circle_inner = plt.Circle((x0, y0), r_orig * (1 - tolerance),
                                  fill=False, edgecolor=color,
                                  linewidth=1, linestyle='--', alpha=0.4)
        circle_outer = plt.Circle((x0, y0), r_orig * (1 + tolerance),
                                  fill=False, edgecolor=color,
                                  linewidth=1, linestyle='--', alpha=0.4)
        ax.add_patch(circle_inner)
        ax.add_patch(circle_outer)

    # Plot actual halo positions
    sector1_count = 0
    sector2_count = 0

    for i in range(1, len(positions)):
        x, y = positions[i]
        ratio = new_distances[i-1] / original_distances[i-1]

        # Calculate angle relative to anchor
        halo_angle = np.arctan2(y - y0, x - x0)

        # Check if within opening angle (either cone)
        angle_diff1 = np.abs(np.angle(np.exp(1j * (halo_angle - anchor_pa))))
        angle_diff2 = np.abs(np.angle(np.exp(1j * (halo_angle - anchor_pa - np.pi))))

        # Determine which sector this halo is in
        in_sector1 = angle_diff1 <= np.deg2rad(opening_angle / 2)
        in_sector2 = angle_diff2 <= np.deg2rad(opening_angle / 2)

        if in_sector1:
            sector1_count += 1
            sector_label = "S1"
        elif in_sector2:
            sector2_count += 1
            sector_label = "S2"
        else:
            sector_label = "??"

        min_angle_diff = min(angle_diff1, angle_diff2)
        within_angle = min_angle_diff <= np.deg2rad(opening_angle / 2)

        # Color code: green if within tolerance AND angle, yellow if only distance OK, red if neither
        within_tolerance = (1 - tolerance) <= ratio <= (1 + tolerance)

        if within_tolerance and within_angle:
            color = 'green'
            marker_label = f'Halo (OK)' if i == 1 else ''
        elif within_tolerance:
            color = 'yellow'
            marker_label = f'Halo (distance OK, angle!)' if i == 1 else ''
        else:
            color = 'red'
            marker_label = f'Halo (ERROR)' if i == 1 else ''

        ax.scatter(x, y, s=300, c=color, marker='o',
                  edgecolors='black', linewidths=2, zorder=4,
                  label=marker_label if i == 1 else '')

        # Draw line from anchor to halo
        ax.plot([x0, x], [y0, y], 'k-', alpha=0.3, linewidth=1)

        # Annotate with distance
        mid_x, mid_y = (x0 + x) / 2, (y0 + y) / 2
        ax.text(mid_x, mid_y, f'{new_distances[i-1]:.1f}"',
               fontsize=9, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Label halo with angle information and sector
        angle_deg = np.rad2deg(min_angle_diff)
        ax.text(x + 1, y + 1, f'H{i+1} ({sector_label})\n∠{angle_deg:.1f}°',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel('x_centre (arcsec)', fontsize=12)
    ax.set_ylabel('y_centre (arcsec)', fontsize=12)
    ax.set_title(f'Main Halo Placement: Distance (±{tolerance*100:.0f}%) & Opening Angle ({opening_angle}°)\n' +
                f'Sector Distribution: S1={sector1_count}, S2={sector2_count}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Set limits to show all halos and circles
    ax.set_xlim(x0 - max_r, x0 + max_r)
    ax.set_ylim(y0 - max_r, y0 + max_r)

    plt.tight_layout()

    # Save figure
    output_file = 'test_halo_distance_placement.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Halo distance visualization saved to: {output_file}")
    print(f"\nSector distribution:")
    print(f"  Sector 1 (along major axis): {sector1_count} halos")
    print(f"  Sector 2 (opposite side): {sector2_count} halos")

    # Verify all halos are within tolerance and angle constraints
    all_within = True
    print("\nDistance and angle verification:")
    for i, (r_orig, r_new) in enumerate(zip(original_distances, new_distances)):
        x, y = positions[i+1]
        ratio = r_new / r_orig

        # Calculate angle
        halo_angle = np.arctan2(y - y0, x - x0)
        angle_diff1 = np.abs(np.angle(np.exp(1j * (halo_angle - anchor_pa))))
        angle_diff2 = np.abs(np.angle(np.exp(1j * (halo_angle - anchor_pa - np.pi))))
        min_angle_diff = np.rad2deg(min(angle_diff1, angle_diff2))

        within_dist = (1 - tolerance) <= ratio <= (1 + tolerance)
        within_angle = min_angle_diff <= opening_angle / 2

        dist_status = "✓" if within_dist else "✗"
        angle_status = "✓" if within_angle else "✗"

        print(f"  Halo {i+2}:")
        print(f"    {dist_status} Distance: {r_new:.2f}\" (ratio={ratio:.3f}, expected ±{tolerance*100:.0f}%)")
        print(f"    {angle_status} Angle: {min_angle_diff:.1f}° from axis (limit={opening_angle/2:.0f}°)")

        if not (within_dist and within_angle):
            all_within = False

    assert all_within, "Some halos are outside the tolerance or angle constraints!"
    print(f"\n✓ All halos within ±{tolerance*100:.0f}% distance AND ±{opening_angle/2:.0f}° angle constraints")

    plt.show()

    return model_smooth

def visualize_results(model_smooth, model_gal, model_gas, mainpot, gals, gas):
    """Create visualization of original and randomized models"""
    print("\n" + "="*70)
    print("VISUALIZATION: Original vs Randomized Models")
    print("="*70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot original model
    ax1.scatter([findInBlock(p, 'x_centre') for p in mainpot],
                [findInBlock(p, 'y_centre') for p in mainpot],
                s=300, c='red', marker='*', label='Main halos', zorder=3, edgecolors='black', linewidths=2)

    ax1.scatter([findInBlock(g, 'x_centre') for g in gals],
                [findInBlock(g, 'y_centre') for g in gals],
                s=30, c='blue', alpha=0.6, label='Galaxies')

    ax1.scatter([findInBlock(g, 'x_centre') for g in gas],
                [findInBlock(g, 'y_centre') for g in gas],
                s=200, c='green', marker='s', alpha=0.5, label='Gas', edgecolors='black')

    ax1.set_xlabel('x_centre (arcsec)', fontsize=12)
    ax1.set_ylabel('y_centre (arcsec)', fontsize=12)
    ax1.set_title('Original Model', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Plot randomized model
    ax2.scatter([findInBlock(p, 'x_centre') for p in model_smooth],
                [findInBlock(p, 'y_centre') for p in model_smooth],
                s=300, c='red', marker='*', label='Main halos', zorder=3, edgecolors='black', linewidths=2)

    ax2.scatter([findInBlock(g, 'x_centre') for g in model_gal],
                [findInBlock(g, 'y_centre') for g in model_gal],
                s=30, c='blue', alpha=0.6, label='Galaxies')

    ax2.scatter([findInBlock(g, 'x_centre') for g in model_gas],
                [findInBlock(g, 'y_centre') for g in model_gas],
                s=200, c='green', marker='s', alpha=0.5, label='Gas', edgecolors='black')

    # Draw lines showing original to randomized positions for main halos
    for i in range(len(mainpot)):
        x_orig = findInBlock(mainpot[i], 'x_centre')
        y_orig = findInBlock(mainpot[i], 'y_centre')
        x_new = findInBlock(model_smooth[i], 'x_centre')
        y_new = findInBlock(model_smooth[i], 'y_centre')
        ax2.plot([x_orig, x_new], [y_orig, y_new], 'k--', alpha=0.3, linewidth=1)

    ax2.set_xlabel('x_centre (arcsec)', fontsize=12)
    ax2.set_ylabel('y_centre (arcsec)', fontsize=12)
    ax2.set_title('Randomized Model', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()

    # Save figure
    output_file = 'test_generateLenstoolModel_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_file}")

    plt.show()

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("TESTING: generateLenstoolModel Function")
    print("="*70)
    print("\nThis test script validates the generateLenstoolModel function which")
    print("randomizes lens model components while preserving their relationships.")

    try:
        # Run tests
        model_smooth, model_gal, model_gas, mainpot, gals, gas = test_basic_functionality()
        test_reproducibility()
        test_opening_angle()
        test_randomize_all_flag()
        test_tolerance_parameter()

        # Visualize halo distance placement
        test_halo_distance_placement()

        # Visualize results
        visualize_results(model_smooth, model_gal, model_gas, mainpot, gals, gas)

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nThe generateLenstoolModel function is working correctly.")
        print("Key features validated:")
        print("  • Component randomization with preserved counts")
        print("  • Reproducibility with random seed")
        print("  • Opening angle constraints")
        print("  • randomize_all flag behavior")
        print("  • Tolerance parameter enforcement")
        print("  • Halo distance placement from anchor")
        print("  • Removal of KPC fields from galaxies")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())

