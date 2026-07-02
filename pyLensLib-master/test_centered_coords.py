#!/usr/bin/env python3
"""Test that delimiting points and galaxy positions are in the same centered coordinate system"""

from pyLensLib.lenstool import getClMembers, getClMembDelimitingPoints, getFoV
import numpy as np
import matplotlib.pyplot as plt

parfile = './apps/test_multi/simulated_model_001.par'
fieldsize = 100.0

# Get FOV
lims = getFoV(parfile)
xcen = 0.5 * (lims[1] + lims[0])
ycen = 0.5 * (lims[3] + lims[2])
print(f"FOV: {lims}")
print(f"FOV center: ({xcen}, {ycen})")

# Get galaxies and center them
df = getClMembers(parfile)
gal_x = df['x_centre'].values - xcen
gal_y = df['y_centre'].values - ycen
print(f"\nCentered galaxy positions:")
print(f"  x range: [{gal_x.min():.2f}, {gal_x.max():.2f}]")
print(f"  y range: [{gal_y.min():.2f}, {gal_y.max():.2f}]")
print(f"  mean: ({gal_x.mean():.2f}, {gal_y.mean():.2f})")

# Get delimiting points (already centered by the function)
x1, x2 = getClMembDelimitingPoints(parfile, dmax=80)
print(f"\nDelimiting points (centered):")
print(f"  x range: [{min(x1):.2f}, {max(x1):.2f}]")
print(f"  y range: [{min(x2):.2f}, {max(x2):.2f}]")
print(f"  mean: ({np.mean(x1):.2f}, {np.mean(x2):.2f})")

# Plot in centered coordinate system
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plot galaxies
ax.scatter(gal_x, gal_y, s=10, c='cyan', alpha=0.5, label='Galaxies (centered)')

# Plot delimiting polygon
ax.plot(x1, x2, 'r-', linewidth=2, label='Delimiting polygon (centered)')
ax.fill(x1, x2, alpha=0.1, color='red')

# Set limits to match image plane
ax.set_xlim(-fieldsize, fieldsize)
ax.set_ylim(-fieldsize, fieldsize)
ax.set_xlabel('x (arcsec, centered)')
ax.set_ylabel('y (arcsec, centered)')
ax.set_title('Galaxies and Delimiting Polygon in Centered Coordinates')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.axvline(0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('test_centered_coords.png', dpi=150)
print("\nPlot saved to test_centered_coords.png")
print("Both should now be in the same coordinate system!")
plt.show()

