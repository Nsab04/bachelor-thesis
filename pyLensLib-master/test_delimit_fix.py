#!/usr/bin/env python3
"""Test script to verify getClMembDelimitingPoints returns correct coordinates"""

from pyLensLib.lenstool import getClMembers, getClMembDelimitingPoints, getFoV
import numpy as np
import matplotlib.pyplot as plt

parfile = './apps/test_multi/simulated_model_001.par'

# Get galaxies
df = getClMembers(parfile)
print(f"Total galaxies: {len(df)}")
print(f"Galaxy x range: [{df['x_centre'].min():.2f}, {df['x_centre'].max():.2f}]")
print(f"Galaxy y range: [{df['y_centre'].min():.2f}, {df['y_centre'].max():.2f}]")

# Get delimiting points
x1, x2 = getClMembDelimitingPoints(parfile, dmax=80)
print(f"\nDelimiting points: {len(x1)} vertices")
print(f"Delimiting x range: [{min(x1):.2f}, {max(x1):.2f}]")
print(f"Delimiting y range: [{min(x2):.2f}, {max(x2):.2f}]")

# Check they're in same coordinate system
gal_center_x = df['x_centre'].mean()
gal_center_y = df['y_centre'].mean()
delim_center_x = np.mean(x1)
delim_center_y = np.mean(x2)

print(f"\nGalaxy center: ({gal_center_x:.2f}, {gal_center_y:.2f})")
print(f"Delimiting center: ({delim_center_x:.2f}, {delim_center_y:.2f})")
print(f"Difference: ({abs(gal_center_x - delim_center_x):.2f}, {abs(gal_center_y - delim_center_y):.2f})")

# Plot to visualize
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plot galaxies
ax.scatter(df['x_centre'], df['y_centre'], s=10, c='cyan', alpha=0.5, label='Galaxies')

# Plot delimiting polygon
ax.plot(x1, x2, 'r-', linewidth=2, label='Delimiting polygon')
ax.fill(x1, x2, alpha=0.1, color='red')

ax.set_xlabel('x_centre (arcsec)')
ax.set_ylabel('y_centre (arcsec)')
ax.set_title('Galaxy Positions and Delimiting Polygon')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('test_delimit_fix.png', dpi=150)
print("\nPlot saved to test_delimit_fix.png")
plt.show()

