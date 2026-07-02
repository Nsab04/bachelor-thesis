# Plot Projected NFW Profile

## Overview

`plot_projected_nfw.py` - A visualization script for the projected NFW (Navarro-Frenk-White) profile using the `_f_proj_nfw` function from pyLensLib.

## Usage

```bash
cd /Users/maxmen3/projects/pyLensLib/apps
python3 plot_projected_nfw.py
```

## Output

The script generates 4 PNG files with comprehensive visualizations:

### 1. `nfw_cumulative_mass.png`
- **Left panel**: Cumulative mass function f(x) on linear scale
- **Right panel**: Same on log-log scale
- Shows how mass accumulates with radius in units of scale radius rs

### 2. `nfw_surface_density.png`
- **Left panel**: Surface density Σ(R) ∝ df/dx (derivative of cumulative mass)
- **Right panel**: Sampling PDF = Σ(R) × R (what's used for galaxy position sampling)
- Shows the radial density profile

### 3. `nfw_different_rs.png`
- **Left panel**: Cumulative mass for different scale radii (10", 30", 50", 100")
- **Right panel**: Surface density for different scale radii
- Demonstrates how the profile shape changes with rs

### 4. `nfw_truncation_effects.png`
- **Left panel**: Effect of truncating the profile at different rmax values
- **Right panel**: Fraction of total mass within each rmax
- Illustrates why truncation changes the distribution shape

## Key Insights Visualized

1. **NFW Profile Shape**: The characteristic NFW profile with:
   - Central cusp (density increases toward center)
   - Transition at r = rs
   - Asymptotic power-law decline at large radii

2. **Truncation Problem**: Shows why truncating at different rmax values creates different density distributions (relevant to the density conservation issue)

3. **Sampling PDF**: The actual probability distribution used for sampling galaxy positions (Σ(R) × 2πR)

4. **Scale Dependence**: How changing rs affects the profile in physical units

## Function Used

`_f_proj_nfw(x)` - Dimensionless cumulative mass function for projected NFW:
- **Input**: x = R/rs (radius in units of scale radius)
- **Output**: f(x) = cumulative projected mass within radius x
- **Formula**: Analytical expression with special functions (arccosh, arccos)

## Technical Details

- Uses matplotlib for visualization
- Logarithmic scales for wide dynamic range
- Color-coded for multiple parameters
- Numerical derivatives computed with numpy.gradient()

## Related to Density Conservation Work

This visualization helps understand:
- Why normalizing CDF to different rmax values changes sampling probabilities
- How the hybrid approach (fixed cluster size) maintains constant density
- The relationship between cumulative mass and surface density

---

Created: January 23, 2026
Part of the pyLensLib hybrid density conservation solution

