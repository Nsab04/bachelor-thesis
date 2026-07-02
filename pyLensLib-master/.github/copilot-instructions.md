# pyLensLib AI Coding Instructions

## Project Overview
pyLensLib simulates gravitational lensing effects at multiple scales (microlensing, galaxy lensing, cluster lensing). It's a scientific Python package focused on astrophysics simulations, particularly strong lensing by galaxy clusters and galaxy-galaxy strong lensing (GGSL).

## Core Architecture

### Class Hierarchy & Inheritance
- **Base Classes:**
  - `genlen` (`pyLensLib/genlen.py`): Base class for lens models. Handles grid setup, critical lines, caustics computation via `setGrid0()`. All deflector models inherit from this.
  - `gensrc` (`pyLensLib/gensrc.py`): Base class for source models. Handles ray tracing via `ray_trace()` method. All source types inherit from this.

- **Lens Models** (inherit from `genlen`):
  - `deflector`: Loads pre-computed deflection angle maps from FITS files
  - `piemd`: Pseudo-Isothermal Elliptical Mass Distribution (analytical)
  - `sie`: Singular Isothermal Ellipsoid (analytical)
  - `compositeModel`: Combines multiple lens models additively

- **Source Models** (inherit from `gensrc`):
  - `sersic`: Extended source with Sersic profile (n=4 default)
  - `sersic_numba`: Numba-optimized version with `@jit(nopython=True, parallel=True)` decorators
  - `pointsrc`: Point source with image-finding algorithms

- **Observational Classes:**
  - `observation`: Handles PSF convolution, photon noise, background, mag↔counts conversions
  - `CriticalLine` & `Caustic` (`critcau.py`): Store critical line/caustic geometries as shapely Polygons

### Key Workflow Pattern (see `GUIDE-Strong-Lensing-by-Galaxy-Cluster.md`)
1. **Create deflector**: Load deflection maps → `deflector(co, angx, angy, zl, zs)`
2. **Set grid**: Call `deflector.setGrid0(theta=...)` to compute κ, γ, α maps on source plane
3. **Compute critical lines**: `genlen.tancl()` finds zero-contours of magnification
4. **Map to caustics**: `genlen.mapCrit2Cau()` uses lens equation β = θ - α(θ)
5. **Create source**: Position source near caustic → `sersic(gl=deflector, ys1=..., ys2=...)`
6. **Ray trace**: Source automatically ray-traces through `gensrc.ray_trace()` during initialization
7. **Add observational effects**: Use `observation.convolve()` for PSF, `addNoise()` for realism

## Coordinate Conventions
- **Image plane**: (θ₁, θ₂) or (theta1, theta2) in arcsec
- **Source plane**: (y1, y2) or (ys1, ys2) in arcsec
- **Deflection angles**: (α₁, α₂) stored as (angx, angy) - can be maps or computed from potential
- **Lens equation**: β = θ - α(θ) implemented in `genlen.mapCrit2Cau()`
- Pixel coordinates use meshgrid convention: `theta1, theta2 = np.meshgrid(thetax, thetay)`

## Critical Code Patterns

### Creating a Lens System
```python
from astropy.cosmology import FlatLambdaCDM
from pyLensLib.deflector import deflector
import astropy.io.fits as pyfits

co = FlatLambdaCDM(Om0=0.3, H0=70.0)
angx_map = pyfits.getdata('deflection_x.fits')
angy_map = pyfits.getdata('deflection_y.fits')
df = deflector(co, angx=angx_map, angy=angy_map, zl=0.5, zs=2.0)

# Set grid covering source plane
fov = 100.0  # arcsec
npix = 500
theta = np.linspace(-fov/2, fov/2, npix)
df.setGrid0(theta=theta)
```

### Multi-Redshift Sources
Sources at different redshifts auto-rescale deflections:
```python
# In sersic.__init__ and pointsrc.__init__:
if self.zs != gl.zs:
    ds = gl.co.angular_diameter_distance(self.zs).value
    dls = gl.co.angular_diameter_distance_z1z2(gl.zl, self.zs).value
    self.rescf = dls/ds * gl.ds/gl.dls  # rescaling factor
```

### Grid Compatibility
Deflector grid determines source grid size. When creating sources:
- Use same `fov` and `npix` as deflector for direct mapping
- Or use `pcx`, `pcy` arrays for custom non-square grids
- Pixel scale must match: `theta[1] - theta[0]` == `pixel_size`

## Tools & Utilities

### lenstool.py Integration
- `getClMembers(best_par='best.par', proftype="81")`: Parse LENSTOOL output files for galaxy catalogs
- `findInBlock()`: Extract parameters from LENSTOOL parameter blocks
- Commonly used with PIEMD profiles (proftype="81") for cluster member galaxies

### SED Modeling (`sedmodel.py`)
- Manages spectral energy distributions for multi-band simulations
- Redshift shifts SEDs using `barak.sed` package (energy conservation: flux scales as 1/(1+z))
- Convert mag→counts→flux: `observation.mag2counts()` requires zero-point for each filter

## Installation & Environment
```bash
pip install .  # from pyLensLib/ directory
```

**Key Dependencies:**
- Scientific: `numpy`, `scipy`, `astropy`
- Performance: `numba` (for `sersic_numba`), `scikit-image`
- Lensing-specific: `shapely`, `geopandas` (critical line geometry)
- SED tools: `barak` (install separately if doing multi-band sims)
- Optional: `PyQt6` (for `lens_interact_qt.py` GUI), `lenstronomy` (alternative image finding)

## Testing & Examples
- **No formal test suite** - validation done via example scripts in `Test/`
- Key examples:
  - `Test/GGSLsims.ipynb`: Full GGSL simulation workflow (follows guide exactly)
  - `Test/simulateGGSL.py`: Production-level GGSL simulations
  - `Test/test_*.py`: Individual module tests (e.g., `test_deflector.py`, `test_sersic.py`)
  - `apps/generate_lens_models.py`: Large-scale model generation pipeline

## Common Pitfalls
1. **Grid mismatch**: Deflector and source grids must have identical `fov` and `npix` for direct lens mapping
2. **Redshift ordering**: Ensure zs > zl, code checks but may silently set `rescf=0.0`
3. **Deflection map units**: Gradients computed assuming pixel scale in header; verify `dtheta` in `deflector.kappa()`
4. **Numba optimization**: Use `sersic_numba` for production; regular `sersic` for debugging (no JIT overhead)
5. **Critical line indexing**: `idcl=0` is largest area, not necessarily "main" cluster critical line

## File Organization
- `pyLensLib/`: Core library modules (lens models, sources, utilities)
- `apps/`: Production scripts for batch processing (e.g., `generate_lens_models.py`)
- `Test/`: Examples, validation scripts, notebooks (not pytest tests)
- `docs/`: Sphinx documentation (incomplete, refer to code docstrings)

## Conventions
- Docstrings use Google style with `Args:` and `Returns:` sections
- Class attributes documented in class-level docstring
- Redshifts: `zl` (lens), `zs` (source), `zsnorm` (normalization reference, often 1.0)
- Cosmology object (`co`) passed explicitly, typically `FlatLambdaCDM(H0=70, Om0=0.3)`
